"""
Reports — generate and retrieve analysis artifacts.

Reports are written to OUTPUT_DIR (/output in the container).
The agent can also read them from the mounted volume.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from service.config import settings
from service.main import get_dataset
from service.models import ReportInfo, ReportList, ReportRequest, SchemaResponse

router = APIRouter(tags=["reports"])

# In-memory registry of generated reports
_report_registry: dict[str, ReportInfo] = {}


@router.post("/reports", response_model=ReportInfo)
async def generate_report(req: ReportRequest):
    """Generate an analysis report and write it to the output volume."""
    ds = get_dataset()
    report_id = uuid.uuid4().hex[:12]
    out_dir = settings.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if req.type == "summary":
        data = _build_summary(ds)
    elif req.type == "full":
        data = _build_full(ds)
    elif req.type == "route":
        if not req.route_code:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=400,
                content={"detail": "route_code is required for type=route"},
            )
        data = _build_route(ds, req.route_code)
    else:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": f"Unknown report type: {req.type}"})

    # Write JSON report
    path = out_dir / f"{report_id}.json"
    path.write_text(json.dumps(data, indent=2, default=str))

    info = ReportInfo(
        report_id=report_id,
        type=req.type,
        status="complete",
        path=path.as_posix(),
    )
    _report_registry[report_id] = info
    return info


@router.get("/reports", response_model=ReportList)
async def list_reports():
    """List all generated reports."""
    # Also scan output dir for reports from batch mode
    out_dir = settings.output_dir / "reports"
    if out_dir.exists():
        for f in out_dir.glob("*.json"):
            rid = f.stem
            if rid not in _report_registry:
                _report_registry[rid] = ReportInfo(
                    report_id=rid,
                    type="unknown",
                    status="complete",
                    path=f.as_posix(),
                )
    return ReportList(reports=list(_report_registry.values()))


@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a generated report by ID."""
    # Check registry first
    if report_id in _report_registry:
        path = Path(_report_registry[report_id].path)
        if path.exists():
            return json.loads(path.read_text())

    # Fall back to disk scan
    path = settings.output_dir / "reports" / f"{report_id}.json"
    if path.exists():
        return json.loads(path.read_text())

    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={"detail": f"Report {report_id} not found"})


@router.get("/schema", response_model=SchemaResponse)
async def dataset_schema():
    """Return the dataset schema documentation (data/SCHEMA.md)."""
    schema_path = settings.data_dir / "data" / "SCHEMA.md"
    if not schema_path.exists():
        # Try alternate location
        schema_path = settings.data_dir / "SCHEMA.md"

    if schema_path.exists():
        content = schema_path.read_text()
    else:
        content = "SCHEMA.md not found in data directory."

    return SchemaResponse(
        title="Traffic Monitor Lizard — Dataset Schema",
        description="Column definitions, data types, relationships, and quality rules",
        content=content,
    )


# ── Report builders ─────────────────────────────────────────────────


def _build_summary(ds) -> dict:
    """Compact summary: route rankings, date range, row counts."""
    df = ds.traffic_df
    rrs = ds.analyzer.calculate_rrs(days_rolling=10)
    labels = dict(zip(ds.routes_df["route_code"], ds.routes_df["label_short"]))

    return {
        "type": "summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "city": ds.city_config.get("display_name"),
        "total_rows": len(df),
        "routes": ds.route_count,
        "date_range": {
            "from": str(df["date"].min()),
            "to": str(df["date"].max()),
        },
        "rrs_rankings": [
            {
                "rank": i + 1,
                "route_code": row["route_code"],
                "label_short": labels.get(row["route_code"], ""),
                "points": round(float(row["points"]), 1),
            }
            for i, (_, row) in enumerate(rrs.iterrows())
        ],
    }


def _build_full(ds) -> dict:
    """Full report: summary + per-route percentiles + anomalies."""
    summary = _build_summary(ds)
    df = ds.traffic_df

    route_details = []
    for rc in ds.routes:
        subset = df[df["route_code"] == rc]
        if subset.empty:
            continue
        route_details.append({
            "route_code": rc,
            "count": len(subset),
            "duration": {
                "mean": round(float(subset["duration"].mean()), 2),
                "median": round(float(subset["duration"].median()), 2),
                "p10": round(float(subset["duration"].quantile(0.10)), 2),
                "p90": round(float(subset["duration"].quantile(0.90)), 2),
            },
            "avg_speed": {
                "mean": round(float(subset["avg_speed"].mean()), 2),
                "median": round(float(subset["avg_speed"].median()), 2),
            },
        })

    summary["type"] = "full"
    summary["route_details"] = route_details
    return summary


def _build_route(ds, route_code: str) -> dict:
    """Single-route deep dive."""
    df = ds.traffic_df[ds.traffic_df["route_code"] == route_code]
    labels = dict(zip(ds.routes_df["route_code"], ds.routes_df["label_short"]))

    if df.empty:
        return {"type": "route", "route_code": route_code, "error": "no data"}

    hourly = df.groupby("hour")["duration"].agg(["mean", "median", "count"]).reset_index()

    return {
        "type": "route",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_code": route_code,
        "label_short": labels.get(route_code, ""),
        "total_readings": len(df),
        "date_range": {"from": str(df["date"].min()), "to": str(df["date"].max())},
        "duration": {
            "mean": round(float(df["duration"].mean()), 2),
            "median": round(float(df["duration"].median()), 2),
            "std": round(float(df["duration"].std()), 2),
            "p10": round(float(df["duration"].quantile(0.10)), 2),
            "p90": round(float(df["duration"].quantile(0.90)), 2),
        },
        "avg_speed": {
            "mean": round(float(df["avg_speed"].mean()), 2),
            "median": round(float(df["avg_speed"].median()), 2),
        },
        "hourly_profile": [
            {"hour": int(row["hour"]), "mean_duration": round(float(row["mean"]), 2), "count": int(row["count"])}
            for _, row in hourly.iterrows()
        ],
    }
