"""
Batch CLI — generate analysis reports without running the HTTP server.

Usage:
  TRAFFIC_MODE=batch python -m service.cli
  TRAFFIC_MODE=batch python -m service.cli --type full
  TRAFFIC_MODE=batch python -m service.cli --type route --route-code "XJG4+7J|5PX4+HQ"
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from service.config import settings
from service.data_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Traffic analysis batch report generator")
    parser.add_argument(
        "--type",
        choices=["summary", "full", "route"],
        default="summary",
        help="Report type to generate",
    )
    parser.add_argument("--route-code", help="Route code (required for type=route)")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--days-rolling", type=int, default=10, help="R3S2 rolling window")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("batch")

    output_dir = Path(args.output_dir) if args.output_dir else settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset from %s ...", settings.data_dir)
    try:
        ds = load_dataset(settings.data_dir, settings.city_config)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    report_id = uuid.uuid4().hex[:12]

    # Build report based on type
    if args.type == "summary":
        data = _build_summary(ds, args.days_rolling)
    elif args.type == "full":
        data = _build_full(ds, args.days_rolling)
    elif args.type == "route":
        if not args.route_code:
            logger.error("--route-code is required for type=route")
            sys.exit(1)
        data = _build_route(ds, args.route_code)
    else:
        logger.error("Unknown report type: %s", args.type)
        sys.exit(1)

    # Write report
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{report_id}.json"
    out_path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Report written to %s", out_path)

    # Write openapi.json if running from API container
    openapi_path = output_dir / "openapi.json"
    if not openapi_path.exists():
        logger.info("(openapi.json not generated — run the API server first)")

    print(f"report_id={report_id}")
    print(f"path={out_path}")


def _build_summary(ds, days_rolling: int) -> dict:
    df = ds.traffic_df
    rrs = ds.analyzer.calculate_rrs(days_rolling=days_rolling)
    labels = dict(zip(ds.routes_df["route_code"], ds.routes_df["label_short"]))
    return {
        "type": "summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "city": ds.city_config.get("display_name"),
        "total_rows": len(df),
        "routes": ds.route_count,
        "date_range": {"from": str(df["date"].min()), "to": str(df["date"].max())},
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


def _build_full(ds, days_rolling: int) -> dict:
    summary = _build_summary(ds, days_rolling)
    df = ds.traffic_df
    route_details = []
    for rc in ds.routes:
        subset = df[df["route_code"] == rc]
        if subset.empty:
            continue
        route_details.append({
            "route_code": rc,
            "count": len(subset),
            "duration_mean": round(float(subset["duration"].mean()), 2),
            "duration_median": round(float(subset["duration"].median()), 2),
            "duration_p10": round(float(subset["duration"].quantile(0.10)), 2),
            "duration_p90": round(float(subset["duration"].quantile(0.90)), 2),
            "avg_speed_mean": round(float(subset["avg_speed"].mean()), 2),
        })
    summary["type"] = "full"
    summary["route_details"] = route_details
    return summary


def _build_route(ds, route_code: str) -> dict:
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
        "duration_mean": round(float(df["duration"].mean()), 2),
        "duration_median": round(float(df["duration"].median()), 2),
        "avg_speed_mean": round(float(df["avg_speed"].mean()), 2),
        "hourly_profile": [
            {"hour": int(r["hour"]), "mean_duration": round(float(r["mean"]), 2), "count": int(r["count"])}
            for _, r in hourly.iterrows()
        ],
    }


if __name__ == "__main__":
    main()
