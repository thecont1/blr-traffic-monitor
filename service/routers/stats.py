"""
Statistical analysis — percentiles, distributions, R³S² scores.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Query

from service.main import get_dataset
from service.models import (
    DistributionBin,
    DistributionResponse,
    PercentileResponse,
    RRSResponse,
    RRSRoute,
)

router = APIRouter(tags=["stats"])


@router.get("/stats/percentiles", response_model=PercentileResponse)
async def percentiles(
    route_code: str = Query(..., description="Route code"),
    metric: str = Query("duration", description="Metric: duration or avg_speed"),
    percentiles: str = Query(
        "10,25,50,75,90", description="Comma-separated percentile values"
    ),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """Compute percentile distribution for a route metric."""
    ds = get_dataset()
    df = ds.traffic_df[ds.traffic_df["route_code"] == route_code]

    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]

    if df.empty:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": f"No data for route_code={route_code}"},
        )

    pcts = [float(p.strip()) for p in percentiles.split(",")]
    values = np.percentile(df[metric].dropna(), pcts)
    pct_dict = {f"p{int(p)}": round(float(v), 2) for p, v in zip(pcts, values)}

    return PercentileResponse(
        route_code=route_code,
        metric=metric,
        percentiles=pct_dict,
        count=len(df),
    )


@router.get("/stats/distribution", response_model=DistributionResponse)
async def distribution(
    route_code: str = Query(..., description="Route code"),
    metric: str = Query("duration", description="Metric: duration or avg_speed"),
    bins: int = Query(20, ge=5, le=100, description="Number of histogram bins"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """Compute histogram distribution for a route metric."""
    ds = get_dataset()
    df = ds.traffic_df[ds.traffic_df["route_code"] == route_code]

    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]

    if df.empty:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": f"No data for route_code={route_code}"},
        )

    counts, edges = np.histogram(df[metric].dropna(), bins=bins)
    bin_list = [
        DistributionBin(lower=round(float(edges[i]), 2), upper=round(float(edges[i + 1]), 2), count=int(counts[i]))
        for i in range(len(counts))
    ]

    return DistributionResponse(
        route_code=route_code,
        metric=metric,
        bins=bin_list,
        count=len(df),
    )


@router.get("/stats/rrs", response_model=RRSResponse)
async def rrs_scores(
    days_rolling: int = Query(10, ge=1, le=90, description="Rolling window in days"),
    ref_date: Optional[str] = Query(None, description="Reference date (YYYY-MM-DD)"),
):
    """Compute R³S² (Rolling Relative Route Scoring System) scores."""
    ds = get_dataset()
    scores = ds.analyzer.calculate_rrs(ref_date=ref_date, days_rolling=days_rolling)

    # Merge with labels
    labels = dict(zip(ds.routes_df["route_code"], ds.routes_df["label_short"]))

    routes_out = []
    for _, row in scores.iterrows():
        rc = row["route_code"]
        routes_out.append(
            RRSRoute(
                route_code=rc,
                label_short=labels.get(rc, rc),
                points=round(float(row["points"]), 1),
            )
        )

    return RRSResponse(
        routes=routes_out,
        days_rolling=days_rolling,
        ref_date=ref_date,
    )
