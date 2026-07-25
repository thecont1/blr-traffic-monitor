"""
Route comparison — side-by-side statistics for multiple routes.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Query

from service.main import get_dataset
from service.models import CompareResponse, RouteStats

router = APIRouter(tags=["compare"])


@router.get("/compare", response_model=CompareResponse)
async def compare_routes(
    routes: str = Query(..., description="Comma-separated route codes"),
    metric: str = Query("duration", description="Primary metric: duration or avg_speed"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """Compare statistics across two or more routes side by side."""
    ds = get_dataset()
    route_codes = [r.strip() for r in routes.split(",") if r.strip()]

    if len(route_codes) < 2:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"detail": "Provide at least 2 comma-separated route codes"},
        )

    df = ds.traffic_df
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]

    # Build label lookup
    labels = dict(zip(ds.routes_df["route_code"], ds.routes_df["label_short"]))

    results = []
    for rc in route_codes:
        subset = df[df["route_code"] == rc]
        if subset.empty:
            continue
        results.append(
            RouteStats(
                route_code=rc,
                label_short=labels.get(rc, rc),
                count=len(subset),
                duration_mean=round(float(subset["duration"].mean()), 2),
                duration_median=round(float(subset["duration"].median()), 2),
                duration_std=round(float(subset["duration"].std()), 2),
                duration_p10=round(float(subset["duration"].quantile(0.10)), 2),
                duration_p90=round(float(subset["duration"].quantile(0.90)), 2),
                avg_speed_mean=round(float(subset["avg_speed"].mean()), 2),
                avg_speed_median=round(float(subset["avg_speed"].median()), 2),
                avg_speed_std=round(float(subset["avg_speed"].std()), 2),
            )
        )

    return CompareResponse(
        routes=results,
        metric=metric,
        date_from=date_from,
        date_to=date_to,
    )
