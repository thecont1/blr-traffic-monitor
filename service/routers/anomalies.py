"""
Anomaly detection — flag unusual traffic readings.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Query

from data_utils import detect_outliers
from service.main import get_dataset
from service.models import AnomalyResponse

router = APIRouter(tags=["anomalies"])


@router.get("/anomalies", response_model=AnomalyResponse)
async def anomalies(
    route_code: Optional[str] = Query(None, description="Filter by route_code (all routes if omitted)"),
    method: str = Query(
        "iqr",
        description="Detection method: iqr, zscore, isolation_forest",
        pattern="^(iqr|zscore|isolation_forest)$",
    ),
    threshold: float = Query(3.0, description="Threshold (interpretation depends on method)"),
    metric: str = Query("duration", description="Metric to check: duration or avg_speed"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
):
    """Detect anomalous traffic readings using statistical methods."""
    ds = get_dataset()
    df = ds.traffic_df.copy()

    if route_code:
        df = df[df["route_code"] == route_code]
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]

    if df.empty:
        return AnomalyResponse(
            rows=[], method=method, threshold=threshold,
            route_code=route_code, anomaly_count=0, total_checked=0,
        )

    # Run outlier detection on the chosen metric
    outlier_mask = detect_outliers(df[metric], method=method, threshold=threshold)
    anomalies_df = df[outlier_mask].head(limit)

    rows = []
    for _, row in anomalies_df.iterrows():
        d = {}
        for col in ["date", "time", "route_code", "duration", "avg_speed", metric]:
            if col in row.index:
                v = row[col]
                d[col] = v.item() if hasattr(v, "item") else v
        rows.append(d)

    return AnomalyResponse(
        rows=rows,
        method=method,
        threshold=threshold,
        route_code=route_code,
        anomaly_count=int(outlier_mask.sum()),
        total_checked=len(df),
    )
