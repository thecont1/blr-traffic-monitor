"""
Traffic data — raw reads and aggregation.
"""

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from service.main import get_dataset
from service.models import AggregateResponse, TrafficResponse

router = APIRouter(tags=["traffic"])


@router.get("/traffic", response_model=TrafficResponse)
async def get_traffic(
    route_code: Optional[str] = Query(None, description="Filter by route_code"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    hour_from: Optional[int] = Query(None, ge=0, le=23, description="Start hour"),
    hour_to: Optional[int] = Query(None, ge=0, le=23, description="End hour"),
    day_of_week: Optional[str] = Query(None, description="Filter by day name (e.g. Monday)"),
    is_weekend: Optional[bool] = Query(None, description="Filter weekend/weekday"),
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """Read raw traffic observations with optional filters and pagination."""
    ds = get_dataset()
    df = ds.traffic_df

    if route_code:
        df = df[df["route_code"] == route_code]
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]
    if hour_from is not None:
        df = df[df["hour"] >= hour_from]
    if hour_to is not None:
        df = df[df["hour"] <= hour_to]
    if day_of_week:
        df = df[df["day_of_week"] == day_of_week]
    if is_weekend is not None:
        df = df[df["is_weekend"] == is_weekend]

    total = len(df)
    page = df.iloc[offset : offset + limit]

    # Convert to plain dicts — timestamps/periods are not JSON-serializable
    rows = []
    for _, row in page.iterrows():
        d = row.to_dict()
        # Strip non-serializable types
        for k, v in d.items():
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
            elif hasattr(v, "item"):
                d[k] = v.item()
        rows.append(d)

    return TrafficResponse(rows=rows, count=len(rows), total=total, offset=offset, limit=limit)


@router.get("/traffic/aggregate", response_model=AggregateResponse)
async def aggregate_traffic(
    route_code: Optional[str] = Query(None, description="Filter by route_code"),
    group_by: str = Query(
        "hour",
        description="Group dimension: hour, day_of_week, month, time_category, is_weekend",
        pattern="^(hour|day_of_week|month|time_category|is_weekend)$",
    ),
    agg: str = Query(
        "mean",
        description="Aggregation: mean, median, p50, p75, p90, min, max, count",
        pattern="^(mean|median|p50|p75|p90|min|max|count)$",
    ),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """Aggregate traffic data by a grouping dimension."""
    ds = get_dataset()
    df = ds.traffic_df

    if route_code:
        df = df[df["route_code"] == route_code]
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]

    if df.empty:
        return AggregateResponse(rows=[], group_by=group_by, agg=agg, route_code=route_code)

    # Build aggregation
    numeric_cols = ["duration", "distance", "avg_speed"]

    if agg == "count":
        result = df.groupby(group_by).size().reset_index(name="count")
        rows = result.to_dict(orient="records")
    else:
        # Map friendly names to pandas agg functions
        agg_map = {
            "mean": "mean",
            "median": "median",
            "p50": lambda x: x.quantile(0.50),
            "p75": lambda x: x.quantile(0.75),
            "p90": lambda x: x.quantile(0.90),
            "min": "min",
            "max": "max",
        }
        agg_func = agg_map[agg]

        result = df.groupby(group_by).agg(
            **{f"{c}_{agg}": (c, agg_func) for c in numeric_cols},
            count=("duration", "size"),
        ).reset_index()
        result.rename(columns={group_by: "group"}, inplace=True)
        rows = []
        for _, row in result.iterrows():
            d = row.to_dict()
            for k, v in d.items():
                if hasattr(v, "item"):
                    d[k] = v.item()
            rows.append(d)

    return AggregateResponse(rows=rows, group_by=group_by, agg=agg, route_code=route_code)
