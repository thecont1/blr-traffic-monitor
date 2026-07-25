"""
Pydantic models for API request/response schemas.

These drive automatic OpenAPI generation. Every endpoint's input
and output shape is declared here.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Health ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    routes: int
    rows: int
    city: str


# ── Routes ──────────────────────────────────────────────────────────

class RouteInfo(BaseModel):
    route_code: str
    label_full: str
    label_short: str
    map_link: str
    accuweather_station: str


class RouteList(BaseModel):
    routes: list[RouteInfo]
    count: int


# ── Traffic ─────────────────────────────────────────────────────────

class TrafficRow(BaseModel):
    date: str
    time: str
    route_code: str
    duration: int
    distance: float
    avg_speed: float
    temp: Optional[float] = None
    realfeel: Optional[float] = None
    humidity: Optional[float] = None
    rsi_flag: Optional[str] = None
    aqi: Optional[float] = None
    day_of_week: Optional[str] = None
    is_weekend: Optional[bool] = None
    time_category: Optional[str] = None


class TrafficResponse(BaseModel):
    rows: list[dict[str, Any]]
    count: int
    total: int
    offset: int
    limit: int


class AggregateRow(BaseModel):
    group: Any
    duration_mean: Optional[float] = None
    duration_median: Optional[float] = None
    distance_mean: Optional[float] = None
    avg_speed_mean: Optional[float] = None
    avg_speed_median: Optional[float] = None
    count: int


class AggregateResponse(BaseModel):
    rows: list[dict[str, Any]]
    group_by: str
    agg: str
    route_code: Optional[str] = None


# ── Compare ─────────────────────────────────────────────────────────

class RouteStats(BaseModel):
    route_code: str
    label_short: str
    count: int
    duration_mean: float
    duration_median: float
    duration_std: float
    duration_p10: float
    duration_p90: float
    avg_speed_mean: float
    avg_speed_median: float
    avg_speed_std: float


class CompareResponse(BaseModel):
    routes: list[RouteStats]
    metric: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None


# ── Stats ───────────────────────────────────────────────────────────

class PercentileResponse(BaseModel):
    route_code: str
    metric: str
    percentiles: dict[str, float]
    count: int


class DistributionBin(BaseModel):
    lower: float
    upper: float
    count: int


class DistributionResponse(BaseModel):
    route_code: str
    metric: str
    bins: list[DistributionBin]
    count: int


class RRSRoute(BaseModel):
    route_code: str
    label_short: str
    points: float


class RRSResponse(BaseModel):
    routes: list[RRSRoute]
    days_rolling: int
    ref_date: Optional[str] = None


# ── Anomalies ───────────────────────────────────────────────────────

class AnomalyRow(BaseModel):
    date: str
    time: str
    route_code: str
    duration: int
    avg_speed: float
    metric_value: float


class AnomalyResponse(BaseModel):
    rows: list[dict[str, Any]]
    method: str
    threshold: float
    route_code: Optional[str] = None
    anomaly_count: int
    total_checked: int


# ── Reports ─────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    type: str = Field(
        default="summary",
        description="Report type: summary, full, route",
        pattern="^(summary|full|route)$",
    )
    route_code: Optional[str] = Field(
        default=None,
        description="Route code (required for type=route)",
    )


class ReportInfo(BaseModel):
    report_id: str
    type: str
    status: str
    path: Optional[str] = None


class ReportList(BaseModel):
    reports: list[ReportInfo]


class SchemaResponse(BaseModel):
    title: str
    description: str
    content: str
