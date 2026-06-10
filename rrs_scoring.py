"""
R³S² — Rolling Relative Route Scoring System
==============================================

Canonical implementation of the R³S² scoring methodology.

R³S² is a rolling, relative, speed-based comparative score for routes.
It compares all routes against each other over a recent rolling window,
based on *mean speed*, to capture general recent route quality.

Architecture:
  1. compute_route_daily_stats()  — per-date × route × TOD mean speed
  2. assign_centered_rank_points() — rank routes per day, assign centered linear points
  3. compute_rrs_rolling_scores() — sum daily points over a rolling window
  4. compute_route_speed_variability() — parallel diagnostic (sigma bands)
  5. assign_sigma_band() — map z-score to one of 8 sigma segments

All canonical computation happens here (Python/pandas). The frontend only
reads derived CSV exports and renders them — no recomputation in the browser.

Derived outputs:
  - route-day table   (one row per date × route × TOD)
  - route-window table (one row per window_end × route × TOD × window_days)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Constants
# ============================================================================

# TOD bucket definitions — must match frontend matchesToD() exactly
# Each bucket is defined by (hour_range, is_weekend, day_of_week_set)
# dow: 0=Sun, 1=Mon, ..., 6=Sat
TOD_RULES: dict[str, dict] = {
    "weekday_morning":   {"weekend": False, "hour_start": 8,  "hour_end": 12},
    "weekday_afternoon": {"weekend": False, "hour_start": 12, "hour_end": 18},
    "weekday_evening":   {"weekend": False, "hour_start": 18, "hour_end": 22},
    "weekends":          {"weekend": True},
    "late_hours":        {"hour_start": 22, "hour_end": 5},   # wraps midnight
    "all":               {},
}

# Minimum data quality gates
MIN_TRIPS_PER_DAY = 1        # a route-day row is valid with >= 1 trip
MIN_DAYS_IN_WINDOW = 7       # a window score is valid with >= 7 of 14 days present
DEFAULT_WINDOW_DAYS = 14     # default rolling window length


# ============================================================================
# TOD bucket derivation
# ============================================================================

def derive_tod_bucket(hour: int, day_of_week: int) -> str:
    """
    Map (hour, day_of_week) to a TOD bucket string.

    Parameters
    ----------
    hour : int
        Hour of day (0–23)
    day_of_week : int
        Day of week (0=Sun, 1=Mon, ..., 6=Sat) — matching Python datetime convention

    Returns
    -------
    str
        One of: weekday_morning, weekday_afternoon, weekday_evening,
                weekends, late_hours
    """
    is_weekend = day_of_week in (5, 6)  # Sat=5, Sun=6

    # Late hours first (wraps midnight)
    if hour >= 22 or hour < 5:
        return "late_hours"

    if is_weekend:
        return "weekends"

    if 8 <= hour < 12:
        return "weekday_morning"
    if 12 <= hour < 18:
        return "weekday_afternoon"
    if 18 <= hour < 22:
        return "weekday_evening"

    # Fallback (shouldn't happen with valid hour)
    return "all"


def matches_tod(hour: int, day_of_week: int, tod: str) -> bool:
    """Check if a (hour, dow) pair matches the given TOD bucket."""
    if tod == "all":
        return True
    return derive_tod_bucket(hour, day_of_week) == tod


# ============================================================================
# Core R³S² functions
# ============================================================================

def compute_route_daily_stats(
    traffic_df: pd.DataFrame,
    tod_bucket: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Return one row per (date, route_code, tod_bucket) with mean speed and trip count.

    Parameters
    ----------
    traffic_df : pd.DataFrame
        Raw traffic data. Must have columns: date (str YYYY-MM-DD), time (str HH:MM),
        route_code, duration, distance.
        Optionally: hour, day_of_week, tod_bucket (pre-computed).
    tod_bucket : str, optional
        Filter to a specific TOD bucket. If None, computes TOD per row from hour/dow.
    start_date, end_date : str, optional
        ISO date strings to filter the date range.

    Returns
    -------
    pd.DataFrame
        Columns: date, route_code, tod_bucket, mean_speed, trip_count
    """
    df = traffic_df.copy()

    # Ensure we have a date column as string
    if "date" not in df.columns:
        raise ValueError("traffic_df must have a 'date' column")

    # Parse timestamp if needed for hour/dow extraction
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        if "time" in df.columns:
            ts = pd.to_datetime(df["date"].astype(str) + "T" + df["time"].astype(str) + ":00", errors="coerce")
            df["hour"] = ts.dt.hour
            df["day_of_week"] = ts.dt.dayofweek  # 0=Mon, 6=Sun
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            df["hour"] = ts.dt.hour
            df["day_of_week"] = ts.dt.dayofweek
        else:
            raise ValueError("Need hour/day_of_week or time/timestamp columns")

    # Compute avg_speed if not present
    if "avg_speed" not in df.columns:
        df["avg_speed"] = 60.0 * df["distance"] / df["duration"]

    # Derive TOD bucket per row if not filtering or if not pre-computed
    if "tod_bucket" not in df.columns:
        if len(df) > 0:
            df["tod_bucket"] = [
                derive_tod_bucket(int(r["hour"]), int(r["day_of_week"]))
                for _, r in df.iterrows()
            ]
        else:
            df["tod_bucket"] = pd.Series(dtype=str)

    # Filter by TOD bucket
    if tod_bucket is not None and tod_bucket != "all":
        df = df[df["tod_bucket"] == tod_bucket].copy()

    # Filter by date range
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]

    if df.empty:
        return pd.DataFrame(columns=["date", "route_code", "tod_bucket", "mean_speed", "trip_count"]).astype({
            "date": str, "route_code": str, "tod_bucket": str,
            "mean_speed": float, "trip_count": int,
        })

    # Group by date × route × TOD
    grouped = (
        df.groupby(["date", "route_code", "tod_bucket"], as_index=False)
        .agg(
            mean_speed=("avg_speed", "mean"),
            trip_count=("avg_speed", "size"),
        )
    )

    grouped["mean_speed"] = grouped["mean_speed"].round(2)

    return grouped


def assign_centered_rank_points(
    daily_route_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each date × tod_bucket: rank routes by mean_speed descending
    and assign centered linear points.

    Parameters
    ----------
    daily_route_stats_df : pd.DataFrame
        Output of compute_route_daily_stats(). Must have:
        date, route_code, tod_bucket, mean_speed, trip_count

    Returns
    -------
    pd.DataFrame
        Columns: date, route_code, tod_bucket, mean_speed, trip_count,
                 daily_rank, participating_routes, rrs_daily_points
    """
    df = daily_route_stats_df.copy()

    if df.empty:
        return df.assign(
            daily_rank=pd.Series(dtype=int),
            participating_routes=pd.Series(dtype=int),
            rrs_daily_points=pd.Series(dtype=float),
        )

    results = []
    for (date, tod), group in df.groupby(["date", "tod_bucket"]):
        group = group.sort_values("mean_speed", ascending=False).reset_index(drop=True)
        n = len(group)

        # Centered linear points: fastest gets +n/2, slowest gets -n/2
        # Special case: single route gets 0 (no comparison possible)
        if n == 1:
            points = np.array([0.0])
        else:
            points = np.linspace(n / 2, -n / 2, n)

        group["daily_rank"] = range(1, n + 1)
        group["participating_routes"] = n
        group["rrs_daily_points"] = np.round(points, 2)
        results.append(group)

    return pd.concat(results, ignore_index=True)


def compute_rrs_rolling_scores(
    daily_points_df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Sum daily points over a rolling window for each route × TOD.

    Parameters
    ----------
    daily_points_df : pd.DataFrame
        Output of assign_centered_rank_points(). Must have:
        date, route_code, tod_bucket, mean_speed, rrs_daily_points, trip_count
    window_days : int
        Rolling window length in days (default 14).
    end_date : str, optional
        ISO date for the window end. Defaults to max date in data.

    Returns
    -------
    pd.DataFrame
        Columns: window_end_date, route_code, tod_bucket, window_days,
                 rrs_rolling_score, rrs_rank, routes_in_window,
                 dates_expected, dates_present, trip_count_window,
                 mean_speed_window, speed_sd_window,
                 completeness_ratio, score_status
    """
    df = daily_points_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if end_date is not None:
        window_end = pd.to_datetime(end_date)
    else:
        window_end = df["date"].max()

    window_start = window_end - timedelta(days=window_days - 1)

    # Filter to window
    window_df = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()

    if window_df.empty:
        return pd.DataFrame(columns=[
            "window_end_date", "route_code", "tod_bucket", "window_days",
            "rrs_rolling_score", "rrs_rank", "routes_in_window",
            "dates_expected", "dates_present", "trip_count_window",
            "mean_speed_window", "speed_sd_window",
            "completeness_ratio", "score_status",
        ])

    # Get unique dates in the window for expected count
    unique_dates_in_window = window_df["date"].nunique()

    results = []
    for (route, tod), group in window_df.groupby(["route_code", "tod_bucket"]):
        dates_present = group["date"].nunique()
        total_trips = group["trip_count"].sum()
        score = group["rrs_daily_points"].sum()
        mean_speed = group["mean_speed"].mean()
        speed_sd = group["mean_speed"].std()

        completeness = dates_present / window_days if window_days > 0 else 0.0

        # Score status
        if dates_present >= MIN_DAYS_IN_WINDOW:
            status = "ok"
        elif dates_present >= 4:
            status = "sparse"
        else:
            status = "insufficient_data"

        results.append({
            "window_end_date": window_end.strftime("%Y-%m-%d"),
            "route_code": route,
            "tod_bucket": tod,
            "window_days": window_days,
            "rrs_rolling_score": round(score, 2),
            "routes_in_window": 0,  # filled below
            "dates_expected": window_days,
            "dates_present": dates_present,
            "trip_count_window": int(total_trips),
            "mean_speed_window": round(mean_speed, 2) if not math.isnan(mean_speed) else 0.0,
            "speed_sd_window": round(speed_sd, 2) if not math.isnan(speed_sd) else 0.0,
            "completeness_ratio": round(completeness, 3),
            "score_status": status,
        })

    out = pd.DataFrame(results)

    # Rank routes by rolling score (descending)
    out["rrs_rank"] = out["rrs_rolling_score"].rank(ascending=False, method="min").astype(int)
    out["routes_in_window"] = len(out)

    # Sort by rank
    out = out.sort_values("rrs_rank").reset_index(drop=True)

    return out


def assign_sigma_band(z: float) -> str:
    """
    Map a z-score to one of 8 sigma segments.

    Segments:
        < -3σ, [-3σ, -2σ), [-2σ, -1σ), [-1σ, 0),
        [0, +1σ), [+1σ, +2σ), [+2σ, +3σ), > +3σ

    Parameters
    ----------
    z : float
        Z-score value

    Returns
    -------
    str
        Sigma band label
    """
    if z < -3:
        return "< -3σ"
    elif z < -2:
        return "[-3σ, -2σ)"
    elif z < -1:
        return "[-2σ, -1σ)"
    elif z < 0:
        return "[-1σ, 0)"
    elif z < 1:
        return "[0, +1σ)"
    elif z < 2:
        return "[+1σ, +2σ)"
    elif z < 3:
        return "[+2σ, +3σ)"
    else:
        return "> +3σ"


def compute_route_speed_variability(
    daily_points_df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-route variability diagnostics over the rolling window.

    Parameters
    ----------
    daily_points_df : pd.DataFrame
        Output of assign_centered_rank_points().
    window_days : int
        Rolling window length in days.
    end_date : str, optional
        Window end date.

    Returns
    -------
    pd.DataFrame
        Columns: route_code, tod_bucket, window_end_date, window_days,
                 mean_speed_window, speed_sd_window, speed_cv,
                 sigma_band_distribution (JSON string), daily_z_scores (JSON string)
    """
    df = daily_points_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if end_date is not None:
        window_end = pd.to_datetime(end_date)
    else:
        window_end = df["date"].max()

    window_start = window_end - timedelta(days=window_days - 1)
    window_df = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()

    if window_df.empty:
        return pd.DataFrame(columns=[
            "route_code", "tod_bucket", "window_end_date", "window_days",
            "mean_speed_window", "speed_sd_window", "speed_cv",
            "sigma_band_distribution", "daily_z_scores",
        ])

    import json

    results = []
    for (route, tod), group in window_df.groupby(["route_code", "tod_bucket"]):
        mu = group["mean_speed"].mean()
        sigma = group["mean_speed"].std()

        if sigma == 0 or math.isnan(sigma):
            z_scores = [0.0] * len(group)
            cv = 0.0
        else:
            z_scores = ((group["mean_speed"] - mu) / sigma).tolist()
            cv = sigma / mu if mu != 0 else 0.0

        # Count sigma bands
        band_counts: dict[str, int] = {
            "< -3σ": 0, "[-3σ, -2σ)": 0, "[-2σ, -1σ)": 0, "[-1σ, 0)": 0,
            "[0, +1σ)": 0, "[+1σ, +2σ)": 0, "[+2σ, +3σ)": 0, "> +3σ": 0,
        }
        for z in z_scores:
            band_counts[assign_sigma_band(z)] += 1

        daily_z = {
            str(row["date"].date()): {"z": round(z, 3), "band": assign_sigma_band(z)}
            for (_, row), z in zip(group.iterrows(), z_scores)
        }

        results.append({
            "route_code": route,
            "tod_bucket": tod,
            "window_end_date": window_end.strftime("%Y-%m-%d"),
            "window_days": window_days,
            "mean_speed_window": round(mu, 2),
            "speed_sd_window": round(sigma, 2) if not math.isnan(sigma) else 0.0,
            "speed_cv": round(cv, 4),
            "sigma_band_distribution": json.dumps(band_counts),
            "daily_z_scores": json.dumps(daily_z),
        })

    return pd.DataFrame(results)


# ============================================================================
# Convenience: compute everything from raw data
# ============================================================================

def compute_all_rrs(
    traffic_df: pd.DataFrame,
    tod_bucket: str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute the full R³S² pipeline from raw traffic data.

    Parameters
    ----------
    traffic_df : pd.DataFrame
        Raw traffic data with date, time, route_code, duration, distance.
    tod_bucket : str, optional
        TOD bucket filter. None = compute for all TODs.
    window_days : int
        Rolling window length.
    end_date : str, optional
        Window end date.

    Returns
    -------
    tuple of (route_day_df, route_window_df, variability_df)
    """
    # Step 1: daily stats
    daily_stats = compute_route_daily_stats(traffic_df, tod_bucket=tod_bucket)

    # Step 2: daily ranking + points
    daily_points = assign_centered_rank_points(daily_stats)

    # Step 3: rolling scores
    rolling_scores = compute_rrs_rolling_scores(daily_points, window_days=window_days, end_date=end_date)

    # Step 4: variability
    variability = compute_route_speed_variability(daily_points, window_days=window_days, end_date=end_date)

    return daily_points, rolling_scores, variability
