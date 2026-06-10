"""
Unit tests for rrs_scoring.py — R³S² Rolling Relative Route Scoring System.

Covers:
  - Ties in daily mean speed
  - Single route on a day
  - Missing routes on some days
  - Sparse rolling windows
  - Full windows
  - Ranking stability over small handcrafted data
  - Sigma band assignment boundaries
"""

import pytest
import numpy as np
import pandas as pd

from rrs_scoring import (
    derive_tod_bucket,
    matches_tod,
    compute_route_daily_stats,
    assign_centered_rank_points,
    compute_rrs_rolling_scores,
    assign_sigma_band,
    compute_route_speed_variability,
    compute_all_rrs,
    DEFAULT_WINDOW_DAYS,
    MIN_DAYS_IN_WINDOW,
)


# ============================================================================
# Fixtures / Helpers
# ============================================================================

def _make_traffic_row(date: str, time: str, route: str, duration: float, distance: float = 10.0):
    """Create a single traffic row dict."""
    return {
        "date": date,
        "time": time,
        "route_code": route,
        "duration": duration,
        "distance": distance,
    }


def _make_traffic_df(rows: list[dict]) -> pd.DataFrame:
    """Create a traffic DataFrame from a list of row dicts."""
    df = pd.DataFrame(rows)
    df["avg_speed"] = 60.0 * df["distance"] / df["duration"]
    ts = pd.to_datetime(df["date"] + "T" + df["time"] + ":00")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    return df


def _make_multi_day_data(
    routes: list[str],
    dates: list[str],
    speed_map: dict[tuple[str, str], float],
    hour: int = 19,
) -> pd.DataFrame:
    """
    Create a multi-day traffic DataFrame.

    speed_map: {(route, date): speed_kmh} — controls the speed for each route on each date.
    Routes/dates not in speed_map get no rows (simulating missing data).
    """
    rows = []
    for date in dates:
        for route in routes:
            speed = speed_map.get((route, date))
            if speed is not None and speed > 0:
                duration = 60.0 * 10.0 / speed  # distance=10km
                rows.append(_make_traffic_row(date, f"{hour:02d}:00", route, duration))
    return _make_traffic_df(rows) if rows else pd.DataFrame(columns=["date", "time", "route_code", "duration", "distance", "avg_speed", "hour", "day_of_week"])


# ============================================================================
# Test: TOD bucket derivation
# ============================================================================

class TestTodBucket:
    def test_weekday_morning(self):
        assert derive_tod_bucket(9, 0) == "weekday_morning"   # Mon 9am
        assert derive_tod_bucket(11, 4) == "weekday_morning"  # Fri 11am

    def test_weekday_afternoon(self):
        assert derive_tod_bucket(14, 1) == "weekday_afternoon"  # Tue 2pm
        assert derive_tod_bucket(17, 3) == "weekday_afternoon"  # Thu 5pm

    def test_weekday_evening(self):
        assert derive_tod_bucket(19, 0) == "weekday_evening"  # Mon 7pm
        assert derive_tod_bucket(21, 2) == "weekday_evening"  # Wed 9pm

    def test_weekends(self):
        assert derive_tod_bucket(10, 5) == "weekends"  # Sat 10am
        assert derive_tod_bucket(15, 6) == "weekends"  # Sun 3pm

    def test_late_hours(self):
        assert derive_tod_bucket(23, 0) == "late_hours"  # Mon 11pm
        assert derive_tod_bucket(2, 3) == "late_hours"   # Thu 2am
        assert derive_tod_bucket(4, 6) == "late_hours"   # Sun 4am

    def test_matches_tod_all(self):
        assert matches_tod(9, 0, "all") is True
        assert matches_tod(23, 5, "all") is True

    def test_matches_tod_specific(self):
        assert matches_tod(19, 0, "weekday_evening") is True
        assert matches_tod(19, 5, "weekday_evening") is False  # Sat evening is 'weekends'


# ============================================================================
# Test: compute_route_daily_stats
# ============================================================================

class TestRouteDailyStats:
    def test_basic_output_shape(self):
        """Should produce one row per (date, route) for the given TOD."""
        df = _make_traffic_df([
            _make_traffic_row("2026-01-05", "19:00", "A", 20),
            _make_traffic_row("2026-01-05", "19:30", "A", 25),
            _make_traffic_row("2026-01-05", "19:00", "B", 30),
        ])
        result = compute_route_daily_stats(df, tod_bucket="weekday_evening")
        assert len(result) == 2  # route A, route B
        assert set(result.columns) >= {"date", "route_code", "tod_bucket", "mean_speed", "trip_count"}

    def test_mean_speed_averaged(self):
        """Multiple trips on same day should be averaged."""
        df = _make_traffic_df([
            _make_traffic_row("2026-01-05", "19:00", "A", 20),  # speed = 30
            _make_traffic_row("2026-01-05", "19:30", "A", 40),  # speed = 15
        ])
        result = compute_route_daily_stats(df, tod_bucket="weekday_evening")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["trip_count"] == 2
        # mean of 30 and 15 = 22.5
        assert abs(row["mean_speed"] - 22.5) < 0.1

    def test_trip_count(self):
        """Should count trips correctly."""
        df = _make_traffic_df([
            _make_traffic_row("2026-01-05", "19:00", "A", 20),
            _make_traffic_row("2026-01-05", "19:15", "A", 22),
            _make_traffic_row("2026-01-05", "19:30", "A", 24),
        ])
        result = compute_route_daily_stats(df, tod_bucket="weekday_evening")
        assert result.iloc[0]["trip_count"] == 3

    def test_empty_input(self):
        """Should return empty DataFrame gracefully."""
        df = pd.DataFrame(columns=["date", "time", "route_code", "duration", "distance"])
        result = compute_route_daily_stats(df)
        assert len(result) == 0


# ============================================================================
# Test: assign_centered_rank_points
# ============================================================================

class TestCenteredRankPoints:
    def test_three_routes_ranking(self):
        """Fastest route gets highest points, slowest gets lowest."""
        daily_stats = pd.DataFrame({
            "date": ["2026-01-05"] * 3,
            "route_code": ["A", "B", "C"],
            "tod_bucket": ["weekday_evening"] * 3,
            "mean_speed": [40.0, 30.0, 20.0],
            "trip_count": [1, 1, 1],
        })
        result = assign_centered_rank_points(daily_stats)
        result = result.sort_values("mean_speed", ascending=False)

        assert result.iloc[0]["daily_rank"] == 1  # A is fastest
        assert result.iloc[0]["rrs_daily_points"] == pytest.approx(1.5)  # n=3, +3/2=1.5
        assert result.iloc[1]["rrs_daily_points"] == pytest.approx(0.0)  # middle
        assert result.iloc[2]["rrs_daily_points"] == pytest.approx(-1.5)  # slowest

    def test_two_routes_points(self):
        """With 2 routes: +1 and -1."""
        daily_stats = pd.DataFrame({
            "date": ["2026-01-05"] * 2,
            "route_code": ["A", "B"],
            "tod_bucket": ["weekday_evening"] * 2,
            "mean_speed": [35.0, 25.0],
            "trip_count": [1, 1],
        })
        result = assign_centered_rank_points(daily_stats)
        points = sorted(result["rrs_daily_points"].tolist(), reverse=True)
        assert points == pytest.approx([1.0, -1.0])

    def test_single_route(self):
        """Single route gets 0 points (no comparison possible)."""
        daily_stats = pd.DataFrame({
            "date": ["2026-01-05"],
            "route_code": ["A"],
            "tod_bucket": ["weekday_evening"],
            "mean_speed": [30.0],
            "trip_count": [1],
        })
        result = assign_centered_rank_points(daily_stats)
        # Single route with no comparison → 0 points
        assert result.iloc[0]["rrs_daily_points"] == pytest.approx(0.0)

    def test_tie_in_speed(self):
        """Tied routes should both get points (order may vary but sum should be symmetric)."""
        daily_stats = pd.DataFrame({
            "date": ["2026-01-05"] * 3,
            "route_code": ["A", "B", "C"],
            "tod_bucket": ["weekday_evening"] * 3,
            "mean_speed": [30.0, 30.0, 20.0],
            "trip_count": [1, 1, 1],
        })
        result = assign_centered_rank_points(daily_stats)
        # Two tied at 30, one at 20 — the tied pair should share ranks 1-2
        assert result["rrs_daily_points"].sum() == pytest.approx(0.0)  # centered sum ≈ 0
        assert len(result) == 3

    def test_participating_routes_count(self):
        """Should report correct number of participating routes."""
        daily_stats = pd.DataFrame({
            "date": ["2026-01-05"] * 5,
            "route_code": ["A", "B", "C", "D", "E"],
            "tod_bucket": ["weekday_evening"] * 5,
            "mean_speed": [50, 40, 30, 20, 10],
            "trip_count": [1] * 5,
        })
        result = assign_centered_rank_points(daily_stats)
        assert all(result["participating_routes"] == 5)


# ============================================================================
# Test: compute_rrs_rolling_scores
# ============================================================================

class TestRollingScores:
    def test_basic_rolling_sum(self):
        """Rolling score should be sum of daily points over the window."""
        daily_points = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "route_code": ["A", "A", "A"],
            "tod_bucket": ["weekday_evening"] * 3,
            "mean_speed": [30.0, 30.0, 30.0],
            "trip_count": [1, 1, 1],
            "rrs_daily_points": [5.0, 3.0, -2.0],
        })
        result = compute_rrs_rolling_scores(daily_points, window_days=7, end_date="2026-01-03")
        assert len(result) == 1
        assert result.iloc[0]["rrs_rolling_score"] == pytest.approx(6.0)  # 5+3-2

    def test_missing_days_contribute_nothing(self):
        """Missing route/day combos should not impute — they contribute 0."""
        daily_points = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-03"],  # missing 01-02
            "route_code": ["A", "A"],
            "tod_bucket": ["weekday_evening"] * 2,
            "mean_speed": [30.0, 30.0],
            "trip_count": [1, 1],
            "rrs_daily_points": [5.0, -2.0],
        })
        result = compute_rrs_rolling_scores(daily_points, window_days=7, end_date="2026-01-03")
        assert result.iloc[0]["rrs_rolling_score"] == pytest.approx(3.0)  # 5 + 0 + (-2)
        assert result.iloc[0]["dates_present"] == 2

    def test_sparse_window_status(self):
        """Windows with < 7 days should be marked sparse or insufficient."""
        daily_points = pd.DataFrame({
            "date": [f"2026-01-0{i}" for i in range(1, 4)],
            "route_code": ["A"] * 3,
            "tod_bucket": ["weekday_evening"] * 3,
            "mean_speed": [30.0] * 3,
            "trip_count": [1] * 3,
            "rrs_daily_points": [1.0] * 3,
        })
        result = compute_rrs_rolling_scores(daily_points, window_days=14, end_date="2026-01-14")
        assert result.iloc[0]["score_status"] == "insufficient_data"  # 3 < 7

    def test_ok_status(self):
        """Windows with >= 7 days should be marked ok."""
        daily_points = pd.DataFrame({
            "date": [f"2026-01-{i:02d}" for i in range(1, 11)],
            "route_code": ["A"] * 10,
            "tod_bucket": ["weekday_evening"] * 10,
            "mean_speed": [30.0] * 10,
            "trip_count": [1] * 10,
            "rrs_daily_points": [1.0] * 10,
        })
        result = compute_rrs_rolling_scores(daily_points, window_days=14, end_date="2026-01-14")
        assert result.iloc[0]["score_status"] == "ok"

    def test_ranking_across_routes(self):
        """Routes with higher rolling scores should rank higher."""
        # Use all weekdays to avoid TOD bucket filtering issues
        # Jan 2026: 5=Mon, 6=Tue, 7=Wed, 8=Thu, 9=Fri, 12=Mon, 13=Tue, 14=Wed, 15=Thu, 16=Fri
        dates = [f"2026-01-{i:02d}" for i in [5, 6, 7, 8, 9, 12, 13, 14, 15, 16]]
        daily_points = pd.DataFrame({
            "date": dates * 2,
            "route_code": ["fast"] * 10 + ["slow"] * 10,
            "tod_bucket": ["weekday_evening"] * 20,
            "mean_speed": [40.0] * 10 + [20.0] * 10,
            "trip_count": [1] * 20,
            "rrs_daily_points": [1.0] * 10 + [-1.0] * 10,
        })
        result = compute_rrs_rolling_scores(daily_points, window_days=14, end_date="2026-01-16")
        fast_row = result[result["route_code"] == "fast"].iloc[0]
        slow_row = result[result["route_code"] == "slow"].iloc[0]
        assert fast_row["rrs_rank"] < slow_row["rrs_rank"]  # lower rank number = better
        assert fast_row["rrs_rolling_score"] > slow_row["rrs_rolling_score"]


# ============================================================================
# Test: assign_sigma_band
# ============================================================================

class TestSigmaBand:
    def test_boundaries(self):
        """Test exact boundary values."""
        assert assign_sigma_band(-3.0) == "[-3σ, -2σ)"
        assert assign_sigma_band(-2.0) == "[-2σ, -1σ)"
        assert assign_sigma_band(-1.0) == "[-1σ, 0)"
        assert assign_sigma_band(0.0) == "[0, +1σ)"
        assert assign_sigma_band(1.0) == "[+1σ, +2σ)"
        assert assign_sigma_band(2.0) == "[+2σ, +3σ)"
        assert assign_sigma_band(3.0) == "> +3σ"

    def test_extremes(self):
        """Extreme z-scores."""
        assert assign_sigma_band(-5.0) == "< -3σ"
        assert assign_sigma_band(5.0) == "> +3σ"

    def test_mid_ranges(self):
        """Values clearly in the middle of bands."""
        assert assign_sigma_band(-2.5) == "[-3σ, -2σ)"
        assert assign_sigma_band(-0.5) == "[-1σ, 0)"
        assert assign_sigma_band(0.5) == "[0, +1σ)"
        assert assign_sigma_band(2.5) == "[+2σ, +3σ)"


# ============================================================================
# Test: compute_route_speed_variability
# ============================================================================

class TestVariability:
    def test_output_has_sigma_bands(self):
        """Variability output should include sigma band distribution as JSON."""
        import json
        daily_points = pd.DataFrame({
            "date": [f"2026-01-{i:02d}" for i in range(1, 15)],
            "route_code": ["A"] * 14,
            "tod_bucket": ["weekday_evening"] * 14,
            "mean_speed": [30, 32, 28, 31, 29, 33, 27, 30, 32, 28, 31, 29, 30, 31],
            "trip_count": [1] * 14,
            "rrs_daily_points": [0.0] * 14,
        })
        result = compute_route_speed_variability(daily_points, window_days=14, end_date="2026-01-14")
        assert len(result) == 1
        band_dist = json.loads(result.iloc[0]["sigma_band_distribution"])
        assert isinstance(band_dist, dict)
        total = sum(band_dist.values())
        assert total == 14

    def test_constant_speed_zero_sd(self):
        """All same speed → SD = 0, CV = 0."""
        daily_points = pd.DataFrame({
            "date": [f"2026-01-{i:02d}" for i in range(1, 8)],
            "route_code": ["A"] * 7,
            "tod_bucket": ["weekday_evening"] * 7,
            "mean_speed": [30.0] * 7,
            "trip_count": [1] * 7,
            "rrs_daily_points": [0.0] * 7,
        })
        result = compute_route_speed_variability(daily_points, window_days=14, end_date="2026-01-14")
        assert result.iloc[0]["speed_sd_window"] == 0.0
        assert result.iloc[0]["speed_cv"] == 0.0


# ============================================================================
# Test: compute_all_rrs (integration)
# ============================================================================

class TestComputeAllRRS:
    def test_full_pipeline(self):
        """End-to-end: raw data → route-day + route-window + variability."""
        routes = ["A", "B", "C"]
        # Use all weekdays: Jan 2026 — 5-9, 12-16, 19-23, 26-30
        dates = [f"2026-01-{i:02d}" for i in [5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20]]
        speed_map = {}
        for d in dates:
            speed_map[("A", d)] = 40.0  # consistently fast
            speed_map[("B", d)] = 30.0  # middle
            speed_map[("C", d)] = 20.0  # consistently slow

        traffic_df = _make_multi_day_data(routes, dates, speed_map, hour=19)
        route_day, route_window, variability = compute_all_rrs(
            traffic_df, tod_bucket="weekday_evening", window_days=14, end_date="2026-01-20"
        )

        # Route-day: 3 routes × 12 days = 36 rows
        assert len(route_day) == 36

        # Route-window: should have 3 routes
        assert len(route_window) == 3

        # Ranking should be A > B > C
        ranks = route_window.set_index("route_code")["rrs_rank"].to_dict()
        assert ranks["A"] < ranks["B"] < ranks["C"]

        # Variability should have 3 entries
        assert len(variability) == 3

    def test_missing_route_data(self):
        """Route with partial data should still get a score (possibly sparse)."""
        routes = ["A", "B"]
        # All weekdays in 2-week window ending 2026-01-16
        # Window: Jan 3-16, weekdays: 5-9, 12-16 = 10 weekdays
        all_dates = [f"2026-01-{i:02d}" for i in [5, 6, 7, 8, 9, 12, 13, 14, 15, 16]]
        speed_map = {}
        for d in all_dates:
            speed_map[("A", d)] = 35.0
        # B only has data for first 3 weekdays
        for d in all_dates[:3]:
            speed_map[("B", d)] = 25.0

        traffic_df = _make_multi_day_data(routes, all_dates, speed_map, hour=19)
        route_day, route_window, variability = compute_all_rrs(
            traffic_df, tod_bucket="weekday_evening", window_days=14, end_date="2026-01-16"
        )

        b_row = route_window[route_window["route_code"] == "B"]
        assert len(b_row) == 1
        assert b_row.iloc[0]["dates_present"] == 3
        assert b_row.iloc[0]["score_status"] == "insufficient_data"  # 3 < 7


# ============================================================================
# Test: completeness ratio and score status
# ============================================================================

class TestCompleteness:
    def test_full_window_completeness(self):
        """14 days present in a 14-day window → completeness = 1.0."""
        daily_points = pd.DataFrame({
            "date": [f"2026-01-{i:02d}" for i in range(1, 15)],
            "route_code": ["A"] * 14,
            "tod_bucket": ["weekday_evening"] * 14,
            "mean_speed": [30.0] * 14,
            "trip_count": [1] * 14,
            "rrs_daily_points": [1.0] * 14,
        })
        result = compute_rrs_rolling_scores(daily_points, window_days=14, end_date="2026-01-14")
        assert result.iloc[0]["completeness_ratio"] == pytest.approx(1.0)
        assert result.iloc[0]["score_status"] == "ok"

    def test_sparse_completeness(self):
        """4 of 14 days → completeness ≈ 0.286, status = sparse."""
        # Use weekday dates to avoid TOD filtering
        daily_points = pd.DataFrame({
            "date": [f"2026-01-{i:02d}" for i in [5, 6, 7, 8]],  # Mon-Thu
            "route_code": ["A"] * 4,
            "tod_bucket": ["weekday_evening"] * 4,
            "mean_speed": [30.0] * 4,
            "trip_count": [1] * 4,
            "rrs_daily_points": [1.0] * 4,
        })
        result = compute_rrs_rolling_scores(daily_points, window_days=14, end_date="2026-01-16")
        assert result.iloc[0]["completeness_ratio"] == pytest.approx(4 / 14, abs=0.001)
        assert result.iloc[0]["score_status"] == "sparse"
