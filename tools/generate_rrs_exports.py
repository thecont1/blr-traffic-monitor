#!/usr/bin/env python3
"""
Generate R³S² derived datasets for frontend consumption.

Reads the raw traffic CSV and produces two derived CSVs:
  - data/rrs-route-day.csv       (route × date × TOD daily scores)
  - data/rrs-route-window.csv    (route × window rolling scores + variability)

These files are committed to the repo and served as static assets
via GitHub raw URLs so the frontend can fetch them directly.

Usage:
    python3 generate_rrs_exports.py
    python3 generate_rrs_exports.py --tod weekday_evening
    python3 generate_rrs_exports.py --window 14 --end-date 2026-06-08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rrs_scoring import (
    compute_all_rrs,
    compute_route_daily_stats,
    assign_centered_rank_points,
    compute_rrs_rolling_scores,
    compute_route_speed_variability,
    DEFAULT_WINDOW_DAYS,
)

DATA_DIR = Path(__file__).parent / "data"
TRAFFIC_CSV = DATA_DIR / "csv-traffic-bangalore.csv"
ROUTES_CSV = DATA_DIR / "csv-routes-bangalore.csv"

OUTPUT_ROUTE_DAY = DATA_DIR / "rrs-route-day.csv"
OUTPUT_ROUTE_WINDOW = DATA_DIR / "rrs-route-window.csv"


def load_traffic_data(csv_path: Path = TRAFFIC_CSV) -> pd.DataFrame:
    """Load and validate the raw traffic CSV."""
    if not csv_path.exists():
        print(f"ERROR: traffic CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    # Basic validation
    required = {"date", "time", "route_code", "duration", "distance"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Drop rows with missing critical fields
    before = len(df)
    df = df.dropna(subset=["date", "route_code", "duration", "distance"])
    df = df[df["duration"] > 0]
    df = df[df["distance"] > 0]
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} invalid rows")

    # Compute avg_speed
    df["avg_speed"] = 60.0 * df["distance"] / df["duration"]
    df = df[df["avg_speed"].between(1, 150)]

    # Parse timestamp components
    ts = pd.to_datetime(df["date"].astype(str) + "T" + df["time"].astype(str) + ":00", errors="coerce")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0=Mon, 6=Sun

    print(f"  Valid rows: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Routes: {df['route_code'].nunique()}")
    print(f"  TOD buckets will be derived from hour/day_of_week")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate R³S² derived datasets")
    parser.add_argument("--tod", type=str, default=None,
                        help="Filter to a single TOD bucket (default: all)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"Rolling window length in days (default: {DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Window end date (default: latest date in data)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    traffic_df = load_traffic_data()

    # Determine TOD buckets to process
    if args.tod:
        tod_buckets = [args.tod]
    else:
        tod_buckets = ["weekday_morning", "weekday_afternoon", "weekday_evening", "weekends", "late_hours"]

    print(f"\nProcessing TOD buckets: {tod_buckets}")
    print(f"Window: {args.window} days")

    # Compute for each TOD bucket and concatenate
    all_route_day = []
    all_route_window = []
    all_variability = []

    for tod in tod_buckets:
        print(f"\n  [{tod}] Computing daily stats...")
        daily_stats = compute_route_daily_stats(traffic_df, tod_bucket=tod)
        print(f"    Route-day rows: {len(daily_stats)}")

        if daily_stats.empty:
            print(f"    No data for {tod}, skipping")
            continue

        print(f"  [{tod}] Assigning daily rank points...")
        daily_points = assign_centered_rank_points(daily_stats)

        print(f"  [{tod}] Computing rolling scores...")
        rolling = compute_rrs_rolling_scores(daily_points, window_days=args.window, end_date=args.end_date)
        print(f"    Window rows: {len(rolling)}")

        print(f"  [{tod}] Computing variability...")
        variability = compute_route_speed_variability(daily_points, window_days=args.window, end_date=args.end_date)

        all_route_day.append(daily_points)
        all_route_window.append(rolling)
        all_variability.append(variability)

    # Concatenate results
    if all_route_day:
        route_day_df = pd.concat(all_route_day, ignore_index=True)
        route_window_df = pd.concat(all_route_window, ignore_index=True)
        variability_df = pd.concat(all_variability, ignore_index=True)
    else:
        print("\nNo data to export!")
        return

    # Merge variability into route-window
    # (variability has extra columns not in rolling scores)
    route_window_full = route_window_df.merge(
        variability_df[["route_code", "tod_bucket", "speed_cv", "sigma_band_distribution", "daily_z_scores"]],
        on=["route_code", "tod_bucket"],
        how="left",
    )

    # Load route labels and merge
    if ROUTES_CSV.exists():
        routes_df = pd.read_csv(ROUTES_CSV)
        label_map = dict(zip(routes_df["route_code"], routes_df["label_short"]))
        route_day_df["route_label"] = route_day_df["route_code"].map(label_map).fillna(route_day_df["route_code"])
        route_window_full["route_label"] = route_window_full["route_code"].map(label_map).fillna(route_window_full["route_code"])

    # Write outputs
    route_day_path = output_dir / "rrs-route-day.csv"
    route_window_path = output_dir / "rrs-route-window.csv"

    route_day_df.to_csv(route_day_path, index=False)
    route_window_full.to_csv(route_window_path, index=False)

    print(f"\n=== Export complete ===")
    print(f"  Route-day:   {len(route_day_df)} rows → {route_day_path}")
    print(f"  Route-window: {len(route_window_full)} rows → {route_window_path}")

    # Print summary for the latest window
    latest_window = route_window_full.sort_values("rrs_rank")
    print(f"\n=== Latest R³S² Rankings ({args.window}-day window) ===")
    for _, row in latest_window.head(10).iterrows():
        print(f"  #{row['rrs_rank']:2d}  {row['rrs_rolling_score']:+7.1f}  "
              f"{row['route_code'][:30]:30s}  ({row['tod_bucket']})  "
              f"{row['dates_present']}/{row['dates_expected']} days  "
              f"μ={row['mean_speed_window']:.1f} km/h")


if __name__ == "__main__":
    main()
