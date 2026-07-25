"""
Data loader — reads CSVs, runs preprocessing, exposes DataFrames.

Called once during FastAPI lifespan startup. The loaded data lives
in memory for the lifetime of the process.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from data_utils import compute_temporal_features, preprocess_traffic_data
from traffic_analyzer import TrafficAnalyzer

logger = logging.getLogger(__name__)


class Dataset:
    """Container for loaded and preprocessed data."""

    def __init__(
        self,
        traffic_df: pd.DataFrame,
        routes_df: pd.DataFrame,
        analyzer: "TrafficAnalyzer",
        city_config: dict,
    ):
        self.traffic_df = traffic_df
        self.routes_df = routes_df
        self.analyzer = analyzer
        self.city_config = city_config

    @property
    def row_count(self) -> int:
        return len(self.traffic_df)

    @property
    def route_count(self) -> int:
        return len(self.routes_df)

    @property
    def routes(self) -> list[str]:
        return sorted(self.traffic_df["route_code"].unique().tolist())


def load_dataset(data_dir: Path, city_config_path: str = "config/city.json") -> Dataset:
    """
    Load all CSVs from data_dir, preprocess, and build a TrafficAnalyzer.

    Parameters
    ----------
    data_dir
        Root directory containing data/ and config/ subdirectories.
        In Docker this is /data which maps to the repo root.
    city_config_path
        Path to city.json relative to data_dir.

    Returns
    -------
    Dataset
        Fully loaded and preprocessed dataset ready for API queries.

    Raises
    ------
    FileNotFoundError
        If any required CSV is missing.
    """

    # --- city config ---
    config_file = data_dir / city_config_path
    if not config_file.exists():
        raise FileNotFoundError(f"City config not found: {config_file}")
    with open(config_file) as f:
        city = json.load(f)
    logger.info("Loaded city config: %s (%s)", city["display_name"], city["city"])

    # --- resolve CSV paths relative to data_dir ---
    traffic_csv = data_dir / city["data"]["traffic_csv"]
    routes_csv = data_dir / city["data"]["routes_csv"]

    for path in (traffic_csv, routes_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    # --- load raw CSVs ---
    logger.info("Loading traffic data from %s ...", traffic_csv)
    traffic_raw = pd.read_csv(traffic_csv)
    logger.info("  %d raw rows loaded", len(traffic_raw))

    logger.info("Loading routes from %s ...", routes_csv)
    routes_df = pd.read_csv(routes_csv)
    logger.info("  %d routes loaded", len(routes_df))

    # --- preprocess ---
    logger.info("Preprocessing traffic data ...")
    traffic_df = preprocess_traffic_data(traffic_raw)
    logger.info("  %d rows after dedup/clean", len(traffic_df))

    logger.info("Computing temporal features ...")
    traffic_df = compute_temporal_features(traffic_df)

    # --- build analyzer ---
    logger.info("Building TrafficAnalyzer ...")
    analyzer = TrafficAnalyzer(traffic_df, routes_df)

    logger.info(
        "Dataset ready: %d rows, %d routes, period %d-%02d to %d-%02d",
        len(traffic_df),
        len(routes_df),
        traffic_df["year"].min(),
        traffic_df["month"].min(),
        traffic_df["year"].max(),
        traffic_df["month"].max(),
    )

    return Dataset(
        traffic_df=traffic_df,
        routes_df=routes_df,
        analyzer=analyzer,
        city_config=city,
    )
