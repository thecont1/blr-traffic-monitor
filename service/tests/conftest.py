"""
Shared fixtures for service tests.

Creates a lightweight test dataset so tests don't need the full 70K-row CSV.
"""

import pandas as pd
import pytest

from data_utils import compute_temporal_features, preprocess_traffic_data
from traffic_analyzer import TrafficAnalyzer


@pytest.fixture
def sample_traffic_df():
    """Small synthetic traffic DataFrame for testing."""
    rows = []
    routes = ["AAA|BBB", "CCC|DDD", "EEE|FFF"]
    for day in range(1, 4):
        for hour in [8, 12, 17]:
            for rc in routes:
                rows.append({
                    "date": f"2025-01-{day:02d}",
                    "time": f"{hour:02d}:00",
                    "route_code": rc,
                    "duration": 20 + hash(rc + str(hour)) % 30,
                    "distance": 10.0 + hash(rc) % 5,
                    "temp": 25,
                    "realfeel": 24,
                    "humidity": 60,
                    "rsi_flag": "",
                    "aqi": 50,
                })
    df = pd.DataFrame(rows)
    df = preprocess_traffic_data(df)
    df = compute_temporal_features(df)
    return df


@pytest.fixture
def sample_routes_df():
    """Small synthetic routes DataFrame."""
    return pd.DataFrame([
        {"route_code": "AAA|BBB", "label_full": "Origin A to Dest B", "label_short": "Route AB", "map_link": "", "accuweather_station": "station-a"},
        {"route_code": "CCC|DDD", "label_full": "Origin C to Dest D", "label_short": "Route CD", "map_link": "", "accuweather_station": "station-b"},
        {"route_code": "EEE|FFF", "label_full": "Origin E to Dest F", "label_short": "Route EF", "map_link": "", "accuweather_station": "station-c"},
    ])


@pytest.fixture
def sample_analyzer(sample_traffic_df, sample_routes_df):
    """TrafficAnalyzer with synthetic data."""
    return TrafficAnalyzer(sample_traffic_df, sample_routes_df)


@pytest.fixture
def sample_dataset(sample_traffic_df, sample_routes_df, sample_analyzer):
    """Dataset-like object for router tests."""
    from service.data_loader import Dataset

    return Dataset(
        traffic_df=sample_traffic_df,
        routes_df=sample_routes_df,
        analyzer=sample_analyzer,
        city_config={"display_name": "Test City", "city": "test"},
    )
