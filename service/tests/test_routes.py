"""
Tests for route listing and traffic read endpoints.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from service.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_list_routes(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/routes")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["routes"]) == 3
        codes = {r["route_code"] for r in body["routes"]}
        assert "AAA|BBB" in codes


def test_traffic_all(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/traffic?limit=100")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] > 0
        assert len(body["rows"]) > 0


def test_traffic_filter_by_route(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/traffic?route_code=AAA|BBB&limit=100")
        assert resp.status_code == 200
        body = resp.json()
        for row in body["rows"]:
            assert row["route_code"] == "AAA|BBB"


def test_traffic_aggregate_by_hour(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/traffic/aggregate?group_by=hour&agg=mean")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_by"] == "hour"
        assert len(body["rows"]) > 0


def test_compare_routes(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/compare?routes=AAA|BBB,CCC|DDD")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["routes"]) == 2


def test_compare_needs_two_routes(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/compare?routes=AAA|BBB")
        assert resp.status_code == 400


def test_percentiles(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/stats/percentiles?route_code=AAA|BBB")
        assert resp.status_code == 200
        body = resp.json()
        assert "p50" in body["percentiles"]


def test_distribution(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/stats/distribution?route_code=AAA|BBB&bins=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["bins"]) == 5


def test_rrs(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/stats/rrs")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["routes"]) == 3


def test_anomalies(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/api/anomalies?method=iqr&threshold=1.5")
        assert resp.status_code == 200
        body = resp.json()
        assert "anomaly_count" in body


def test_schema(client, sample_dataset):
    with patch("service.main._dataset", sample_dataset):
        # Patch the data_dir to point somewhere with SCHEMA.md
        from unittest.mock import MagicMock
        with patch("service.routers.reports.settings") as mock_settings:
            from pathlib import Path
            mock_settings.data_dir = Path(__file__).resolve().parent.parent.parent
            resp = client.get("/api/schema")
            assert resp.status_code == 200
            body = resp.json()
            assert "SCHEMA" in body["title"] or body["content"] != ""
