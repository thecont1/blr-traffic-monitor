"""
Tests for health endpoints (/livez, /readyz).
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from service.main import app, _dataset


@pytest.fixture
def client():
    return TestClient(app)


def test_livez_returns_200(client):
    """Liveness always returns 200 regardless of dataset state."""
    resp = client.get("/livez")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readyz_returns_503_when_no_data(client):
    """Readiness returns 503 if dataset is not loaded."""
    with patch("service.main._dataset", None):
        resp = client.get("/readyz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"


def test_readyz_returns_200_when_loaded(client, sample_dataset):
    """Readiness returns 200 with stats when dataset is loaded."""
    with patch("service.main._dataset", sample_dataset):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["routes"] == 3
        assert body["rows"] > 0
        assert body["city"] == "Test City"
