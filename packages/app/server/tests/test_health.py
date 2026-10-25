"""Tests for health endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["device"] == "cpu"
    assert data["num_languages"] == 3
    assert "X-Request-ID" in response.headers
