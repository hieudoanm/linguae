"""Tests for middleware."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_request_id_echo(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-123"


def test_gzip_compression(client: TestClient) -> None:
    response = client.get("/health", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200


def test_response_time_header(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Response-Time-Seconds" in response.headers
