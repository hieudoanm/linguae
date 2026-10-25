"""Tests for detection endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_detect_json(client: TestClient) -> None:
    response = client.post("/detect", json={"text": "Hello world", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["language"] == "eng"
    assert data["confidence"] == 0.99
    assert isinstance(data["top_k"], list)
    assert "X-Request-ID" in response.headers


def test_detect_json_default_top_k(client: TestClient) -> None:
    response = client.post("/detect", json={"text": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "eng"


def test_detect_form(client: TestClient) -> None:
    response = client.post("/detect/form", data={"text": "Bonjour", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Bonjour"
    assert data["language"] == "eng"


def test_detect_form_default_top_k(client: TestClient) -> None:
    response = client.post("/detect/form", data={"text": "Bonjour"})
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "eng"


def test_detect_batch_json(client: TestClient) -> None:
    response = client.post("/detect/batch", json={"texts": ["Hello", "World"], "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert all(p["language"] == "eng" for p in data["predictions"])


def test_detect_batch_json_default_top_k(client: TestClient) -> None:
    response = client.post("/detect/batch", json={"texts": ["Hello"]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1


def test_detect_file(client: TestClient) -> None:
    content = b"Hello world\nBonjour\n\n"
    response = client.post("/detect/file", files={"file": ("test.txt", content, "text/plain")}, data={"top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_detect_file_no_filename(client: TestClient) -> None:
    response = client.post("/detect/file", files={"file": ("", b"hello", "text/plain")})
    assert response.status_code in (400, 422)


def test_detect_file_empty_lines(client: TestClient) -> None:
    response = client.post("/detect/file", files={"file": ("test.txt", b"\n\n   \n", "text/plain")})
    assert response.status_code == 400
