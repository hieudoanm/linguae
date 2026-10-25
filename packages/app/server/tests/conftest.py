"""Tests for the Linguae FastAPI server."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_detector(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock the detector to avoid loading the real artifact."""
    artifact = {
        "device": "cpu",
        "idx2label": {0: "eng", 1: "fra", 2: "spa"},
        "vocab": {"<pad>": 0, "<unk>": 1, "a": 2},
        "max_len": 100,
    }

    class MockDetector:
        _artifact = artifact

        @classmethod
        def get(cls) -> dict[str, Any]:
            return cls._artifact

        @classmethod
        def reset(cls) -> None:
            pass

    def mock_detect(text: str, top_k: int = 3) -> dict[str, Any]:
        return {
            "text": text,
            "language": "eng",
            "confidence": 0.99,
            "top_k": [
                {"language": "eng", "confidence": 0.99},
                {"language": "fra", "confidence": 0.01},
            ],
        }

    def mock_detect_batch(texts: list[str], top_k: int = 3) -> list[dict[str, Any]]:
        return [mock_detect(t, top_k) for t in texts]

    import linguae_server.detector as detector_module

    monkeypatch.setattr(detector_module.Detector, "get", MockDetector.get)
    monkeypatch.setattr(detector_module, "detect", mock_detect)
    monkeypatch.setattr(detector_module, "detect_batch", mock_detect_batch)

    return artifact


@pytest.fixture
def client(mock_detector: dict[str, Any]) -> TestClient:
    """Create a test client with mocked detector."""
    from linguae_server.app import create_app
    from linguae_server.middlewares import add_middlewares

    app = create_app()
    app = add_middlewares(app)
    return TestClient(app)
