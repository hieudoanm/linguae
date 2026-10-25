"""Tests for detector module."""
from __future__ import annotations

from typing import Any

import pytest


def test_detector_singleton() -> None:
    from linguae_server import detector

    detector.Detector.reset()
    d1 = detector.Detector()
    d2 = detector.Detector()
    assert d1 is d2


def test_detect_batch_with_top_k() -> None:
    artifact = {
        "device": "cpu",
        "idx2label": {0: "eng", 1: "fra"},
    }

    class MockDetector:
        @classmethod
        def get(cls) -> dict[str, Any]:
            return artifact

    import linguae_server.detector as detector_module

    def mock_predict_batch(texts, artifact, batch_size=256):
        return [{"text": t, "language": "eng", "confidence": 0.9} for t in texts]

    detector_module.Detector.get = MockDetector.get
    detector_module.predict_batch = mock_predict_batch
    result = detector_module.detect_batch(["a", "b"], top_k=5)
    assert len(result) == 2
