"""Detector wrapper that loads the Linguae artifact and exposes a simple API."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Self

from .config import settings
from .inference import load_artifact, predict_batch, predict_text

# Map language codes (e.g. "eng") to English names (e.g. "English").
_CODE_TO_NAME: dict[str, str] = {}
_LABELS_PATH = Path(settings.labels_path)
if _LABELS_PATH.is_file():
    with _LABELS_PATH.open(encoding="utf-8") as _handle:
        _reader = csv.DictReader(_handle)
        for _row in _reader:
            _code = (_row.get("label") or "").strip()
            _name = (_row.get("name") or "").strip()
            if _code:
                _CODE_TO_NAME[_code] = _name


def _with_language_name(prediction: dict[str, Any]) -> dict[str, Any]:
    """Attach the human-readable language name to a prediction dict."""
    prediction["language_name"] = _CODE_TO_NAME.get(prediction.get("language", ""))
    return prediction


class Detector:
    """Lazy singleton wrapper around the Linguae artifact."""

    _instance: Self | None = None
    _artifact: dict[str, Any] | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get(cls) -> dict[str, Any]:
        """Load (or return cached) the artifact."""
        if cls._artifact is None:
            cls._artifact = load_artifact(
                str(settings.artifact_dir), device=settings.device
            )
        return cls._artifact

    @classmethod
    def reset(cls) -> None:
        """Drop the cached artifact (useful for testing)."""
        cls._artifact = None


def detect(text: str, top_k: int = 3) -> dict[str, Any]:
    """Predict one language label for a single string."""
    artifact = Detector.get()
    result = predict_text(text, artifact, top_k=top_k)
    return _with_language_name(result)


def detect_batch(texts: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    """Predict one language label per string in a list."""
    artifact = Detector.get()
    predictions = predict_batch(texts, artifact, batch_size=256)
    return [_with_language_name(prediction) for prediction in predictions]
