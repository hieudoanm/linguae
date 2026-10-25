"""Detector wrapper that loads the Linguae artifact and exposes a simple API."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Self

from .config import settings

# Allow the artifact's standalone inference module to be imported.
_ARTIFACT_DIR = Path(settings.artifact_dir)
if str(_ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ARTIFACT_DIR))

try:
    from linguae_inference import load_artifact, predict_batch, predict_text  # type: ignore
except Exception:  # pragma: no cover - artifact not present at import time  # noqa: BLE001
    load_artifact = None  # type: ignore
    predict_batch = None  # type: ignore
    predict_text = None  # type: ignore

# Map language codes (e.g. "eng") to English names (e.g. "English").
_CODE_TO_NAME: dict[str, str] = {}
_LABELS_PATH = Path(settings.labels_path)
if _LABELS_PATH.is_file():
    with _LABELS_PATH.open(encoding="utf-8") as _handle:
        _reader = csv.DictReader(_handle, delimiter=";")
        for _row in _reader:
            _code = (_row.get("Label") or "").strip()
            _name = (_row.get("English") or "").strip()
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
            if load_artifact is None:
                raise RuntimeError(
                    "linguae_inference module not available in artifact directory"
                )
            cls._artifact = load_artifact(str(_ARTIFACT_DIR), device=settings.device)
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
