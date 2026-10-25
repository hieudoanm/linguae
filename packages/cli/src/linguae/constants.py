"""Constants and configuration for the Linguae CLI."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ARTIFACT = Path("packages") / "models" / "artifacts" / "linguae"
REQUIRED_ARTIFACT_FILES = (
    "config.json",
    "vocabulary.json",
    "labels.json",
    "model.pt",
    "linguae_inference.py",
)
INFERENCE_MODULE = "_linguae_artifact_inference"