"""Tests for linguae.constants module."""

from __future__ import annotations

from pathlib import Path

from linguae.constants import (
    DEFAULT_ARTIFACT,
    INFERENCE_MODULE,
    REQUIRED_ARTIFACT_FILES,
)


def test_default_artifact_path():
    """Test that DEFAULT_ARTIFACT is a Path with expected components."""
    assert isinstance(DEFAULT_ARTIFACT, Path)
    assert "packages" in DEFAULT_ARTIFACT.parts
    assert "models" in DEFAULT_ARTIFACT.parts
    assert "artifacts" in DEFAULT_ARTIFACT.parts
    assert "linguae" in DEFAULT_ARTIFACT.parts


def test_required_artifact_files():
    """Test that REQUIRED_ARTIFACT_FILES contains expected files."""
    assert isinstance(REQUIRED_ARTIFACT_FILES, tuple)
    assert "config.json" in REQUIRED_ARTIFACT_FILES
    assert "vocabulary.json" in REQUIRED_ARTIFACT_FILES
    assert "labels.json" in REQUIRED_ARTIFACT_FILES
    assert "model.pt" in REQUIRED_ARTIFACT_FILES
    assert "linguae_inference.py" in REQUIRED_ARTIFACT_FILES
    assert len(REQUIRED_ARTIFACT_FILES) == 5


def test_inference_module_name():
    """Test that INFERENCE_MODULE has expected value."""
    assert INFERENCE_MODULE == "_linguae_artifact_inference"