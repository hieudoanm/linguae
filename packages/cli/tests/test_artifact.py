"""Tests for linguae.artifact module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from linguae.artifact import (
    _artifact_candidates,
    _missing_artifact_files,
    load_detector,
    load_inference_module,
    resolve_artifact,
)
from linguae.constants import REQUIRED_ARTIFACT_FILES


def test_missing_artifact_files_none_missing(tmp_path):
    """Test _missing_artifact_files when all files present."""
    for fname in REQUIRED_ARTIFACT_FILES:
        (tmp_path / fname).write_text("", encoding="utf-8")
    missing = _missing_artifact_files(tmp_path)
    assert missing == []


def test_missing_artifact_files_some_missing(tmp_path):
    """Test _missing_artifact_files when some files missing."""
    (tmp_path / "config.json").write_text("", encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("", encoding="utf-8")
    missing = _missing_artifact_files(tmp_path)
    assert "labels.json" in missing
    assert "model.pt" in missing
    assert "linguae_inference.py" in missing
    assert len(missing) == 3


def test_missing_artifact_files_all_missing(tmp_path):
    """Test _missing_artifact_files when directory is empty."""
    missing = _missing_artifact_files(tmp_path)
    assert set(missing) == set(REQUIRED_ARTIFACT_FILES)


def test_artifact_candidates_includes_pyinstaller_path(monkeypatch):
    """Test that _artifact_candidates includes PyInstaller path when frozen."""
    # Simulate frozen state by directly modifying sys attributes
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/fake/meipass", raising=False)
    candidates = _artifact_candidates()
    assert Path("/fake/meipass/artifacts/linguae") in candidates


def test_artifact_candidates_includes_cwd_and_parents(tmp_path, monkeypatch):
    """Test that _artifact_candidates includes cwd and parents."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    candidates = _artifact_candidates()
    # Should include cwd / DEFAULT_ARTIFACT and cwd / artifacts / linguae
    assert any("packages/models/artifacts/linguae" in str(c) for c in candidates)
    assert any("artifacts/linguae" in str(c) for c in candidates)


def test_resolve_artifact_explicit_valid(tmp_path):
    """Test resolve_artifact with explicit valid path."""
    for fname in REQUIRED_ARTIFACT_FILES:
        (tmp_path / fname).write_text("", encoding="utf-8")
    result = resolve_artifact(tmp_path)
    assert result == tmp_path.resolve()


def test_resolve_artifact_explicit_not_dir(tmp_path):
    """Test resolve_artifact with explicit path that's not a directory."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("", encoding="utf-8")
    with pytest.raises(typer.Exit) as exc_info:
        resolve_artifact(file_path)
    assert exc_info.value.exit_code == 1


def test_resolve_artifact_explicit_missing_files(tmp_path):
    """Test resolve_artifact with explicit path missing required files."""
    (tmp_path / "config.json").write_text("", encoding="utf-8")
    with pytest.raises(typer.Exit) as exc_info:
        resolve_artifact(tmp_path)
    assert exc_info.value.exit_code == 1


def test_resolve_artifact_env_var_valid(tmp_path, monkeypatch):
    """Test resolve_artifact with valid LINGUAE_ARTIFACT_DIR."""
    for fname in REQUIRED_ARTIFACT_FILES:
        (tmp_path / fname).write_text("", encoding="utf-8")
    monkeypatch.setenv("LINGUAE_ARTIFACT_DIR", str(tmp_path))
    # Need to clear module cache to re-read env var
    import importlib
    import linguae.artifact
    importlib.reload(linguae.artifact)
    result = linguae.artifact.resolve_artifact(None)
    assert result == tmp_path.resolve()


def test_resolve_artifact_env_var_invalid(tmp_path, monkeypatch):
    """Test resolve_artifact with invalid LINGUAE_ARTIFACT_DIR."""
    monkeypatch.setenv("LINGUAE_ARTIFACT_DIR", str(tmp_path))
    import importlib
    import linguae.artifact
    importlib.reload(linguae.artifact)
    with pytest.raises(typer.Exit) as exc_info:
        linguae.artifact.resolve_artifact(None)
    assert exc_info.value.exit_code == 1


def test_load_inference_module(tmp_path):
    """Test load_inference_module loads the module."""
    module_path = tmp_path / "linguae_inference.py"
    module_path.write_text("def foo(): return 'bar'", encoding="utf-8")
    module = load_inference_module(tmp_path)
    assert hasattr(module, "foo")
    assert module.foo() == "bar"


def test_load_inference_module_missing_file(tmp_path):
    """Test load_inference_module with missing file."""
    with pytest.raises(typer.Exit) as exc_info:
        load_inference_module(tmp_path)
    assert exc_info.value.exit_code == 1


def test_load_detector(tmp_path):
    """Test load_detector loads module and calls load_artifact."""
    module_path = tmp_path / "linguae_inference.py"
    module_path.write_text("""
def load_artifact(directory, device):
    return {'directory': directory, 'device': device}
""", encoding="utf-8")
    module, artifact = load_detector(tmp_path, "cpu")
    assert artifact["device"] == "cpu"
    assert artifact["directory"] == tmp_path


def test_load_detector_load_error(tmp_path):
    """Test load_detector handles load_artifact error."""
    module_path = tmp_path / "linguae_inference.py"
    module_path.write_text("""
def load_artifact(directory, device):
    raise RuntimeError("load failed")
""", encoding="utf-8")
    with pytest.raises(typer.Exit) as exc_info:
        load_detector(tmp_path, "cpu")
    assert exc_info.value.exit_code == 1