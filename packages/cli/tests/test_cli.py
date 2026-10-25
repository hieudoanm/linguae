"""Tests for linguae.cli main module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

from linguae.cli import app


runner = CliRunner()


def test_version_option(monkeypatch):
    """Test --version option prints version and exits."""
    monkeypatch.setattr("linguae.version.version", lambda: "1.2.3")
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "linguae 1.2.3" in result.output


def test_version_short_option(monkeypatch):
    """Test -V option prints version and exits."""
    monkeypatch.setattr("linguae.version.version", lambda: "1.2.3")
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "linguae 1.2.3" in result.output


def test_help_option():
    """Test --help option shows help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Detect the language" in result.output
    assert "--artifact" in result.output
    assert "--confidence" in result.output
    assert "--json" in result.output


def test_no_args_shows_help():
    """Test that no args shows help and exits."""
    result = runner.invoke(app, [], input="")
    assert result.exit_code in (1, 2)


def _setup_single_text_mocks(monkeypatch):
    """Set up mocks for single text tests."""
    mock_gather = MagicMock()
    mock_resolve_artifact = MagicMock()
    mock_load_detector = MagicMock()
    mock_module = MagicMock()

    mock_gather.return_value = (["Hello world"], False)
    mock_resolve_artifact.return_value = Path("/fake/artifact")
    mock_module.predict_text.return_value = {
        "text": "Hello world",
        "language": "eng",
        "confidence": 0.951234,
        "top_k": [
            {"language": "eng", "score": 0.951234},
            {"language": "sco", "score": 0.03},
            {"language": "ibo", "score": 0.01},
            {"language": "bcl", "score": 0.005},
        ],
    }
    mock_module.predict_batch.return_value = [
        {"text": "Line 1", "language": "eng", "confidence": 0.9},
        {"text": "Line 2", "language": "fra", "confidence": 0.8},
    ]
    mock_load_detector.return_value = (mock_module, {"device": "cpu"})

    monkeypatch.setattr("linguae.input.gather_texts", mock_gather)
    monkeypatch.setattr("linguae.artifact.resolve_artifact", mock_resolve_artifact)
    monkeypatch.setattr("linguae.artifact.load_detector", mock_load_detector)

    return mock_gather, mock_resolve_artifact, mock_load_detector, mock_module


def _setup_batch_mocks(monkeypatch):
    """Set up mocks for batch/file tests."""
    mock_gather = MagicMock()
    mock_resolve_artifact = MagicMock()
    mock_load_detector = MagicMock()
    mock_module = MagicMock()

    mock_gather.return_value = (["Line 1", "Line 2"], True)
    mock_resolve_artifact.return_value = Path("/fake/artifact")
    mock_module.predict_batch.return_value = [
        {"text": "Line 1", "language": "eng", "confidence": 0.9},
        {"text": "Line 2", "language": "fra", "confidence": 0.8},
    ]
    mock_load_detector.return_value = (mock_module, {"device": "cpu"})

    monkeypatch.setattr("linguae.input.gather_texts", mock_gather)
    monkeypatch.setattr("linguae.artifact.resolve_artifact", mock_resolve_artifact)
    monkeypatch.setattr("linguae.artifact.load_detector", mock_load_detector)

    return mock_gather, mock_resolve_artifact, mock_load_detector, mock_module


def test_main_single_text(monkeypatch):
    """Test main with single text argument."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_single_text_mocks(monkeypatch)

    result = runner.invoke(app, ["Hello world"])
    assert result.exit_code == 0
    assert "eng" in result.output
    mock_gather.assert_called_once()
    mock_resolve_artifact.assert_called_once()


def test_main_with_confidence(monkeypatch):
    """Test main with --confidence option."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_single_text_mocks(monkeypatch)

    result = runner.invoke(app, ["--confidence", "Hello world"])
    assert result.exit_code == 0
    assert "eng\t0.9512" in result.output


def test_main_with_json(monkeypatch):
    """Test main with --json option."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_single_text_mocks(monkeypatch)

    result = runner.invoke(app, ["--json", "Hello world"])
    assert result.exit_code == 0
    assert '"language": "eng"' in result.output
    assert '"confidence": 0.951234' in result.output
    assert '"top_k"' in result.output


def test_main_batch_from_file(monkeypatch, tmp_path):
    """Test main with --file option."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_batch_mocks(monkeypatch)

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\n", encoding="utf-8")

    result = runner.invoke(app, ["--file", str(file_path)])
    assert result.exit_code == 0
    assert "eng" in result.output
    assert "fra" in result.output


def test_main_with_device_option(monkeypatch):
    """Test main with --device option."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_single_text_mocks(monkeypatch)

    result = runner.invoke(app, ["--device", "mps", "Hello world"])
    assert result.exit_code == 0
    mock_load_detector.assert_called_once_with(Path("/fake/artifact"), "mps")


def test_main_with_top_k_option(monkeypatch):
    """Test main with --top-k option."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_single_text_mocks(monkeypatch)

    result = runner.invoke(app, ["--top-k", "5", "--json", "Hello world"])
    assert result.exit_code == 0
    import json
    data = json.loads(result.output.strip())
    assert len(data["top_k"]) == 4


def test_main_with_artifact_option(monkeypatch):
    """Test main with --artifact option."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_single_text_mocks(monkeypatch)

    result = runner.invoke(app, ["--artifact", "/custom/artifact", "Hello world"])
    assert result.exit_code == 0
    mock_resolve_artifact.assert_called_once_with(Path("/custom/artifact"))


def test_main_from_stdin(monkeypatch):
    """Test main reading from stdin."""
    mock_gather, mock_resolve_artifact, mock_load_detector, mock_module = _setup_single_text_mocks(monkeypatch)

    result = runner.invoke(app, [], input="Hello from stdin\n")
    assert result.exit_code == 0
    assert "eng" in result.output


def test_main_artifact_not_found(monkeypatch):
    """Test main when artifact not found."""
    mock_gather = MagicMock()
    mock_gather.return_value = (["Hello world"], False)
    monkeypatch.setattr("linguae.input.gather_texts", mock_gather)
    monkeypatch.setattr("linguae.artifact.resolve_artifact", MagicMock(side_effect=typer.Exit(1)))

    result = runner.invoke(app, ["Hello world"])
    assert result.exit_code == 1


def test_main_load_detector_fails(monkeypatch):
    """Test main when load_detector fails."""
    mock_gather = MagicMock()
    mock_gather.return_value = (["Hello world"], False)
    monkeypatch.setattr("linguae.input.gather_texts", mock_gather)
    monkeypatch.setattr("linguae.artifact.resolve_artifact", MagicMock(return_value=Path("/fake/artifact")))
    monkeypatch.setattr("linguae.artifact.load_detector", MagicMock(side_effect=typer.Exit(1)))

    result = runner.invoke(app, ["Hello world"])
    assert result.exit_code == 1