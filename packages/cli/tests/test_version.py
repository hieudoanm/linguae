"""Tests for linguae.version module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from linguae.version import version


def test_version_returns_string():
    """Test that version returns a string."""
    result = version()
    assert isinstance(result, str)
    assert len(result) > 0


@patch("linguae.version.importlib.metadata.version")
def test_version_from_metadata(mock_version):
    """Test that version uses importlib.metadata when available."""
    mock_version.return_value = "1.2.3"
    result = version()
    assert result == "1.2.3"
    mock_version.assert_called_once_with("linguae")


@patch("linguae.version.importlib.metadata.version")
def test_version_fallback_on_error(mock_version):
    """Test that version falls back to default when metadata not found."""
    import importlib.metadata
    mock_version.side_effect = importlib.metadata.PackageNotFoundError
    result = version()
    assert result == "0.0.1"