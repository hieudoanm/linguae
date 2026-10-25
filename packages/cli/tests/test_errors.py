"""Tests for linguae.errors module."""

from __future__ import annotations

import pytest
import typer

from linguae.errors import fail


def test_fail_raises_exit():
    """Test that fail raises typer.Exit with correct code."""
    with pytest.raises(typer.Exit) as exc_info:
        fail("test error message", code=2)
    assert exc_info.value.exit_code == 2


def test_fail_default_code():
    """Test that fail uses default code 1."""
    with pytest.raises(typer.Exit) as exc_info:
        fail("test error message")
    assert exc_info.value.exit_code == 1