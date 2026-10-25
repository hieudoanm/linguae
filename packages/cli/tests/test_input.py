"""Tests for linguae.input module."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from linguae.errors import fail
from linguae.input import gather_texts


def test_gather_texts_from_argument():
    """Test gathering texts from text argument."""
    ctx = MagicMock(spec=typer.Context)
    texts, is_batch = gather_texts(ctx, "Hello world", None)
    assert texts == ["Hello world"]
    assert is_batch is False


def test_gather_texts_from_argument_strips_whitespace():
    """Test that text argument is stripped."""
    ctx = MagicMock(spec=typer.Context)
    texts, _ = gather_texts(ctx, "  Hello world  ", None)
    assert texts == ["Hello world"]


def test_gather_texts_empty_argument_raises():
    """Test that empty text argument raises error."""
    ctx = MagicMock(spec=typer.Context)
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, "", None)
    assert exc_info.value.exit_code == 2


def test_gather_texts_whitespace_only_argument_raises():
    """Test that whitespace-only text argument raises error."""
    ctx = MagicMock(spec=typer.Context)
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, "   ", None)
    assert exc_info.value.exit_code == 2


def test_gather_texts_from_file(tmp_path):
    """Test gathering texts from file."""
    ctx = MagicMock(spec=typer.Context)
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\n\nLine 3\n", encoding="utf-8")
    texts, is_batch = gather_texts(ctx, None, file_path)
    assert texts == ["Line 1", "Line 2", "Line 3"]
    assert is_batch is True


def test_gather_texts_from_file_skips_empty_lines(tmp_path):
    """Test that file reading skips empty lines."""
    ctx = MagicMock(spec=typer.Context)
    file_path = tmp_path / "test.txt"
    file_path.write_text("\n\nLine 1\n\n\nLine 2\n\n", encoding="utf-8")
    texts, _ = gather_texts(ctx, None, file_path)
    assert texts == ["Line 1", "Line 2"]


def test_gather_texts_empty_file_raises(tmp_path):
    """Test that empty file raises error."""
    ctx = MagicMock(spec=typer.Context)
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, None, file_path)
    assert exc_info.value.exit_code == 2


def test_gather_texts_file_with_only_whitespace_raises(tmp_path):
    """Test that file with only whitespace raises error."""
    ctx = MagicMock(spec=typer.Context)
    file_path = tmp_path / "whitespace.txt"
    file_path.write_text("   \n\n\t\n", encoding="utf-8")
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, None, file_path)
    assert exc_info.value.exit_code == 2


def test_gather_texts_file_not_found():
    """Test that non-existent file raises error."""
    ctx = MagicMock(spec=typer.Context)
    file_path = Path("/nonexistent/path/test.txt")
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, None, file_path)
    assert exc_info.value.exit_code == 2


def test_gather_texts_both_text_and_file_raises():
    """Test that providing both text and file raises error."""
    ctx = MagicMock(spec=typer.Context)
    file_path = Path("/tmp/test.txt")
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, "Hello", file_path)
    assert exc_info.value.exit_code == 2


def test_gather_texts_from_stdin(monkeypatch):
    """Test gathering texts from stdin."""
    ctx = MagicMock(spec=typer.Context)
    stdin_content = "Hello from stdin\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_content))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    texts, is_batch = gather_texts(ctx, None, None)
    assert texts == ["Hello from stdin"]
    assert is_batch is False


def test_gather_texts_empty_stdin_raises(monkeypatch):
    """Test that empty stdin raises error."""
    ctx = MagicMock(spec=typer.Context)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, None, None)
    assert exc_info.value.exit_code == 2


def test_gather_texts_tty_shows_help(monkeypatch):
    """Test that TTY stdin shows help and exits."""
    ctx = MagicMock(spec=typer.Context)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with pytest.raises(typer.Exit) as exc_info:
        gather_texts(ctx, None, None)
    assert exc_info.value.exit_code == 1
    ctx.get_help.assert_called_once()