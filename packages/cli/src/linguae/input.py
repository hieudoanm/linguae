"""Input text gathering for the Linguae CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from linguae.errors import fail


def gather_texts(
    ctx: typer.Context,
    text: str | None,
    input_file: Path | None,
) -> tuple[list[str], bool]:
    """Collect input texts and report whether they came from a file."""
    if input_file is not None and text is not None:
        fail("provide either TEXT or --file, not both", 2)

    if input_file is not None:
        try:
            content = input_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            fail(f"could not read {input_file}: {error}", 2)
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            fail(f"{input_file} contains no non-empty lines", 2)
        return lines, True

    if text is not None:
        stripped = text.strip()
        if not stripped:
            fail("text is empty", 2)
        return [stripped], False

    if sys.stdin.isatty():
        typer.echo(ctx.get_help())
        raise typer.Exit(1)

    content = sys.stdin.read()
    if not content.strip():
        fail("no input text received on stdin", 2)
    return [content.strip()], False