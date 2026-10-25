"""Error handling utilities for the Linguae CLI."""

from __future__ import annotations

import typer
from typing import NoReturn


def fail(message: str, code: int = 1) -> NoReturn:
    """Print an error and exit with the given status code."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)