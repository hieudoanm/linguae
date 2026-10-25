"""Version utilities for the Linguae CLI."""

from __future__ import annotations

import importlib.metadata


def version() -> str:
    """Return the installed CLI version, tolerating source-tree runs."""
    try:
        return importlib.metadata.version("linguae")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.1"