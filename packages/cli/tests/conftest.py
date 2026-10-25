"""Pytest configuration for linguae tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is in path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module state between tests."""
    yield
    # Clear any modules that might have cached state
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("linguae"):
            del sys.modules[mod_name]