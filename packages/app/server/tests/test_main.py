"""Tests for main module."""
from __future__ import annotations

from unittest.mock import patch


def test_main() -> None:
    with patch("linguae_server.main.uvicorn.run") as mock_run:
        from linguae_server.main import main

        main()
        mock_run.assert_called_once()
