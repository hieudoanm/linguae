"""Entry point for the Linguae FastAPI server."""

from __future__ import annotations

import uvicorn

from .app import app
from .config import settings
from .log import logger


def main() -> None:
    """Run the Linguae server with uvicorn."""
    logger.info(f"Starting Linguae server on {settings.host}:{settings.port}")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
