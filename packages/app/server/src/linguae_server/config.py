"""Configuration for the Linguae FastAPI server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Runtime settings, overridable via environment variables."""

    host: str = os.environ.get("LINGUAE_SERVER_HOST", "0.0.0.0")
    port: int = int(os.environ.get("LINGUAE_SERVER_PORT", "8000"))
    artifact_dir: Path = Path(
        os.environ.get(
            "LINGUAE_ARTIFACT_DIR",
            str(
                Path(__file__).resolve().parents[5]
                / "packages"
                / "models"
                / "artifacts"
                / "linguae"
            ),
        )
    )
    labels_path: Path = Path(
        os.environ.get(
            "LINGUAE_LABELS_PATH",
            str(
                Path(__file__).resolve().parents[5]
                / "packages"
                / "models"
                / "data"
                / "txt"
                / "wili-2018"
                / "labels.csv"
            ),
        )
    )
    device: str = os.environ.get("LINGUAE_DEVICE", "cpu")
    max_texts: int = int(os.environ.get("LINGUAE_MAX_TEXTS", "100"))
    max_text_length: int = int(os.environ.get("LINGUAE_MAX_TEXT_LENGTH", "5000"))
    max_batch_bytes: int = int(
        os.environ.get("LINGUAE_MAX_BATCH_BYTES", str(2 * 1024 * 1024))
    )
    default_top_k: int = int(os.environ.get("LINGUAE_DEFAULT_TOP_K", "3"))

    # Rate limiting
    rate_limit_enabled: bool = os.environ.get("LINGUAE_RATE_LIMIT", "true") == "true"
    rate_limit_requests: int = int(os.environ.get("LINGUAE_RATE_LIMIT_REQUESTS", "60"))
    rate_limit_window_seconds: int = int(
        os.environ.get("LINGUAE_RATE_LIMIT_WINDOW", "60")
    )

    # CORS
    cors_enabled: bool = os.environ.get("LINGUAE_CORS", "true") == "true"
    cors_origins: list[str] = field(
        default_factory=lambda: os.environ.get("LINGUAE_CORS_ORIGINS", "*").split(",")
    )
    cors_allow_credentials: bool = (
        os.environ.get("LINGUAE_CORS_CREDENTIALS", "true") == "true"
    )
    cors_allow_methods: list[str] = field(
        default_factory=lambda: os.environ.get("LINGUAE_CORS_METHODS", "*").split(",")
    )
    cors_allow_headers: list[str] = field(
        default_factory=lambda: os.environ.get("LINGUAE_CORS_HEADERS", "*").split(",")
    )

    # Timeout
    request_timeout_seconds: float = float(os.environ.get("LINGUAE_TIMEOUT", "30.0"))

    # Request ID
    request_id_header: str = os.environ.get("LINGUAE_REQUEST_ID_HEADER", "X-Request-ID")


settings = Settings()
