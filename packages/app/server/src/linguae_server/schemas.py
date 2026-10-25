"""Pydantic models for the Linguae server API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TextDetectionRequest(BaseModel):
    """Request body for single-text detection."""

    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to classify."
    )
    top_k: int = Field(
        default=3, ge=1, le=20, description="Number of alternatives to include."
    )


class BatchDetectionRequest(BaseModel):
    """Request body for batch detection."""

    texts: list[str] = Field(
        ..., min_length=1, max_length=100, description="Texts to classify."
    )
    top_k: int = Field(
        default=3, ge=1, le=20, description="Number of alternatives to include."
    )


class PredictionResponse(BaseModel):
    """Single prediction result."""

    text: str
    language: str
    language_name: str | None = None
    confidence: float
    top_k: list[dict[str, Any]] | None = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction result."""

    predictions: list[PredictionResponse]
    count: int


class FormDetectionRequest(BaseModel):
    """Form-encoded request body."""

    text: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=3, ge=1, le=20)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model: str = "linguae-character-cnn"
    device: str
    num_languages: int
    version: str = "0.0.1"
