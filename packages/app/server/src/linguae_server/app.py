"""FastAPI application for the Linguae language detector."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .detector import Detector, detect, detect_batch
from .log import logger
from .middlewares import add_middlewares
from .schemas import (
    BatchDetectionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TextDetectionRequest,
)


def create_app() -> FastAPI:
    """Build the FastAPI application with middlewares and routes."""
    app = FastAPI(
        title="Linguae Language Detection API",
        description=(
            "Detect the language of text with the Linguae character-level model. "
            "Accepts JSON bodies or form uploads. One language label per string."
        ),
        version="0.0.1",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Health -----------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        logger.debug("Health check requested")
        artifact = Detector.get()
        logger.trace(
            f"Artifact loaded with {len(artifact['idx2label'])} languages on {artifact['device']}"
        )
        return HealthResponse(
            status="ok",
            device=str(artifact["device"]),
            num_languages=len(artifact["idx2label"]),
        )

    # --- Single text (JSON) ----------------------------------------------

    @app.post("/detect", response_model=PredictionResponse, tags=["detect"])
    async def detect_json(request: TextDetectionRequest) -> PredictionResponse:
        logger.info(f"Detect request (json) top_k={request.top_k}")
        try:
            result = detect(request.text, top_k=request.top_k)
        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.debug(
            f"Detected language: {result['language']} ({result['confidence']:.4f})"
        )
        return PredictionResponse(**result)

    # --- Single text (form) ----------------------------------------------

    @app.post("/detect/form", response_model=PredictionResponse, tags=["detect"])
    async def detect_form(
        text: Annotated[str, Form(..., min_length=1, max_length=5000)],
        top_k: Annotated[int, Form(ge=1, le=20)] = 3,
    ) -> PredictionResponse:
        logger.info(f"Detect request (form) top_k={top_k}")
        try:
            result = detect(text, top_k=top_k)
        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.debug(
            f"Detected language: {result['language']} ({result['confidence']:.4f})"
        )
        return PredictionResponse(**result)

    # --- Batch (JSON) -----------------------------------------------------

    @app.post("/detect/batch", response_model=BatchPredictionResponse, tags=["detect"])
    async def detect_batch_json(
        request: BatchDetectionRequest,
    ) -> BatchPredictionResponse:
        logger.info(
            f"Batch detect request (json) count={len(request.texts)} top_k={request.top_k}"
        )
        try:
            predictions = detect_batch(request.texts, top_k=request.top_k)
        except Exception as exc:
            logger.error(f"Batch detection failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.debug(f"Batch detected {len(predictions)} items")
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**p) for p in predictions],
            count=len(predictions),
        )

    # --- Batch (file upload) ----------------------------------------------

    @app.post("/detect/file", response_model=BatchPredictionResponse, tags=["detect"])
    async def detect_file(
        file: UploadFile,
        top_k: Annotated[int, Form(ge=1, le=20)] = 3,
    ) -> BatchPredictionResponse:
        if not file.filename:
            logger.warning("File upload without filename")
            raise HTTPException(status_code=400, detail="no file provided")
        try:
            content = await file.read()
            text = content.decode("utf-8")
        except Exception as exc:
            logger.warning(f"Could not read uploaded file: {exc}")
            raise HTTPException(
                status_code=400, detail=f"could not read file: {exc}"
            ) from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            logger.warning("Uploaded file contains no non-empty lines")
            raise HTTPException(
                status_code=400, detail="file contains no non-empty lines"
            )
        logger.info(
            f"Batch detect request (file) filename={file.filename} count={len(lines)} top_k={top_k}"
        )
        try:
            predictions = detect_batch(lines, top_k=top_k)
        except Exception as exc:
            logger.error(f"File batch detection failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.debug(f"File batch detected {len(predictions)} items")
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**p) for p in predictions],
            count=len(predictions),
        )

    # --- Exception handler -------------------------------------------------

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.fatal(f"Unhandled exception [request_id={request_id}]: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal server error",
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    return app


app = create_app()
app = add_middlewares(app)
