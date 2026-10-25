"""Artifact resolution and loading for the Linguae CLI."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from linguae.constants import DEFAULT_ARTIFACT, INFERENCE_MODULE, REQUIRED_ARTIFACT_FILES
from linguae.errors import fail


def _missing_artifact_files(directory: Path) -> list[str]:
    """Return the required artifact files that are absent from a directory."""
    return [name for name in REQUIRED_ARTIFACT_FILES if not (directory / name).is_file()]


def _artifact_candidates() -> list[Path]:
    """Build a deduplicated list of likely artifact locations."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    # PyInstaller bundle: artifacts are extracted to sys._MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        add(Path(sys._MEIPASS) / "artifacts" / "linguae")

    for root in (Path.cwd(), *Path.cwd().parents):
        add(root / DEFAULT_ARTIFACT)
        add(root / "artifacts" / "linguae")

    # Works for editable installs: <repo>/packages/cli/src/linguae/cli.py
    module_file = Path(__file__).resolve()
    if len(module_file.parents) >= 4:
        add(module_file.parents[3] / "models" / "artifacts" / "linguae")

    return candidates


def resolve_artifact(explicit: Path | None) -> Path:
    """Resolve and validate the artifact directory."""
    if explicit is not None:
        directory = explicit.expanduser().resolve()
        if not directory.is_dir():
            fail(f"artifact directory not found: {directory}")
        missing = _missing_artifact_files(directory)
        if missing:
            fail(f"{directory} is missing: {', '.join(missing)}")
        return directory

    environment = os.environ.get("LINGUAE_ARTIFACT_DIR")
    if environment:
        directory = Path(environment).expanduser().resolve()
        if directory.is_dir() and not _missing_artifact_files(directory):
            return directory
        fail(f"LINGUAE_ARTIFACT_DIR={environment!r} is not a valid artifact directory")

    for candidate in _artifact_candidates():
        if candidate.is_dir() and not _missing_artifact_files(candidate):
            return candidate.resolve()

    fail(
        "could not find the Linguae artifact. Pass --artifact PATH or set "
        "LINGUAE_ARTIFACT_DIR. Expected files: "
        + ", ".join(REQUIRED_ARTIFACT_FILES)
    )


def load_inference_module(artifact_dir: Path) -> Any:
    """Import the artifact's standalone inference module."""
    module_path = artifact_dir / "linguae_inference.py"
    spec = importlib.util.spec_from_file_location(INFERENCE_MODULE, module_path)
    if spec is None or spec.loader is None:
        fail(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[INFERENCE_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        fail(f"could not import {module_path}: {error}")
    return module


def load_detector(artifact_dir: Path, device: str) -> tuple[Any, dict]:
    """Load the inference module and model artifact."""
    module = load_inference_module(artifact_dir)
    try:
        artifact = module.load_artifact(artifact_dir, device=device)
    except Exception as error:
        fail(f"could not load artifact {artifact_dir}: {error}")
    return module, artifact