"""Convert the WiLI 2018 text/label files into a single CSV dataset.

This script mirrors ``notebooks/wili_2018_txt_to_csv.ipynb`` and is intended for
headless runs. It reads the four WiLI 2018 split files, validates that texts and
labels line up, and writes ``data/csv/wili-2018.csv`` with ``text`` and
``language`` columns. Progress and dataset statistics are reported through the
standard :mod:`logging` module.

The converted CSV supplies the *long* (paragraph-length) examples for
``scripts/train_language_detector.py``. The training pipeline also reads
``data/csv/tatoeba.tsv`` (headerless ``id``, ``language``, ``text``) for *short*
sentences; that file is downloaded separately and is not produced here.

Usage::

    cd packages/models
    uv run python scripts/wili_2018_txt_to_csv.py

Environment overrides (all optional)::

    LINGUAE_LOG_LEVEL     DEBUG, INFO (default), WARNING, ...
    LINGUAE_WILI_DIR      directory with x_train.txt, y_train.txt, ...
    LINGUAE_WILI_CSV      output CSV path
    LINGUAE_PACKAGE_ROOT  override package-root discovery
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import pandas as pd

LOG_LEVEL = os.environ.get("LINGUAE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("linguae.wili")
SCRIPT_STARTED = time.perf_counter()


def log_section(title: str) -> None:
    """Log a stage banner with elapsed wall-clock time."""
    logger.info("=" * 78)
    logger.info("STEP: %s (t+%.1fs)", title, time.perf_counter() - SCRIPT_STARTED)
    logger.info("=" * 78)


def find_package_root(start: Path | None = None) -> Path:
    """Locate packages/models by walking up from the script or current directory."""
    override = os.environ.get("LINGUAE_PACKAGE_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / "pyproject.toml").is_file() and (root / "data").is_dir():
            return root
        raise FileNotFoundError(
            f"LINGUAE_PACKAGE_ROOT={override!r} does not look like packages/models."
        )
    origins = [Path(start)] if start else [Path.cwd()]
    script_file = globals().get("__file__")
    if script_file:
        origins.append(Path(str(script_file)).resolve().parent)
    for origin in origins:
        for candidate in (origin.resolve(), *origin.resolve().parents):
            for option in (candidate, candidate / "packages" / "models"):
                if (
                    (option / "pyproject.toml").is_file()
                    and (option / "data").is_dir()
                    and (option / "notebooks").is_dir()
                ):
                    return option
    raise FileNotFoundError(
        "Could not locate packages/models. Run the script from inside the repository, "
        "or set LINGUAE_PACKAGE_ROOT=/absolute/path/to/packages/models."
    )


def read_lines(path: Path) -> list[str]:
    """Read a UTF-8 text file and return its lines, logging size and count."""
    if not path.is_file():
        raise FileNotFoundError(f"missing WiLI 2018 file: {path}")
    started = time.perf_counter()
    lines = path.read_text(encoding="utf-8").splitlines()
    logger.info(
        "read %-12s %8d lines | %7.1f MiB | %.2f s",
        path.name,
        len(lines),
        path.stat().st_size / (1024 * 1024),
        time.perf_counter() - started,
    )
    return lines


def sha256_of(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


log_section("1. Resolve paths")
BASE = find_package_root()
DATA_DIR = Path(
    os.environ.get("LINGUAE_WILI_DIR", str(BASE / "data" / "txt" / "wili-2018"))
).expanduser()
OUT = Path(
    os.environ.get("LINGUAE_WILI_CSV", str(BASE / "data" / "csv" / "wili-2018.csv"))
).expanduser()
logger.info("package root: %s", BASE)
logger.info("input dir   : %s", DATA_DIR)
logger.info("output CSV  : %s", OUT)

log_section("2. Read the WiLI 2018 split files")
x_train = read_lines(DATA_DIR / "x_train.txt")
y_train = read_lines(DATA_DIR / "y_train.txt")
x_test = read_lines(DATA_DIR / "x_test.txt")
y_test = read_lines(DATA_DIR / "y_test.txt")

log_section("3. Validate texts and labels line up")
pairs = [
    ("train", x_train, y_train),
    ("test", x_test, y_test),
]
for split_name, texts, labels in pairs:
    assert len(texts) == len(labels), (split_name, len(texts), len(labels))
    assert texts, f"{split_name} split is empty"
    logger.info(
        "%-5s split: %d texts, %d labels (matched)",
        split_name,
        len(texts),
        len(labels),
    )
    blank_texts = sum(1 for text in texts if not text.strip())
    blank_labels = sum(1 for label in labels if not label.strip())
    logger.info(
        "%-5s split: %d blank texts, %d blank labels",
        split_name,
        blank_texts,
        blank_labels,
    )
    if blank_texts or blank_labels:
        logger.warning(
            "%s split contains blank values; they will be kept as-is", split_name
        )

log_section("4. Build the combined DataFrame")
started = time.perf_counter()
df_train = pd.DataFrame({"text": x_train, "language": y_train})
df_test = pd.DataFrame({"text": x_test, "language": y_test})
df = pd.concat([df_train, df_test], ignore_index=True)
logger.info(
    "combined train=%d + test=%d -> %d rows in %.2f s",
    len(df_train),
    len(df_test),
    len(df),
    time.perf_counter() - started,
)
logger.info("columns: %s", list(df.columns))
logger.info("dtypes:\n%s", df.dtypes.to_string())
logger.info(
    "character length: min=%d median=%.0f max=%d",
    df["text"].str.len().min(),
    df["text"].str.len().median(),
    df["text"].str.len().max(),
)
logger.info("unique languages: %d", df["language"].nunique())
logger.info("duplicate text rows: %d", df.duplicated(subset=["text"]).sum())
logger.info(
    "texts with conflicting labels: %d",
    int(df.groupby("text")["language"].nunique().gt(1).sum()),
)

label_counts = df["language"].value_counts()
logger.info("rows per language:")
logger.info(label_counts.sort_index().to_string())
logger.info(
    "examples per language: min=%d median=%.0f max=%d",
    label_counts.min(),
    label_counts.median(),
    label_counts.max(),
)
logger.info("most common languages:")
logger.info(label_counts.head().to_string())

log_section("5. Write the CSV atomically")
OUT.parent.mkdir(parents=True, exist_ok=True)
temporary = OUT.with_name(OUT.name + ".tmp")
started = time.perf_counter()
df.to_csv(temporary, index=False, encoding="utf-8")
os.replace(temporary, OUT)
elapsed = time.perf_counter() - started
logger.info(
    "wrote %s (%d rows, %.1f MiB) in %.2f s",
    OUT,
    len(df),
    OUT.stat().st_size / (1024 * 1024),
    elapsed,
)
logger.info("sha256 %s  %s", sha256_of(OUT), OUT.name)

log_section("6. Verify the written CSV")
verification = pd.read_csv(OUT, dtype=str, keep_default_na=False, na_values=[])
logger.info(
    "reloaded %d rows x %d columns",
    len(verification),
    len(verification.columns),
)
assert list(verification.columns) == ["text", "language"], verification.columns
assert len(verification) == len(df), (len(verification), len(df))
logger.info("row count and columns verified")

logger.info("=" * 78)
logger.info(
    "WiLI 2018 conversion complete in %.1f s", time.perf_counter() - SCRIPT_STARTED
)
logger.info("output: %s", OUT)
logger.info("=" * 78)
