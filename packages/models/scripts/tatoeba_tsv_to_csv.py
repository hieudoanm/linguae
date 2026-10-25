"""Convert the Tatoeba sentence-pair TSV into a CSV dataset.

This script mirrors ``notebooks/wili_2018_txt_to_csv.ipynb`` and is intended for
headless runs. It reads ``data/csv/tatoeba.tsv`` (headerless, tab-separated
``id``, ``language``, ``text``), drops rows with ``\\N`` labels, and writes
``data/csv/tatoeba.csv`` with ``id``, ``language`` and ``text`` columns.
Progress and dataset statistics are reported through the standard
:mod:`logging` module.

The converted CSV supplies the *short* (conversational sentence) examples for
``scripts/train_language_detector.py`` alongside ``data/csv/wili-2018.csv``
(*long* paragraphs from WiLI-2018). Both sources share one label space.

Usage::

    cd packages/models
    uv run python scripts/tatoeba_tsv_to_csv.py

Environment overrides (all optional)::

    LINGUAE_LOG_LEVEL       DEBUG, INFO (default), WARNING, ...
    LINGUAE_TATOEBA_TSV     input TSV path (default data/csv/tatoeba.tsv)
    LINGUAE_TATOEBA_CSV     output CSV path (default data/csv/tatoeba.csv)
    LINGUAE_PACKAGE_ROOT    override package-root discovery
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
logger = logging.getLogger("linguae.tatoeba")
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


def sha256_of(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


log_section("1. Resolve paths")
BASE = find_package_root()
IN_TSV = Path(
    os.environ.get("LINGUAE_TATOEBA_TSV", str(BASE / "data" / "csv" / "tatoeba.tsv"))
).expanduser()
OUT_CSV = Path(
    os.environ.get(
        "LINGUAE_TATOEBA_CSV", str(BASE / "data" / "csv" / "tatoeba.csv")
    )
).expanduser()
logger.info("package root: %s", BASE)
logger.info("input TSV   : %s", IN_TSV)
logger.info("output CSV  : %s", OUT_CSV)

log_section("2. Read the Tatoeba TSV")
if not IN_TSV.is_file():
    raise FileNotFoundError(f"missing Tatoeba TSV: {IN_TSV}")

started = time.perf_counter()
df = pd.read_csv(
    IN_TSV,
    sep="\t",
    header=None,
    names=["id", "language", "text"],
    dtype=str,
    keep_default_na=False,
    na_values=[],
    quoting=3,  # csv.QUOTE_NONE
)
logger.info(
    "read %-12s %8d rows | %7.1f MiB | %.2f s",
    IN_TSV.name,
    len(df),
    IN_TSV.stat().st_size / (1024 * 1024),
    time.perf_counter() - started,
)
logger.info("columns: %s", list(df.columns))

log_section("3. Validate and normalise")
blank_texts = int(df["text"].str.strip().eq("").sum())
blank_labels = int(df["language"].str.strip().eq("").sum())
nan_labels = int(df["language"].eq("\\N").sum())
logger.info(
    "%-5s split: %d blank texts, %d blank labels, %d \\\\N labels",
    "tatoeba",
    blank_texts,
    blank_labels,
    nan_labels,
)
if blank_texts or blank_labels or nan_labels:
    logger.warning(
        "Tatoeba TSV contains blank or \\\\N values; rows with empty language "
        "labels will be excluded from training."
    )

label_counts = df["language"].value_counts()
logger.info("rows per language (top 10):")
logger.info(label_counts.head(10).to_string())
logger.info(
    "examples per language: min=%d median=%.0f max=%d",
    label_counts.min(),
    label_counts.median(),
    label_counts.max(),
)
logger.info("unique languages: %d", df["language"].nunique())

log_section("4. Build the cleaned DataFrame")
# Drop rows with empty or \\N language labels before writing.
before = len(df)
df = df.loc[df["language"].str.strip().ne("")].copy()
df = df.loc[df["language"].ne("\\N")].copy()
dropped = before - len(df)
if dropped:
    logger.info("dropped %d row(s) with empty or \\\\N language labels", dropped)
df = df.reset_index(drop=True)
logger.info(
    "combined -> %d rows in %.2f s",
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
logger.info("duplicate text rows: %d", df.duplicated(subset=["text"]).sum())
logger.info(
    "texts with conflicting labels: %d",
    int(df.groupby("text")["language"].nunique().gt(1).sum()),
)

log_section("5. Write the CSV atomically")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
temporary = OUT_CSV.with_name(OUT_CSV.name + ".tmp")
started = time.perf_counter()
df.to_csv(temporary, index=False, encoding="utf-8")
os.replace(temporary, OUT_CSV)
elapsed = time.perf_counter() - started
logger.info(
    "wrote %s (%d rows, %.1f MiB) in %.2f s",
    OUT_CSV,
    len(df),
    OUT_CSV.stat().st_size / (1024 * 1024),
    elapsed,
)
logger.info("sha256 %s  %s", sha256_of(OUT_CSV), OUT_CSV.name)

log_section("6. Verify the written CSV")
verification = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False, na_values=[])
logger.info(
    "reloaded %d rows x %d columns",
    len(verification),
    len(verification.columns),
)
assert list(verification.columns) == ["id", "language", "text"], verification.columns
assert len(verification) == len(df), (len(verification), len(df))
logger.info("row count and columns verified")

logger.info("=" * 78)
logger.info(
    "Tatoeba conversion complete in %.1f s", time.perf_counter() - SCRIPT_STARTED
)
logger.info("output: %s", OUT_CSV)
logger.info("=" * 78)
