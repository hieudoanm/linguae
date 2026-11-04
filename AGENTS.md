# AGENTS.md

## Project Overview

Linguae is a local-first language identification project. It trains a
character-level CNN from scratch in PyTorch and ships a command-line tool that
detects the language of text. No pretrained models, external inference APIs, or
other ML frameworks are used.

**Problem:** Reliable language identification should run locally without
network calls or pretrained weights.
**Solution:** A character-level CNN trained on Tatoeba (short sentences),
WiLI-2018 (long paragraphs), and a per-language WiLI word dataset, served by a
small Rust CLI binary that runs inference with Candle.

## Architecture

### Monorepo layout

```
linguae/
├── packages/
│   ├── models/                 # Model training + artifacts
│   │   ├── notebooks/          # Training notebooks
│   │   ├── scripts/            # Headless training script
│   │   ├── models/             # language_detector.pt
│   │   ├── artifacts/linguae/  # CLI artifact (config.json, vocabulary.json, labels.json, model.pt, linguae_inference.py)
│   │   ├── pyproject.toml
│   │   └── Makefile
│   └── app/                    # FastAPI server + Next.js/Tauri app
│       ├── server/
│       └── web/
├── crates/                     # Rust workspace
│   ├── linguae-core/           # Artifact loading, preprocessing, CharCNN (Candle)
│   ├── linguae-cli/            # `linguae` binary (clap)
│   └── linguae-data/           # Dataset conversion + word extraction
│       └── data/               # Raw and processed datasets
│           ├── csv/tatoeba.tsv        # Short sentences (id, language, text)
│           ├── csv/wili-2018.csv      # Long texts (text, language)
│           ├── csv/wili-2018-words.csv # Frequent words (text, language, count)
│           └── processed/labels.json
├── .github/workflows/ci.yaml   # CI + GitHub release
├── AGENTS.md
├── LICENSE
└── README.md
```

### Pipeline

1. **Data**: `crates/linguae-data` converts `crates/linguae-data/data/txt/wili-2018/*.txt` to `crates/linguae-data/data/csv/wili-2018.csv` (long texts), `crates/linguae-data/data/csv/tatoeba.tsv` to `crates/linguae-data/data/csv/tatoeba.csv` (short sentences), and extracts `crates/linguae-data/data/csv/wili-2018-words.csv` (frequent words per language)
2. **Train**: `packages/models/scripts/train_language_detector.py` loads the three sources, aliases labels, drops conflicting/duplicate texts, subsamples sentence sources per source and words per language, and writes `crates/linguae-data/data/processed/labels.json` plus `artifacts/images/*.png`
3. **Export**: Build the CLI artifact directory `packages/models/artifacts/linguae` containing `config.json`, `vocabulary.json`, `labels.json`, `model.pt`, `linguae_inference.py`, `metrics.json`, `manifest.json`
4. **Serve (HTTP)**: `packages/app/server` exposes a FastAPI API that loads the artifact and predicts one language per string over JSON/form endpoints with rate limiting, CORS, GZip, request ID, and timeout middlewares.
5. **Serve (CLI)**: `crates/linguae-cli` exposes the `linguae` command (clap) which loads the artifact directly and predicts one language per string.

### Single Entry Point

- Command: `linguae` (via `crates/linguae-cli`)
  - Behavior: reads text, a file, or stdin and prints a language code (ISO 639-3).
- Command: `linguae-server` (via `packages/app/server/pyproject.toml` scripts: `linguae-server = "linguae_server.main:main"`)
  - Behavior: runs uvicorn on `0.0.0.0:8000` by default, serving `/detect` (JSON and form), `/detect/batch`, `/detect/file`, and `/health`.

## Environment Variables

- `LINGUAE_DATA_DIR`: Dataset directory (default: `crates/linguae-data/data`).
- `LINGUAE_ARTIFACT_DIR`: Artifact directory (default: `packages/models/artifacts/linguae`).
- The Rust CLI resolves artifacts relative to `--artifact`, `LINGUAE_ARTIFACT_DIR`, the current directory and its parents, or the executable and its parents.

## Tech Stack

### Runtime

- **Rust** 1.85+ for the CLI and data pipeline (`cargo`, Candle, clap)
- **Python** 3.14.8 for training and the HTTP server (`.python-version`, `pyproject.toml: requires-python = "==3.14.8"`)
- **Virtual env**: `uv` with `.venv`

### Core Dependencies

| Package       | Version   | Purpose                          |
| ------------- | --------- | -------------------------------- |
| `candle-core` | `0.11.1`  | Tensor runtime (CLI inference)   |
| `candle-nn`   | `0.11.1`  | Layers + PyTorch `.pt` loading   |
| `clap`        | `4.6`     | CLI framework                    |
| `torch`       | `==2.14.1` | Model + training                |
| `numpy`       | `==2.5.4`  | Array ops                       |
| `pandas`      | `==3.1.0`  | Data handling                   |
| `fastapi`     | `==0.162.0`| HTTP API framework (server)     |
| `uvicorn`     | `==0.54.0` | ASGI server (server)            |
| `starlette`   | `==1.10.0` | ASGI toolkit / middlewares (server) |
| `pydantic`    | `==2.14.1` | Request/response validation (server) |

### Dev Dependencies

| Package       | Version   | Purpose                |
| ------------- | --------- | ---------------------- |
| `jupyter`     | `==1.1.1`  | Notebooks              |
| `nbconvert`   | `==7.16.6` | Notebook execution     |
| `ipykernel`   | `==7.4.0`  | Notebook kernel        |
| `ruff`        | latest     | Linting + formatting   |

### Tooling

- **Rust**: `cargo` (workspace in `crates/`)
- **Python package manager**: `uv`
- **Python lint/format**: `ruff`
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yaml`)
- **Release**: `softprops/action-gh-release` -> GitHub Releases tagged `latest`

## Build & Test

```bash
# Model package
cd packages/models
make install      # uv sync
make csv          # build the WiLI, Tatoeba and word CSVs with linguae-data
make notebook     # alias for csv
make train        # run scripts/train_language_detector.py
make train-notebook  # execute the Jupyter training notebook instead

# Rust workspace (CLI + data pipeline)
cd crates
cargo test --workspace
cargo build --release -p linguae
cargo run --release -p linguae-data -- all
./target/release/linguae "Bonjour tout le monde"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Server package
cd packages/app/server
make install      # uv sync
make run          # uv run linguae-server
make test         # pytest
make lint         # ruff check src
```

## GitHub Actions (`.github/workflows/ci.yaml`)

CI workflow runs on push/PR to master/main:
1. **Lint (Rust)**: `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`
2. **Test (Rust)**: `cargo test --workspace`
3. **Build**: `cargo build --release -p linguae` (zipped as `crates/dist/linguae.zip`)
4. **Lint/Test (Python)**: `ruff check src` and `pytest` (server package)
5. **Upload artifact**: CLI binary uploaded for the release job

On push to master/main, the release job:
- Deletes old "latest" release and tag
- Creates new "latest" release with binary using `softprops/action-gh-release@v3.0.3`

## Code Style

- Rust 1.85+ for the CLI; Python 3.14.8 for training and the server
- Type hints/type annotations where appropriate
- Keep files under ~200 lines
- Minimal, focused implementation
- Formatting via `cargo fmt` (Rust) and `ruff` (Python)
- Linting via `cargo clippy` (Rust) and `ruff check` (Python)