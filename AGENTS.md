# AGENTS.md

## Project Overview

Linguae is a local-first language identification project. It trains a
character-level CNN from scratch with Candle and ships a command-line tool that
detects the language of text. No pretrained models, external inference APIs, or
other ML frameworks are used.

**Problem:** Reliable language identification should run locally without
network calls or pretrained weights.
**Solution:** A character-level CNN trained on Tatoeba (short sentences),
WiLI-2018 (long paragraphs), and a per-language WiLI word dataset, served by a
small Rust CLI binary and a FastAPI server. Training and inference both run in
Rust; the Python server is inference-only.

## Architecture

### Monorepo layout

```
linguae/
├── packages/
│   └── app/                    # FastAPI server + Next.js/Tauri app
│       ├── server/
│       └── web/
├── crates/                     # Rust workspace
│   ├── linguae-core/           # Artifact loading, preprocessing, CharCNN (Candle)
│   ├── linguae-cli/            # `linguae` binary (clap)
│   ├── linguae-data/           # Dataset conversion + word extraction
│   │   └── data/               # Raw and processed datasets
│   │       ├── csv/tatoeba.tsv        # Short sentences (id, language, text)
│   │       ├── csv/wili-2018.csv      # Long texts (text, language)
│   │       ├── csv/wili-2018-words.csv # Frequent words (text, language, count)
│   │       └── processed/labels.json
│   └── linguae-train/          # Rust training pipeline (`linguae-train`)
│       ├── artifacts/linguae/  # Exported artifact (config.json, vocabulary.json,
│       │                       # labels.json, model.safetensors, metrics.json,
│       │                       # manifest.json, CSVs)
│       └── artifacts/images/   # Training plots
├── .github/workflows/ci.yaml   # CI + GitHub release
├── AGENTS.md
├── LICENSE
└── README.md
```

### Pipeline

1. **Data**: `crates/linguae-data` converts `crates/linguae-data/data/txt/wili-2018/*.txt` to `crates/linguae-data/data/csv/wili-2018.csv` (long texts), `crates/linguae-data/data/csv/tatoeba.tsv` to `crates/linguae-data/data/csv/tatoeba.csv` (short sentences), and extracts `crates/linguae-data/data/csv/wili-2018-words.csv` (frequent words per language)
2. **Train**: `crates/linguae-train` loads the three sources, aliases labels, drops conflicting/duplicate texts, subsamples sentence sources per source and words per language, trains the CharCNN with Candle, and writes `crates/linguae-data/data/processed/labels.json` plus `crates/linguae-train/artifacts/images/*.svg`
3. **Export**: Build the artifact directory `crates/linguae-train/artifacts/linguae` containing `config.json`, `vocabulary.json`, `labels.json`, `model.safetensors`, `metrics.json`, `manifest.json`, and evaluation CSVs
4. **Serve (HTTP)**: `packages/app/server` exposes a FastAPI API that loads the artifact and predicts one language per string over JSON/form endpoints with rate limiting, CORS, GZip, request ID, and timeout middlewares.
5. **Serve (CLI)**: `crates/linguae-cli` exposes the `linguae` command (clap) which loads the artifact directly and predicts one language per string.

### Single Entry Point

- Command: `linguae` (via `crates/linguae-cli`)
  - Behavior: reads text, a file, or stdin and prints a language code (ISO 639-3).
- Command: `linguae-train` (via `crates/linguae-train`)
  - Behavior: trains, converts legacy artifacts, or verifies an artifact.
- Command: `linguae-server` (via `packages/app/server/pyproject.toml` scripts: `linguae-server = "linguae_server.main:main"`)
  - Behavior: runs uvicorn on `0.0.0.0:8000` by default, serving `/detect` (JSON and form), `/detect/batch`, `/detect/file`, and `/health`.

## Environment Variables

- `LINGUAE_DATA_DIR`: Dataset directory (default: `crates/linguae-data/data`).
- `LINGUAE_ARTIFACT_DIR`: Artifact directory (default: `crates/linguae-train/artifacts/linguae`).
- `LINGUAE_DEVICE`: Inference device for the server (default: `cpu`).
- Training honours the same `LINGUAE_*` variables as the previous Python
  trainer; see `crates/linguae-train/README.md` for the full list.
- The Rust CLI resolves artifacts relative to `--artifact`, `LINGUAE_ARTIFACT_DIR`, the current directory and its parents, or the executable and its parents.

## Tech Stack

### Runtime

- **Rust** 1.85+ for the CLI, data pipeline, and training (`cargo`, Candle, clap, plotters)
- **Python** 3.14.8 for the HTTP server (`.python-version`, `pyproject.toml: requires-python = "==3.14.8"`)
- **Virtual env**: `uv` with `.venv`

### Core Dependencies

| Package       | Version   | Purpose                          |
| ------------- | --------- | -------------------------------- |
| `candle-core` | `0.11.1`  | Tensor runtime (training + CLI inference) |
| `candle-nn`   | `0.11.1`  | Layers, optimizers, safetensors  |
| `plotters`    | `0.3`     | SVG training plots               |
| `clap`        | `4.6`     | CLI framework                    |
| `torch`       | `==2.15.0`| Server inference only            |
| `safetensors` | `==0.8.0` | Server weight loading            |
| `fastapi`     | `==0.162.0`| HTTP API framework (server)     |
| `uvicorn`     | `==0.54.0` | ASGI server (server)            |
| `starlette`   | `==1.10.0` | ASGI toolkit / middlewares (server) |
| `pydantic`    | `==2.14.2` | Request/response validation (server) |

### Dev Dependencies

| Package       | Version   | Purpose                |
| ------------- | --------- | ---------------------- |
| `ruff`        | latest     | Linting + formatting   |

### Tooling

- **Rust**: `cargo` (workspace in `crates/`)
- **Python package manager**: `uv`
- **Python lint/format**: `ruff`
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yaml`)
- **Release**: `softprops/action-gh-release` -> GitHub Releases tagged `latest`

## Build & Test

```bash
# Data preparation
cd crates
cargo run --release -p linguae-data -- all

# Training and export
cargo run --release -p linguae-train -- train
cargo run --release -p linguae-train -- verify --device cpu

# Rust workspace (CLI + data + training)
cd crates
cargo test --workspace
cargo build --release -p linguae
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

- Rust 1.85+ for the CLI, data pipeline, and training; Python 3.14.8 for the server
- Type hints/type annotations where appropriate
- Keep files under ~200 lines
- Minimal, focused implementation
- Formatting via `cargo fmt` (Rust) and `ruff` (Python)
- Linting via `cargo clippy` (Rust) and `ruff check` (Python)
