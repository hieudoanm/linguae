# AGENTS.md

## Project Overview

Linguae is a local-first language identification project. It trains a
character-level CNN from scratch in PyTorch and ships a command-line tool that
detects the language of text. No pretrained models, external inference APIs, or
other ML frameworks are used.

**Problem:** Reliable language identification should run locally without
network calls or pretrained weights.
**Solution:** A character-level CNN trained on the WiLI-2018 dataset, packaged
as a standalone CLI binary built with PyInstaller.

## Architecture

### Monorepo layout

```
linguae/
├── packages/
│   ├── models/                 # Model training + data
│   │   ├── notebooks/          # wili-2018-txt-to-csv.ipynb, train_language_detector.ipynb
│   │   ├── data/               # Raw and processed datasets
│   │   │   ├── csv/wili-2018.csv
│   │   │   └── processed/dataset.csv, metrics.json, misclassified.csv
│   │   ├── models/             # language_detector.pt
│   │   ├── artifacts/linguae/  # CLI artifact (config.json, vocabulary.json, labels.json, model.pt, linguae_inference.py)
│   │   ├── pyproject.toml
│   │   └── Makefile
│   ├── cli/                    # Command-line interface
│   │   ├── src/linguae/
│   │   │   ├── __init__.py     # exports __version__
│   │   │   ├── cli.py          # Typer CLI
│   │   │   └── __main__.py
│   │   ├── pyproject.toml
│   │   ├── linguae.spec        # PyInstaller spec
│   │   └── Makefile
│   └── server/                 # FastAPI HTTP server
│       ├── src/linguae_server/
│       │   ├── __init__.py     # exports __version__
│       │   ├── app.py          # FastAPI app + routes
│       │   ├── config.py       # Runtime settings (env-driven)
│       │   ├── detector.py     # Lazy singleton wrapper around the artifact
│       │   ├── main.py         # Uvicorn entry point
│       │   ├── middlewares.py  # RequestID, RateLimit, CORS, GZip, Timeout
│       │   └── schemas.py      # Pydantic request/response models
│       ├── pyproject.toml
│       ├── Makefile
│       ├── README.md
│       └── AGENTS.md
├── .github/workflows/ci.yaml   # CI + GitHub release
├── AGENTS.md
├── LICENSE
└── README.md
```

### Pipeline

1. **Data**: `data/txt/wili-2018/*.txt` -> `notebooks/wili-2018-txt-to-csv.ipynb` -> `data/csv/wili-2018.csv` -> `data/processed/dataset.csv`
2. **Train**: `notebooks/train_language_detector.ipynb` -> `models/language_detector.pt`, `data/processed/metrics.json`, `data/processed/misclassified.csv`, `data/processed/training_curves.png`, `data/processed/confusion_matrix.png`
3. **Export**: Build the CLI artifact directory `packages/models/artifacts/linguae` containing `config.json`, `vocabulary.json`, `labels.json`, `model.pt`, `linguae_inference.py`
4. **Serve (HTTP)**: `packages/server` exposes a FastAPI API that loads the artifact and predicts one language per string over JSON/form endpoints with rate limiting, CORS, GZip, request ID, and timeout middlewares.
5. **Serve (CLI)**: `packages/cli` exposes the `linguae` command (Typer) which loads the artifact and predicts one language per string.

### Single Entry Point

- Command: `linguae` (via `packages/cli/pyproject.toml` scripts: `linguae = "linguae.cli:app"`)
  - Behavior: reads text, a file, or stdin and prints a language code (ISO 639-1 where available).
- Command: `linguae-server` (via `packages/server/pyproject.toml` scripts: `linguae-server = "linguae_server.main:main"`)
  - Behavior: runs uvicorn on `0.0.0.0:8000` by default, serving `/detect` (JSON and form), `/detect/batch`, `/detect/file`, and `/health`.

## Environment Variables

- `LINGUAE_ARTIFACT_DIR`: Artifact directory (default: `packages/models/artifacts/linguae`).
- `PYTHONPATH`: not required; the CLI resolves artifacts relative to cwd, the module file, or `sys._MEIPASS` when frozen.

## Tech Stack

### Runtime

- **Python** 3.14.8 (`.python-version`, `pyproject.toml: requires-python = "==3.14.8"`)
- **Virtual env**: `uv` with `.venv`

### Core Dependencies

| Package       | Version   | Purpose                          |
| ------------- | --------- | -------------------------------- |
| `torch`       | `==2.14.1` | Model + training + inference    |
| `numpy`       | `==2.5.4`  | Array ops                       |
| `pandas`      | `==3.1.0`  | Data handling                   |
| `typer`       | `==0.27.3` | CLI framework                   |
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
| `pyinstaller` | `>=6.22.3` | Binary builds          |
| `ruff`        | latest     | Linting + formatting   |

### Tooling

- **Package manager**: `uv`
- **Lint/Format**: `ruff`
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yaml`)
- **Release**: `softprops/action-gh-release` -> GitHub Releases tagged `latest`

## Build & Test

```bash
# Model package
cd packages/models
make install      # uv sync
make csv          # build data/csv/wili-2018.csv from raw txt
make notebook     # alias for csv
# Then run the training notebook manually in Jupyter.

# CLI package
cd packages/cli
make install      # uv sync
make run          # uv run linguae "Bonjour tout le monde"
make smoke        # printf 'Hello world' | uv run linguae --confidence
make build        # pyinstaller --onefile --name linguae linguae.spec

# Server package
cd packages/server
make install      # uv sync
make run          # uv run linguae-server
make test         # pytest
make lint         # ruff check src
```

## GitHub Actions (`.github/workflows/ci.yaml`)

CI workflow runs on push/PR to master/main:
1. **Lint**: `ruff check src` (cli + server)
2. **Build**: `pyinstaller --onefile --name linguae linguae.spec`
3. **Test**: `pytest` (server package)
4. **Upload artifact**: Binary uploaded for release job

On push to master/main, the release job:
- Deletes old "latest" release and tag
- Creates new "latest" release with binary using `softprops/action-gh-release@v3.0.3`

## Code Style

- Python 3.14.8
- Type hints where appropriate
- Keep files under ~200 lines
- Minimal, focused implementation
- Formatting via `ruff`