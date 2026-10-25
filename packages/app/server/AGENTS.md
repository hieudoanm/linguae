# AGENTS.md

## Project Overview

`server` is the FastAPI server package for Linguae. It exposes the trained
language detector over HTTP with JSON and form-upload endpoints, plus a full
set of production middlewares (request ID, rate limiting, CORS, GZip, timeout).

## Architecture

### Source Structure

```
packages/server/
├── src/linguae_server/
│   ├── __init__.py     # Public API, exports __version__
│   ├── app.py          # FastAPI app + routes
│   ├── config.py       # Runtime settings (env-driven)
│   ├── detector.py     # Lazy singleton wrapper around the artifact
│   ├── main.py         # Uvicorn entry point
│   ├── middlewares.py  # RequestID, RateLimit, Timeout, CORS, GZip
│   └── schemas.py      # Pydantic request/response models
├── pyproject.toml
├── README.md
└── AGENTS.md
```

### Pipeline

1. The server loads the artifact from `packages/models/artifacts/linguae`
   (configurable via `LINGUAE_ARTIFACT_DIR`).
2. `detector.py` wraps `linguae_inference.py` (from the artifact) in a lazy
   singleton so the model loads once at startup.
3. `app.py` defines five routes: `/health`, `/detect`, `/detect/form`,
   `/detect/batch`, `/detect/file`.
4. Middlewares are applied in `add_middlewares()` in this order (outermost
   first): GZip → CORS → Timeout → RateLimit → RequestID.

### Single Entry Point

- Command: `linguae-server` (via `pyproject.toml` scripts:
  `linguae-server = "linguae_server.main:main"`)
- Behavior: runs uvicorn on `0.0.0.0:8000` by default.

## Tech Stack

### Runtime

- **Python** 3.14.8 (`.python-version`, `pyproject.toml: requires-python = "==3.14.8"`)
- **Virtual env**: `uv` with `.venv`

### Core Dependencies

| Package             | Version     | Purpose                     |
| ------------------- | ----------- | --------------------------- |
| `torch`             | `==2.14.1`  | Model inference             |
| `numpy`             | `==2.5.4`   | Array ops                   |
| `fastapi`           | `==0.162.0` | HTTP API framework          |
| `uvicorn[standard]` | `==0.54.0`  | ASGI server                 |
| `pydantic`          | `==2.14.1`  | Request/response validation |
| `python-multipart`  | `==0.0.32`  | Form/file upload parsing    |
| `starlette`         | `==1.10.0`  | ASGI toolkit (middlewares)  |

### Dev Dependencies

| Package       | Version    | Purpose                |
| ------------- | ---------- | ---------------------- |
| `pytest`      | `>=9.1.1`  | Testing                |
| `pytest-cov`  | `>=7.1.0`  | Test coverage          |
| `pytest-mock` | `>=3.17.0` | Mocking                |
| `httpx`       | `>=0.28.0` | Async HTTP test client |
| `ruff`        | `>=0.10.0` | Linting + formatting   |

### Tooling

- **Package manager**: `uv`
- **Lint/Format**: `ruff`
- **CI/CD**: GitHub Actions (root `.github/workflows/ci.yaml` builds the CLI binary)

## Build & Test

```bash
make install      # uv sync
make run          # uv run linguae-server
make test         # pytest
make lint         # ruff check src
```

## GitHub Actions (`.github/workflows/ci.yaml`)

CI workflow runs on push/PR to master/main:

1. **Lint**: `ruff check src`
2. **Test**: `pytest`
3. **Build**: `pyinstaller --onefile --name linguae linguae.spec`

## Code Style

- Python 3.14.8
- Type hints where appropriate
- Keep files under ~200 lines
- Minimal, focused implementation
- Formatting via `ruff`
