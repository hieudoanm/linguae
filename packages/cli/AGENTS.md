# AGENTS.md

## Project Overview

`cli` is the command-line interface for Linguae. It loads the trained artifact
in `packages/models/artifacts/linguae` and predicts one language per input
string. The model performs **single-label classification per string**; it does
not detect mixed-language spans.

## Architecture

### Source Structure

```
packages/cli/
├── src/linguae/
│   ├── __init__.py     # Public API, exports __version__
│   ├── cli.py          # CLI logic (Typer app)
│   └── __main__.py     # Entry point for `python -m linguae`
├── linguae.spec        # PyInstaller spec
├── pyproject.toml
├── Makefile
├── README.md
└── examples/           # Example usage scripts and sample texts
```

### Single Entry Point

- Command: `linguae` (via `pyproject.toml` scripts: `linguae = "linguae.cli:app"`)
- Behavior: reads TEXT, a file via `--file`, or stdin and prints one language
  code per input.

### Artifact Resolution

The CLI searches for the artifact in this order:

1. Explicit `--artifact PATH` or `LINGUAE_ARTIFACT_DIR` environment variable.
2. PyInstaller bundle: `sys._MEIPASS/artifacts/linguae` when frozen.
3. Walk up from cwd: `<root>/packages/models/artifacts/linguae` and
   `<root>/artifacts/linguae`.
4. Editable install: `<repo>/packages/models/artifacts/linguae` derived from
   `__file__`.

The artifact directory must contain `config.json`, `vocabulary.json`,
`labels.json`, `model.pt`, and `linguae_inference.py`.

## Tech Stack

### Runtime

- **Python** 3.14.8 (`.python-version`, `pyproject.toml: requires-python = "==3.14.8"`)
- **Virtual env**: `uv` with `.venv`

### Core Dependencies

| Package       | Version   | Purpose                          |
| ------------- | --------- | -------------------------------- |
| `torch`       | `==2.14.1` | Model inference                 |
| `numpy`       | `==2.5.4`  | Array ops                       |
| `typer`       | `==0.27.3` | CLI framework                   |

### Dev Dependencies

| Package       | Version   | Purpose                |
| ------------- | --------- | ---------------------- |
| `pyinstaller` | `>=6.22.3` | Binary builds          |
| `ruff`        | latest     | Linting + formatting   |

### Tooling

- **Package manager**: `uv`
- **Lint/Format**: `ruff`
- **CI/CD**: GitHub Actions (root `.github/workflows/ci.yaml`)
- **Release**: `softprops/action-gh-release` -> GitHub Releases tagged `latest`

## Build & Test

```bash
make install      # uv sync
make run          # uv run linguae "Bonjour tout le monde"
make smoke        # printf 'Hello world' | uv run linguae --confidence
make build        # pyinstaller --onefile --name linguae linguae.spec
```

## GitHub Actions (`.github/workflows/ci.yaml`)

CI workflow runs on push/PR to master/main:
1. **Lint**: `ruff check src`
2. **Build**: `pyinstaller --onefile --name linguae linguae.spec`
3. **Upload artifact**: Binary uploaded for release job

On push to master/main, the release job:
- Deletes old "latest" release and tag
- Creates new "latest" release with binary using `softprops/action-gh-release@v3.0.3`

## Code Style

- Python 3.14.8
- Type hints where appropriate
- Keep files under ~200 lines
- Minimal, focused implementation
- Formatting via `ruff`