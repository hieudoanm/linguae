# AGENTS.md

## Project Overview

`models` is the training package for Linguae. It contains the WiLI-2018 data
pipeline, the character-level CNN training notebook, and the exported CLI
artifact consumed by `packages/cli`.

## Architecture

### Source Structure

```
packages/models/
├── notebooks/                  # wili-2018-txt-to-csv.ipynb, train_language_detector.ipynb
├── data/
│   ├── csv/wili-2018.csv      # Combined text + language dataset
│   ├── processed/              # Training outputs
│   │   ├── dataset.csv
│   │   ├── metrics.json
│   │   ├── misclassified.csv
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   └── txt/wili-2018/          # Raw WiLI-2018 data
├── models/                     # language_detector.pt
├── artifacts/linguae/          # CLI artifact (config.json, vocabulary.json, labels.json, model.pt, linguae_inference.py)
├── pyproject.toml
├── Makefile
└── README.md
```

### Pipeline

1. **Data**: `data/txt/wili-2018/*.txt` -> `notebooks/wili-2018-txt-to-csv.ipynb` -> `data/csv/wili-2018.csv` -> `data/processed/dataset.csv`
2. **Train**: `notebooks/train_language_detector.ipynb` -> `models/language_detector.pt`, `data/processed/metrics.json`, `data/processed/misclassified.csv`, `data/processed/training_curves.png`, `data/processed/confusion_matrix.png`
3. **Export**: Build the CLI artifact directory `artifacts/linguae` containing `config.json`, `vocabulary.json`, `labels.json`, `model.pt`, `linguae_inference.py`

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

### Dev Dependencies

| Package       | Version   | Purpose                |
| ------------- | --------- | ---------------------- |
| `jupyter`     | `==1.1.1`  | Notebooks              |
| `nbconvert`   | `==7.16.6` | Notebook execution     |
| `ipykernel`   | `==7.4.0`  | Notebook kernel        |

### Tooling

- **Package manager**: `uv`
- **CI/CD**: GitHub Actions (root `.github/workflows/ci.yaml` builds the CLI binary)

## Build & Test

```bash
make install      # uv sync
make csv          # build data/csv/wili-2018.csv from raw txt
make notebook     # alias for csv
```

Then run `notebooks/train_language_detector.ipynb` in Jupyter to train the
model and produce the artifact in `artifacts/linguae`.

## Data

The WiLI-2018 dataset (without labels) is used for training. The raw files live
in `data/txt/wili-2018/`:

- `x_train.txt`, `y_train.txt` (117,500 examples)
- `x_test.txt`, `y_test.txt` (117,500 examples)
- `labels.csv` maps ISO 639-3 codes to language names and families.

`data/csv/wili-2018.csv` is the combined CSV with columns `text,language`.
`data/processed/dataset.csv` is the cleaned version used by the training
notebook.

## Code Style

- Python 3.14.8
- Type hints where appropriate
- Keep files under ~200 lines
- Minimal, focused implementation
- Formatting via `ruff`