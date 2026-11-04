# AGENTS.md

## Project Overview

`models` is the training package for Linguae. It combines the Tatoeba short-
sentence corpus, the WiLI-2018 long-text corpus, and a per-language WiLI word
dataset, trains the character-level CNN, and exports the artifact consumed by
the Rust CLI (`crates/`) and the FastAPI server (`packages/app/server`).

## Architecture

### Source Structure

```
packages/models/
├── notebooks/                  # Exploratory Jupyter notebooks (*.ipynb)
├── scripts/                    # Headless training script
│   └── train_language_detector.py
├── artifacts/
│   ├── images/                 # Training visualisations (Matplotlib + VisualTorch)
│   │   ├── accuracy_by_length.png
│   │   ├── confusion_matrix.png
│   │   ├── dataset_quality.png
│   │   ├── layer_dimensions.png
│   │   ├── network_architecture.png
│   │   ├── prediction_probabilities.png
│   │   └── training_curves.png
│   └── linguae/                # CLI artifact (config.json, vocabulary.json, labels.json, model.pt, linguae_inference.py, metrics.json, manifest.json)
├── pyproject.toml
├── Makefile
└── README.md

crates/linguae-data/
└── data/                       # Raw and processed datasets
    ├── csv/
    │   ├── languages.csv       # WiLI label → name + family lookup
    │   ├── tatoeba.tsv         # Short sentences (id, language, text)
    │   ├── wili-2018.csv       # Long texts (text, language)
    │   └── wili-2018-words.csv # Frequent words (text, language, count)
    ├── processed/
    │   └── labels.json
    └── txt/wili-2018/          # Raw WiLI-2018 data
```

### Pipeline

1. **Data**: `crates/linguae-data` converts `crates/linguae-data/data/txt/wili-2018/*.txt` to `crates/linguae-data/data/csv/wili-2018.csv` (long texts), `crates/linguae-data/data/csv/tatoeba.tsv` to `crates/linguae-data/data/csv/tatoeba.csv` (short sentences), and extracts `crates/linguae-data/data/csv/wili-2018-words.csv` (frequent words per language)
2. **Train**: `scripts/train_language_detector.py` loads the three sources, aliases labels, drops conflicting/duplicate texts, subsamples sentence sources per source and words per language, and writes `crates/linguae-data/data/processed/labels.json` plus `artifacts/images/*.png`
3. **Visualise**: VisualTorch renders the CNN architecture; Matplotlib renders layer dimensions, training curves, the confusion matrix and prediction probabilities
4. **Export**: Build the CLI artifact directory `artifacts/linguae` containing `config.json`, `vocabulary.json`, `labels.json`, `model.pt`, `linguae_inference.py`, `metrics.json`, `manifest.json`

## Tech Stack

### Runtime

- **Python** 3.14.8 (`.python-version`, `pyproject.toml: requires-python = "==3.14.8"`)
- **Virtual env**: `uv` with `.venv`

### Core Dependencies

| Package  | Version    | Purpose                      |
| -------- | ---------- | ---------------------------- |
| `torch`  | `==2.14.1` | Model + training + inference |
| `numpy`  | `==2.5.4`  | Array ops                    |
| `pandas` | `==3.1.0`  | Data handling                |

### Dev Dependencies

| Package     | Version    | Purpose            |
| ----------- | ---------- | ------------------ |
| `jupyter`   | `==1.1.1`  | Notebooks          |
| `nbconvert` | `==7.16.6` | Notebook execution |
| `ipykernel` | `==7.4.0`  | Notebook kernel    |

### Tooling

- **Package manager**: `uv`
- **CI/CD**: GitHub Actions (root `.github/workflows/ci.yaml` builds the CLI binary)

## Build & Test

```bash
make install      # uv sync
make csv          # build the WiLI, Tatoeba and word CSVs
make wili         # build only crates/linguae-data/data/csv/wili-2018.csv
make tatoeba      # build only crates/linguae-data/data/csv/tatoeba.csv
make words        # build only crates/linguae-data/data/csv/wili-2018-words.csv
make notebook     # alias for csv
make train        # run scripts/train_language_detector.py
make train-notebook  # execute the Jupyter version instead
```

The CSV targets run the Rust `linguae-data` pipeline from `crates/linguae-data`
and require the raw WiLI-2018 files plus `crates/linguae-data/data/csv/tatoeba.tsv`.

Training writes `labels.json` and plots to `crates/linguae-data/data/processed/` and exports the
artifact to `artifacts/linguae`. `make train` and `make train-notebook` default
`LINGUAE_ALLOW_OVERWRITE=1` so retraining replaces the existing artifact; pass
`LINGUAE_ALLOW_OVERWRITE=0` to keep the non-empty-directory guard.

## Data

Training uses three datasets with a shared label space:

- `crates/linguae-data/data/csv/tatoeba.tsv` — headerless TSV (`id`, `language`, `text`) of short
  conversational sentences. Rows with `\N` labels are discarded.
- `crates/linguae-data/data/csv/wili-2018.csv` — `text,language` CSV of long sentences and
  Wikipedia paragraphs, generated from the raw files in `crates/linguae-data/data/txt/wili-2018/`
  (`x_train.txt`, `y_train.txt`, `x_test.txt`, `y_test.txt`, 117,500 each).
- `crates/linguae-data/data/csv/wili-2018-words.csv` — `text,language,count` CSV of the most
  frequent words per language (up to 100,000), generated from the same WiLI
  splits. The trainer adds this optional source only when the file exists.

Tatoeba label variants are aliased to WiLI codes (`cmn` → `zho`, `swh`/`swc` →
`swa`, etc.), texts with conflicting labels are dropped, and `(text, language)`
duplicates are removed. Sentence sources are subsampled to 1000 rows per
language by default; the word source uses its own cap (100,000 per language by
default). Labels with fewer than 100 examples are dropped.

WiLI texts are longer than the 256-character input window (median ~271), so long
inputs are classified from their first 256 characters.

`crates/linguae-data/data/csv/languages.csv` maps WiLI ISO 639-3 codes to names and families; it does
not cover Tatoeba-only labels.

### Training environment overrides

`LINGUAE_DATA_DIR`, `LINGUAE_WILI_CSV`, `LINGUAE_TATOEBA_TSV`,
`LINGUAE_WILI_WORDS_CSV`, `LINGUAE_DISABLE_WILI`, `LINGUAE_DISABLE_TATOEBA`,
`LINGUAE_DISABLE_WILI_WORDS`, `LINGUAE_DATASET_PATH` (single custom CSV),
`LINGUAE_MIN_EXAMPLES_PER_LANGUAGE`, `LINGUAE_MAX_ROWS_PER_LANGUAGE_PER_SOURCE`,
`LINGUAE_MAX_WORDS_PER_LANGUAGE`, `LINGUAE_MAX_ROWS_PER_LANGUAGE`,
`LINGUAE_MAX_LENGTH`, `LINGUAE_BATCH_SIZE`, `LINGUAE_EPOCHS`,
`LINGUAE_PROCESSED_DIR`, `LINGUAE_PLOTS_DIR`, `LINGUAE_ARTIFACT_DIR`,
`LINGUAE_ALLOW_OVERWRITE`.

## Code Style

- Python 3.14.8
- Type hints where appropriate
- Keep files under ~200 lines
- Minimal, focused implementation
- Formatting via `ruff`
