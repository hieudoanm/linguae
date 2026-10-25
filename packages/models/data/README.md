# Linguae Data

Data pipeline and artifacts for the Linguae character-level language detector.

Training combines two datasets:

- **Tatoeba** — short conversational sentences (`csv/tatoeba.tsv`).
- **WiLI-2018** — long sentences and Wikipedia paragraphs (`csv/wili-2018.csv`).

This directory also holds the raw WiLI-2018 dataset, processed training outputs,
and derived plots. The `csv/`, `processed/`, and `txt/` directories are
gitignored — they are generated locally from the raw data or downloaded
separately.

## Structure

```
data/
├── csv/
│   ├── languages.csv          # ISO 639-3 code → name + family lookup (WiLI)
│   ├── tatoeba.tsv            # Short sentences: id, language, text (headerless)
│   └── wili-2018.csv          # Long text: text, language (header row)
├── processed/
│   └── labels.json            # Label → index mapping for the model
└── txt/wili-2018/             # Raw WiLI-2018 dataset
    ├── README.txt
    ├── labels.csv
    ├── urls.txt
    ├── x_train.txt            # Training texts (117,500 examples)
    ├── y_train.txt            # Training labels (117,500 examples)
    ├── x_test.txt             # Test texts (117,500 examples)
    └── y_test.txt             # Test labels (117,500 examples)
```

## Pipeline

```
data/txt/wili-2018/*.txt
  → notebooks/wili_2018_txt_to_csv.py
  → data/csv/wili-2018.csv            (long: text, language)

data/csv/tatoeba.tsv                  (short: id, language, text, source file)

data/csv/wili-2018.csv + data/csv/tatoeba.tsv
  → notebooks/train_language_detector.py
  → data/processed/labels.json
  → artifacts/images/*.png
  → artifacts/linguae/
```

### Steps

1. **Raw text → CSV** — `notebooks/wili_2018_txt_to_csv.py` reads the raw
   `x_train.txt`/`y_train.txt`/`x_test.txt`/`y_test.txt` pairs and produces a
   single CSV with columns `text, language`.
2. **Load both sources** — the training script reads the WiLI CSV and the
   headerless Tatoeba TSV (`id`, `language`, `text`), validating each source
   separately.
3. **Normalise and combine** — labels are aliased where Tatoeba uses a variant
   code, contradictory labels and duplicate `(text, language)` pairs are
   dropped, and each source is subsampled per language.
4. **Train** — `notebooks/train_language_detector.py` splits the combined frame
   per language and trains the character-level CNN, writing plots and
   `labels.json`.
5. **Export** — a self-contained artifact is written to `artifacts/linguae/`.

## Datasets

### Tatoeba (`csv/tatoeba.tsv`)

Headerless TSV with three columns: `id`, `language`, `text`. The training script
uses `language` and `text` only. Rows whose label is `\N` are discarded, and
texts with contradictory labels across sources are dropped. Tatoeba provides the
short-sentence examples.

### WiLI-2018 (`csv/wili-2018.csv`)

CSV with a header and columns `text, language`, generated from the raw
`txt/wili-2018/` files. It provides the long-sentence and paragraph examples.
WiLI's texts are longer than the model's 256-character input window (median ~271
characters), so long inputs are classified from their first 256 characters.

### Label mapping

Tatoeba and WiLI use different ISO 639-3 conventions. The trainer applies a
small alias table (`cmn` → `zho`, `yue` → `zh-yue`, `pes` → `fas`, `lvs` → `lav`,
`npi` → `nep`, `zsm`/`zlm` → `msa`, `kmr` → `kur`, `swh`/`swc` → `swa`,
`gsw` → `als`, `bua` → `bxr`, `gom` → `kok`, `kpv` → `kom`) so a language keeps
one output class. `languages.csv` remains the WiLI label/name/family lookup; it
does not list Tatoeba-only labels.

## Regenerating Data

```bash
cd packages/models
uv sync
make csv      # build data/csv/wili-2018.csv from raw txt files
make train    # run notebooks/train_language_detector.py
```

`make notebook` is an alias for `make csv`; `make train-notebook` executes the
Jupyter training notebook instead.

## Key Files

| File                                            | Description                                                 |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `csv/languages.csv`                             | WiLI ISO 639-3 code → language name + family lookup         |
| `csv/wili-2018.csv`                             | Long-text dataset: one row per text with its language label |
| `csv/tatoeba.tsv`                               | Short-text dataset: headerless `id`, `language`, `text`     |
| `processed/labels.json`                         | Label index map consumed by the model                       |
| `artifacts/images/network_architecture.png`     | VisualTorch flow view of the character-level CNN            |
| `artifacts/images/layer_dimensions.png`         | Per-layer input/output shapes and parameter counts          |
| `artifacts/images/training_curves.png`          | Training/validation loss and accuracy over epochs           |
| `artifacts/images/confusion_matrix.png`         | Per-class confusion matrix heatmap                          |
| `artifacts/images/prediction_probabilities.png` | Top-k softmax scores for sample inputs                      |
| `artifacts/images/dataset_quality.png`          | Per-source length distribution and class balance            |
| `artifacts/images/accuracy_by_length.png`       | Accuracy stratified by text length                          |

## Notes

- The raw `txt/wili-2018/` files (WiLI-2018) are large and not committed to git.
  Download them from the URLs in `txt/wili-2018/urls.txt`.
- `csv/tatoeba.tsv` is also local-only; it is not generated by this package.
- `csv/` and `processed/` are generated artifacts — regenerate locally via the
  Makefile targets above.
- `labels.json` maps language codes to integer indices for the model's output
  layer; it is regenerated during training.
- Default filtering keeps labels with at least 100 examples and at most 1000
  rows per source. Override with `LINGUAE_MIN_EXAMPLES_PER_LANGUAGE` and
  `LINGUAE_MAX_ROWS_PER_LANGUAGE_PER_SOURCE`.
