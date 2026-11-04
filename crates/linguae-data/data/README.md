# Linguae Data

Data pipeline and artifacts for the Linguae character-level language detector.

Training combines three datasets:

- **Tatoeba** — short conversational sentences (`csv/tatoeba.tsv`).
- **WiLI-2018** — long sentences and Wikipedia paragraphs (`csv/wili-2018.csv`).
- **WiLI words** — the most frequent words per language
  (`csv/wili-2018-words.csv`), for one- to few-word strings.

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
│   ├── wili-2018.csv          # Long text: text, language (header row)
│   └── wili-2018-words.csv    # Frequent words: text, language, count
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
  → crates/linguae-data wili
  → data/csv/wili-2018.csv            (long: text, language)

data/csv/tatoeba.tsv
  → crates/linguae-data tatoeba
  → data/csv/tatoeba.csv              (converted: id, language, text)

data/txt/wili-2018/*.txt
  → crates/linguae-data words
  → data/csv/wili-2018-words.csv      (words: text, language, count)

data/csv/wili-2018.csv + data/csv/tatoeba.tsv + data/csv/wili-2018-words.csv
  → crates/linguae-train (cargo run -p linguae-train -- train)
  → data/processed/labels.json
  → crates/linguae-train/artifacts/images/*.svg
  → crates/linguae-train/artifacts/linguae/
```

### Steps

1. **Raw text → CSV** — `cargo run -p linguae-data -- wili` reads the raw
   `x_train.txt`/`y_train.txt`/`x_test.txt`/`y_test.txt` pairs and produces a
   single CSV with columns `text, language`.
2. **Tatoeba → CSV** — `cargo run -p linguae-data -- tatoeba` converts the
   headerless `id`, `language`, `text` TSV and drops rows with empty or `\N`
   labels.
3. **Extract words** — `cargo run -p linguae-data -- words` tokenises the WiLI
   splits and writes the most frequent words per language with their counts.
4. **Load the sources** — the Rust trainer reads the WiLI CSV, the Tatoeba
   TSV, and (when present) the word CSV, validating each source separately.
   `tatoeba.csv` is available for tooling that prefers CSV.
5. **Normalise and combine** — labels are aliased where Tatoeba uses a variant
   code, contradictory labels and duplicate `(text, language)` pairs are
   dropped, and each source is subsampled per language. The word source uses
   its own per-language cap.
6. **Train** — `cargo run -p linguae-train -- train` splits the combined frame
   per language and trains the character-level CNN with Candle, writing
   `labels.json`, SVG plots, and the exported artifact.
7. **Export** — a self-contained safetensors artifact is written to
   `crates/linguae-train/artifacts/linguae/`.

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

### WiLI words (`csv/wili-2018-words.csv`)

CSV with a header and columns `text, language, count`, generated from the WiLI
train and test splits. It keeps up to 100,000 of the most frequent words per
language, ordered by language, descending frequency, then word. Tokens are
NFC-normalised, lowercased Unicode words; numbers, punctuation, and tokens
longer than 64 characters are dropped, and scripts without spaces (for example
Chinese and Japanese) are split into characters. This source improves detection
of one- to few-word strings and is optional: the trainer adds it only when the
file exists.

### Label mapping

Tatoeba and WiLI use different ISO 639-3 conventions. The trainer applies a
small alias table (`cmn` → `zho`, `yue` → `zh-yue`, `pes` → `fas`, `lvs` → `lav`,
`npi` → `nep`, `zsm`/`zlm` → `msa`, `kmr` → `kur`, `swh`/`swc` → `swa`,
`gsw` → `als`, `bua` → `bxr`, `gom` → `kok`, `kpv` → `kom`) so a language keeps
one output class. `languages.csv` remains the WiLI label/name/family lookup; it
does not list Tatoeba-only labels.

## Regenerating Data

```bash
cd crates
cargo run --release -p linguae-data -- all   # build the WiLI, Tatoeba and word CSVs
cargo run --release -p linguae-train -- train
```

The data targets require the raw WiLI-2018 files and `csv/tatoeba.tsv`. Run
`wili`, `tatoeba`, and `words` individually to build one dataset at a time.
`make csv` and `make train` in `crates/linguae-train` wrap the same commands.
See `crates/linguae-train/README.md` for training options and environment
variables.

## Key Files

| File                                            | Description                                                 |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `csv/languages.csv`                             | WiLI ISO 639-3 code → language name + family lookup         |
| `csv/wili-2018.csv`                             | Long-text dataset: one row per text with its language label |
| `csv/tatoeba.tsv`                               | Short-text dataset: headerless `id`, `language`, `text`     |
| `csv/tatoeba.csv`                               | Converted short-text dataset: `id`, `language`, `text`      |
| `csv/wili-2018-words.csv`                       | Frequent words per language: `text`, `language`, `count`    |
| `processed/labels.json`                         | Label index map written by the trainer                      |
| `crates/linguae-train/artifacts/images/training_curves.svg` | Training/validation loss and accuracy over epochs  |
| `crates/linguae-train/artifacts/images/confusion_matrix.svg` | Per-class confusion matrix heatmap                |
| `crates/linguae-train/artifacts/images/dataset_quality.svg` | Per-source length distribution and class balance   |
| `crates/linguae-train/artifacts/images/accuracy_by_length.svg` | Accuracy stratified by text length             |

## Notes

- The raw `txt/wili-2018/` files (WiLI-2018) are large and not committed to git.
  Download them from the URLs in `txt/wili-2018/urls.txt`.
- `csv/tatoeba.tsv` is also local-only; it is not generated by this package.
- `csv/` and `processed/` are generated artifacts — regenerate locally via the
  Makefile targets above.
- `labels.json` maps language codes to integer indices for the model's output
  layer; it is regenerated during training.
- Default filtering keeps labels with at least 100 examples and at most 1000
  rows per sentence source. Override with `LINGUAE_MIN_EXAMPLES_PER_LANGUAGE`
  and `LINGUAE_MAX_ROWS_PER_LANGUAGE_PER_SOURCE`.
- The word source is capped separately by `LINGUAE_MAX_WORDS_PER_LANGUAGE`
  (default 100,000 per language). Disable it with `LINGUAE_DISABLE_WILI_WORDS=1`.
