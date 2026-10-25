# Linguae Train

Rust training pipeline for the Linguae character-level language detector. It
loads the shared datasets under `crates/linguae-data/data`, cleans and splits
them, builds a character vocabulary, trains a CharCNN with
[Candle](https://github.com/huggingface/candle), evaluates it, renders SVG plots
with [plotters](https://github.com/plotters-rs/plotters), and exports a
safetensors artifact that both `linguae-core` and the Python server can load.

No Python, PyTorch, or pandas is required to train or export the model.

## Usage

Run from `crates/`:

```bash
# Train with the defaults and write the artifact
cargo run --release -p linguae-train -- train

# Short smoke run
LINGUAE_EPOCHS=1 cargo run --release -p linguae-train -- train --allow-overwrite

# Convert a legacy PyTorch artifact to safetensors
cargo run --release -p linguae-train -- convert \
  --source /path/to/legacy/artifact \
  --destination artifacts/linguae \
  --overwrite

# Load an artifact and print sample predictions
cargo run --release -p linguae-train -- verify --device cpu
```

Running the binary without a subcommand is equivalent to `train`:

```bash
cargo run --release -p linguae-train
```

## Subcommands

### `train`

The default command. It performs the full pipeline:

1. Load and validate the WiLI-2018, Tatoeba, and WiLI-word sources.
2. Normalise text, alias label variants, drop contradictory labels and
   duplicates, and subsample per source and language.
3. Split deterministically (80/10/10 by default), grouping duplicate texts.
4. Build the character vocabulary from the training split only.
5. Train the CharCNN with AdamW, cross-entropy, gradient clipping, and early
   stopping on validation loss.
6. Evaluate on the held-out test split and check for leakage.
7. Write SVG plots and the exported artifact.
8. Reload `model.safetensors` and assert predictions match the in-memory model.

| Option                | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| `--data-dir PATH`     | Data root containing `csv/wili-2018.csv` and `csv/tatoeba.tsv`.         |
| `--artifact-dir PATH` | Export directory (default `crates/linguae-train/artifacts/linguae`).    |
| `--plots-dir PATH`    | SVG output directory (default `crates/linguae-train/artifacts/images`). |
| `--device TEXT`       | `auto` (default), `cpu`, `cuda`, or `metal`.                            |
| `--epochs INT`        | Maximum epochs.                                                         |
| `--batch-size INT`    | Mini-batch size.                                                        |
| `--max-length INT`    | Maximum characters per example.                                         |
| `--seed INT`          | Random seed.                                                            |
| `--no-plots`          | Skip plot generation.                                                   |
| `--allow-overwrite`   | Replace a non-empty artifact directory.                                 |

### `convert`

Convert a legacy PyTorch artifact (containing `model.pt`) into the safetensors
layout. It copies the metadata files, converts every tensor, and rewrites
`config.json` and `manifest.json` to the `linguae.language-detector/2` schema.

| Option               | Description                                         |
| -------------------- | --------------------------------------------------- |
| `--source PATH`      | Legacy artifact directory containing `model.pt`.    |
| `--destination PATH` | Destination directory for the safetensors artifact. |
| `--overwrite`        | Replace a non-empty destination directory.          |

### `verify`

Load an artifact with `linguae-core` and print predictions for the supplied
strings (or a small multilingual default sample).

| Option            | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| `--artifact PATH` | Artifact directory; discovered automatically when omitted. |
| `--device TEXT`   | `cpu` (default), `cuda`, or `metal`.                       |
| `TEXT...`         | Strings to classify.                                       |

## Configuration

All options can also be set through the same `LINGUAE_*` environment variables
used by the previous Python trainer:

| Variable                                   | Default                               | Purpose                                  |
| ------------------------------------------ | ------------------------------------- | ---------------------------------------- |
| `LINGUAE_DATA_DIR`                         | discovered `crates/linguae-data/data` | Dataset root                             |
| `LINGUAE_TRAIN_ROOT`                       | this crate                            | Root for the default artifact/plot paths |
| `LINGUAE_PACKAGE_ROOT`                     | this crate                            | Legacy alias for `LINGUAE_TRAIN_ROOT`    |
| `LINGUAE_WILI_CSV`                         | `<data>/csv/wili-2018.csv`            | Long-form CSV                            |
| `LINGUAE_TATOEBA_TSV`                      | `<data>/csv/tatoeba.tsv`              | Short-form TSV                           |
| `LINGUAE_WILI_WORDS_CSV`                   | `<data>/csv/wili-2018-words.csv`      | Word CSV                                 |
| `LINGUAE_DISABLE_WILI`                     | `0`                                   | Skip the WiLI source                     |
| `LINGUAE_DISABLE_TATOEBA`                  | `0`                                   | Skip the Tatoeba source                  |
| `LINGUAE_DISABLE_WILI_WORDS`               | `0`                                   | Skip the word source                     |
| `LINGUAE_DATASET_PATH`                     | unset                                 | Use one custom `text,language` CSV       |
| `LINGUAE_MIN_EXAMPLES_PER_LANGUAGE`        | `100`                                 | Drop rarer labels                        |
| `LINGUAE_MAX_ROWS_PER_LANGUAGE_PER_SOURCE` | `1000`                                | Per-source cap (`0` keeps all)           |
| `LINGUAE_MAX_WORDS_PER_LANGUAGE`           | `100000`                              | Word cap (`0` keeps all)                 |
| `LINGUAE_MAX_ROWS_PER_LANGUAGE`            | `0`                                   | Combined-frame cap (`0` keeps all)       |
| `LINGUAE_MAX_ROWS`                         | unset                                 | Overall row cap                          |
| `LINGUAE_MAX_LENGTH`                       | `256`                                 | Character sequence length                |
| `LINGUAE_EMBEDDING_DIM`                    | `32`                                  | Embedding size                           |
| `LINGUAE_NUM_FILTERS`                      | `128,128,128`                         | Filters per kernel                       |
| `LINGUAE_KERNEL_SIZES`                     | `3,5,7`                               | Convolution kernel sizes                 |
| `LINGUAE_DROPOUT`                          | `0.3`                                 | Dropout probability                      |
| `LINGUAE_BATCH_SIZE`                       | `256`                                 | Batch size                               |
| `LINGUAE_EPOCHS`                           | `10`                                  | Maximum epochs                           |
| `LINGUAE_LEARNING_RATE`                    | `1e-3`                                | AdamW learning rate                      |
| `LINGUAE_WEIGHT_DECAY`                     | `1e-4`                                | AdamW weight decay                       |
| `LINGUAE_GRAD_CLIP_NORM`                   | `5.0`                                 | Gradient clipping                        |
| `LINGUAE_EARLY_STOPPING_PATIENCE`          | `2`                                   | Epochs without improvement               |
| `LINGUAE_MAX_MISCLASSIFIED_EXAMPLES`       | `2000`                                | Rows written to the CSV                  |
| `LINGUAE_MODEL_VERSION`                    | `1.0.0`                               | Version recorded in metadata             |
| `LINGUAE_SEED`                             | `42`                                  | Random seed                              |
| `LINGUAE_DEVICE`                           | `auto`                                | Device preference                        |
| `LINGUAE_PROGRESS_EVERY`                   | `200`                                 | Log every N batches                      |
| `LINGUAE_PROCESSED_DIR`                    | `<data>/processed`                    | Processed output directory               |
| `LINGUAE_PLOTS_DIR`                        | `artifacts/images`                    | Plot output directory                    |
| `LINGUAE_ARTIFACT_DIR`                     | `artifacts/linguae`                   | Artifact output directory                |
| `LINGUAE_NO_PLOTS`                         | `0`                                   | Disable plots                            |
| `RUST_LOG`                                 | `info`                                | Logging filter (via `env_logger`)        |

## Artifact

`train` writes:

- `config.json`, `vocabulary.json`, `labels.json`
- `model.safetensors` (Candle, f32)
- `metrics.json` (overall, per-source, per-language, length and confidence data)
- `manifest.json` (dataset checksums and training metadata)
- `confusion_matrix.csv`, `misclassified_examples.csv`, `training_history.csv`

Plots are written as SVG to `artifacts/images/`:

- `dataset_quality.svg`
- `training_curves.svg`
- `confusion_matrix.svg`
- `accuracy_by_length.svg`

The Python server and `linguae-core` both load `model.safetensors`. Legacy
`model.pt` artifacts remain supported by `linguae-core`.

The committed artifact under `artifacts/linguae` was produced by `convert` from
the legacy PyTorch artifact, so its metrics and the retained PNG plots describe
that training run. Run `train` to regenerate the artifact and SVG plots with the
Rust pipeline.

## Data preparation

The datasets are produced by `linguae-data`. Run from `crates/`:

```bash
cargo run --release -p linguae-data -- wili
cargo run --release -p linguae-data -- tatoeba
cargo run --release -p linguae-data -- words
cargo run --release -p linguae-data -- all
```

See `crates/README.md` and `crates/linguae-data/data/README.md` for the source
formats and filtering rules.

## Test

```bash
cargo test -p linguae-train
cargo clippy -p linguae-train --all-targets -- -D warnings
```

## Notes

- Default device is `auto`, which selects CUDA or Metal only when the crate is
  built with the matching `cuda` or `metal` feature; otherwise it uses CPU. Pass
  `--device cpu` to force CPU.
- Inputs are truncated to `max_length` (256 characters by default). The model
  predicts exactly one language per string; it does not detect mixed-language
  spans.
