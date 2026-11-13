# Linguae Rust Workspace

Rust implementation of the `linguae` command, the data-preparation pipeline, and
the training pipeline. The CLI loads the trained artifact in
`crates/linguae-train/artifacts/linguae` and prints one prediction per input
string.

The workspace contains four crates:

```text
crates/
├── linguae-core/   # artifact loading, text preprocessing, CharCNN forward pass
├── linguae-cli/    # `linguae` binary (clap)
├── linguae-data/   # dataset conversion + word extraction (`linguae-data`)
└── linguae-train/  # training, evaluation, plots, export (`linguae-train`)
```

`linguae-core` reads the artifact directly (`config.json`, `vocabulary.json`,
`labels.json`, `model.safetensors`) with [Candle](https://github.com/huggingface/candle),
so the CLI does not need Python or PyTorch at runtime. Legacy `model.pt`
artifacts are also accepted.

`linguae-data` converts the raw WiLI-2018 split files and the Tatoeba TSV into
the CSVs consumed by training, and extracts the most frequent words per language
from WiLI for short-string detection.

`linguae-train` ports the former Python trainer to Rust: it cleans and splits
the sources, builds the vocabulary, trains the CharCNN, evaluates it, renders
SVG plots, and exports the safetensors artifact. See
[`linguae-train/README.md`](linguae-train/README.md).

## Build

```bash
cd crates
cargo build --release -p linguae
./target/release/linguae "Bonjour tout le monde"
```

## Usage

```bash
linguae "Bonjour tout le monde, comment allez-vous aujourd'hui dans ce beau monde"
# fra

linguae --confidence "Hello world, this is a longer English text to test language detection"
# eng	0.6114

printf 'Привет, мир, это более длинный русский текст' | linguae
# rus

linguae --file sentences.txt
# one language code per non-empty line

linguae --json "Bonjour tout le monde, comment allez-vous ?"
# {"text":"...","language":"fra","confidence":0.826987,"top_k":[...]}
```

`--file` and `--json` can be combined; JSON output is emitted as one object per
line so it pipes cleanly into `jq`.

## Options

| Option                | Description                                                                            |
| --------------------- | -------------------------------------------------------------------------------------- |
| `TEXT`                | Text to classify. When omitted, stdin is read.                                         |
| `--file, -f PATH`     | Classify each non-empty line of a UTF-8 file.                                          |
| `--artifact, -a PATH` | Artifact directory. Defaults to `crates/linguae-train/artifacts/linguae`.              |
| `--device TEXT`       | Inference device: `cpu` (default), `cuda`, or `metal` (`mps` is an alias for `metal`). |
| `--top-k, -k INT`     | Number of alternatives included in JSON output (1–20, default 3).                      |
| `--confidence, -c`    | Append the confidence score in plain-text output.                                      |
| `--json`              | Emit JSON Lines with text, language, confidence and alternatives.                      |
| `--version, -V`       | Print the CLI version.                                                                 |

The artifact directory can also be set with `LINGUAE_ARTIFACT_DIR`. The model
performs **single-label classification per string**; it does not detect
mixed-language spans.

`cuda` and `metal` require building with the matching feature:

```bash
cargo build --release -p linguae --features cuda
cargo build --release -p linguae --features metal
```

## Data preparation

Run from `crates/` (the crate discovers `linguae-data/data` automatically, or
set `LINGUAE_DATA_DIR` to override it):

```bash
cd crates
cargo run --release -p linguae-data -- wili
cargo run --release -p linguae-data -- tatoeba
cargo run --release -p linguae-data -- words
# or all three in order
cargo run --release -p linguae-data -- all
```

| Subcommand | Input                                                    | Output                                                              |
| ---------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| `wili`     | `linguae-data/data/txt/wili-2018/{x,y}_{train,test}.txt` | `linguae-data/data/csv/wili-2018.csv` (`text,language`)             |
| `tatoeba`  | `linguae-data/data/csv/tatoeba.tsv`                      | `linguae-data/data/csv/tatoeba.csv` (`id,language,text`)            |
| `words`    | WiLI train + test splits                                 | `linguae-data/data/csv/wili-2018-words.csv` (`text,language,count`) |

`words` keeps up to 100,000 of the most frequent words per language, ordered by
language, then descending frequency, then word. Pass `--max-words N` to change
the cap (`0` keeps every word) or `--train-only` to exclude the WiLI test split.
Tokens are NFC-normalised, lowercased Unicode words; numbers, punctuation, and
tokens longer than 64 characters are dropped, and scripts without spaces (for
example Chinese and Japanese) are split into characters.

Every subcommand honours `LINGUAE_DATA_DIR`, `LINGUAE_WILI_DIR`,
`LINGUAE_WILI_CSV`, `LINGUAE_TATOEBA_TSV`, `LINGUAE_TATOEBA_CSV`, and
`LINGUAE_WILI_WORDS_CSV`. Outputs are written atomically, and the reported
SHA-256 covers the completed file.

## Training

`linguae-train` ports the Python trainer to Rust. It reads the prepared CSVs,
trains the CharCNN, evaluates it, writes SVG plots, and exports
`model.safetensors` plus the metadata files:

```bash
cd crates
cargo run --release -p linguae-train -- train
cargo run --release -p linguae-train -- verify --device cpu
```

`train` is the default subcommand; `convert` migrates a legacy `model.pt`
artifact and `verify` loads an artifact and prints sample predictions. See
[`linguae-train/README.md`](linguae-train/README.md) for all options and
environment variables.

## Test

```bash
cargo test --workspace
```

`linguae-core/tests/parity.rs` compares the Rust forward pass against golden
predictions generated by the Python artifact (`tests/fixtures/golden.json`).
Labels must match exactly and confidence/top-k scores must agree within `1e-5`.

`linguae-data/tests/` builds temporary WiLI-2018 and Tatoeba fixtures and covers
each conversion, word extraction, error handling, and the CLI subcommands.

## Artifact resolution

The CLI searches for the artifact in this order:

1. Explicit `--artifact PATH` or `LINGUAE_ARTIFACT_DIR`.
2. The current directory and its parents:
   `<root>/crates/linguae-train/artifacts/linguae`, then `<root>/artifacts/linguae`.
3. The executable's directory and its parents (up to four levels).

The artifact must contain `config.json`, `vocabulary.json`, `labels.json`, and
either `model.safetensors` (preferred) or a legacy `model.pt`.

## Note on text length

The model truncates inputs to 256 characters. It performs best on texts of at
least ~50–100 characters; very short texts may produce low-confidence or
incorrect predictions.
