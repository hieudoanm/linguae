# Linguae

**Detect the language of text with a local character-level model — no cloud, no pretrained weights, no network calls.**

Linguae is a local-first language identification project. It trains a
character-level CNN from scratch in PyTorch on the Tatoeba (short sentences),
WiLI-2018 (long paragraphs), and per-language WiLI word datasets and ships a
standalone command-line tool that predicts exactly one language per input
string. Everything runs on your own machine.

[Documentation](AGENTS.md) · [🚀 Quick Start](#get-started) · [🔍 See It In Action](#see-it-in-action) · [CLI Usage](crates/README.md) · [Model Training](packages/models/README.md)

![Linguae](assets/og/og-light.png)

## The Problem

Language identification still depends on cloud APIs, pretrained weights, or
third-party services — none of which are acceptable for offline, privacy-sensitive,
or air-gapped workflows. Even open-source models often bundle large pretrained
backports that are hard to inspect or shrink.

## The Solution

Linguae trains a small character-level CNN from scratch in PyTorch and ships
a small Rust CLI binary that runs inference with [Candle](https://github.com/huggingface/candle).
The model performs **single-label classification per string**; it does not
detect mixed-language spans.

- **Local-first & private** — Inference runs entirely on your machine. Your text never leaves your environment.
- **From text to language in one command** — `linguae "Bonjour tout le monde"` prints `fra`.
- **Honest, task-aware model** — One language label per string, with optional confidence scores and top-k alternatives.
- **Reproducible by design** — Fixed random seed, pinned dependency versions, and a deterministic train/val/test split that groups exact duplicate texts within a single split.
- **No pretrained models** — The character-level CNN is trained from scratch on the Tatoeba, WiLI-2018, and WiLI word datasets.
- **Short and long input** — Tatoeba supplies short conversational sentences; WiLI-2018 supplies long sentences and paragraphs; the WiLI word dataset covers one- to few-word strings, all sharing one label space.

## Features

- **Character-level CNN** — Embedding, 1D convolutions (kernel sizes 3, 5, 7), masked max-pooling that ignores padding, and a fully connected classification head.
- **Single-label prediction** — One ISO 639-3 language code per input string.
- **Confidence & top-k** — `--confidence` appends a softmax score; `--json` emits JSON Lines with text, language, confidence, and alternatives.
- **Batch mode** — `--file PATH` classifies each non-empty line of a UTF-8 file.
- **Device selection** — `--device cpu`, `cuda`, or `metal` (`mps` is an alias for `metal`).
- **Standalone binary** — A small Rust binary distributed via GitHub Releases tagged `latest`; no Python runtime required.

## See It in Action

![Linguae demo](assets/demo/demo.png)

Or run the commands directly:

```bash
linguae "Bonjour tout le monde, comment allez-vous aujourd'hui dans ce beau monde"
# fra

linguae --confidence "Hello world, this is a longer English text to test language detection"
# eng	0.6114

printf 'Привет, мир, это более длинный русский текст' | linguae
# rus

linguae --json "Bonjour tout le monde, comment allez-vous ?"
# {"text": "Bonjour tout le monde, comment allez-vous ?", "language": "fra", "confidence": 0.826987, ...}
```

## Get Started

### Prerequisites

- [Rust](https://www.rust-lang.org/) 1.85+ (CLI)
- [Python](https://www.python.org/) 3.14.8 and [uv](https://github.com/astral-sh/uv) (training only)

### Install the CLI

```bash
cd crates
cargo build --release -p linguae
```

### Train the model (optional)

```bash
cd packages/models
uv sync
make csv                      # build the WiLI, Tatoeba and word CSVs
make train                    # run scripts/train_language_detector.py
```

### Run

```bash
crates/target/release/linguae "Bonjour tout le monde"
```

## HTTP Server and Web App

The `packages/app/server` package exposes the same language detector as a FastAPI HTTP API. The `packages/app` package contains a Next.js web app to interact with it.

### Install and Run

```bash
# Start both with docker-compose
docker-compose up --build

# Or run separately
cd packages/app/server
uv sync
linguae-server

cd packages/app
npm install
npm run dev
```

The server binds to `0.0.0.0:8000` by default. The web app runs on `http://localhost:3000`.
Open <http://127.0.0.1:8000/docs> for the interactive API docs.

### Endpoints

| Method | Path            | Body         | Description                |
| ------ | --------------- | ------------ | -------------------------- |
| GET    | `/health`       | —            | Health check               |
| POST   | `/detect`       | JSON         | Single text detection      |
| POST   | `/detect/form`  | Form-encoded | Single text (form upload)  |
| POST   | `/detect/batch` | JSON         | Batch of texts             |
| POST   | `/detect/file`  | File upload  | Batch from UTF-8 text file |

### Examples

Detect a single language via JSON:

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Bonjour tout le monde, comment allez-vous ?","top_k":3}'
# {"text":"...","language":"fra","confidence":0.99,"top_k":[...]}
```

Detect via form upload:

```bash
curl -X POST http://127.0.0.1:8000/detect/form \
  -d "text=Hello world, this is English" \
  -d "top_k=3"
```

Batch detection via JSON:

```bash
curl -X POST http://127.0.0.1:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Bonjour tout le monde","Hello world"],"top_k":2}'
```

Batch detection from an uploaded file:

```bash
curl -X POST http://127.0.0.1:8000/detect/file \
  -F "file=@texts.txt"
```

### Configuration

All settings are overridable via environment variables (see `packages/app/server/README.md`):

| Variable                      | Default                       | Description                |
| ----------------------------- | ----------------------------- | -------------------------- |
| `LINGUAE_SERVER_HOST`         | `0.0.0.0`                     | Bind host                  |
| `LINGUAE_SERVER_PORT`         | `8000`                        | Bind port                  |
| `LINGUAE_ARTIFACT_DIR`        | `../models/artifacts/linguae` | Artifact directory         |
| `LINGUAE_DEVICE`              | `cpu`                         | Torch device               |
| `LINGUAE_RATE_LIMIT`          | `true`                        | Enable rate limiting       |
| `LINGUAE_RATE_LIMIT_REQUESTS` | `60`                          | Max requests per window    |
| `LINGUAE_RATE_LIMIT_WINDOW`   | `60`                          | Window size in seconds     |
| `LINGUAE_CORS`                | `true`                        | Enable CORS                |
| `LINGUAE_CORS_ORIGINS`        | `*`                           | Comma-separated origins    |
| `LINGUAE_TIMEOUT`             | `30.0`                        | Request timeout in seconds |

### Middlewares

- **Request ID** — `X-Request-ID` header injected and echoed on every response.
- **Rate limiting** — sliding window keyed by client IP (default 60 req/60s).
- **CORS** — configurable origins, methods, and headers.
- **GZip compression** — responses above 500 bytes are compressed.
- **Timeout** — `X-Response-Time-Seconds` header on every response.

## Use Cases

- **Rapid prototyping** — Detect the language of text snippets without setting up a full NLP pipeline.
- **Offline workflows** — Keep sensitive data on-device while classifying text.
- **Teaching & learning** — Explore character-level CNNs, train/val/test splitting, and evaluation end-to-end.
- **Model portability** — Ship a single binary that runs on any Linux machine without Python installed.

## Why This Project?

- **Focused scope** — Does one thing well: predict one language per string from a character-level CNN. No AutoML bloat, no pretrained weights.
- **Transparent** — Clear training pipeline, per-metric explanations, and reproducibility metadata.
- **Local-first** — Runs entirely on your machine; no vendor lock-in or external dependencies.
- **Production-minded demo** — Real PyTorch training + Rust/Candle inference implementation with CI, linting, and GitHub Releases.

## Training Visualisations

The model is trained end-to-end on Tatoeba (short sentences) and WiLI-2018
(long paragraphs). The following visualisations are produced in
`packages/models/artifacts/images/`:

![Network architecture](packages/models/artifacts/images/network_architecture.png)

![Layer dimensions](packages/models/artifacts/images/layer_dimensions.png)

![Training curves](packages/models/artifacts/images/training_curves.png)

![Confusion matrix](packages/models/artifacts/images/confusion_matrix.png)

![Prediction probabilities](packages/models/artifacts/images/prediction_probabilities.png)

![Dataset quality](packages/models/artifacts/images/dataset_quality.png)

![Accuracy by length](packages/models/artifacts/images/accuracy_by_length.png)


## Project Status and Trust

- **Maturity:** Software demonstration (not intended for production safety-critical use).
- **Dataset:** Tatoeba short sentences plus WiLI-2018 long paragraphs, combined into one label space (235,000 WiLI examples before filtering).
- **Architecture:** Character-level CNN trained from scratch in PyTorch, exported as a small Rust CLI binary that runs inference with Candle.
- **License:** [MIT License](LICENSE) — see the `LICENSE` file for full terms.
- **Tech stack:** Rust 1.85+ (CLI, Candle, clap), Python 3.14.8 (training), PyTorch 2.14.1, NumPy 2.5.4, pandas 3.1.0, uv, ruff.
- **Note on text length:** The model is trained on both short Tatoeba sentences and long WiLI paragraphs, and it truncates inputs to its first 256 characters. Very short texts can still produce low-confidence or incorrect predictions.

## Community and Contributions

Contributions, bug reports, and feature suggestions are welcome. See:

- [AGENTS.md](AGENTS.md) — architecture, pipeline, dependencies, build/test commands
- [crates/README.md](crates/README.md) — Rust CLI usage, artifact resolution, parity tests
- [packages/models/AGENTS.md](packages/models/AGENTS.md) — training pipeline and data

To report issues or propose changes, open an issue or pull request in this repository.

## Call to Action

**[Get Started](#get-started)** — Run `linguae` locally in seconds and classify your first string.

## License

This project is licensed under the [MIT License](LICENSE).
