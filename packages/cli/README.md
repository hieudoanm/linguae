# Linguae CLI

Detect the language of a piece of text from the command line. The `linguae`
binary loads the trained artifact in `packages/models/artifacts/linguae` and
prints one prediction per input.

## Install

```bash
cd packages/cli
uv sync
```

## Usage

```bash
linguae "Bonjour tout le monde, comment allez-vous aujourd'hui dans ce beau monde"
# fra

linguae --confidence "Hello world, this is a longer English text to test language detection"
# eng	0.9998

printf 'Привет, мир, это более длинный русский текст для проверки определения языка' | linguae
# rus

linguae --file sentences.txt
# one language code per non-empty line

linguae --json "Bonjour tout le monde, comment allez-vous ?"
# {"text": "Bonjour tout le monde, comment allez-vous ?", "language": "fra", "confidence": 0.9998, ...}
```

`--file` and `--json` can be combined; JSON output is emitted as one object per
line so it pipes cleanly into `jq`.

## Options

| Option | Description |
| --- | --- |
| `TEXT` | Text to classify. When omitted, stdin is read. |
| `--file, -f PATH` | Classify each non-empty line of a UTF-8 file. |
| `--artifact, -a PATH` | Artifact directory. Defaults to `packages/models/artifacts/linguae`. |
| `--device TEXT` | Torch device for inference (`cpu` by default; `mps` or `cuda` also work). |
| `--top-k, -k INT` | Number of alternatives included in JSON output (default 3). |
| `--confidence, -c` | Append the confidence score in plain-text output. |
| `--json` | Emit JSON Lines with text, language, confidence and alternatives. |
| `--version` | Print the CLI version. |

The artifact directory can also be set with `LINGUAE_ARTIFACT_DIR`. The model
performs **single-label classification per string**; it does not detect
mixed-language spans.

## Note on text length

The underlying model was trained on the WiLI-2018 dataset, which consists of
paragraphs typically 100–1000+ characters long. The model performs best on
texts of at least ~50–100 characters. Very short texts (e.g. "Hello world",
"Bonjour") may produce low-confidence or incorrect predictions because there
is insufficient character-level signal for reliable classification.

For production use with short texts, consider fine-tuning on a dataset
representative of your target text lengths.
