# Model and architecture

Linguae trains a **character-level convolutional neural network** from scratch
with [Candle](https://github.com/huggingface/candle) — no pretrained weights,
no downloads at inference time.

## Why character-level

Character n-grams capture script and orthography directly, which makes the
model robust across languages with different writing systems and avoids the
vocabulary problems of word-level models.

## Training data

| Dataset | Content |
| --- | --- |
| Tatoeba | Short sentences, many language pairs |
| WiLI-2018 | Long paragraphs across ~235 languages |
| WiLI word lists | Per-language vocabulary |

The `linguae-data` crate builds the CSV corpora; `linguae-train` trains the
CNN and exports the artifact consumed by the CLI and HTTP server.
