# linguae documentation

**linguae** detects the language of text with a local character-level model —
no cloud, no pretrained weights, no network calls. It trains a character-level
CNN from scratch with Candle on the Tatoeba, WiLI-2018, and per-language WiLI
word datasets, and ships a standalone CLI that predicts exactly one language
per input string.

An MCP server (`linguae mcp`) exposes language detection as a tool for agents —
see [MCP server](guides/mcp-server.md).

## Sections

- **Getting started** — build the CLI, train the model, run a prediction.
- **Concepts** — the model architecture and training data.
- **Guides** — the FastAPI HTTP server, Next.js web app, and MCP server.
