# MCP server

`linguae` ships a [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio. Agents can detect languages through the same local model
as the CLI — no network calls, no pretrained weights.

## Run

```sh
linguae mcp
```

The server reads newline-delimited JSON-RPC on stdin and writes responses on
stdout. It delegates every tool call to the `linguae` binary itself.

## Client configuration

```json
{
  "mcpServers": {
    "linguae": {
      "command": "linguae",
      "args": ["mcp"]
    }
  }
}
```

The model artifact resolves exactly as on the CLI (`--artifact`,
`LINGUAE_ARTIFACT_DIR`, or the default search path).

## Tools

| Tool              | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `detect_language` | Detect the language of text; returns an ISO 639-3 code and scores  |

`detect_language` accepts `text` or `file` (one is required), plus optional
`top_k`, `device`, and `artifact`. Output is always JSON Lines, one prediction
per line. See [Train and run](../getting-started/train-and-run.md) for model
basics.


### Streamable HTTP

`linguae mcp --http 127.0.0.1:8080` also accepts `--http <addr>` (and `--token <token>`, or the
`MCP_AUTH_TOKEN` environment variable) to serve the same MCP surface over
streamable HTTP at `http://<addr>/mcp` — the transport used by ChatGPT
connectors and remote Claude clients. When a token is configured, every
request must carry `Authorization: Bearer <token>`.
