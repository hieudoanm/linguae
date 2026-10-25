# Linguae Server

FastAPI server for the Linguae language detector. Accepts JSON bodies or form
uploads and returns one language prediction per string.

## Endpoints

| Method | Path            | Description                        |
| ------ | --------------- | ---------------------------------- |
| GET    | `/health`       | Health check                       |
| POST   | `/detect`       | Single text (JSON body)            |
| POST   | `/detect/form`  | Single text (form-encoded)         |
| POST   | `/detect/batch` | Batch of texts (JSON body)         |
| POST   | `/detect/file`  | Batch from uploaded UTF-8 file     |

## Middlewares

- **Request ID** — `X-Request-ID` header injected on every request and echoed
  in the response.
- **Rate limiting** — sliding-window limiter keyed by client IP (default 60
  requests per 60 seconds).
- **CORS** — configurable origins, methods, and headers.
- **GZip compression** — responses above 500 bytes are compressed.
- **Timeout** — responses are tagged with `X-Response-Time-Seconds`.

## Install

```bash
cd packages/server
uv sync
```

## Run

```bash
linguae-server
# or
uv run linguae-server
```

Open <http://127.0.0.1:8000/docs> for the interactive API docs.

## Configuration

All settings are overridable via environment variables:

| Variable                     | Default            | Description                    |
| ---------------------------- | ------------------ | ------------------------------ |
| `LINGUAE_SERVER_HOST`        | `0.0.0.0`          | Bind host                      |
| `LINGUAE_SERVER_PORT`        | `8000`             | Bind port                      |
| `LINGUAE_ARTIFACT_DIR`       | `../models/artifacts/linguae` | Artifact directory |
| `LINGUAE_DEVICE`             | `cpu`              | Torch device                   |
| `LINGUAE_RATE_LIMIT`         | `true`             | Enable rate limiting           |
| `LINGUAE_RATE_LIMIT_REQUESTS`| `60`               | Max requests per window        |
| `LINGUAE_RATE_LIMIT_WINDOW`  | `60`               | Window size in seconds         |
| `LINGUAE_CORS`               | `true`             | Enable CORS                    |
| `LINGUAE_CORS_ORIGINS`       | `*`                | Comma-separated origins        |
| `LINGUAE_TIMEOUT`            | `30.0`             | Request timeout in seconds     |