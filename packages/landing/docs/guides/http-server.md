# HTTP server and web app

`packages/app/server` exposes the same detector as a FastAPI API;
`packages/app/web` is a Next.js web app on top of it.

## Run

```bash
docker-compose up --build        # both services

# or separately
cd packages/app/server && uv sync && linguae-server   # :8000
cd packages/app/web && npm install && npm run dev          # :3000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

## Endpoints

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| GET | `/health` | — | Health check |
| POST | `/detect` | JSON | Single text detection |
| POST | `/detect/form` | Form | Single text (form upload) |
| POST | `/detect/batch` | JSON | Batch of texts |
| POST | `/detect/file` | File | Batch from a UTF-8 text file |

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Bonjour tout le monde","top_k":3}'
# {"text":"...","language":"fra","confidence":0.99,"top_k":[...]}
```

## Configuration

Environment variables (see `packages/app/server/README.md`):

| Variable | Default | Description |
| --- | --- | --- |
| `LINGUAE_SERVER_HOST` | `0.0.0.0` | Bind host |
| `LINGUAE_SERVER_PORT` | `8000` | Bind port |
| `LINGUAE_ARTIFACT_DIR` | `crates/linguae-train/artifacts/linguae` | Artifact directory |
| `LINGUAE_DEVICE` | `cpu` | Torch device |
| `LINGUAE_RATE_LIMIT` | `true` | Enable rate limiting |
| `LINGUAE_CORS` | `true` | Enable CORS |
| `LINGUAE_TIMEOUT` | `30.0` | Request timeout in seconds |

Built-in middlewares: request IDs (`X-Request-ID`), sliding-window rate
limiting (60 req/60s per IP by default), CORS, and gzip compression.
