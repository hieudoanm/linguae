# Contributing to linguae

Thanks for contributing to linguae, a local-first language-identification
tool: a character-level CNN trained from scratch (Candle/Rust) plus a CLI, an
HTTP server, and a Next.js web app.

## Repository layout

- `crates/` — Rust workspace: `linguae` CLI, `linguae-data` (datasets),
  `linguae-train` (training), plus shared crates.
- `packages/app/server` — FastAPI HTTP server (Python, `uv`).
- `packages/app` — Next.js web app.
- `docker-compose.yaml` — runs server + web together.

See [AGENTS.md](AGENTS.md) for the full architecture and conventions.

## Setup

```bash
git clone https://github.com/hieudoanm/linguae.git
cd linguae

# Rust CLI
cd crates && cargo build --release -p linguae

# Or the full stack
pnpm install
docker-compose up --build
```

Requires Rust 1.85+, Python 3.14.8 with `uv` (server), and Node/pnpm (web).

## Checks

```bash
cd crates
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

cd ../packages/app   # frontend
pnpm typecheck && pnpm lint && pnpm test && pnpm build
```

## Conventions

- The model performs single-label classification per string — do not add
  mixed-language span detection without a spec change.
- Inference must stay local-first: no network calls, no cloud dependencies.
- Keep the CLI, HTTP API, and web app behaviour in parity; update
  `crates/README.md` and `packages/app/server/README.md` when behaviour
  changes.

## License

By contributing, you agree that your contributions will be licensed under the
repository license in [`LICENSE`](LICENSE).
