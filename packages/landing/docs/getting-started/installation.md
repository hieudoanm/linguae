# Installation

## Prerequisites

- [Rust](https://www.rust-lang.org/) 1.85+ — CLI, data pipeline, training
- [Python](https://www.python.org/) 3.14.8 and [uv](https://github.com/astral-sh/uv) — HTTP server only

## Install the CLI

```bash
cd crates
cargo build --release -p linguae
```

The binary lands at `crates/target/release/linguae`.

## From a release

```bash
./scripts/install.sh        # prebuilt binary, or builds from source
./scripts/install-local.sh  # always build from this checkout
```
