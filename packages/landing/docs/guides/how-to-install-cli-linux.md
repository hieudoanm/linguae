# How to install the linguae CLI on Linux

For x86_64 Linux distributions. The Linguae CLI for the language-detection
pipeline.

## Option 1: install script (recommended)

```sh
curl -fsSL https://github.com/hieudoanm/linguae/master/scripts/install.sh | bash
```

The script detects `linux/x86_64` and downloads
`cli-linguae-x86_64-unknown-linux-gnu` from the
[latest release](https://github.com/hieudoanm/linguae/releases/latest). It
installs to `~/.local/bin/linguae` — override with `INSTALL_DIR=/path`.

## Option 2: manual download

1. Open the [downloads page](https://hieudoanm.github.io/linguae/downloads/) and
   download **Linux** under **CLI** — `cli-linguae-x86_64-unknown-linux-gnu`.
2. Make it executable and move it onto your `PATH`:

```sh
chmod +x cli-linguae-x86_64-unknown-linux-gnu
mv cli-linguae-x86_64-unknown-linux-gnu ~/.local/bin/linguae
```

3. Verify:

```sh
linguae --version
```

## Troubleshooting

- `linguae: command not found` — ensure `~/.local/bin` is on your `PATH` (most
  distributions already include it).
- `No such file or directory` on a glibc system — the binary is built against
  glibc; on musl-based distros (Alpine) build from source instead.
