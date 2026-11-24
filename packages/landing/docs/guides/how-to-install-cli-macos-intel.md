# How to install the linguae CLI on macOS (Intel)

For Intel-based Macs. The Linguae CLI for the language-detection pipeline.

## Option 1: install script (recommended)

```sh
curl -fsSL https://github.com/hieudoanm/linguae/master/scripts/install.sh | bash
```

The script detects `macos/x86_64` and downloads
`cli-linguae-x86_64-apple-darwin` from the
[latest release](https://github.com/hieudoanm/linguae/releases/latest). It
installs to `~/.local/bin/linguae` — override with `INSTALL_DIR=/path`.

## Option 2: manual download

1. Open the [downloads page](https://hieudoanm.github.io/linguae/downloads/) and
   download **macOS (Intel)** under **CLI** — `cli-linguae-x86_64-apple-darwin`.
2. Make it executable and move it onto your `PATH`:

```sh
chmod +x cli-linguae-x86_64-apple-darwin
mv cli-linguae-x86_64-apple-darwin ~/.local/bin/linguae
```

3. Verify:

```sh
linguae --version
```

## Troubleshooting

- `linguae: command not found` — the install directory is not on your `PATH`.
  Add `export PATH="$HOME/.local/bin:$PATH"` to your shell profile.
- Gatekeeper warning — the binary is unsigned. Right-click → Open, or run
  `xattr -d com.apple.quarantine` on the installed binary.
