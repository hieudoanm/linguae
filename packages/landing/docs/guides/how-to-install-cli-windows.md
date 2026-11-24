# How to install the linguae CLI on Windows

For 64-bit Windows. The Linguae CLI for the language-detection pipeline.

The release binary is a native `.exe`. The install script is bash, so on Windows
run it from **Git Bash** or **WSL** — or download the binary manually.

## Option 1: install script (recommended)

```sh
curl -fsSL https://github.com/hieudoanm/linguae/master/scripts/install.sh | bash
```

The script detects Windows and downloads
`cli-linguae-x86_64-pc-windows-msvc.exe`, installing it to
`~/.local/bin/linguae` (or the path you pass as `INSTALL_DIR`).

## Option 2: manual download

1. Open the [downloads page](https://hieudoanm.github.io/linguae/downloads/) and
   download **Windows** under **CLI** —
   `cli-linguae-x86_64-pc-windows-msvc.exe`.
2. Rename it to `linguae.exe` and place it in a directory on your `PATH` (for
   example `%LOCALAPPDATA%\Programs\linguae\` or `C:\tools\`).
3. Open a new terminal and verify:

```powershell
linguae --version
```

## Troubleshooting

- `linguae is not recognized` — the directory is not on `PATH`. Add it via
  Settings → Environment Variables, or call the binary by full path.
- SmartScreen warning — the binary is unsigned. Choose **More info → Run
  anyway**.
