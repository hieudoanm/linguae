## Linguae — latest build

Fast, local-first language detection and text tooling: identify the
language of text or files on CPU or GPU, plus a desktop app for
interactive use. Ships a CLI (`linguae`) and an MCP server.

### Download

CLI binaries:

| Platform | Asset |
| --- | --- |
| Linux x86_64 | `linguae-x86_64-unknown-linux-gnu` |
| macOS Apple Silicon | `linguae-aarch64-apple-darwin` |
| Windows x86_64 | `linguae-x86_64-pc-windows-msvc.exe` |

Desktop bundles for Linux (`.AppImage`/`.deb`/`.rpm`), macOS (`.dmg`)
and Windows (`.msi`/`.exe`) are in the assets below.

### Quick start

```sh
echo "Bonjour le monde" | linguae detect
linguae mcp
```

### Documentation

- [README](https://github.com/hieudoanm/linguae#readme)
- [Docs site](https://hieudoanm.github.io/linguae/docs/)
