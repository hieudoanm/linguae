# Linguae Web

**Desktop app for Linguae language identification — built with Next.js + Tauri, distributed for Linux, macOS, and Windows.**

Linguae Web packages the Linguae language detector into a cross-platform desktop
application. It is built with Next.js for the UI and Tauri (Rust) for the native
shell, and is released as installable bundles for every supported operating system
via GitHub Releases tagged `latest`.

## Release Artifacts

Each `web-release` run produces bundles for all three platforms:

- **Linux** — `.AppImage`, `.deb`
- **macOS** — `.dmg`, `.app`
- **Windows** — `.msi`, `.exe` (NSIS)

Download the bundle matching your OS from the `latest` GitHub Release.

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 26.x
- [pnpm](https://pnpm.io/) 12.x
- [Rust](https://www.rust-lang.org/tools/install) (stable)
- [Tauri prerequisites](https://tauri.app/start/prerequisites/) for your OS

### Install and Run

```bash
cd packages/app/web
pnpm install
pnpm run tauri          # dev mode
pnpm run tauri:build    # production build
```

### Build Web Only

```bash
cd packages/app/web
pnpm install
pnpm run build          # outputs static export to out/
```

## Development

```bash
pnpm run dev            # Next.js dev server on http://localhost:3000
pnpm run lint           # ESLint
pnpm run test:run       # Vitest
```

## Release Process

On push to `master`/`main`, CI runs `web-build` (lint, test, build, upload `out`
artifact) followed by `web-release`, which builds Tauri bundles for Linux, macOS,
and Windows and uploads them to the `latest` GitHub Release.

## License

This project is licensed under the [MIT License](../../../LICENSE).