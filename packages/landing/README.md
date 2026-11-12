# Linguae landing

Static landing page for Linguae. No build step, no dependencies, no JavaScript —
just open `index.html` or serve the directory:

```bash
python3 -m http.server 8080 -d packages/landing
```

All asset paths are relative, so the page works at any mount point (e.g. GitHub
Pages at `/linguae/`). Light theme only; terminal mockups stay dark. Deployed to
GitHub Pages by the `landing-deploy` job in `.github/workflows/ci.yaml` on
pushes to `master`.
