# AGENTS.md — packages/landing

Static marketing page for Linguae. **No build step, no package.json, no
JavaScript.** The entire site is `index.html` + `styles.css` + `svg/` assets.

## Layout

```
landing/
├── index.html    # Single-page markup: nav, hero, features, demo, how-it-works, install, footer
├── styles.css    # All styling — design tokens + components, hand-written
├── svg/          # Logo, favicon, OG images (referenced with relative paths)
└── README.md
```

## Rules

- **Keep it static.** Do not add frameworks, bundlers, npm packages, or
  `<script>` tags. If interactivity is ever needed, prefer CSS-only solutions.
- **Relative paths only.** Assets are referenced as `svg/...`, `styles.css` —
  never absolute `/...` paths, so the page works under any base path
  (GitHub Pages serves it at `/linguae/`).
- **Light theme only.** The page palette is light; terminal mockups
  (`.terminal`) intentionally stay dark in all contexts. Do not add a
  `prefers-color-scheme` dark theme.
- **Design tokens live in `:root`** at the top of `styles.css` — colors,
  radii, fonts, spacing. Reuse them; don't hardcode new hex values in
  components.
- **Unslop design rules** apply (see `notes/ai-unslop-design.md` in the
  hieudoanm notes repo): no filler badges, monospace only for code/terminals,
  consistent icon family (24px viewBox, 1.5px stroke, `currentColor`), teal
  `--accent` for CTAs/accents only, real CLI output in demos — no mock
  illustrations, `tabular-nums` for numbers, `text-wrap: balance` on headings.
- No gradients, no pure `#000`/`#fff` — use the gray-shaded palette.

## Verify

There is no test/build tooling. Sanity-check by serving the directory and
opening `index.html`:

```bash
python3 -m http.server 8080 -d packages/landing
```

Deployed automatically to GitHub Pages by `landing-deploy` in
`.github/workflows/ci.yaml` on `master` pushes.
