# AGENTS.md — packages/landing

Static marketing page for Linguae. **No build step, no package.json, no
JavaScript except a tiny year-injection snippet.** The entire site is flat files — `index.html` + `styles.css` +
assets at the root, no subdirectories.

## Layout

```
landing/
├── index.html    # Single-page markup: nav, hero, features, demo, how-it-works, install, CTA, footer
├── 404.html      # GitHub Pages 404 page, reuses styles.css
├── styles.css    # All styling — design tokens + components, hand-written
├── logo.svg      # Brand mark (light theme)
├── favicon.svg   # Browser favicon
├── og.svg        # Social share image
├── robots.txt    # Allows all crawlers, points at sitemap.xml
├── sitemap.xml   # Single URL: https://hieudoanm.github.io/linguae/
├── humans.txt    # Project/team credits
├── LICENSE       # Copy of the repo MIT license (dir is uploaded verbatim)
└── README.md, AGENTS.md
```

## Rules

- **Keep it flat.** No subdirectories — all assets live at the root. Do not
  add frameworks, bundlers, npm packages, or `<script>` tags beyond the footer year snippet. If
  interactivity is ever needed, prefer CSS-only solutions.
- **Relative paths only.** Assets are referenced as `logo.svg`,
  `styles.css` — never absolute `/...` paths, so the page works under any
  base path (GitHub Pages serves it at `/linguae/`).
- **One light theme only** — `#ffffff` surfaces, `#f9fafb` alt sections,
  `#111827` text, `#e5e7eb` borders (same palette family as the llm.rs,
  neo.bank, and transcribers landings). Terminal mockups (`.terminal`)
  intentionally stay dark. Do not add theme variants, `prefers-color-scheme`
  handling, or `*-light`/`*-dark` asset duplicates.
- **Design tokens live in `:root`** at the top of `styles.css` — colors,
  radii, fonts, spacing, shadows. Reuse them; don't hardcode new hex values
  in components.
- **Unslop design rules** apply (see `notes/ai-unslop-design.md` in the
  hieudoanm notes repo): no filler badges, monospace only for code/terminals,
  consistent icon family (24px viewBox, 1.5px stroke, `currentColor`), teal
  `--accent` for CTAs/accents only, real CLI output in demos — no mock
  illustrations, `tabular-nums` for numbers, `text-wrap: balance` on headings.
- No gradients. Accent is teal (`--accent` family); reserve it for CTAs,
  icons, and small highlights.

## Verify

There is no test/build tooling. Sanity-check by serving the directory and
opening `index.html`:

```bash
python3 -m http.server 8080 -d packages/landing
```

Deployed automatically to GitHub Pages by `landing-deploy` in
`.github/workflows/ci.yaml` on `master` pushes.

## Conventions

- **Fonts:** Inter (sans) loaded via the Google Fonts `<link>` in `index.html`; mono stack `ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace`. Tokens: `--font-sans` / `--font-mono`. Do not add other typefaces.

## Footer

The footer follows the shared contract: `© {year} {repo-name}` (lowercase
repo name, dynamic year via `<span id="year">` + the year script), a
one-line tagline/license descriptor, and a right-aligned `<nav
aria-label="Footer">` with exactly two links — `GitHub` (repo root) and
`README` (`<repo>#readme`), both `target="_blank" rel="noreferrer"`.
