# @linguae/landing

Marketing landing page for [Linguae](https://github.com/hieudoanm/linguae),
built with Next.js, Tailwind CSS, and daisyUI.

## Stack

- Next.js 16 (App Router, static export)
- Tailwind CSS 4 + daisyUI 5 with custom `linguae` / `linguae-dark` themes
  (light/dark, no gradients)
- Vitest + Testing Library for unit tests
- Prettier with `prettier-plugin-tailwindcss`

## Commands

```bash
pnpm dev            # next dev
pnpm build          # static export to out/
pnpm test:run       # vitest run (single pass)
pnpm test:coverage  # vitest run --coverage
pnpm lint           # eslint src
pnpm format         # prettier --write .
pnpm format:check   # prettier --check .
```
