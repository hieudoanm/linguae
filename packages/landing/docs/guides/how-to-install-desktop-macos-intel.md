# How to install the Linguae desktop app on macOS (Intel)

For Macs with an Intel chip.

## Option 1: disk image (recommended)

1. Open the [downloads page](https://hieudoanm.github.io/linguae/downloads/) and
   download **macOS (Intel)** under **Desktop** —
   `desktop-linguae-x86_64-apple-darwin.dmg`.
2. Open the `.dmg` and drag **Linguae.app** into `/Applications`.
3. Eject the disk image and launch Linguae from `/Applications`.

## Option 2: zipped app bundle

If the `.dmg` does not work, download
`desktop-linguae-x86_64-apple-darwin.app.zip` instead, unzip it, and move
**Linguae.app** to `/Applications`:

```sh
unzip desktop-linguae-x86_64-apple-darwin.app.zip
mv "Linguae.app" /Applications/   # adjust to the unzipped name
```

## First launch (Gatekeeper)

The app is unsigned, so macOS may block the first launch. Either right-click
**Linguae.app** → **Open** → **Open**, or strip the quarantine attribute:

```sh
xattr -dr com.apple.quarantine /Applications/Linguae.app
```

## Verifying

Launch Linguae from `/Applications`, or from a terminal:

```sh
open -a "Linguae"
```
