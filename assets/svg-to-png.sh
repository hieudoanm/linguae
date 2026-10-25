#!/usr/bin/env bash
# Convert all SVG files in the project to PNG using librsvg (rsvg-convert)
# Usage: svg-to-png.sh [base_dir]
# Walks the directory tree and converts every *.svg to a sibling *.png

set -euo pipefail

BASE_DIR="${1:-.}"

if ! command -v rsvg-convert >/dev/null 2>&1; then
  echo "rsvg-convert not found. Install librsvg:"
  echo "  macOS:  brew install librsvg"
  echo "  Debian: sudo apt-get install librsvg2-bin"
  echo "  Arch:   sudo pacman -S librsvg"
  exit 1
fi

while IFS= read -r -d '' svg; do
  png="${svg%.svg}.png"
  echo "Converting $svg -> $png"
  rsvg-convert -o "$png" "$svg"
done < <(find "$BASE_DIR" -type f -name "*.svg" -print0)

echo "Done."