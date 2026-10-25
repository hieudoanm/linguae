#!/bin/bash
# Generate the Tauri icon set from the Linguae logo SVG.
# Usage: ./create-icons.sh
set -euo pipefail

# Resolve the repo root relative to this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SCRIPT_DIR}/../../../../assets/logo/logo-mark-light.svg"
TMP_PNG=".icon-1024.png"

rsvg-convert -w 1024 -h 1024 -b none "$SOURCE" -o "$TMP_PNG"
pnpm tauri icon "$TMP_PNG"
rm -f "$TMP_PNG"