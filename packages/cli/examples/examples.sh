#!/usr/bin/env bash
# Run linguae binary on examples.txt
# Usage: ./examples.sh [binary_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_BINARY="$CLI_ROOT/dist/linguae"
BINARY="${1:-$DEFAULT_BINARY}"
EXAMPLES_FILE="$SCRIPT_DIR/examples.txt"

if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary not found at $BINARY"
    echo "Build it first with: cd $CLI_ROOT && pyinstaller linguae.spec"
    exit 1
fi

if [[ ! -f "$EXAMPLES_FILE" ]]; then
    echo "Error: Examples file not found at $EXAMPLES_FILE"
    exit 1
fi

echo "=== Linguae CLI Examples ==="
echo "Binary: $BINARY"
echo "Examples: $EXAMPLES_FILE"
echo

# Read non-comment, non-empty lines from examples.txt
while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    # Display first 80 chars (handling UTF-8 properly with printf)
    display="$(printf '%.80s' "$line")"
    [[ ${#line} -gt 80 ]] && display="${display}..."

    echo "Input:  $display"
    echo -n "Output: "
    "$BINARY" --confidence "$line"
    echo
done < "$EXAMPLES_FILE"

echo "=== Batch mode (--file) ==="
echo "Running: $BINARY --file $EXAMPLES_FILE"
"$BINARY" --file "$EXAMPLES_FILE"

echo
echo "=== JSON output (--json) ==="
echo "Running: $BINARY --file $EXAMPLES_FILE --json"
"$BINARY" --file "$EXAMPLES_FILE" --json