#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/crates"

echo "Building CLI..."
cargo build --release -p linguae

BIN="$ROOT/crates/target/release/linguae"

echo
echo "Running against examples/examples.txt"
echo "--------------------------------------"

# Collect results
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

maxlen=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  out=$("$BIN" --confidence "$line")
  code=$(echo "$out" | awk '{print $1}')
  score=$(echo "$out" | awk '{print $2}')
  pct=$(awk -v s="$score" 'BEGIN { printf "%.2f%%", s*100 }')
  printf '%s\t%s\t%s\n' "$line" "$code" "$pct" >> "$tmp"
  [ ${#line} -gt $maxlen ] && maxlen=${#line}
done < "$ROOT/examples/examples.txt"

# Print table
{
  printf 'TEXT\tLANG\tCONFIDENCE\n'
  cat "$tmp"
} | column -t -s $'\t'
