#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash package_pngs.sh /path/to/samples contributor_name" >&2
  echo "Example: bash package_pngs.sh /home/maureenz/LlamaGen_6S976/samples maureenz" >&2
  exit 1
fi

SOURCE_ROOT="$1"
CONTRIBUTOR="$2"
OUT="${CONTRIBUTOR}_samples.tar.gz"
TMPDIR="$(mktemp -d)"
CATEGORIES=(clean tokens watermarked)

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source directory does not exist: $SOURCE_ROOT" >&2
  exit 1
fi

case "$CONTRIBUTOR" in
  *[!A-Za-z0-9_-]*|"")
    echo "Contributor name must contain only letters, numbers, underscore, or dash." >&2
    exit 1
    ;;
esac

mkdir -p "$TMPDIR/$CONTRIBUTOR"

total=0
for category in "${CATEGORIES[@]}"; do
  source_dir="$SOURCE_ROOT/$category"
  dest_dir="$TMPDIR/$CONTRIBUTOR/$category"
  mkdir -p "$dest_dir"

  if [[ ! -d "$source_dir" ]]; then
    echo "Warning: missing category folder, skipping: $source_dir" >&2
    continue
  fi

  rsync -a "$source_dir/" "$dest_dir/"
  count="$(find "$dest_dir" -type f | wc -l)"
  total=$((total + count))
done

if [[ "$total" -eq 0 ]]; then
  echo "No sample files found under: $SOURCE_ROOT" >&2
  exit 1
fi

tar -C "$TMPDIR" -czf "$OUT" "$CONTRIBUTOR"

echo "Created $OUT with $total sample files."
echo "Send it to the other collaborators so they can place it under aggregated_samples/tarballs/."
