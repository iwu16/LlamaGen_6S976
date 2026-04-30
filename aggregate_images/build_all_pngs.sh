#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${REDTEAM_DATASET_ROOT:-/home/reugene/orcd/pool/redteam_images}"
INCOMING="$DATASET_ROOT/incoming"
ALL="$DATASET_ROOT/all_pngs"
MANIFEST="$DATASET_ROOT/manifests/all_pngs.tsv"

if [[ ! -d "$INCOMING" ]]; then
  echo "Incoming directory does not exist: $INCOMING" >&2
  echo "Run setup_reugene_dataset.sh first." >&2
  exit 1
fi

mkdir -p "$ALL" "$DATASET_ROOT/manifests"

: > "$MANIFEST"

find "$INCOMING" -mindepth 1 -maxdepth 1 -type d | sort |
  while IFS= read -r contributor_dir; do
    contributor="$(basename "$contributor_dir")"
    find "$contributor_dir" -type f \( -iname '*.png' \) -print0 |
      while IFS= read -r -d '' file; do
        base="$(basename "$file")"
        dest="$ALL/${contributor}_${base}"
        if [[ -e "$dest" ]]; then
          stem="${base%.*}"
          ext="${base##*.}"
          hash="$(printf '%s' "$file" | sha1sum | awk '{print substr($1,1,10)}')"
          dest="$ALL/${contributor}_${stem}_${hash}.${ext}"
        fi
        cp -n "$file" "$dest"
        printf '%s\t%s\t%s\n' "$contributor" "$file" "$dest" >> "$MANIFEST"
      done
  done

count="$(find "$ALL" -maxdepth 1 -type f -iname '*.png' | wc -l)"

echo "Built combined PNG folder: $ALL"
echo "Combined PNG count: $count"
echo "Manifest: $MANIFEST"

