#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${AGGREGATED_SAMPLES_ROOT:-$PWD/aggregated_samples}"
INCOMING="$DATASET_ROOT/incoming"
CATEGORIES=(clean tokens watermarked)
MANIFEST="$DATASET_ROOT/manifests/aggregated_samples.tsv"

if [[ ! -d "$INCOMING" ]]; then
  echo "Incoming directory does not exist: $INCOMING" >&2
  echo "Run setup_reugene_dataset.sh first." >&2
  exit 1
fi

mkdir -p "$DATASET_ROOT/manifests"
for category in "${CATEGORIES[@]}"; do
  mkdir -p "$DATASET_ROOT/$category"
  find "$DATASET_ROOT/$category" -mindepth 1 -type f -delete
done

: > "$MANIFEST"

find "$INCOMING" -mindepth 1 -maxdepth 1 -type d | sort |
  while IFS= read -r contributor_dir; do
    contributor="$(basename "$contributor_dir")"
    for category in "${CATEGORIES[@]}"; do
      category_dir="$contributor_dir/$category"
      [[ -d "$category_dir" ]] || continue

      find "$category_dir" -type f -print0 |
        while IFS= read -r -d '' file; do
          base="$(basename "$file")"
          dest="$DATASET_ROOT/$category/${contributor}_${base}"
          if [[ -e "$dest" ]]; then
            stem="${base%.*}"
            ext="${base##*.}"
            hash="$(printf '%s' "$file" | sha1sum | awk '{print substr($1,1,10)}')"
            if [[ "$stem" == "$ext" ]]; then
              dest="$DATASET_ROOT/$category/${contributor}_${base}_${hash}"
            else
              dest="$DATASET_ROOT/$category/${contributor}_${stem}_${hash}.${ext}"
            fi
          fi
          cp -n "$file" "$dest"
          printf '%s\t%s\t%s\t%s\n' "$contributor" "$category" "$file" "$dest" >> "$MANIFEST"
        done
    done
  done

clean_count="$(find "$DATASET_ROOT/clean" -type f | wc -l)"
tokens_count="$(find "$DATASET_ROOT/tokens" -type f | wc -l)"
watermarked_count="$(find "$DATASET_ROOT/watermarked" -type f | wc -l)"

echo "Built aggregated samples under: $DATASET_ROOT"
echo "clean files: $clean_count"
echo "tokens files: $tokens_count"
echo "watermarked files: $watermarked_count"
echo "Manifest: $MANIFEST"
