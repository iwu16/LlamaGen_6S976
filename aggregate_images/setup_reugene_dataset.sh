#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${AGGREGATED_SAMPLES_ROOT:-$PWD/aggregated_samples}"
SOURCE_ROOT="${1:-$PWD/samples}"
USER_NAME="${2:-$(id -un)}"
CATEGORIES=(clean tokens watermarked)

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source directory does not exist: $SOURCE_ROOT" >&2
  exit 1
fi

case "$USER_NAME" in
  *[!A-Za-z0-9_-]*|"")
    echo "Contributor name must contain only letters, numbers, underscore, or dash." >&2
    exit 1
    ;;
esac

mkdir -p "$DATASET_ROOT/incoming/$USER_NAME"
mkdir -p "$DATASET_ROOT/tarballs"
mkdir -p "$DATASET_ROOT/manifests"
mkdir -p "$DATASET_ROOT/clean" "$DATASET_ROOT/tokens" "$DATASET_ROOT/watermarked"

for category in "${CATEGORIES[@]}"; do
  source_dir="$SOURCE_ROOT/$category"
  dest_dir="$DATASET_ROOT/incoming/$USER_NAME/$category"
  mkdir -p "$dest_dir"

  if [[ ! -d "$source_dir" ]]; then
    echo "Warning: missing category folder, skipping: $source_dir" >&2
    continue
  fi

  rsync -a "$source_dir/" "$dest_dir/"
done

find "$DATASET_ROOT/incoming/$USER_NAME" -type f | sort \
  > "$DATASET_ROOT/manifests/${USER_NAME}_files.txt"

count="$(wc -l < "$DATASET_ROOT/manifests/${USER_NAME}_files.txt")"

echo "Initialized aggregated samples root: $DATASET_ROOT"
echo "Copied $count sample files for $USER_NAME"
echo "Next: place collaborator tarballs in $DATASET_ROOT/tarballs"
