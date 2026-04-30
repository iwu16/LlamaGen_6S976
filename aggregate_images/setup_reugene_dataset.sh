#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${REDTEAM_DATASET_ROOT:-/home/reugene/orcd/pool/redteam_images}"
SOURCE_DIR="${1:-/home/reugene/LlamaGen_6S976/samples/clean}"
USER_NAME="reugene"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$DATASET_ROOT/incoming/$USER_NAME"
mkdir -p "$DATASET_ROOT/tarballs"
mkdir -p "$DATASET_ROOT/all_pngs"
mkdir -p "$DATASET_ROOT/manifests"

rsync -av --include='*/' --include='*.png' --include='*.PNG' --exclude='*' \
  "$SOURCE_DIR/" "$DATASET_ROOT/incoming/$USER_NAME/"

find "$DATASET_ROOT/incoming/$USER_NAME" -type f \( -iname '*.png' \) | sort \
  > "$DATASET_ROOT/manifests/${USER_NAME}_files.txt"

count="$(wc -l < "$DATASET_ROOT/manifests/${USER_NAME}_files.txt")"

echo "Initialized red-team dataset root: $DATASET_ROOT"
echo "Copied $count PNG files for $USER_NAME"
echo "Next: place contributor tarballs in $DATASET_ROOT/tarballs"

