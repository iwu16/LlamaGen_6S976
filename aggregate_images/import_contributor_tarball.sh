#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash aggregate_images/import_contributor_tarball.sh /path/to/contributor_pngs.tar.gz contributor_name" >&2
  exit 1
fi

TARBALL="$1"
CONTRIBUTOR="$2"
DATASET_ROOT="${REDTEAM_DATASET_ROOT:-/home/reugene/orcd/pool/redteam_images}"
DEST="$DATASET_ROOT/incoming/$CONTRIBUTOR"
TMPDIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [[ ! -f "$TARBALL" ]]; then
  echo "Tarball does not exist: $TARBALL" >&2
  exit 1
fi

case "$CONTRIBUTOR" in
  *[!A-Za-z0-9_-]*|"")
    echo "Contributor name must contain only letters, numbers, underscore, or dash." >&2
    exit 1
    ;;
esac

mkdir -p "$DEST" "$DATASET_ROOT/manifests"

tar -xzf "$TARBALL" -C "$TMPDIR"

if [[ ! -d "$TMPDIR/$CONTRIBUTOR" ]]; then
  echo "Expected tarball to contain a top-level directory named: $CONTRIBUTOR" >&2
  echo "Found:" >&2
  find "$TMPDIR" -maxdepth 2 -type d >&2
  exit 1
fi

rsync -av --include='*/' --include='*.png' --include='*.PNG' --exclude='*' \
  "$TMPDIR/$CONTRIBUTOR/" "$DEST/"

find "$DEST" -type f \( -iname '*.png' \) | sort \
  > "$DATASET_ROOT/manifests/${CONTRIBUTOR}_files.txt"

count="$(wc -l < "$DATASET_ROOT/manifests/${CONTRIBUTOR}_files.txt")"

echo "Imported $count PNG files for $CONTRIBUTOR into $DEST"
