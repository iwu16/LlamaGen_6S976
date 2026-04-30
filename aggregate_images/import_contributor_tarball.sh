#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash aggregate_images/import_contributor_tarball.sh /path/to/contributor_samples.tar.gz contributor_name" >&2
  exit 1
fi

TARBALL="$1"
CONTRIBUTOR="$2"
DATASET_ROOT="${AGGREGATED_SAMPLES_ROOT:-$PWD/aggregated_samples}"
DEST="$DATASET_ROOT/incoming/$CONTRIBUTOR"
TMPDIR="$(mktemp -d)"
CATEGORIES=(clean tokens watermarked)

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

for category in "${CATEGORIES[@]}"; do
  if [[ -d "$TMPDIR/$CONTRIBUTOR/$category" ]]; then
    mkdir -p "$DEST/$category"
    rsync -a "$TMPDIR/$CONTRIBUTOR/$category/" "$DEST/$category/"
  fi
done

find "$DEST" -type f | sort \
  > "$DATASET_ROOT/manifests/${CONTRIBUTOR}_files.txt"

count="$(wc -l < "$DATASET_ROOT/manifests/${CONTRIBUTOR}_files.txt")"

echo "Imported $count sample files for $CONTRIBUTOR into $DEST"
