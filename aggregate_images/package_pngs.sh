#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash package_pngs.sh /path/to/pngs contributor_name" >&2
  echo "Example: bash package_pngs.sh /home/maureenz/orcd/pool/my_samples maureenz" >&2
  exit 1
fi

SOURCE_DIR="$1"
CONTRIBUTOR="$2"
OUT="${CONTRIBUTOR}_pngs.tar.gz"
TMPDIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

case "$CONTRIBUTOR" in
  *[!A-Za-z0-9_-]*|"")
    echo "Contributor name must contain only letters, numbers, underscore, or dash." >&2
    exit 1
    ;;
esac

mkdir -p "$TMPDIR/$CONTRIBUTOR"

find "$SOURCE_DIR" -type f \( -iname '*.png' \) -print0 |
  while IFS= read -r -d '' file; do
    base="$(basename "$file")"
    cp -n "$file" "$TMPDIR/$CONTRIBUTOR/$base"
  done

count="$(find "$TMPDIR/$CONTRIBUTOR" -type f -iname '*.png' | wc -l)"

if [[ "$count" -eq 0 ]]; then
  echo "No PNG files found under: $SOURCE_DIR" >&2
  exit 1
fi

tar -C "$TMPDIR" -czf "$OUT" "$CONTRIBUTOR"

echo "Created $OUT with $count PNG files."
echo "Send it to reugene, for example:"
echo "scp $OUT reugene@orcd-login.mit.edu:/home/reugene/orcd/pool/redteam_images/tarballs/"

