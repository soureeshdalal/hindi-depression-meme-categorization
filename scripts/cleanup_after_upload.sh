#!/usr/bin/env bash
# Delete local data after GitHub + Drive uploads are verified.
#
# Usage:
#   bash scripts/cleanup_after_upload.sh --dry-run          # show what would be deleted
#   bash scripts/cleanup_after_upload.sh --archive-only   # delete archive/ (~4.4 GB)
#   bash scripts/cleanup_after_upload.sh --include-images # also delete local images (~2.9 GB)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
DELETE_ARCHIVE=false
DELETE_IMAGES=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --archive-only) DELETE_ARCHIVE=true ;;
    --include-images) DELETE_ARCHIVE=true; DELETE_IMAGES=true ;;
    *)
      echo "Usage: $0 [--dry-run] [--archive-only | --include-images]"
      exit 1
      ;;
  esac
done

if ! $DELETE_ARCHIVE && ! $DELETE_IMAGES; then
  echo "Pick a mode: --archive-only or --include-images (add --dry-run to preview)"
  exit 1
fi

delete_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "  skip (not found): $path"
    return
  fi
  local size
  size=$(du -sh "$path" 2>/dev/null | cut -f1)
  if $DRY_RUN; then
    echo "  would delete ($size): $path"
  else
    echo "  deleting ($size): $path"
    rm -rf "$path"
  fi
}

echo "=== Cleanup plan ==="
echo "Repo: $REPO_ROOT"
echo ""

if $DELETE_ARCHIVE; then
  echo "Archive:"
  delete_path "$REPO_ROOT/archive"
  echo ""
fi

if $DELETE_IMAGES; then
  echo "Image folders (must be on Google Drive first):"
  for path in \
    "$REPO_ROOT/00_dataset/raw/train" \
    "$REPO_ROOT/00_dataset/raw/flat/test" \
    "$REPO_ROOT/00_dataset/raw/flat/validation" \
    "$REPO_ROOT/00_dataset/raw/val_images_flat" \
    "$REPO_ROOT/00_dataset/translated/translated_train" \
    "$REPO_ROOT/00_dataset/translated/translated_categorized_memes" \
    "$REPO_ROOT/00_dataset/hinglish/images" \
    "$REPO_ROOT/06_explanations/colab_experiments/data/test_images" \
    "$REPO_ROOT/10_docs/meme_dataset_build/images"
  do
    delete_path "$path"
    # Restore .gitkeep placeholder if not dry-run
    if ! $DRY_RUN && [[ "$path" == *"/00_dataset/"* || "$path" == *"/06_explanations/"* || "$path" == *"/10_docs/"* ]]; then
      mkdir -p "$path" && touch "$path/.gitkeep"
    fi
  done
  echo ""
fi

if $DRY_RUN; then
  echo "Dry run complete — no files deleted."
else
  echo "Done. Freed space locally."
  echo "GitHub + Google Drive are now your only copies."
fi
