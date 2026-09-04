#!/usr/bin/env bash
# Prepare a single folder on your Desktop ready to drag into Google Drive.
#
# Usage:
#   bash scripts/prepare_drive_upload.sh
#
# Then drag ~/Desktop/categorization-images-to-upload  into Google Drive.

set -euo pipefail

REPO="/Users/soureeshdalal/Desktop/categorization"
DEST="$HOME/Desktop/categorization-images-to-upload"

echo "Preparing upload folder at:"
echo "  $DEST"
echo ""
echo "This creates a copy (~3 GB). Make sure you have enough disk space."
read -p "Continue? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

rm -rf "$DEST"
mkdir -p "$DEST"

copy_dir() {
  local src="$REPO/$1"
  local dst="$DEST/$1"
  if [[ -d "$src" ]]; then
    echo "  Copying $1 ..."
    mkdir -p "$(dirname "$dst")"
    rsync -a "$src/" "$dst/"
  else
    echo "  SKIP (not found): $1"
  fi
}

echo "Copying required folders..."
copy_dir "00_dataset/raw/train"
copy_dir "00_dataset/raw/flat"
copy_dir "00_dataset/raw/val_images_flat"
copy_dir "00_dataset/translated/translated_train"
copy_dir "00_dataset/translated/translated_categorized_memes"
copy_dir "00_dataset/hinglish/images"

echo "Copying optional folders..."
copy_dir "06_explanations/colab_experiments/data/test_images"
copy_dir "10_docs/meme_dataset_build/images"

echo ""
echo "Done."
du -sh "$DEST"
echo ""
echo "Next steps:"
echo "  1. Open https://drive.google.com"
echo "  2. New → Folder → name it: categorization-images"
echo "  3. Open that folder"
echo "  4. Drag this entire folder into Drive:"
echo "       $DEST"
echo "  5. Wait for upload to finish (may take 30–60 min)"
echo "  6. Right-click folder → Share → Anyone with the link → Viewer"
echo "  7. Copy the link and send it to update README.md"
