#!/usr/bin/env bash
# Download dataset images from Google Drive into the repo.
#
# Setup:
#   1. Upload categorization-images/ to Google Drive (see docs/DATA_ON_GDRIVE.md)
#   2. Share folder: Anyone with the link → Viewer
#   3. Set GDRIVE_FOLDER_ID below (from the folder URL)
#
# Usage:
#   pip install gdown
#   bash scripts/download_images.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Configure after uploading to Drive ─────────────────────────────────────
# Folder URL example: https://drive.google.com/drive/folders/1ABC...xyz
# Folder ID is the part after /folders/
GDRIVE_FOLDER_ID="${GDRIVE_FOLDER_ID:-1XQgvJPdJCgnWx0Jt7F6IhAik_C5uorhb}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown..."
  pip install gdown
fi

STAGING="$REPO_ROOT/.gdrive_download"
rm -rf "$STAGING"
mkdir -p "$STAGING"

echo "Downloading categorization-images from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "$STAGING" --remaining-ok

# Merge downloaded tree into repo (expects categorization-images/ at top of staging)
SRC="$STAGING/categorization-images"
if [[ ! -d "$SRC" ]]; then
  # gdown may flatten — use staging root if structure differs
  SRC="$STAGING"
fi

echo "Merging into repo root..."
for sub in 00_dataset 06_explanations 10_docs; do
  if [[ -d "$SRC/$sub" ]]; then
    rsync -a "$SRC/$sub/" "$REPO_ROOT/$sub/"
  fi
done

rm -rf "$STAGING"
echo "Done. Verifying..."
python3 "$REPO_ROOT/scripts/verify_images.py"
