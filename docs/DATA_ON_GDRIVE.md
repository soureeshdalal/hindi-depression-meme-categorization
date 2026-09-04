# Dataset Images on Google Drive

All **image files** live on Google Drive. GitHub holds code, labels (JSON/CSV), documentation, metrics, and analysis — everything except images and the local `archive/` folder.

After cloning the repo, download the images and place them in the paths below (or run `scripts/download_images.sh` once your Drive link is configured).

---

## Google Drive link

> **Replace this after you upload:**

| Dataset | Link | Size (approx.) |
|---------|------|----------------|
| **All project images** | _[Add your Google Drive folder link here]_ | ~2.9 GB |

Upload the folder `categorization-images/` (see structure below) and set sharing to **Anyone with the link → Viewer** (required for `gdown`).

---

## What to upload to Google Drive

Mirror this structure inside a Drive folder named **`categorization-images`**:

```
categorization-images/
├── 00_dataset/
│   ├── raw/
│   │   ├── train/                    # 8,814 English training images (TR-*.jpg)
│   │   ├── flat/
│   │   │   ├── test/                 # 662 original test images
│   │   │   └── validation/           # 361 validation images
│   │   └── val_images_flat/          # 353 flat validation images
│   ├── translated/
│   │   ├── translated_train/         # 2,077 Devanagari training images
│   │   └── translated_categorized_memes/  # 650 test images, 13 category folders
│   └── hinglish/
│       └── images/                   # 44 Hinglish Roman-script variants
├── 06_explanations/
│   └── colab_experiments/
│       └── data/
│           └── test_images/          # 650 flat test images (Colab bundle)
└── 10_docs/
    └── meme_dataset_build/
        └── images/                   # Pinterest pin images for dataset doc
```

### Upload from your Mac

```bash
# Option A — drag-and-drop in browser
# 1. Create folder "categorization-images" on Google Drive
# 2. Upload these local folders preserving the structure above:
cd /Users/soureeshdalal/Desktop/categorization

# Core dataset (required)
#   00_dataset/raw/train
#   00_dataset/raw/flat
#   00_dataset/raw/val_images_flat
#   00_dataset/translated/translated_train
#   00_dataset/translated/translated_categorized_memes
#   00_dataset/hinglish/images

# Optional (duplicates or supplementary)
#   06_explanations/colab_experiments/data/test_images
#   10_docs/meme_dataset_build/images
```

```bash
# Option B — zip and upload (faster for many small files)
cd /Users/soureeshdalal/Desktop/categorization
zip -r ~/Desktop/categorization-images.zip \
  00_dataset/raw/train \
  00_dataset/raw/flat \
  00_dataset/raw/val_images_flat \
  00_dataset/translated/translated_train \
  00_dataset/translated/translated_categorized_memes \
  00_dataset/hinglish/images \
  06_explanations/colab_experiments/data/test_images \
  10_docs/meme_dataset_build/images
# Upload categorization-images.zip to Drive, then unzip in Drive or after download
```

---

## After uploading — update the repo

1. Copy the **shared folder link** from Google Drive.
2. Paste it in:
   - This file (`docs/DATA_ON_GDRIVE.md`) — table above
   - Root `README.md` — **Dataset images** section
   - `scripts/download_images.sh` — `GDRIVE_FOLDER_ID` or `GDRIVE_URL`

---

## Download images after cloning GitHub

### Manual

1. Open the Google Drive link.
2. Download the `categorization-images` folder (or zip).
3. Merge into the repo root so paths match, e.g.:
   - Drive: `categorization-images/00_dataset/raw/train/` → repo: `00_dataset/raw/train/`

### Script (after link is set)

```bash
pip install gdown
bash scripts/download_images.sh
```

See `scripts/download_images.sh` for details.

---

## Verification

```bash
python3 scripts/verify_images.py
```

Prints file counts per folder and flags missing directories.

---

## What stays on GitHub (no Drive needed)

| Included on GitHub | Examples |
|--------------------|----------|
| Labels & metadata | `train.json`, `test.json`, `val.json`, all CSVs |
| Code & scripts | All `.py`, `.sh`, notebooks |
| Documentation | `10_docs/`, thesis PDF, timeline |
| Metrics & predictions | `07_metrics/`, prediction CSVs |
| Analysis (non-image) | HTML/JSON reports in `08_analysis/` |
| Pipeline outputs | `09_runs/` (CSVs, logs) |

---

## `archive/` — not backed up

The `archive/` folder (~4.4 GB) is **not** on GitHub or Drive. Delete it locally after upload — see [`LOCAL_CLEANUP.md`](LOCAL_CLEANUP.md).
