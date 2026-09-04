# Upload Images to Google Drive

Simple step-by-step guide. Total upload: **~3 GB**, mostly one folder (`00_dataset`).

---

## What you are uploading

| Folder on your Mac | What's inside | Size |
|--------------------|---------------|------|
| **`00_dataset/`** | All main images (required) | ~2.7 GB |
| ↳ `raw/train/` | 8,814 English training images | 1.0 GB |
| ↳ `raw/flat/test/` | 662 original test images | 65 MB |
| ↳ `raw/flat/validation/` | 361 validation images | 33 MB |
| ↳ `raw/val_images_flat/` | 353 flat validation images | 170 MB |
| ↳ `translated/translated_train/` | 2,077 Hindi training images | 922 MB |
| ↳ `translated/translated_categorized_memes/` | 650 Hindi test images (13 categories) | 490 MB |
| ↳ `hinglish/images/` | 44 Hinglish test images | 23 MB |
| **`06_explanations/.../test_images/`** | Colab copy of test images (optional) | 325 MB |
| **`10_docs/meme_dataset_build/images/`** | Dataset doc pins (optional) | 13 MB |

**Minimum upload:** just `00_dataset/` — that covers everything important.

**Do NOT upload:** `archive/` (not backed up anywhere, delete later).

---

## Method A — Drag one folder (easiest)

### Step 1 — Open Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Click **+ New** → **Folder**
3. Name it: **`categorization-images`**
4. Open that folder (double-click it)

### Step 2 — Open Finder

1. Open Finder
2. Go to: **`Desktop` → `categorization`**
3. You should see the folder **`00_dataset`**

### Step 3 — Upload

1. **Drag the entire `00_dataset` folder** from Finder into the open Google Drive browser window
2. Wait for the upload — **30–60 minutes** depending on your internet
3. Do not close the browser tab until it says upload complete

Your Drive should look like:

```
categorization-images/
└── 00_dataset/
    ├── raw/
    ├── translated/
    └── hinglish/
```

### Step 4 — (Optional) Upload two extra folders

Only if you want the Colab duplicate and dataset doc pins:

1. In Drive, inside `categorization-images`, create folders:
   - `06_explanations/colab_experiments/data/`
   - `10_docs/meme_dataset_build/`
2. From Finder, drag:
   - `categorization/06_explanations/colab_experiments/data/test_images` → into `06_explanations/colab_experiments/data/` on Drive
   - `categorization/10_docs/meme_dataset_build/images` → into `10_docs/meme_dataset_build/` on Drive

Skip this if you want to keep it simple — `00_dataset` alone is enough.

### Step 5 — Share the folder

1. In Google Drive, go back to **`categorization-images`** (the top folder)
2. Right-click → **Share**
3. Under **General access**, click **Restricted** → change to **Anyone with the link**
4. Role: **Viewer**
5. Click **Copy link**
6. Save the link somewhere — you'll paste it into the GitHub README

---

## Method B — Prepare a copy on Desktop first

Use this if drag-and-drop from the project folder feels messy.

```bash
cd /Users/soureeshdalal/Desktop/categorization
bash scripts/prepare_drive_upload.sh
```

This creates **`Desktop/categorization-images-to-upload/`** with the correct folder structure.

Then:
1. Open [drive.google.com](https://drive.google.com)
2. Create folder **`categorization-images`**
3. Drag **everything inside** `categorization-images-to-upload/` into it
4. Share → Anyone with the link → Viewer → Copy link

Delete `categorization-images-to-upload/` from Desktop after upload finishes (saves ~3 GB).

---

## After upload — paste the link on GitHub

Send me the Drive link, or update these two files yourself:

1. **`README.md`** — replace the placeholder in the "Dataset images" table
2. **`docs/DATA_ON_GDRIVE.md`** — replace the placeholder in the table below

Then push:

```bash
cd /Users/soureeshdalal/Desktop/categorization
git add README.md docs/DATA_ON_GDRIVE.md
git commit -m "Add Google Drive link for dataset images"
git push
```

---

## Google Drive link

| Dataset | Link |
|---------|------|
| **All project images** | [Google Drive](https://drive.google.com/drive/folders/1XQgvJPdJCgnWx0Jt7F6IhAik_C5uorhb?usp=sharing) (~2.7 GB) |

---

## Verify upload (optional)

Open the Drive link in an incognito window and spot-check:
- [ ] `00_dataset/translated/translated_categorized_memes/UNIVERSAL_HUMAN/test/` has `.jpg` files
- [ ] `00_dataset/translated/translated_train/` has files
- [ ] `00_dataset/raw/train/` has files

---

## Download later (on another machine)

After cloning from GitHub:

```bash
git clone https://github.com/soureeshdalal/hindi-depression-meme-categorization.git
cd hindi-depression-meme-categorization

# Manual: download categorization-images from Drive, merge 00_dataset/ into repo root
python3 scripts/verify_images.py
```

---

## What stays on GitHub (no Drive needed)

Labels (`train.json`, `test.json`), all code, thesis, CSVs, metrics, docs — already pushed.

See [`LOCAL_CLEANUP.md`](LOCAL_CLEANUP.md) for deleting local files after upload is verified.
