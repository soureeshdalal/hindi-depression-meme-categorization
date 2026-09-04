# GitHub Setup Guide

Push **code, docs, labels, metrics, and analysis** to a **private** GitHub repo. **Images** stay on Google Drive — see [`DATA_ON_GDRIVE.md`](DATA_ON_GDRIVE.md).

Estimated GitHub repo size: **~200 MB** (no images, no `archive/`).

---

## Overview

| Location | What |
|----------|------|
| **GitHub** | Scripts, JSON/CSV labels, docs, thesis, metrics, predictions, analysis |
| **Google Drive** | All `.jpg` / image files (~2.9 GB) |
| **Local only** | `archive/` (superseded experiments; optional Drive backup) |

---

## Step 1 — Upload images to Google Drive

Follow [`DATA_ON_GDRIVE.md`](DATA_ON_GDRIVE.md):

1. Create folder `categorization-images` on Drive
2. Upload the image folders listed there
3. Share: **Anyone with the link → Viewer**
4. Paste the link in `README.md` and `docs/DATA_ON_GDRIVE.md`

Do this **before** or **after** the GitHub push — order doesn't matter.

---

## Step 2 — Create a private GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Name: e.g. `hindi-depression-meme-categorization`
3. Visibility: **Private**
4. Do **not** add README, .gitignore, or license
5. Create repository

---

## Step 3 — Initialize git and commit

```bash
cd /Users/soureeshdalal/Desktop/categorization

git init
git add .
git status   # confirm: no .jpg files, no archive/

git commit -m "$(cat <<'EOF'
Initial commit: Hindi depression meme categorization project.

Phase-based layout, chronological docs, thesis, and scripts.
Images excluded — hosted on Google Drive (see docs/DATA_ON_GDRIVE.md).
EOF
)"
```

**What `.gitignore` excludes:**
- All images (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`) → Google Drive
- `archive/` → local only, delete after upload
- Model checkpoints (`.pt`, `.bin`, `.safetensors`)
- macOS junk, Python cache, logs

---

## Step 4 — Push to GitHub

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

Use a [Personal Access Token](https://github.com/settings/tokens) if prompted (scope: `repo`).

---

## Step 5 — Verify

On GitHub, confirm:
- [ ] `README.md` renders with Drive link section
- [ ] `docs/PROJECT_TIMELINE.md` is readable
- [ ] `10_docs/thesis/Soureesh_Dalal_Thesis.pdf` is present
- [ ] `train.json`, `test.json` are present
- [ ] **No** `.jpg` files in the repo
- [ ] Image folders contain only `.gitkeep`

---

## Cloning on another machine

```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME

pip install -r requirements.txt
pip install -r requirements_multimodal.txt

# Download images from Google Drive
pip install gdown
export GDRIVE_FOLDER_ID=your_folder_id
bash scripts/download_images.sh

# Or download manually — see docs/DATA_ON_GDRIVE.md
python3 scripts/verify_images.py
```

---

## No Git LFS needed

Images are on Drive, so you do **not** need Git LFS or paid storage packs for this setup.

---

## Freeing local disk (after upload)

Once GitHub and Drive are verified, delete local copies:

```bash
bash scripts/cleanup_after_upload.sh --archive-only      # ~4.4 GB
bash scripts/cleanup_after_upload.sh --include-images    # + ~2.9 GB images
```

See [`LOCAL_CLEANUP.md`](LOCAL_CLEANUP.md). **`archive/` is not backed up anywhere** — it is deleted locally only.
