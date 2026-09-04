# Local Cleanup (after GitHub + Drive upload)

Once both remotes are verified, delete local copies to free disk space. **Do not run this until uploads are confirmed.**

---

## Checklist before deleting anything

### GitHub verified
- [ ] Repo is private and pushed
- [ ] `README.md`, `docs/PROJECT_TIMELINE.md`, thesis PDF visible on GitHub
- [ ] `train.json`, `test.json`, prediction CSVs, scripts present
- [ ] No `.jpg` files on GitHub (images excluded correctly)

### Google Drive verified
- [ ] `categorization-images/` folder uploaded with correct structure
- [ ] Share link works (Anyone with the link → Viewer)
- [ ] Link pasted in `README.md` and `docs/DATA_ON_GDRIVE.md`
- [ ] Spot-check: open a few images in Drive

---

## What to delete

| Target | Frees | Safe? |
|--------|-------|-------|
| **`archive/`** | ~4.4 GB | Yes — not backed up anywhere; superseded duplicates |
| **Image folders** (optional) | ~2.9 GB | Yes — if Drive upload verified; re-download via `scripts/download_images.sh` |
| **Entire project folder** (optional) | ~8 GB | Yes — if GitHub + Drive both verified; re-clone + download images later |

---

## Quick cleanup (archive only)

```bash
cd /Users/soureeshdalal/Desktop/categorization
bash scripts/cleanup_after_upload.sh --archive-only
```

## Full cleanup (archive + local images)

Only after Drive is verified:

```bash
bash scripts/cleanup_after_upload.sh --include-images
```

## Maximum cleanup (remove entire project)

After GitHub + Drive verified, if you want the folder gone entirely:

```bash
cd /Users/soureeshdalal/Desktop
rm -rf categorization
```

To restore later:

```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git categorization
cd categorization
pip install -r requirements.txt gdown
export GDRIVE_FOLDER_ID=your_folder_id
bash scripts/download_images.sh
```

---

## What you keep remotely

| Remote | Contents |
|--------|----------|
| **GitHub** | Code, docs, thesis, labels, CSVs, metrics, analysis (~200 MB) |
| **Google Drive** | All images (~2.9 GB) |
| **Nowhere** | `archive/` — delete locally, no backup |
