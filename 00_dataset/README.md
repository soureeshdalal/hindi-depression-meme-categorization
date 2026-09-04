# 00 — Dataset

All raw, translated, and labeled data for the project.

## Image files (Google Drive)

Image folders below contain **`.gitkeep` placeholders** in GitHub. After cloning, download images from Google Drive — see [`docs/DATA_ON_GDRIVE.md`](../docs/DATA_ON_GDRIVE.md).

```
00_dataset/
├── labels/          ← on GitHub (train.json, test.json, val.json)
├── raw/
│   ├── train/       ← images on Drive (8,814 English TR-*)
│   ├── flat/        ← images on Drive (test + validation)
│   └── val_images_flat/
├── translated/
│   ├── translated_train/              ← images on Drive (2,077 Devanagari TR-*)
│   └── translated_categorized_memes/  ← images on Drive (650 Devanagari TE-*, 13 categories)
│       └── {CATEGORY}/test/*.jpg
├── hinglish/
│   └── images/      ← images on Drive (44 Hinglish TE-*)
└── adaptation/
    └── csvs/        ← on GitHub
```

## Policy

- **`train.json`** is used **only** for fine-tuning supervision.
- **`test.json`** is used **only** for evaluation metrics after predictions exist.
- See [`10_docs/GOLD_LABELS_POLICY.md`](../10_docs/GOLD_LABELS_POLICY.md).

## Symlinks at repo root

For backward compatibility: `train.json`, `test.json`, `train/`, `translated_train/`, `translated_categorized_memes/`, `hinglish_memes/` all symlink here.
