#!/usr/bin/env python3
"""Verify image folders exist and report file counts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FOLDERS = [
    ("00_dataset/raw/train", "English training images"),
    ("00_dataset/raw/flat/test", "Original test images"),
    ("00_dataset/raw/flat/validation", "Original validation images"),
    ("00_dataset/raw/val_images_flat", "Flat validation images"),
    ("00_dataset/translated/translated_train", "Devanagari training images"),
    ("00_dataset/translated/translated_categorized_memes", "Devanagari test (by category)"),
    ("00_dataset/hinglish/images", "Hinglish subset"),
    ("06_explanations/colab_experiments/data/test_images", "Colab flat test images (optional)"),
    ("10_docs/meme_dataset_build/images", "Dataset doc pins (optional)"),
]

IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def count_images(path: Path) -> int:
    if not path.is_dir():
        return -1
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMG_EXT)


def main() -> None:
    print("Image folder verification\n")
    missing = []
    for rel, label in FOLDERS:
        path = ROOT / rel
        n = count_images(path)
        if n < 0:
            status = "MISSING"
            missing.append(rel)
        elif n == 0:
            status = "EMPTY (download from Google Drive)"
            missing.append(rel)
        else:
            status = f"OK ({n} images)"
        print(f"  [{status:40}] {rel}")
        print(f"    {label}")

    print()
    if missing:
        print(f"Action needed: download images — see docs/DATA_ON_GDRIVE.md")
        raise SystemExit(1)
    print("All image folders present.")


if __name__ == "__main__":
    main()
