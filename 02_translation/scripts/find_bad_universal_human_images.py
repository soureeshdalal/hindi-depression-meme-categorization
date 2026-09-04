#!/usr/bin/env python3
"""
Find UNIVERSAL_HUMAN translated images that Pillow cannot read.
Use the output list with translate_universal_human.py --only-ids to retranslate only those.
Run from project root: python 02_translation_scripts/find_bad_universal_human_images.py
"""

import argparse
from pathlib import Path

from PIL import Image

TRANSLATED_DIR = "translated_categorized_memes/UNIVERSAL_HUMAN"


def pillow_can_read(image_path):
    """Return True if Pillow can open and load the image."""
    try:
        with Image.open(image_path) as im:
            im.load()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Find UNIVERSAL_HUMAN images Pillow cannot read")
    parser.add_argument(
        "--translated_dir",
        type=str,
        default=TRANSLATED_DIR,
        help="Path to translated UNIVERSAL_HUMAN folder (test/ and validation/ inside)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bad_universal_human_ids.txt",
        help="Output file: one meme ID per line (e.g. TE-123)",
    )
    args = parser.parse_args()

    base = Path(args.translated_dir)
    if not base.exists():
        print(f"❌ Directory not found: {base}")
        return

    bad_ids = []
    for subdir in ("test", "validation"):
        folder = base / subdir
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            meme_id = path.stem
            if not pillow_can_read(path):
                bad_ids.append(meme_id)
                print(f"  bad: {path.relative_to(base)}")

    out_path = Path(args.output)
    out_path.write_text("\n".join(bad_ids) + ("\n" if bad_ids else ""), encoding="utf-8")
    print(f"\n✅ Found {len(bad_ids)} images Pillow cannot read.")
    print(f"   Wrote IDs to: {out_path}")
    if bad_ids:
        print(f"   Retranslate with: python 02_translation_scripts/translate_universal_human.py --only-ids {out_path}")


if __name__ == "__main__":
    main()
