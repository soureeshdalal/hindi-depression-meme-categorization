"""
Prepare Colab experiment data package.

Creates:
  - colab_experiments/data/test_images/*.jpg|png (flattened test images)
  - colab_experiments/data/metadata.csv
  - colab_experiments/data/explanations.csv (if source exists)
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def collect_test_images(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for cat in sorted(root.iterdir()):
        if not cat.is_dir():
            continue
        test_dir = cat / "test"
        if not test_dir.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in test_dir.glob(ext):
                out[p.name] = str(p)
    return out


def copy_flat(images: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in images.items():
        shutil.copy2(src, out_dir / name)


def prepare_metadata(src_csv: Path, images: dict[str, str], out_csv: Path) -> int:
    required = ["image_id", "category", "ocr_text", "human_label"]
    out_rows = []
    before = 0
    with src_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required metadata columns: {missing}")
        for row in reader:
            before += 1
            iid = row["image_id"]
            if iid not in images:
                continue
            out_rows.append(
                {
                    "image_id": iid,
                    "category": row["category"],
                    "ocr_text": row["ocr_text"],
                    "human_label": row["human_label"],
                    "image_path": f"test_images/{iid}",
                }
            )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "category", "ocr_text", "human_label", "image_path"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    dropped = before - len(out_rows)
    return dropped


def prepare_explanations(src_path: Path | None, out_csv: Path, image_ids: set[str]) -> bool:
    if src_path is None or not src_path.exists():
        return False

    with src_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    if "image_id" not in fields:
        return False

    exp_field = "explanation_implicit"
    if exp_field not in fields:
        # Handle legacy schemas
        for cand in ("explanation", "explanation_direct", "implicit_explanation"):
            if cand in fields:
                exp_field = cand
                break
    if exp_field not in fields:
        return False

    seen = set()
    out_rows = []
    for r in rows:
        iid = r.get("image_id", "")
        if iid not in image_ids or iid in seen:
            continue
        seen.add(iid)
        out_rows.append(
            {
                "image_id": iid,
                "explanation_implicit": r.get(exp_field, ""),
            }
        )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "explanation_implicit"])
        writer.writeheader()
        writer.writerows(out_rows)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root",
    )
    ap.add_argument(
        "--metadata_src",
        type=Path,
        default=Path("unimodal/BART_Large_predictions.csv"),
        help="Source metadata CSV relative to repo root",
    )
    ap.add_argument(
        "--explanations_src",
        type=Path,
        default=Path("06_explanations/legacy_data/phase1_implicit_explanations.csv"),
        help="Source explanations CSV relative to repo root (optional)",
    )
    ap.add_argument(
        "--skip_copy_images",
        action="store_true",
        help="Only generate metadata/explanations; skip flattening image copies",
    )
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = repo_root / "colab_experiments" / "data"
    test_img_root = data_root / "test_images"
    meta_out = data_root / "metadata.csv"
    expl_out = data_root / "explanations.csv"

    image_root = repo_root / "translated_categorized_memes"
    metadata_src = (repo_root / args.metadata_src).resolve()
    explanations_src = (repo_root / args.explanations_src).resolve()

    images = collect_test_images(image_root)
    if not images:
        raise RuntimeError(f"No test images found under: {image_root}")

    if not args.skip_copy_images:
        copy_flat(images, test_img_root)
    dropped = prepare_metadata(metadata_src, images, meta_out)
    ok = prepare_explanations(explanations_src, expl_out, set(images.keys()))

    print(f"Discovered images: {len(images)}")
    if args.skip_copy_images:
        print("Skipped image copying by request.")
    print(f"Wrote metadata: {meta_out}")
    if dropped:
        print(f"Metadata rows dropped (not found in image set): {dropped}")
    if ok:
        print(f"Wrote explanations: {expl_out}")
    else:
        print("No usable explanations source found. You can add explanations later.")


if __name__ == "__main__":
    main()
