"""
Shared helpers for MAMA-Memeia-style fine-tuned multimodal baselines (CLIP, ViT+BERT, VisualBERT).

**Gold labels (no leakage):**
- ``train.json`` — human labels for the **training split** only (paired with ``translated_train/``, TR-* ids).
  Used **only** to compute training loss. Never used to score the test set during training.
- ``test.json`` — human labels for the **test split** (TE-* ids). Used **only** in ``03_scripts/compute_metrics*.py``
  after predictions exist. **No fine-tuning script in this repo reads test.json.**

Evaluates on ``translated_categorized_memes/*/test/`` images **without** loading their gold labels during FT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

NUM_LABELS = 7
SEED = 42

SYMPTOM_NAMES = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]


def assert_train_json_not_test_gold(train_json_path: str | Path) -> None:
    """
    Refuse accidental fine-tuning on held-out test annotations.

    In this dataset, train split uses TR-* ``sample_id`` values and test uses TE-*.
    ``test.json`` must only be consumed by metric scripts, never as ``--train_json``.
    """
    path = Path(train_json_path)
    if path.name.lower() == "test.json":
        raise ValueError(
            "Refusing to fine-tune using test.json: it holds held-out human gold labels for "
            "**metrics only**. Use train.json (training split) for supervision; pass test.json "
            "only to compute_metrics_finetuned.py / compute_metrics.py after inference."
        )
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    ids = [(item.get("sample_id") or "").strip().upper() for item in data if item.get("sample_id")]
    if not ids:
        return
    n_te = sum(1 for s in ids if s.startswith("TE-"))
    n_tr = sum(1 for s in ids if s.startswith("TR-"))
    if n_te > 0 and n_te >= n_tr:
        raise ValueError(
            f"{path}: file looks like **test** annotations (TE-* ids dominate: TE={n_te}, TR={n_tr}). "
            "Do not use test gold labels for fine-tuning."
        )


def load_training_paths_and_labels(train_json_path: str | Path, train_image_dir: str | Path):
    """Return list of (image_path_str, label_list[7]) for existing files."""
    assert_train_json_not_test_gold(train_json_path)
    with open(train_json_path) as f:
        data = json.load(f)
    base = Path(train_image_dir)
    samples = []
    missing = 0
    for item in data:
        sid = (item.get("sample_id") or "").strip()
        if not sid:
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            p = base / f"{sid}{ext}"
            if p.exists():
                vec = [0.0] * NUM_LABELS
                for cat in item.get("meme_depressive_categories", []):
                    if cat in SYMPTOM_NAMES:
                        vec[SYMPTOM_NAMES.index(cat)] = 1.0
                samples.append((str(p), vec))
                break
        else:
            missing += 1
    if missing:
        print(f"  Warning: {missing} train rows missing under {train_image_dir}", file=sys.stderr)
    return samples


def collect_test_images(data_dir: str | Path) -> list[dict[str, Any]]:
    images = []
    base_path = Path(data_dir)
    if not base_path.exists():
        return images
    for cat_folder in sorted(base_path.iterdir()):
        if cat_folder.is_dir():
            test_folder = cat_folder / "test"
            if test_folder.exists():
                for img_file in sorted(test_folder.iterdir()):
                    if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        images.append(
                            {
                                "image_path": str(img_file),
                                "category": cat_folder.name,
                                "filename": img_file.name,
                            }
                        )
    return images


def ocr_paths(reader, paths: list[str], desc: str = "OCR") -> list[str]:
    out = []
    for p in tqdm(paths, desc=desc):
        try:
            results = reader.readtext(p, detail=0)
            text = " ".join(results).strip()
            out.append(text if text else "[NO_TEXT]")
        except Exception as e:
            print(f"  [OCR] {p}: {e}", file=sys.stderr)
            out.append("[OCR_ERROR]")
    return out


def load_pil(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def labels_to_tensor(labels_batch: list) -> torch.Tensor:
    return torch.tensor(np.array(labels_batch, dtype=np.float32))
