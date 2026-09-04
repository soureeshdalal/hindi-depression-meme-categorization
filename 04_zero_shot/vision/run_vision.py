#!/usr/bin/env python3
"""
Vision-only benchmarking pipeline for Hindi mental health memes.

MAMA-Memeia Table 2 "Unimodal Image" baselines: **ViT-B/16**, **ResNet-50**, **EfficientNet**
(Tan & Le 2019 — default **B0** as in common RESTORE-style setups; pass `--efficientnet b7` for B7).
Train on **translated_train/** with **train.json** (TR-* training gold only).
Inference on **/test/** images without reading test gold; **test.json** is for metrics scripts only.

One CSV per model. Never overwrites an existing CSV — skips that model instead.

Usage:
  python vision/run_vision.py \
      --data_dir translated_categorized_memes \
      --train_dir translated_train \
      --train_json train.json \
      --output_dir vision

  # Run a single model:
  python vision/run_vision.py \
      --data_dir translated_categorized_memes \
      --train_dir translated_train \
      --train_json train.json \
      --only_model ViT
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from multimodal.mama_finetune_common import assert_train_json_not_test_gold

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.models as tv_models

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
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

# (arch_key, short_name, display_name)
def default_model_configs(eff_key: str):
    tag = "B0" if eff_key == "efficientnet_b0" else "B7"
    return [
        ("vit_b_16", "ViT", "ViT-B/16"),
        ("resnet50", "ResNet", "ResNet-50"),
        (eff_key, f"EfficientNet_{tag}", f"EfficientNet-{tag}"),
    ]

# Standard ImageNet preprocessing
TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path):
    """Load image as RGB PIL Image. Returns None on failure."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def build_model(arch_key, num_labels):
    """
    Build a torchvision model with its final layer replaced for multi-label output.
    Returns logits (no sigmoid) — use BCEWithLogitsLoss for training.
    """
    if arch_key == "vit_b_16":
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features  # 768
        model.heads.head = nn.Linear(in_features, num_labels)

    elif arch_key == "resnet50":
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Linear(in_features, num_labels)

    elif arch_key == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_labels)

    elif arch_key == "efficientnet_b7":
        model = tv_models.efficientnet_b7(weights=tv_models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_labels)

    else:
        raise ValueError(f"Unknown arch key: {arch_key}")

    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(train_json_path, train_dir):
    """
    Load train.json and match sample_ids to images in train_dir.
    Returns list of (image_path_str, label_vector) tuples.
    Missing images are skipped with a warning.
    """
    assert_train_json_not_test_gold(train_json_path)
    with open(train_json_path) as f:
        data = json.load(f)

    train_path = Path(train_dir)
    samples = []
    missing = 0

    for item in data:
        sample_id = item["sample_id"]  # e.g. "TR-1"
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = train_path / f"{sample_id}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            missing += 1
            continue

        vec = [0.0] * NUM_LABELS
        for cat in item.get("meme_depressive_categories", []):
            if cat in SYMPTOM_NAMES:
                vec[SYMPTOM_NAMES.index(cat)] = 1.0
        samples.append((str(img_path), vec))

    if missing:
        print(f"  Warning: {missing} train.json entries had no matching image in {train_dir}")
    return samples


def collect_test_images(base_dir):
    """Walk translated_categorized_memes/*/test/ — same as existing pipeline."""
    images = []
    base_path = Path(base_dir)
    if not base_path.exists():
        return images
    for cat_folder in sorted(base_path.iterdir()):
        if cat_folder.is_dir():
            test_folder = cat_folder / "test"
            if test_folder.exists():
                for img_file in sorted(test_folder.iterdir()):
                    if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        images.append({
                            "image_path": str(img_file),
                            "category":   cat_folder.name,
                            "filename":   img_file.name,
                        })
    return images


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fine_tune(model, train_samples, device, epochs=10, batch_size=16, lr=1e-4):
    """Fine-tune model in-place using BCEWithLogitsLoss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train_samples)
    indices = list(range(n))

    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0
        bar = tqdm(range(0, n, batch_size), desc=f"  Epoch {epoch + 1}/{epochs}", leave=False)

        for start in bar:
            batch_idx = indices[start: start + batch_size]
            imgs, lbls = [], []

            for i in batch_idx:
                path, label = train_samples[i]
                img = load_image(path)
                if img is None:
                    continue
                imgs.append(TRANSFORM_TRAIN(img))
                lbls.append(label)

            if not imgs:
                continue

            try:
                batch_imgs = torch.stack(imgs).to(device)
                batch_lbls = torch.tensor(lbls, dtype=torch.float32).to(device)

                optimizer.zero_grad()
                logits = model(batch_imgs)
                loss = loss_fn(logits, batch_lbls)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                bar.set_postfix(loss=f"{loss.item():.4f}")

            except Exception as e:
                print(f"\n  [Train batch error] {e}", file=sys.stderr)
                continue

        avg = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {avg:.4f}")

    model.eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, test_images, device, batch_size=32, threshold=0.5):
    """Run batched inference. Returns (pred_strings, raw_strings)."""
    model.eval()
    all_preds, all_raw = [], []

    for start in tqdm(range(0, len(test_images), batch_size), desc="  Inference"):
        batch = test_images[start: start + batch_size]
        imgs, valid_mask = [], []

        for img_info in batch:
            img = load_image(img_info["image_path"])
            if img is not None:
                imgs.append(TRANSFORM_TEST(img))
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not any(valid_mask):
            for _ in batch:
                all_preds.append("[INFERENCE_ERROR]")
                all_raw.append("[INFERENCE_ERROR]")
            continue

        try:
            batch_imgs = torch.stack(imgs).to(device)
            with torch.no_grad():
                logits = model(batch_imgs)
                probs = torch.sigmoid(logits).cpu().numpy()

            prob_idx = 0
            for v in valid_mask:
                if v:
                    prob = probs[prob_idx]
                    prob_idx += 1
                    binary = (prob >= threshold).astype(int)
                    all_preds.append("[" + ",".join(map(str, binary.tolist())) + "]")
                    all_raw.append("[" + ",".join(f"{x:.2f}" for x in prob.tolist()) + "]")
                else:
                    all_preds.append("[INFERENCE_ERROR]")
                    all_raw.append("[INFERENCE_ERROR]")

        except Exception as e:
            print(f"\n  [Inference error] batch {start}: {e}", file=sys.stderr)
            for _ in batch:
                all_preds.append("[INFERENCE_ERROR]")
                all_raw.append("[INFERENCE_ERROR]")

    return all_preds, all_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune vision-only models for meme depression symptom classification"
    )
    parser.add_argument("--data_dir",   required=True, help="Path to translated_categorized_memes")
    parser.add_argument(
        "--train_dir",
        default="translated_train",
        help="Folder of translated train images (TR-*.jpg), aligned with train.json sample_id",
    )
    parser.add_argument("--train_json", required=True, help="Path to train.json")
    parser.add_argument("--output_dir", default="vision", help="Directory to save per-model CSVs")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--only_model", type=str, default=None,
                        help="Short name of a single model to run (ViT, ResNet, or EfficientNet)")
    parser.add_argument(
        "--efficientnet",
        choices=("b0", "b7"),
        default="b0",
        help="EfficientNet variant (MAMA cites EfficientNet; B0 is the standard light baseline)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"=== VISION-ONLY FINE-TUNED PIPELINE ===")
    print(f"Device: {device}")

    # Load training data
    print(f"\nLoading training data...")
    train_samples = load_training_data(args.train_json, args.train_dir)
    print(f"  {len(train_samples)} training samples with images found")

    if not train_samples:
        print("No training samples found. Check --train_dir and --train_json. Exiting.")
        return

    # Collect test images
    print(f"\nCollecting test images from {args.data_dir}...")
    test_images = collect_test_images(args.data_dir)
    if args.max_images:
        test_images = test_images[: args.max_images]
    print(f"  {len(test_images)} test images found")

    if not test_images:
        print("No test images found. Exiting.")
        return

    eff_key = f"efficientnet_{args.efficientnet}"
    model_configs = default_model_configs(eff_key)

    # Filter models
    configs = model_configs
    if args.only_model:
        configs = [
            c for c in model_configs
            if c[1] == args.only_model or c[0] == args.only_model
        ]
        if not configs:
            names = [c[1] for c in model_configs]
            print(f"Model '{args.only_model}' not found. Available: {names}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    # Run each model
    for arch_key, short_name, display_name in configs:
        out_csv = Path(args.output_dir) / f"{short_name}_predictions.csv"

        if out_csv.exists():
            print(f"\n[SKIP] {display_name}: {out_csv} already exists")
            continue

        print(f"\n{'=' * 60}")
        print(f"[MODEL] {display_name}")
        print(f"{'=' * 60}")

        try:
            print(f"  Building {display_name} (pretrained ImageNet)...")
            model = build_model(arch_key, NUM_LABELS).to(device)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Parameters: {n_params:.1f}M")

            print(f"  Fine-tuning for {args.epochs} epochs "
                  f"on {len(train_samples)} images (batch={TRAIN_BATCH_SIZE}, lr={LEARNING_RATE})...")
            fine_tune(
                model, train_samples, device,
                epochs=args.epochs,
                batch_size=TRAIN_BATCH_SIZE,
                lr=LEARNING_RATE,
            )

            print("  Running inference on test set...")
            preds, raw = run_inference(
                model, test_images, device,
                batch_size=INFER_BATCH_SIZE, threshold=0.5,
            )

            df = pd.DataFrame({
                "image_id":            [x["filename"] for x in test_images],
                "category":            [x["category"] for x in test_images],
                f"{short_name}_pred":  preds,
                f"{short_name}_raw":   raw,
                "human_label":         [""] * len(test_images),
            })
            df.to_csv(out_csv, index=False)
            print(f"  Saved {len(df)} rows → {out_csv}")

        except Exception as e:
            print(f"\n  [ERROR] {display_name} failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"  Skipping {display_name}, continuing with next model.\n")

        finally:
            try:
                del model
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
