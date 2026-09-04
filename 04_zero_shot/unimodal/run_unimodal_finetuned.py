#!/usr/bin/env python3
"""
Fine-tuned unimodal text benchmarking pipeline for Hindi mental health memes.

Uses **train.json** only for **training-split** supervision (TR-* ids + ``translated_train/``).
**test.json** is never read here; held-out metrics use ``03_scripts/compute_metrics_finetuned.py``.

Loads labels from train.json, runs Easy OCR on **translated_train/** images
(same Devanagari domain as test), fine-tunes a classification head (+ backbone)
for each model, then runs inference on translated **test** memes (MAMA-Memeia-style
supervised setup without English-train / Hindi-test mismatch).

One CSV per model. Never overwrites an existing CSV — skips that model instead.

Usage:
  python unimodal/run_unimodal_finetuned.py \
      --data_dir translated_categorized_memes \
      --train_json train.json \
      --train_image_dir translated_train \
      --output_dir unimodal/finetuned

  # Run a single model:
  python unimodal/run_unimodal_finetuned.py \
      --data_dir translated_categorized_memes \
      --train_json train.json \
      --train_image_dir translated_train \
      --output_dir unimodal/finetuned \
      --only_model MentalBERT_ft
"""

import argparse
import json
import os
import random
import ssl
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from multimodal.mama_finetune_common import assert_train_json_not_test_gold

# Gated HF models: `export HF_TOKEN=...` (no token in repo)
HF_TOKEN = os.environ.get("HF_TOKEN")

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Hyperparameters (MAMA-Memeia / RESTOREx-style supervised text baselines)
# See: 01_documentation/MAMA_MEMEIA_FINETUNING_ALIGNMENT.md
# BERT-family: 3 epochs, AdamW 2e-5, BCE-with-logits, multi-label ×7 — Table 2 "OCR" rows.
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_LABELS = 7
MAX_TOKEN_LEN = 128
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

# MAMA-Memeia Table 2 "Unimodal Text" (OCR) checkpoints + Hindi extensions.
# Preset `mama`: exact four families from the paper (English BERT checkpoint; weak on Devanagari OCR).
# Preset `extended`: multilingual BERT + your original run_unimodal.py lineup (no duplicate MAMA rows).
# Preset `all`: mama + extended (recommended for full tables + Hindi coverage).

PRESET_MAMA = [
    ("google-bert/bert-base-uncased",      "BERT_ft",            "BERT-base-uncased (FT, MAMA Table 2)"),
    ("AIMH/mental-bert-large-cased",       "MentalBERT_ft",      "MentalBERT-Large (FT)"),
    ("facebook/bart-base",                 "BART_ft",            "BART-Base (FT)"),
    ("Tianlin668/MentalBART",              "MentalBART_ft",      "MentalBART (FT)"),
]

PRESET_EXTENDED = [
    ("google-bert/bert-base-multilingual-cased", "BERT_mBERT_ft", "BERT-multilingual (FT, Devanagari OCR)"),
    ("ai4bharat/IndicBERTv2-SS",          "IndicBERTv2_SS_ft",  "IndicBERTv2-SS (FT)"),
    ("ai4bharat/IndicBERTv2-MLM-Sam-TLM", "IndicBERTv2_Sam_ft", "IndicBERTv2-MLM-Sam-TLM (FT)"),
    ("google/muril-base-cased",            "MuRIL_ft",           "MuRIL-base (FT)"),
    ("mental/mental-roberta-base",         "MentalRoBERTa_ft",   "MentalRoBERTa-Base (FT)"),
    ("facebook/bart-large",                "BART_Large_ft",      "BART-Large (FT)"),
]

PRESET_ALL = PRESET_MAMA + PRESET_EXTENDED

MODEL_PRESETS = {
    "mama": PRESET_MAMA,
    "extended": PRESET_EXTENDED,
    "all": PRESET_ALL,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_ids_and_labels(train_json_path):
    """
    Load train.json → parallel (sample_ids, labels_matrix [n, 7]).
    sample_id e.g. 'TR-1' → image translated_train/TR-1.jpg
    """
    assert_train_json_not_test_gold(train_json_path)
    with open(train_json_path) as f:
        data = json.load(f)
    ids, labels = [], []
    for item in data:
        sid = (item.get("sample_id") or "").strip()
        if not sid:
            continue
        vec = [0.0] * NUM_LABELS
        for cat in item.get("meme_depressive_categories", []):
            if cat in SYMPTOM_NAMES:
                vec[SYMPTOM_NAMES.index(cat)] = 1.0
        ids.append(sid)
        labels.append(vec)
    return ids, np.array(labels, dtype=np.float32)


def ocr_translated_train_images(sample_ids, labels, image_dir, reader):
    """
    OCR each translated train image; skip missing files. Returns aligned
    (texts, labels) for fine-tuning.
    """
    base = Path(image_dir)
    texts, kept_labels = [], []
    missing = 0
    for sid, lab in zip(sample_ids, labels):
        img_path = base / f"{sid}.jpg"
        if not img_path.exists():
            img_path = base / f"{sid}.jpeg"
        if not img_path.exists():
            img_path = base / f"{sid}.png"
        if not img_path.exists():
            missing += 1
            continue
        try:
            results = reader.readtext(str(img_path), detail=0)
            text = " ".join(results).strip()
            texts.append(text if text else "[NO_TEXT]")
            kept_labels.append(lab)
        except Exception as e:
            print(f"  [OCR train error] {img_path}: {e}", file=sys.stderr)
            texts.append("[OCR_ERROR]")
            kept_labels.append(lab)

    if missing:
        print(f"  Warning: {missing} train.json rows had no image under {image_dir}")
    return texts, np.array(kept_labels, dtype=np.float32)


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
                            "category": cat_folder.name,
                            "filename": img_file.name,
                        })
    return images


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextClassifier(nn.Module):
    """Pretrained encoder + linear multi-label classification head."""

    def __init__(self, model_name, num_labels=7, token=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token,
        )
        # BART-style uses d_model; BERT-style uses hidden_size
        hidden = (
            self.encoder.config.d_model
            if hasattr(self.encoder.config, "d_model")
            else self.encoder.config.hidden_size
        )
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        """Returns raw logits (no sigmoid)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        if cls.dtype != self.classifier.weight.dtype:
            cls = cls.to(self.classifier.weight.dtype)
        return self.classifier(cls)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fine_tune(model, tokenizer, texts, labels, device,
              epochs=3, batch_size=16, lr=2e-5):
    """Fine-tune model in place using BCEWithLogitsLoss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(texts)
    indices = list(range(n))

    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0
        bar = tqdm(range(0, n, batch_size), desc=f"  Epoch {epoch + 1}/{epochs}", leave=False)

        for start in bar:
            batch_idx = indices[start: start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            batch_labels = torch.tensor(
                labels[batch_idx], dtype=torch.float32
            ).to(device)

            try:
                enc = tokenizer(
                    batch_texts,
                    max_length=MAX_TOKEN_LEN,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, batch_labels)
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

def run_inference(model, tokenizer, texts, device,
                  batch_size=64, threshold=0.5):
    """Run batched inference. Returns (pred_strings, raw_strings)."""
    model.eval()
    all_preds, all_raw = [], []

    for start in tqdm(range(0, len(texts), batch_size), desc="  Inference"):
        batch = texts[start: start + batch_size]
        try:
            enc = tokenizer(
                batch,
                max_length=MAX_TOKEN_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()

            for prob in probs:
                binary = (prob >= threshold).astype(int)
                all_preds.append("[" + ",".join(map(str, binary.tolist())) + "]")
                all_raw.append("[" + ",".join(f"{x:.2f}" for x in prob.tolist()) + "]")

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
        description="Fine-tune text models for meme depression symptom classification"
    )
    parser.add_argument("--data_dir",    required=True, help="Path to translated_categorized_memes")
    parser.add_argument("--train_json",  required=True, help="Path to train.json (labels + sample_id)")
    parser.add_argument(
        "--train_image_dir",
        default="translated_train",
        help="Translated train images (TR-*.jpg), OCR'd for text fine-tuning",
    )
    parser.add_argument(
        "--output_dir",
        default="unimodal/finetuned",
        help="Directory to save per-model *_ft_predictions.csv",
    )
    parser.add_argument("--epochs",      type=int, default=EPOCHS)
    parser.add_argument("--max_images",  type=int, default=None)
    parser.add_argument("--only_model",  type=str, default=None,
                        help="Short name or display name of a single model to run")
    parser.add_argument(
        "--preset",
        choices=("mama", "extended", "all"),
        default="all",
        help="mama=Table2 text (BERT-uncased,MentalBERT,BART,MentalBART); extended=Hindi extras; all=both",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"=== FINE-TUNED UNIMODAL TEXT PIPELINE ===")
    print(f"Device: {device}")

    # Load training ids + labels (text comes from translated images, not json ocr_text)
    print(f"\nLoading training labels from {args.train_json}...")
    train_ids, train_label_mat = load_training_ids_and_labels(args.train_json)
    print(f"  {len(train_ids)} labeled train rows")

    # Collect test images
    print(f"\nCollecting test images from {args.data_dir}...")
    images = collect_test_images(args.data_dir)
    if args.max_images:
        images = images[: args.max_images]
    print(f"  {len(images)} test images found")

    if not images:
        print("No test images found. Exiting.")
        return

    # OCR reader once for train + test (translated domain)
    print("\n[OCR] Initializing reader (hi, en)...")
    reader = easyocr.Reader(["hi", "en"], gpu=(str(device) != "cpu"))

    print(f"\n[OCR] Translated train images under {args.train_image_dir}...")
    train_texts, train_labels = ocr_translated_train_images(
        train_ids, train_label_mat, args.train_image_dir, reader
    )
    print(f"  {len(train_texts)} train samples with OCR (for fine-tuning)")
    if not train_texts:
        print("No training samples after OCR / image match. Exiting.")
        return

    print("\n[OCR] Test images...")
    ocr_texts = []
    for img in tqdm(images, desc="OCR test"):
        try:
            results = reader.readtext(img["image_path"], detail=0)
            text = " ".join(results).strip()
            ocr_texts.append(text if text else "[NO_TEXT]")
        except Exception as e:
            print(f"  [OCR error] {img['image_path']}: {e}", file=sys.stderr)
            ocr_texts.append("[OCR_ERROR]")
    extractable = sum(1 for t in ocr_texts if t not in ("[NO_TEXT]", "[OCR_ERROR]"))
    print(f"OCR complete. {extractable}/{len(images)} test images had extractable text.\n")

    # Filter models
    configs = MODEL_PRESETS[args.preset]
    if args.only_model:
        configs = [
            c for c in configs
            if c[1] == args.only_model or c[2] == args.only_model
        ]
        if not configs:
            names = [c[1] for c in MODEL_PRESETS[args.preset]]
            print(f"Model '{args.only_model}' not found in preset '{args.preset}'. Short names: {names}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    # Run each model
    for model_id, short_name, display_name in configs:
        out_csv = Path(args.output_dir) / f"{short_name}_predictions.csv"

        if out_csv.exists():
            print(f"[SKIP] {display_name}: {out_csv} already exists")
            continue

        print(f"\n{'=' * 60}")
        print(f"[MODEL] {display_name}")
        print(f"{'=' * 60}")

        try:
            print(f"  Loading {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, token=HF_TOKEN
            )
            model = TextClassifier(model_id, NUM_LABELS, HF_TOKEN).to(device)

            print(f"  Fine-tuning for {args.epochs} epochs "
                  f"on {len(train_texts)} samples (batch={TRAIN_BATCH_SIZE}, lr={LEARNING_RATE})...")
            fine_tune(
                model, tokenizer, train_texts, train_labels, device,
                epochs=args.epochs, batch_size=TRAIN_BATCH_SIZE, lr=LEARNING_RATE,
            )

            print("  Running inference on test set...")
            preds, raw = run_inference(
                model, tokenizer, ocr_texts, device,
                batch_size=INFER_BATCH_SIZE, threshold=0.5,
            )

            df = pd.DataFrame({
                "image_id":              [x["filename"] for x in images],
                "category":              [x["category"] for x in images],
                "ocr_text":              ocr_texts,
                f"{short_name}_pred":    preds,
                f"{short_name}_raw":     raw,
                "human_label":           [""] * len(images),
            })
            df.to_csv(out_csv, index=False)
            print(f"  Saved {len(df)} rows → {out_csv}")

        except Exception as e:
            print(f"\n  [ERROR] {display_name} failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"  Skipping {display_name}, continuing with next model.\n")

        finally:
            # Free GPU/MPS memory before next model
            try:
                del model
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
