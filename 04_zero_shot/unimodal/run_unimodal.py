#!/usr/bin/env python3
"""
Unimodal benchmarking pipeline for Hindi mental health memes.
Extracts text via OCR, runs 3 text classification models, saves predictions to CSV.
"""

import argparse
import os
import ssl
import sys
from datetime import datetime
from pathlib import Path

# HuggingFace token for gated models (MentalBERT, etc.)
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# Allow EasyOCR/HuggingFace model downloads on environments with strict or missing SSL certs
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
# Hardware: AWS g5.12xlarge, 4× A10G GPUs (96 GB VRAM)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
DEVICE = "cuda:0"
NUM_LABELS = 7

SYMPTOM_NAMES = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

# Model configs: (model_id, short_name, display_name, fixed_threshold, use_adaptive)
# - fixed_threshold: Used for _pred column (model's original-style binarization).
# - use_adaptive: If True, also compute _pred_adaptive from per-symptom median thresholds; raw kept in _raw.
# We keep both: raw scores, fixed-threshold preds, and (for adaptive models) adaptive-threshold preds.
MODEL_CONFIGS = [
    ("ai4bharat/IndicBERTv2-SS", "IndicBERTv2_SS", "IndicBERTv2-SS", 0.5, False),
    ("ai4bharat/IndicBERTv2-MLM-Sam-TLM", "IndicBERTv2_Sam", "IndicBERTv2-MLM-Sam-TLM", 0.5, False),
    ("google/muril-base-cased", "MuRIL", "MuRIL-base", 0.5, False),
    ("AIMH/mental-bert-large-cased", "MentalBERT", "MentalBERT-Large", 0.4, True),   # fixed 0.4 + adaptive
    ("mental/mental-roberta-base", "MentalRoBERTa", "MentalRoBERTa-Base", 0.5, False),
    ("facebook/bart-base", "BART", "BART-Base", 0.3, True),   # fixed 0.3 + adaptive
    ("facebook/bart-large", "BART_Large", "BART-Large", 0.5, False),
]


def collect_test_images(base_dir):
    """Returns list of dicts: {image_path, category, filename}. Only test/ folders."""
    images = []
    base_path = Path(base_dir)
    if not base_path.exists():
        return images
    for category_folder in sorted(base_path.iterdir()):
        if category_folder.is_dir():
            test_folder = category_folder / "test"
            if test_folder.exists():
                for img_file in sorted(test_folder.iterdir()):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        images.append(
                            {
                                "image_path": str(img_file),
                                "category": category_folder.name,
                                "filename": img_file.name,
                            }
                        )
    return images


def extract_text(reader, image_path):
    """Extract text from image using EasyOCR. Returns placeholder on failure."""
    try:
        results = reader.readtext(image_path, detail=0)
        text = " ".join(results).strip()
        return text if text else "[NO_TEXT]"
    except Exception as e:
        print(f"  [OCR error] {image_path}: {e}", file=sys.stderr)
        return "[OCR_ERROR]"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class UnimodalClassifier(nn.Module):
    def __init__(self, model_name, num_labels=7, token=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name, 
            use_safetensors=True, 
            trust_remote_code=True,
            token=token
        )
        # Handle BART (encoder-decoder) vs BERT (encoder-only)
        if hasattr(self.encoder.config, 'd_model'):
            # BART-style: use d_model
            hidden_size = self.encoder.config.d_model
        else:
            # BERT-style: use hidden_size
            hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        # Ensure classifier matches encoder dtype (for BART float16 compatibility)
        if hasattr(self.encoder, 'dtype'):
            self.classifier = self.classifier.to(dtype=self.encoder.dtype)

    def forward(self, input_ids, attention_mask):
        # For BART, model(**inputs) returns encoder outputs with last_hidden_state
        # For BERT/RoBERTa, same structure
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use first token (CLS for BERT, <s> for BART)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Ensure dtype match (BART may be float16, classifier needs to match)
        if cls_output.dtype != next(self.classifier.parameters()).dtype:
            cls_output = cls_output.to(dtype=next(self.classifier.parameters()).dtype)
        logits = self.classifier(cls_output)
        return self.sigmoid(logits)


def run_inference(model, tokenizer, texts, device, batch_size=64, fixed_threshold=0.5, adaptive_threshold=False):
    """
    Run batched inference. Returns (preds_fixed, preds_adaptive_or_none, raw).
    - preds_fixed: binary [0,1] per symptom using fixed_threshold.
    - preds_adaptive: same but using per-symptom median threshold (only if adaptive_threshold=True).
    - raw: string of score lists, e.g. "[0.3,0.8,...]".
    """
    model.eval()
    all_raw = []
    all_probs = []
    n = len(texts)

    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        try:
            encoded = tokenizer(
                batch_texts,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
                probs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = probs.cpu().numpy()
            all_probs.append(probs)
            for i in range(len(batch_texts)):
                all_raw.append("[" + ",".join(f"{x:.2f}" for x in probs[i].tolist()) + "]")
        except Exception as e:
            print(f"  [Inference error] batch {start}: {e}", file=sys.stderr)
            for _ in batch_texts:
                all_raw.append("[INFERENCE_ERROR]")
            all_probs.append(np.full((len(batch_texts), 7), np.nan))

    # Build fixed-threshold predictions (and adaptive if requested)
    preds_fixed = []
    preds_adaptive = [] if adaptive_threshold else None
    adaptive_thresh = np.full(7, fixed_threshold)

    if adaptive_threshold and len(all_probs) > 0:
        valid = [p for p in all_probs if not np.any(np.isnan(p))]
        if valid:
            all_probs_array = np.vstack(valid)
            adaptive_thresh = np.median(all_probs_array, axis=0)
            print(f"  Adaptive thresholds (median per symptom): {adaptive_thresh.round(3).tolist()}")
        else:
            adaptive_thresh = np.full(7, fixed_threshold)

    prob_idx = 0
    for start in range(0, n, batch_size):
        if prob_idx < len(all_probs):
            probs = all_probs[prob_idx]
            prob_idx += 1
            if np.any(np.isnan(probs)):
                for _ in range(probs.shape[0]):
                    preds_fixed.append("[INFERENCE_ERROR]")
                    if preds_adaptive is not None:
                        preds_adaptive.append("[INFERENCE_ERROR]")
            else:
                binary_fixed = (probs >= fixed_threshold).astype(np.int32)
                for i in range(probs.shape[0]):
                    preds_fixed.append("[" + ",".join(map(str, binary_fixed[i].tolist())) + "]")
                if preds_adaptive is not None:
                    binary_adaptive = (probs >= adaptive_thresh).astype(np.int32)
                    for i in range(probs.shape[0]):
                        preds_adaptive.append("[" + ",".join(map(str, binary_adaptive[i].tolist())) + "]")
        else:
            break

    return preds_fixed, preds_adaptive, all_raw


def main():
    parser = argparse.ArgumentParser(description="Unimodal benchmarking pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to translated_categorized_memes")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: unimodal_predictions_YYYYMMDD_HHMMSS.csv, never overwrites)",
    )
    parser.add_argument("--max_images", type=int, default=None, help="If set, only process this many images (for testing)")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"unimodal_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print("=== UNIMODAL BENCHMARKING PIPELINE ===")
    images = collect_test_images(args.data_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
        print(f"Processing first {len(images)} images (--max_images={args.max_images})")
    print(f"Found {len(images)} test images across categories")

    if not images:
        print("No test images found. Exiting.")
        return

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if str(device) != DEVICE:
        print(f"Note: Using {device} (CUDA not available)")

    # ---- OCR ----
    print("\n[OCR] Extracting text from all images...")
    reader = easyocr.Reader(["hi", "en"], gpu=(str(device).startswith("cuda")))
    ocr_texts = []
    for img_info in tqdm(images, desc="OCR"):
        text = extract_text(reader, img_info["image_path"])
        ocr_texts.append(text)
    ocr_errors = sum(1 for t in ocr_texts if t == "[OCR_ERROR]")
    extractable = sum(1 for t in ocr_texts if t not in ("[NO_TEXT]", "[OCR_ERROR]"))
    print(f"OCR complete. {extractable}/{len(images)} images had extractable text.\n")

    # ---- Models ----
    results = {
        "image_id": [x["filename"] for x in images],
        "category": [x["category"] for x in images],
        "ocr_text": ocr_texts,
        "human_label": [""] * len(images),
    }
    inference_errors = 0

    for idx, (model_id, short_name, display_name, fixed_threshold, use_adaptive) in enumerate(MODEL_CONFIGS, 1):
        print(f"[MODEL {idx}/{len(MODEL_CONFIGS)}] {display_name}")
        print(f"Loading model from {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
            model = UnimodalClassifier(model_id, num_labels=NUM_LABELS, token=HF_TOKEN)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"  Failed to load model: {e}", file=sys.stderr)
            results[f"{short_name}_pred"] = ["[INFERENCE_ERROR]"] * len(images)
            results[f"{short_name}_raw"] = ["[INFERENCE_ERROR]"] * len(images)
            if use_adaptive:
                results[f"{short_name}_pred_adaptive"] = ["[INFERENCE_ERROR]"] * len(images)
            inference_errors += len(images)
            continue

        threshold_str = f"fixed={fixed_threshold}" + (" + adaptive" if use_adaptive else "")
        print(f"Running inference ({threshold_str})...")
        preds_fixed, preds_adaptive, raw = run_inference(
            model, tokenizer, ocr_texts, device,
            batch_size=BATCH_SIZE,
            fixed_threshold=fixed_threshold,
            adaptive_threshold=use_adaptive,
        )
        for i, (p, r) in enumerate(zip(preds_fixed, raw)):
            if p == "[INFERENCE_ERROR]":
                inference_errors += 1
        results[f"{short_name}_pred"] = preds_fixed
        results[f"{short_name}_raw"] = raw
        if preds_adaptive is not None:
            results[f"{short_name}_pred_adaptive"] = preds_adaptive
        print()

    # ---- Save CSV ----
    df = pd.DataFrame(results)
    # Build column order dynamically from all models that ran
    col_order = ["image_id", "category", "ocr_text"]
    for _, short_name, _, _, use_adaptive in MODEL_CONFIGS:
        for suffix in ["_pred", "_pred_adaptive", "_raw"]:
            if suffix == "_pred_adaptive" and not use_adaptive:
                continue
            col = f"{short_name}{suffix}"
            if col in df.columns:
                col_order.append(col)
    col_order.append("human_label")
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(args.output, index=False)
    print(f"Saving results to {args.output}...")
    print(f"Done. {len(df)} rows saved.\n")

    print("=== SUMMARY ===")
    print(f"Total images: {len(images)}")
    print(f"OCR errors: {ocr_errors}")
    print(f"Inference errors: {inference_errors}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
