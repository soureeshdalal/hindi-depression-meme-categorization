#!/usr/bin/env python3
# === COLAB SETUP ===
# Before running on Colab:
# 1. Mount Google Drive
# 2. Set BASE_DIR to the root of your project on Drive
# 3. Install: pip install transformers scikit-learn pandas torch tqdm
# All paths below are relative to BASE_DIR
"""
Zero-shot unimodal text inference on the 581 memes with non-empty Phase-1 explanations.

Input modes (same n=581 for comparability with explanation-only runs):
  - ocr: OCR text from colab_experiments/data/metadata.csv
  - ocr+explanation: ocr_text + " [SEP] " + explanation

Outputs under unimodal/581_results/ (does not write into explanation_runs/).
"""

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Repository root — override on Colab after mounting Drive
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

EXPL_CSV = BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv"
METADATA_CSV = BASE_DIR / "colab_experiments" / "data" / "metadata.csv"
OUT_DIR = BASE_DIR / "unimodal" / "581_results"

HF_TOKEN = os.environ.get("HF_TOKEN")
BATCH_SIZE = 64
NUM_LABELS = 7
EXPECTED_N = 581

SYMPTOM_NAMES = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

# (hf_id, folder_name, display_name, fixed_threshold, use_adaptive)
# Must match explanation_runs/scripts/run_unimodal.py
MODEL_CONFIGS = [
    ("ai4bharat/IndicBERTv2-SS", "IndicBERTv2_SS", "IndicBERTv2-SS", 0.5, False),
    ("ai4bharat/IndicBERTv2-MLM-Sam-TLM", "IndicBERTv2_Sam", "IndicBERTv2-Sam", 0.5, False),
    ("google/muril-base-cased", "MuRIL", "MuRIL", 0.5, False),
    ("AIMH/mental-bert-large-cased", "MentalBERT", "MentalBERT", 0.4, True),
    ("mental/mental-roberta-base", "MentalRoBERTa", "MentalRoBERTa", 0.5, False),
    ("facebook/bart-base", "BART_Base", "BART-Base", 0.3, True),
    ("facebook/bart-large", "BART_Large", "BART-Large", 0.5, False),
    ("Tianlin668/MentalBART", "MentalBART", "MentalBART", 0.3, True),
]

INPUT_MODES = (
    ("ocr", "ocr_text_input"),
    ("ocr+explanation", "ocr_plus_explanation"),
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class UnimodalClassifier(nn.Module):
    def __init__(self, model_id: str, num_labels: int = 7, token: str = None):
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(
                model_id,
                use_safetensors=True,
                trust_remote_code=True,
                token=token,
            )
        except Exception:
            self.encoder = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=token,
            )
        cfg = self.encoder.config
        hidden = getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size", 768)
        self.classifier = nn.Linear(hidden, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_hs = getattr(out, "encoder_last_hidden_state", None)
        cls = enc_hs[:, 0, :] if enc_hs is not None else out.last_hidden_state[:, 0, :]
        if cls.dtype != self.classifier.weight.dtype:
            cls = cls.to(dtype=self.classifier.weight.dtype)
        return self.sigmoid(self.classifier(cls))


def parse_label_vector(s):
    try:
        vals = [int(x) for x in str(s).strip().strip("[]").split(",")]
        return vals if len(vals) == NUM_LABELS else None
    except Exception:
        return None


def format_vector(arr):
    return "[" + ",".join(map(str, arr.tolist())) + "]"


def compute_metrics(gold_col, pred_col):
    pairs = [
        (parse_label_vector(g), parse_label_vector(p))
        for g, p in zip(gold_col, pred_col)
    ]
    pairs = [(g, p) for g, p in pairs if g is not None and p is not None]
    if not pairs:
        return None
    G = np.array([x[0] for x in pairs])
    P = np.array([x[1] for x in pairs])
    return {
        "n_samples": len(G),
        "macro_f1": round(float(f1_score(G, P, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(G, P, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(G, P, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(G, P, average="macro", zero_division=0)), 4),
    }


def run_inference_batched(model, tokenizer, texts, batch_size, fixed_threshold, use_adaptive):
    """Same as explanation_runs/scripts/run_unimodal.py — fixed + optional adaptive median thresholds."""
    model.eval()
    all_probs = []

    t0 = time.time()
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        try:
            enc = tokenizer(
                batch, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                probs = model(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                ).cpu().float().numpy()
            all_probs.append(probs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise
            for _ in batch:
                all_probs.append(np.full((1, NUM_LABELS), np.nan))

        done = min(start + batch_size, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"\r  {done}/{n}  ETA {eta:.0f}s  ", end="", flush=True)
    print()

    prob_matrix = np.vstack(all_probs)

    def apply_threshold(thresh_vec):
        preds = []
        for row in prob_matrix:
            if np.any(np.isnan(row)):
                preds.append("[INFERENCE_ERROR]")
            else:
                preds.append(format_vector((row >= thresh_vec).astype(int)))
        return preds

    preds_fixed = apply_threshold(np.full(NUM_LABELS, fixed_threshold))

    preds_adaptive = None
    if use_adaptive:
        valid = prob_matrix[~np.any(np.isnan(prob_matrix), axis=1)]
        med = np.median(valid, axis=0) if len(valid) else np.full(NUM_LABELS, fixed_threshold)
        print(f"  Adaptive thresholds (per-symptom median): {med.round(3).tolist()}")
        preds_adaptive = apply_threshold(med)

    raw_scores = [
        "[INFERENCE_ERROR]" if np.any(np.isnan(r))
        else "[" + ",".join(f"{x:.4f}" for x in r) + "]"
        for r in prob_matrix
    ]

    return preds_fixed, preds_adaptive, raw_scores


def predictions_path(out_dir: Path, folder: str, input_mode: str) -> Path:
    return out_dir / f"{folder}_{input_mode}_predictions.csv"


def load_and_merge_data():
    expl = pd.read_csv(EXPL_CSV)
    valid = expl["explanation"].notna() & (expl["explanation"].astype(str).str.strip() != "")
    expl = expl.loc[valid].copy()
    meta = pd.read_csv(METADATA_CSV)[["image_id", "ocr_text"]]
    df = expl.merge(meta, on="image_id", how="inner")
    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} rows after inner join on image_id; got {len(df)}. "
            "Check phase1_explanations_v2.csv and metadata.csv alignment."
        )
    df["ocr_text_input"] = df["ocr_text"].fillna("").astype(str).str.strip()
    df["explanation_input"] = df["explanation"].fillna("").astype(str).str.strip()
    df["ocr_plus_explanation"] = (
        df["ocr_text_input"] + " [SEP] " + df["explanation_input"]
    )
    return df


def run_one_combo(
    model_id,
    folder,
    display_name,
    fixed_threshold,
    use_adaptive,
    df,
    input_mode,
    col_name,
    out_dir: Path,
):
    out_csv = predictions_path(out_dir, folder, input_mode)
    if out_csv.exists():
        print(f"  Skip (exists): {out_csv.name}")
        return

    texts = df[col_name].tolist()
    image_ids = df["image_id"].tolist()
    categories = df["category"].tolist()
    gold_col = df["gold_labels"].tolist()
    ocr_col = df["ocr_text_input"].tolist()
    expl_col = df["explanation_input"].tolist()

    print(f"  Sanity check on 3 rows ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=HF_TOKEN
        )
        mdl = UnimodalClassifier(model_id, num_labels=NUM_LABELS, token=HF_TOKEN)
        mdl.to(DEVICE)
        mdl.eval()
        enc = tokenizer(
            texts[:3], max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out = mdl(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        print(f"  Sanity OK — output shape {out.shape}, sample: {out[0].cpu().tolist()}")
    except Exception as e:
        print(f"  ERROR: Sanity check failed: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return

    print(f"  Full run on {len(texts)} rows (batch_size={BATCH_SIZE}) ...")
    bsz = BATCH_SIZE
    while True:
        try:
            preds_fixed, _, raw_scores = run_inference_batched(
                mdl, tokenizer, texts, bsz, fixed_threshold, use_adaptive
            )
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bsz > 1:
                torch.cuda.empty_cache()
                bsz = bsz // 2
                print(f"  OOM — retrying with batch_size={bsz}", file=sys.stderr)
            else:
                print(f"  ERROR: Inference failed: {e}\n{traceback.format_exc()}", file=sys.stderr)
                del mdl
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return

    out_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "category": categories,
            "gold_labels": gold_col,
            "ocr_text": ocr_col,
            "explanation": expl_col,
            "predicted_labels": preds_fixed,
            "raw_scores": raw_scores,
        }
    )

    out_df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv.name}")

    m = compute_metrics(gold_col, preds_fixed)
    if m:
        print(
            f"  macro_f1={m['macro_f1']}  weighted_f1={m['weighted_f1']}  "
            f"P={m['macro_precision']}  R={m['macro_recall']}"
        )

    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_metrics_summary(out_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(out_dir.glob("*_predictions.csv")):
        if f.name == "metrics_summary.csv":
            continue
        stem = f.stem
        if not stem.endswith("_predictions"):
            continue
        stem = stem[: -len("_predictions")]
        if stem.endswith("_ocr+explanation"):
            folder_guess = stem[: -len("_ocr+explanation")]
            input_mode = "ocr+explanation"
        elif stem.endswith("_ocr"):
            folder_guess = stem[: -len("_ocr")]
            input_mode = "ocr"
        else:
            continue

        display_name = None
        for _, fd, disp, _, _ in MODEL_CONFIGS:
            if fd == folder_guess:
                display_name = disp
                break
        if display_name is None:
            continue

        sub = pd.read_csv(f)
        m = compute_metrics(sub["gold_labels"].tolist(), sub["predicted_labels"].tolist())
        if not m:
            continue
        rows.append(
            {
                "model": display_name,
                "input_mode": input_mode,
                "n_samples": m["n_samples"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "input_mode"]).reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Unimodal OCR / OCR+explanation on n=581 subset.")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Project root (default: parent of unimodal/). Overrides BASE_DIR for this run.",
    )
    return p.parse_args()


def main():
    global BASE_DIR, EXPL_CSV, METADATA_CSV, OUT_DIR

    args = parse_args()
    if args.base_dir is not None:
        BASE_DIR = args.base_dir.resolve()
        EXPL_CSV = BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv"
        METADATA_CSV = BASE_DIR / "colab_experiments" / "data" / "metadata.csv"
        OUT_DIR = BASE_DIR / "unimodal" / "581_results"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "metrics_summary.csv"

    print("=" * 70)
    print("UNIMODAL 581 — OCR & OCR+EXPLANATION")
    print(f"  BASE_DIR : {BASE_DIR}")
    print(f"  Device   : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  Expl CSV : {EXPL_CSV}")
    print(f"  Meta CSV : {METADATA_CSV}")
    print(f"  Output   : {OUT_DIR}")
    print("=" * 70 + "\n")

    df = load_and_merge_data()
    print(f"Loaded {len(df)} rows (non-empty explanations ∩ metadata).\n")

    assert len(MODEL_CONFIGS) == 8, "Expected 8 models in MODEL_CONFIGS"

    for model_id, folder, display_name, fixed_threshold, use_adaptive in MODEL_CONFIGS:
        for input_mode, col_name in INPUT_MODES:
            print(f"\n{'='*60}")
            print(f"[MODEL] {display_name}  ({model_id})  |  input_mode={input_mode}")
            run_one_combo(
                model_id,
                folder,
                display_name,
                fixed_threshold,
                use_adaptive,
                df,
                input_mode,
                col_name,
                OUT_DIR,
            )

    if summary_path.exists():
        print(
            f"\n{summary_path.name} already exists — not overwriting. "
            "Delete it to regenerate from on-disk predictions.",
            file=sys.stderr,
        )
    else:
        summary_df = build_metrics_summary(OUT_DIR)
        if summary_df.empty:
            print("No prediction files found for metrics summary.", file=sys.stderr)
        else:
            summary_df.to_csv(summary_path, index=False)
            print(f"\nWrote {summary_path}")
            print(summary_df.to_string(index=False))
            if (summary_df["n_samples"] != EXPECTED_N).any():
                bad = summary_df.loc[summary_df["n_samples"] != EXPECTED_N]
                print(f"WARNING: Some rows have n_samples != {EXPECTED_N}:\n{bad}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
