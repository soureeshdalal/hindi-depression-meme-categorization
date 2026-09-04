#!/usr/bin/env python3
"""
Unimodal text benchmarking on Gemini-generated structured explanations.

Input:  phase-1 explanations CSV (default under gemini_runs/run_v2/)
Output: explanation_runs/unimodal/<ModelName>/predictions.csv
        explanation_runs/unimodal/metrics_summary.csv, metrics_per_label.csv
        explanation_runs/errors.log

No fine-tuning: pre-trained weights + randomly-initialized nn.Linear(hidden_size, 7)
 classification head + sigmoid. Input = explanation text (NOT OCR, NOT images).

CLI:
  python explanation_runs/scripts/run_unimodal.py \\
    --repo-root . \\
    --explanations-csv gemini_runs/run_v2/phase1_explanations_v2.csv
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths (set by apply_path_config() in main)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
BASE_DIR     = Path(__file__).resolve().parent.parent.parent  # repo root
EXPL_CSV     = BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv"
BASELINE_CSV = BASE_DIR / "metrics" / "no_finetuning" / "metrics_summary.csv"
OUT_ROOT     = SCRIPT_DIR.parent / "unimodal"
ERRORS_LOG   = SCRIPT_DIR.parent / "errors.log"


def parse_args():
    repo_default = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(
        description="Unimodal classifiers on Gemini explanation text (vs OCR baselines in metrics/).",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=repo_default,
        help="Repository root (folder containing train.json, translated_categorized_memes/, …).",
    )
    p.add_argument(
        "--explanations-csv",
        type=Path,
        default=None,
        help="Phase-1 CSV with columns image_id, category, gold_labels, explanation.",
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="metrics/no_finetuning/metrics_summary.csv for comparison_table Δ vs OCR.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: explanation_runs/unimodal).",
    )
    return p.parse_args()


def apply_path_config(args: argparse.Namespace) -> None:
    global BASE_DIR, EXPL_CSV, BASELINE_CSV, OUT_ROOT, ERRORS_LOG
    BASE_DIR = args.repo_root.resolve()
    EXPL_CSV = (
        args.explanations_csv.resolve()
        if args.explanations_csv
        else (BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv")
    )
    BASELINE_CSV = (
        args.baseline_csv.resolve()
        if args.baseline_csv
        else (BASE_DIR / "metrics" / "no_finetuning" / "metrics_summary.csv")
    )
    OUT_ROOT = (
        args.output_dir.resolve()
        if args.output_dir
        else (SCRIPT_DIR.parent / "unimodal")
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Keep errors.log next to explanation_runs outputs (isolated run = under runs/pipeline_*/…)
    ERRORS_LOG = OUT_ROOT.parent / "errors.log"

HF_TOKEN   = os.environ.get("HF_TOKEN")  # export HF_TOKEN for gated models; no token in repo
BATCH_SIZE = 64
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

# (hf_id, folder_name, display_name, fixed_threshold, use_adaptive)
MODEL_CONFIGS = [
    ("ai4bharat/IndicBERTv2-SS",           "IndicBERTv2_SS",   "IndicBERTv2-SS",        0.5, False),
    ("ai4bharat/IndicBERTv2-MLM-Sam-TLM",  "IndicBERTv2_Sam",  "IndicBERTv2-Sam",       0.5, False),
    ("google/muril-base-cased",            "MuRIL",            "MuRIL",                 0.5, False),
    ("AIMH/mental-bert-large-cased",       "MentalBERT",       "MentalBERT",            0.4, True),
    ("mental/mental-roberta-base",         "MentalRoBERTa",    "MentalRoBERTa",         0.5, False),
    ("facebook/bart-base",                 "BART_Base",        "BART-Base",             0.3, True),
    ("facebook/bart-large",               "BART_Large",       "BART-Large",            0.5, False),
    ("Tianlin668/MentalBART",              "MentalBART",       "MentalBART",            0.3, True),
]


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_error(model_name: str, msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{model_name}] {msg}\n"
    with open(ERRORS_LOG, "a") as f:
        f.write(line)
    print(f"  ERROR: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
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
        # For encoder-decoder models (BART family), prefer the encoder's hidden
        # states for classification — decoder output is optimised for generation.
        enc_hs = getattr(out, "encoder_last_hidden_state", None)
        cls = enc_hs[:, 0, :] if enc_hs is not None else out.last_hidden_state[:, 0, :]
        if cls.dtype != self.classifier.weight.dtype:
            cls = cls.to(dtype=self.classifier.weight.dtype)
        return self.sigmoid(self.classifier(cls))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    result = {
        "n_samples":        len(G),
        "macro_f1":         round(float(f1_score(G, P, average="macro",    zero_division=0)), 4),
        "weighted_f1":      round(float(f1_score(G, P, average="weighted", zero_division=0)), 4),
        "macro_precision":  round(float(precision_score(G, P, average="macro", zero_division=0)), 4),
        "macro_recall":     round(float(recall_score(G, P, average="macro",    zero_division=0)), 4),
        "per_label": {},
    }
    for i, sym in enumerate(SYMPTOM_NAMES):
        result["per_label"][sym] = {
            "f1":        round(float(f1_score(G[:, i],        P[:, i], zero_division=0)), 4),
            "precision": round(float(precision_score(G[:, i], P[:, i], zero_division=0)), 4),
            "recall":    round(float(recall_score(G[:, i],    P[:, i], zero_division=0)), 4),
        }
    return result


def check_degenerate(pred_col, model_name):
    vecs = [parse_label_vector(p) for p in pred_col if parse_label_vector(p) is not None]
    if not vecs:
        return
    arr = np.array(vecs)
    col_means = arr.mean(axis=0)
    if np.all(col_means == 0) or np.all(col_means == 1):
        log_error(model_name, "degenerate predictions — all-zeros or all-ones across dataset")
    elif np.any(col_means == 0) or np.any(col_means == 1):
        bad = [SYMPTOM_NAMES[i] for i in range(NUM_LABELS)
               if col_means[i] == 0 or col_means[i] == 1]
        log_error(model_name, f"degenerate per-label predictions for: {bad}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference_batched(model, tokenizer, texts, batch_size, fixed_threshold, use_adaptive):
    """Run inference, returning (preds_fixed, preds_adaptive|None, raw_scores)."""
    model.eval()
    all_probs = []

    t0 = time.time()
    n  = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start: start + batch_size]
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

        done  = min(start + batch_size, n)
        elapsed = time.time() - t0
        rate  = done / elapsed if elapsed > 0 else 0
        eta   = (n - done) / rate if rate > 0 else 0
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
        med   = np.median(valid, axis=0) if len(valid) else np.full(NUM_LABELS, fixed_threshold)
        print(f"  Adaptive thresholds (per-symptom median): {med.round(3).tolist()}")
        preds_adaptive = apply_threshold(med)

    raw_scores = [
        "[INFERENCE_ERROR]" if np.any(np.isnan(r))
        else "[" + ",".join(f"{x:.4f}" for x in r) + "]"
        for r in prob_matrix
    ]

    return preds_fixed, preds_adaptive, raw_scores


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------
def run_model(model_id, folder, display_name, fixed_threshold, use_adaptive,
              image_ids, categories, gold_col, texts):
    out_dir  = OUT_ROOT / folder
    out_dir.mkdir(exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    met_json = out_dir / "metrics.json"

    # Resume: skip already-processed rows
    done_ids = set()
    if pred_csv.exists():
        try:
            existing = pd.read_csv(pred_csv)
            done_ids = set(existing["image_id"].tolist())
            print(f"  Resume: {len(done_ids)} rows already done, skipping them.")
        except Exception:
            done_ids = set()

    # Filter to unprocessed
    todo_mask = [iid not in done_ids for iid in image_ids]
    if not any(todo_mask):
        print(f"  All {len(image_ids)} rows already done — skipping model.")
        return

    t_ids  = [x for x, m in zip(image_ids,  todo_mask) if m]
    t_cats = [x for x, m in zip(categories, todo_mask) if m]
    t_gold = [x for x, m in zip(gold_col,   todo_mask) if m]
    t_txts = [x for x, m in zip(texts,      todo_mask) if m]

    # --- Sanity check (3 rows) ---
    print(f"  Sanity check on 3 rows ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=HF_TOKEN
        )
        mdl = UnimodalClassifier(model_id, num_labels=NUM_LABELS, token=HF_TOKEN)
        mdl.to(DEVICE)
        mdl.eval()
        enc = tokenizer(t_txts[:3], max_length=512, padding=True,
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = mdl(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        print(f"  Sanity OK — output shape {out.shape}, sample: {out[0].cpu().tolist()}")
    except Exception as e:
        log_error(display_name, f"Sanity check failed: {e}\n{traceback.format_exc()}")
        return

    # --- Full run ---
    print(f"  Full run on {len(t_txts)} rows (batch_size={BATCH_SIZE}) ...")
    bsz = BATCH_SIZE
    while True:
        try:
            preds_fixed, preds_adaptive, raw_scores = run_inference_batched(
                mdl, tokenizer, t_txts, bsz, fixed_threshold, use_adaptive
            )
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bsz > 1:
                torch.cuda.empty_cache()
                bsz = bsz // 2
                log_error(display_name, f"OOM at batch {bsz*2}, retrying with batch {bsz}")
                print(f"  OOM — retrying with batch_size={bsz}")
            else:
                log_error(display_name, f"Inference failed: {e}\n{traceback.format_exc()}")
                del mdl; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return

    # --- Save CSV (append mode if resuming) ---
    new_rows = pd.DataFrame({
        "image_id":          t_ids,
        "category":          t_cats,
        "gold_labels":       t_gold,
        "predicted_labels":  preds_fixed,
        "raw_scores":        raw_scores,
    })
    if use_adaptive:
        new_rows["predicted_labels_adaptive"] = preds_adaptive

    write_header = not pred_csv.exists()
    new_rows.to_csv(pred_csv, mode="a", header=write_header, index=False)
    print(f"  Saved predictions → {pred_csv}")

    # --- Metrics (over all rows in output CSV) ---
    full_df        = pd.read_csv(pred_csv)
    check_degenerate(full_df["predicted_labels"].tolist(), display_name)
    m = compute_metrics(full_df["gold_labels"], full_df["predicted_labels"])
    if m:
        m["model"]   = display_name
        m["input"]   = "explanation"
        m["folder"]  = folder
        with open(met_json, "w") as f:
            json.dump(m, f, indent=2)
        print(f"  macro_f1={m['macro_f1']}  weighted_f1={m['weighted_f1']}  "
              f"P={m['macro_precision']}  R={m['macro_recall']}")
        if use_adaptive:
            m_a = compute_metrics(full_df["gold_labels"], full_df["predicted_labels_adaptive"])
            if m_a:
                print(f"  [adaptive] macro_f1={m_a['macro_f1']}  "
                      f"P={m_a['macro_precision']}  R={m_a['macro_recall']}")

    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate_metrics():
    summary_rows   = []
    per_label_rows = []
    for _, folder, display_name, _, _ in MODEL_CONFIGS:
        met_json = OUT_ROOT / folder / "metrics.json"
        if not met_json.exists():
            continue
        with open(met_json) as f:
            m = json.load(f)
        summary_rows.append({
            "model":           display_name,
            "input":           "explanation",
            "n_samples":       m.get("n_samples", 0),
            "macro_f1":        m.get("macro_f1", ""),
            "weighted_f1":     m.get("weighted_f1", ""),
            "macro_precision": m.get("macro_precision", ""),
            "macro_recall":    m.get("macro_recall", ""),
        })
        for sym, vals in m.get("per_label", {}).items():
            per_label_rows.append({
                "model":   display_name,
                "input":   "explanation",
                "symptom": sym,
                **vals,
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "metrics_summary.csv", index=False)
        pd.DataFrame(per_label_rows).to_csv(OUT_ROOT / "metrics_per_label.csv", index=False)

    return pd.DataFrame(summary_rows) if summary_rows else None


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(summary_df):
    if summary_df is None or summary_df.empty:
        return

    print("\n" + "=" * 90)
    print("FINAL COMPARISON TABLE (explanation input vs OCR baseline)")
    print("=" * 90)

    # Load baseline
    baseline = {}
    if BASELINE_CSV.exists():
        try:
            b = pd.read_csv(BASELINE_CSV)
            for _, row in b.iterrows():
                baseline[str(row["model"])] = row
        except Exception as e:
            print(f"  [Warning] Could not load baseline: {e}")

    header = f"{'model':<22} {'input':<11} {'n':>5} {'macro_f1':>9} {'wt_f1':>8} {'prec':>7} {'rec':>7}"
    if baseline:
        header += f"  {'Δmacro_f1':>10}"
    print(header)
    print("-" * len(header))

    for _, row in summary_df.iterrows():
        mname = str(row["model"])
        line  = (f"{mname:<22} {'explanation':<11} {int(row['n_samples']):>5} "
                 f"{float(row['macro_f1']):>9.4f} {float(row['weighted_f1']):>8.4f} "
                 f"{float(row['macro_precision']):>7.4f} {float(row['macro_recall']):>7.4f}")
        if baseline and mname in baseline:
            base_f1 = float(baseline[mname]["macro_f1"])
            delta   = float(row["macro_f1"]) - base_f1
            sign    = "+" if delta >= 0 else ""
            line   += f"  {sign}{delta:>9.4f}"
        print(line)

    print("=" * 90)

    # Save to file
    comp_path = OUT_ROOT.parent / "comparison_table.txt"
    with open(comp_path, "w") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for _, row in summary_df.iterrows():
            mname = str(row["model"])
            line  = (f"{mname:<22} {'explanation':<11} {int(row['n_samples']):>5} "
                     f"{float(row['macro_f1']):>9.4f} {float(row['weighted_f1']):>8.4f} "
                     f"{float(row['macro_precision']):>7.4f} {float(row['macro_recall']):>7.4f}")
            if baseline and mname in baseline:
                base_f1 = float(baseline[mname]["macro_f1"])
                delta   = float(row["macro_f1"]) - base_f1
                sign    = "+" if delta >= 0 else ""
                line   += f"  {sign}{delta:>9.4f}"
            f.write(line + "\n")
    print(f"\nComparison saved → {comp_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    apply_path_config(args)

    print("=" * 70)
    print("UNIMODAL EXPLANATION BENCHMARKING")
    print(f"  Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  Input   : {EXPL_CSV}")
    print(f"  Output  : {OUT_ROOT}")
    print("=" * 70 + "\n")

    # Load explanations
    df    = pd.read_csv(EXPL_CSV)
    valid = df["explanation"].notna() & (df["explanation"].str.strip() != "")
    df    = df[valid].reset_index(drop=True)
    print(f"Loaded {len(df)} rows (skipped {(~valid).sum()} empty/safety-blocked)\n")

    image_ids  = df["image_id"].tolist()
    categories = df["category"].tolist()
    gold_col   = df["gold_labels"].tolist()
    texts      = df["explanation"].tolist()

    for cfg in MODEL_CONFIGS:
        model_id, folder, display_name, fixed_threshold, use_adaptive = cfg
        print(f"\n{'='*60}")
        print(f"[MODEL] {display_name}  ({model_id})")
        try:
            run_model(
                model_id, folder, display_name, fixed_threshold, use_adaptive,
                image_ids, categories, gold_col, texts,
            )
        except Exception as e:
            log_error(display_name, f"Unhandled exception: {e}\n{traceback.format_exc()}")
            print(f"  FAILED — continuing to next model.\n", file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n\n[Aggregating metrics ...]")
    summary_df = aggregate_metrics()
    print_comparison(summary_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
