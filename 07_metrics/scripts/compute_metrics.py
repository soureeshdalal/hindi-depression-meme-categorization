#!/usr/bin/env python3
"""
compute_metrics.py

Steps:
1. Reads human labels from repo-root ``test.json`` (category strings -> 7-D binary vector).
2. Fills the human_label column in every unimodal and multimodal prediction CSV.
   (Only human_label is written; all other columns are untouched.)
3. Computes macro-F1, weighted-F1, macro-Precision, macro-Recall for every model
   that has at least MIN_VALID_SAMPLES rows with both a human label and a valid prediction.
4. Saves under ``metrics/no_finetuning/``:
   - metrics_summary.csv
   - metrics_per_label.csv
   - metrics_report.txt

Prediction CSV paths are relative to the **repository root** (same layout as ``unimodal/*_predictions.csv``).

Nothing except the human_label column in existing CSVs is changed.
The JSON file is not modified.
"""

import ast
import csv
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # repo root (parent of 03_scripts/)
JSON_PATH = ROOT / "test.json"
OUT_DIR = ROOT / "metrics" / "no_finetuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_VALID_SAMPLES = 50   # skip models with fewer valid (label + pred) pairs

# Symptom order (index 0..6): MUST match the binary vector convention used in
# all prediction CSVs.
SYMPTOMS = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

LABEL_TO_IDX = {s: i for i, s in enumerate(SYMPTOMS)}

# (csv_path, pred_column, model_display_name)
MODEL_CONFIGS = [
    # Unimodal
    ("unimodal/IndicBERTv2_SS_predictions.csv",  "IndicBERTv2_SS_pred",  "IndicBERTv2-SS"),
    ("unimodal/IndicBERTv2_Sam_predictions.csv", "IndicBERTv2_Sam_pred", "IndicBERTv2-Sam"),
    ("unimodal/MuRIL_predictions.csv",           "MuRIL_pred",           "MuRIL"),
    ("unimodal/MentalBERT_predictions.csv",      "MentalBERT_pred",      "MentalBERT"),
    ("unimodal/MentalRoBERTa_predictions.csv",   "MentalRoBERTa_pred",   "MentalRoBERTa"),
    ("unimodal/BART_predictions.csv",            "BART_pred",            "BART-Base"),
    ("unimodal/BART_Large_predictions.csv",      "BART_Large_pred",      "BART-Large"),
    ("unimodal/MentalBART_predictions.csv",      "MentalBART_pred",      "MentalBART"),
    # Multimodal (only models with full or near-full runs)
    ("multimodal/PALO/PALO_predictions.csv",           "PALO_pred",      "PALO-7B"),
    ("multimodal/LLaVA/LLaVA_predictions.csv",         "LLaVA_pred",     "LLaVA-1.5"),
    ("multimodal/LLaVA_NeXT/LLaVA_NeXT_predictions.csv", "LLaVA_NeXT_pred", "LLaVA-NeXT"),
    ("multimodal/BLIP2/BLIP2_predictions.csv",         "BLIP2_pred",     "BLIP-2"),
    ("multimodal/IDEFICS/IDEFICS_predictions.csv",     "IDEFICS_pred",   "IDEFICS"),
    ("multimodal/BLIP/BLIP_predictions.csv",           "BLIP_pred",      "BLIP"),
    # Skipped: mBLIP (2 rows), MiniCPM_V (1 row), InstructBLIP (all errors), Chitrarth (2 rows)
]

# ---------------------------------------------------------------------------
# Step 1: Build human label lookup from JSON
# ---------------------------------------------------------------------------

def build_label_dict(json_path):
    """
    Returns dict: sample_id (str, e.g. 'TE-8') -> binary list [0/1 x 7].
    Entries with no categories get [0, 0, 0, 0, 0, 0, 0].
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    label_dict = {}
    for entry in data:
        sid = entry["sample_id"]
        vec = [0] * 7
        for cat in entry.get("meme_depressive_categories", []):
            idx = LABEL_TO_IDX.get(cat)
            if idx is not None:
                vec[idx] = 1
        label_dict[sid] = vec
    return label_dict


def image_id_to_sample_id(image_id):
    """Strip extension: 'TE-8.jpg' -> 'TE-8'."""
    return re.sub(r'\.(jpg|jpeg|png)$', '', image_id, flags=re.IGNORECASE)


def vec_to_str(vec):
    return "[" + ",".join(str(v) for v in vec) + "]"


# ---------------------------------------------------------------------------
# Step 2: Fill human_label in a CSV and return rows with valid pairs
# ---------------------------------------------------------------------------

def update_and_collect(csv_path, pred_col, label_dict):
    """
    Reads csv_path, fills human_label where the sample_id is in label_dict,
    writes the CSV back (same columns, only human_label changed).
    Returns list of (y_true_vec, y_pred_vec) for rows with both valid.
    """
    path = ROOT / csv_path
    if not path.exists():
        print(f"  [SKIP] File not found: {csv_path}")
        return []

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames[:]
        rows = list(reader)

    if pred_col not in fieldnames:
        print(f"  [SKIP] Column '{pred_col}' not in {csv_path}")
        return []

    updated_rows = []
    valid_pairs = []
    filled = 0
    skipped_no_label = 0
    skipped_bad_pred = 0

    for row in rows:
        sample_id = image_id_to_sample_id(row.get("image_id", ""))
        if sample_id in label_dict:
            label_vec = label_dict[sample_id]
            row["human_label"] = vec_to_str(label_vec)
            filled += 1

            # Parse prediction
            pred_str = row.get(pred_col, "").strip()
            if pred_str.startswith("[INFERENCE_ERROR") or pred_str.startswith("[ERROR") or not pred_str:
                skipped_bad_pred += 1
            else:
                try:
                    pred_vec = ast.literal_eval(pred_str)
                    if isinstance(pred_vec, list) and len(pred_vec) == 7:
                        valid_pairs.append((label_vec, pred_vec))
                except Exception:
                    skipped_bad_pred += 1
        else:
            skipped_no_label += 1

        updated_rows.append(row)

    # Write back (only human_label changed)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"  {csv_path}")
    print(f"    human_label filled: {filled}  |  valid pairs: {len(valid_pairs)}"
          f"  |  no JSON match: {skipped_no_label}  |  bad pred: {skipped_bad_pred}")
    return valid_pairs


# ---------------------------------------------------------------------------
# Step 3: Compute multilabel metrics from (y_true, y_pred) pairs
# ---------------------------------------------------------------------------

def compute_metrics(pairs):
    """
    pairs: list of (label_vec [7], pred_vec [7])
    Returns dict with:
        macro_f1, weighted_f1, macro_precision, macro_recall
        per_label: list of 7 dicts {f1, precision, recall, support}
    Pure Python, no numpy.
    """
    n = len(pairs)
    per_label = []
    support_total = 0
    weighted_f1_sum = 0.0

    for j in range(7):
        tp = fp = fn = support = 0
        for yt, yp_raw in pairs:
            t = int(yt[j])
            # clamp prediction to 0/1
            p = min(1, max(0, int(yp_raw[j])))
            if t == 1:
                support += 1
                if p == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if p == 1:
                    fp += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_label.append({
            "symptom": SYMPTOMS[j],
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   support,
        })
        support_total += support
        weighted_f1_sum += f1 * support

    macro_f1        = round(sum(pl["f1"]        for pl in per_label) / 7, 4)
    macro_precision = round(sum(pl["precision"] for pl in per_label) / 7, 4)
    macro_recall    = round(sum(pl["recall"]    for pl in per_label) / 7, 4)
    weighted_f1     = round(weighted_f1_sum / support_total if support_total > 0 else 0.0, 4)

    return {
        "n_samples":        n,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "macro_precision":  macro_precision,
        "macro_recall":     macro_recall,
        "per_label":        per_label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 1: Building human label dict from JSON")
    print("=" * 60)
    label_dict = build_label_dict(JSON_PATH)
    print(f"  Loaded {len(label_dict)} human-labeled samples from JSON")

    print()
    print("=" * 60)
    print("STEP 2: Filling human_label columns + collecting valid pairs")
    print("=" * 60)

    all_results = []   # list of {model, metrics}
    all_per_label = [] # list of {model, symptom, ...}

    for csv_rel, pred_col, model_name in MODEL_CONFIGS:
        print(f"\n[{model_name}]")
        pairs = update_and_collect(csv_rel, pred_col, label_dict)
        if len(pairs) < MIN_VALID_SAMPLES:
            print(f"  [SKIP metrics] Only {len(pairs)} valid pairs (< {MIN_VALID_SAMPLES})")
            continue
        metrics = compute_metrics(pairs)
        all_results.append({
            "model":           model_name,
            "n_samples":       metrics["n_samples"],
            "macro_f1":        metrics["macro_f1"],
            "weighted_f1":     metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall":    metrics["macro_recall"],
        })
        for pl in metrics["per_label"]:
            all_per_label.append({
                "model":     model_name,
                "symptom":   pl["symptom"],
                "precision": pl["precision"],
                "recall":    pl["recall"],
                "f1":        pl["f1"],
                "support":   pl["support"],
            })

    # ---------- Save summary CSV ----------
    summary_path = OUT_DIR / "metrics_summary.csv"
    summary_cols = ["model", "n_samples", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {summary_path}")

    # ---------- Save per-label CSV ----------
    per_label_path = OUT_DIR / "metrics_per_label.csv"
    per_label_cols = ["model", "symptom", "precision", "recall", "f1", "support"]
    with open(per_label_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_label_cols)
        writer.writeheader()
        writer.writerows(all_per_label)
    print(f"Saved: {per_label_path}")

    # ---------- Save human-readable report ----------
    report_path = OUT_DIR / "metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        # Overall summary table
        f.write("METRICS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Evaluation on matched samples from test.json.\n")
        f.write(f"Skipped models with fewer than {MIN_VALID_SAMPLES} valid prediction+label pairs.\n")
        f.write(f"Metrics are multilabel (averaged across 7 PHQ-9 symptoms).\n\n")

        col_w = [20, 10, 12, 14, 18, 14]
        header = ["Model", "N Samples", "Macro-F1", "Weighted-F1", "Macro-Precision", "Macro-Recall"]
        f.write("  ".join(h.ljust(w) for h, w in zip(header, col_w)) + "\n")
        f.write("-" * 96 + "\n")
        for r in sorted(all_results, key=lambda x: -x["macro_f1"]):
            row = [
                r["model"], str(r["n_samples"]),
                str(r["macro_f1"]), str(r["weighted_f1"]),
                str(r["macro_precision"]), str(r["macro_recall"]),
            ]
            f.write("  ".join(v.ljust(w) for v, w in zip(row, col_w)) + "\n")

        f.write("\n\n")
        f.write("PER-LABEL BREAKDOWN\n")
        f.write("=" * 80 + "\n")

        # Group by model
        models_seen = []
        for r in all_results:
            if r["model"] not in models_seen:
                models_seen.append(r["model"])

        for model_name in models_seen:
            f.write(f"\n{model_name}\n")
            f.write("-" * 60 + "\n")
            col_w2 = [26, 12, 10, 10, 10]
            h2 = ["Symptom", "Support", "Precision", "Recall", "F1"]
            f.write("  ".join(h.ljust(w) for h, w in zip(h2, col_w2)) + "\n")
            rows_for_model = [pl for pl in all_per_label if pl["model"] == model_name]
            for pl in rows_for_model:
                row = [
                    pl["symptom"], str(pl["support"]),
                    str(pl["precision"]), str(pl["recall"]), str(pl["f1"]),
                ]
                f.write("  ".join(v.ljust(w) for v, w in zip(row, col_w2)) + "\n")

    print(f"Saved: {report_path}")

    # ---------- Print report to console ----------
    print()
    with open(report_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
