#!/usr/bin/env python3
"""
compute_metrics_hinglish.py

Calculates Macro-F1, Weighted-F1, Macro-Precision, Macro-Recall for all
unimodal models run on the 44 Hinglish memes (hinglish_unimodal/).

Steps:
1. Reads human labels from repo-root ``test.json``.
2. Fills the human_label column in each CSV in ``hinglish_unimodal/``.
3. Computes metrics for every model.
4. Saves results to ``metrics/hinglish/hinglish_metrics_*.csv`` (and report).
"""

import ast
import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
JSON_PATH  = ROOT / "test.json"
INPUT_DIR  = ROOT / "hinglish_unimodal"
OUTPUT_DIR = ROOT / "metrics" / "hinglish"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only 44 samples — set threshold to 1 so no model is skipped
MIN_VALID_SAMPLES = 1

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

MODEL_CONFIGS = [
    ("IndicBERTv2_SS_predictions.csv",  "IndicBERTv2_SS_pred",  "IndicBERTv2-SS"),
    ("IndicBERTv2_Sam_predictions.csv", "IndicBERTv2_Sam_pred", "IndicBERTv2-Sam"),
    ("MuRIL_predictions.csv",           "MuRIL_pred",           "MuRIL"),
    ("MentalBERT_predictions.csv",      "MentalBERT_pred",      "MentalBERT"),
    ("MentalRoBERTa_predictions.csv",   "MentalRoBERTa_pred",   "MentalRoBERTa"),
    ("BART_predictions.csv",            "BART_pred",            "BART-Base"),
    ("BART_Large_predictions.csv",      "BART_Large_pred",      "BART-Large"),
    ("MentalBART_predictions.csv",      "MentalBART_pred",      "MentalBART"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_label_dict(json_path):
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
    return re.sub(r'\.(jpg|jpeg|png)$', '', image_id, flags=re.IGNORECASE)


def vec_to_str(vec):
    return "[" + ",".join(str(v) for v in vec) + "]"


def update_and_collect(csv_path, pred_col, label_dict):
    if not csv_path.exists():
        print(f"  [SKIP] Not found: {csv_path}")
        return []

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames[:]
        rows = list(reader)

    if pred_col not in fieldnames:
        print(f"  [SKIP] Column '{pred_col}' missing in {csv_path.name}")
        return []

    valid_pairs = []
    filled = skipped_no_label = skipped_bad_pred = 0

    for row in rows:
        sid = image_id_to_sample_id(row.get("image_id", ""))
        if sid in label_dict:
            label_vec = label_dict[sid]
            row["human_label"] = vec_to_str(label_vec)
            filled += 1

            pred_str = row.get(pred_col, "").strip()
            if pred_str.startswith("[INFERENCE_ERROR") or pred_str.startswith("[ERROR") or not pred_str:
                skipped_bad_pred += 1
            else:
                try:
                    pred_vec = ast.literal_eval(pred_str)
                    if isinstance(pred_vec, list) and len(pred_vec) == 7:
                        valid_pairs.append((label_vec, pred_vec))
                    else:
                        skipped_bad_pred += 1
                except Exception:
                    skipped_bad_pred += 1
        else:
            skipped_no_label += 1

    # Write human_label back into CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {csv_path.name}")
    print(f"    filled: {filled}  |  valid pairs: {len(valid_pairs)}"
          f"  |  no JSON match: {skipped_no_label}  |  bad pred: {skipped_bad_pred}")
    return valid_pairs


def compute_metrics(pairs):
    n = len(pairs)
    per_label = []
    support_total = 0
    weighted_f1_sum = 0.0

    for j in range(7):
        tp = fp = fn = support = 0
        for yt, yp_raw in pairs:
            t = int(yt[j])
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
            "symptom":   SYMPTOMS[j],
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   support,
        })
        support_total    += support
        weighted_f1_sum  += f1 * support

    macro_f1        = round(sum(pl["f1"]        for pl in per_label) / 7, 4)
    macro_precision = round(sum(pl["precision"] for pl in per_label) / 7, 4)
    macro_recall    = round(sum(pl["recall"]    for pl in per_label) / 7, 4)
    weighted_f1     = round(weighted_f1_sum / support_total if support_total > 0 else 0.0, 4)

    return {
        "n_samples":       n,
        "macro_f1":        macro_f1,
        "weighted_f1":     weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "per_label":       per_label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Hinglish Metrics")
    print("=" * 60)

    label_dict = build_label_dict(JSON_PATH)
    print(f"Human labels loaded: {len(label_dict)} entries\n")

    all_results   = []
    all_per_label = []

    for csv_file, pred_col, model_name in MODEL_CONFIGS:
        print(f"[{model_name}]")
        csv_path = INPUT_DIR / csv_file
        pairs = update_and_collect(csv_path, pred_col, label_dict)
        if len(pairs) < MIN_VALID_SAMPLES:
            print(f"  [SKIP metrics] {len(pairs)} valid pairs\n")
            continue
        m = compute_metrics(pairs)
        all_results.append({
            "model":           model_name,
            "n_samples":       m["n_samples"],
            "macro_f1":        m["macro_f1"],
            "weighted_f1":     m["weighted_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall":    m["macro_recall"],
        })
        for pl in m["per_label"]:
            all_per_label.append({"model": model_name, **pl})
        print()

    # Summary CSV
    summary_path = OUTPUT_DIR / "hinglish_metrics_summary.csv"
    cols = ["model", "n_samples", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved: {summary_path}")

    # Per-label CSV
    per_label_path = OUTPUT_DIR / "hinglish_metrics_per_label.csv"
    pl_cols = ["model", "symptom", "precision", "recall", "f1", "support"]
    with open(per_label_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pl_cols)
        writer.writeheader()
        writer.writerows(all_per_label)
    print(f"Saved: {per_label_path}")

    # Human-readable report
    report_path = OUTPUT_DIR / "hinglish_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HINGLISH MEMES — METRICS SUMMARY (Unimodal Models)\n")
        f.write("=" * 80 + "\n")
        f.write(f"44 Hinglish memes evaluated against human labels from test.json.\n")
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

        f.write("\n\nPER-LABEL BREAKDOWN\n")
        f.write("=" * 80 + "\n")

        for r in all_results:
            mname = r["model"]
            f.write(f"\n{mname}\n")
            f.write("-" * 60 + "\n")
            col_w2 = [26, 12, 10, 10, 10]
            f.write("  ".join(h.ljust(w) for h, w in zip(
                ["Symptom", "Support", "Precision", "Recall", "F1"], col_w2)) + "\n")
            for pl in [x for x in all_per_label if x["model"] == mname]:
                row = [pl["symptom"], str(pl["support"]),
                       str(pl["precision"]), str(pl["recall"]), str(pl["f1"])]
                f.write("  ".join(v.ljust(w) for v, w in zip(row, col_w2)) + "\n")

    print(f"Saved: {report_path}\n")

    # Print report
    with open(report_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
