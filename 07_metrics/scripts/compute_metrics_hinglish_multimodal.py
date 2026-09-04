#!/usr/bin/env python3
"""
compute_metrics_hinglish_multimodal.py

Calculates Macro-F1, Weighted-F1, Macro-Precision, Macro-Recall for all
multimodal models run on the 44 Hinglish memes (hinglish_multimodal/).

Human labels are read directly from the `human_label` column already
present in each model's CSV (populated during the unimodal pipeline run).

Outputs (appended to existing unimodal metrics files):
  metrics/hinglish_metrics_summary.csv
  metrics/hinglish_metrics_per_label.csv
  metrics/hinglish_metrics_report.txt
"""

import ast
import csv
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE       = Path(__file__).resolve().parent
INPUT_DIR  = BASE / "hinglish_multimodal"
OUTPUT_DIR = BASE / "metrics"
OUTPUT_DIR.mkdir(exist_ok=True)

SYMPTOMS = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

# (subfolder, pred_column, display_name)
MODEL_CONFIGS = [
    ("LLaVA",        "LLaVA_pred",        "LLaVA-1.5-7B"),
    ("LLaVA_NeXT",   "LLaVA_NeXT_pred",   "LLaVA-NeXT-7B"),
    ("BLIP2",        "BLIP2_pred",        "BLIP-2-Flan-T5-XL"),
    ("PALO",         "PALO_pred",         "PALO-7B"),
    ("IDEFICS",      "IDEFICS_pred",      "IDEFICS-9B"),
    ("mBLIP",        "mBLIP_pred",        "mBLIP-mT0-XL"),
    ("BLIP",         "BLIP_pred",         "BLIP-Base"),
    ("Chitrarth",    "Chitrarth_pred",    "Chitrarth-7.5B"),
    ("MiniCPM_V",    "MiniCPM_V_pred",    "MiniCPM-V-3B"),
    ("InstructBLIP", "InstructBLIP_pred", "InstructBLIP-7B"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_csv(subfolder: str) -> Path | None:
    """Return the single CSV in hinglish_multimodal/{subfolder}/."""
    folder = INPUT_DIR / subfolder
    if not folder.exists():
        return None
    csvs = sorted(folder.glob("*.csv"))
    return csvs[-1] if csvs else None


def load_pairs(csv_path: Path, pred_col: str):
    """
    Returns list of (human_label_vec, pred_vec) for rows where both are valid.
    Skips rows with missing human_label, INFERENCE_ERROR, or unparseable preds.
    """
    pairs = []
    skipped_no_label = skipped_bad_pred = 0

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if pred_col not in (reader.fieldnames or []):
            print(f"  [SKIP] Column '{pred_col}' not found in {csv_path.name}")
            return []

        for row in reader:
            human_raw = row.get("human_label", "").strip()
            pred_raw  = row.get(pred_col, "").strip()

            # Skip unlabelled rows
            if not human_raw or human_raw in ("nan", "[]", ""):
                skipped_no_label += 1
                continue

            # Skip error / missing predictions
            if not pred_raw or "INFERENCE_ERROR" in pred_raw or "ERROR" in pred_raw:
                skipped_bad_pred += 1
                continue

            try:
                human_vec = ast.literal_eval(human_raw)
                pred_vec  = ast.literal_eval(pred_raw)
                if (isinstance(human_vec, list) and len(human_vec) == 7
                        and isinstance(pred_vec, list) and len(pred_vec) == 7):
                    pairs.append((human_vec, pred_vec))
                else:
                    skipped_bad_pred += 1
            except Exception:
                skipped_bad_pred += 1

    print(f"  {csv_path.name}")
    print(f"    valid pairs: {len(pairs)}"
          f"  |  no label: {skipped_no_label}"
          f"  |  bad pred: {skipped_bad_pred}")
    return pairs


def compute_metrics(pairs):
    per_label = []
    support_total    = 0
    weighted_f1_sum  = 0.0

    for j in range(7):
        tp = fp = fn = support = 0
        for yt, yp in pairs:
            t = int(yt[j])
            p = min(1, max(0, int(yp[j])))
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
        support_total   += support
        weighted_f1_sum += f1 * support

    return {
        "n_samples":       len(pairs),
        "macro_f1":        round(sum(pl["f1"]        for pl in per_label) / 7, 4),
        "weighted_f1":     round(weighted_f1_sum / support_total if support_total > 0 else 0.0, 4),
        "macro_precision": round(sum(pl["precision"] for pl in per_label) / 7, 4),
        "macro_recall":    round(sum(pl["recall"]    for pl in per_label) / 7, 4),
        "per_label":       per_label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Hinglish Multimodal Metrics")
    print("=" * 60)

    all_results   = []
    all_per_label = []

    for subfolder, pred_col, display_name in MODEL_CONFIGS:
        print(f"\n[{display_name}]")
        csv_path = find_csv(subfolder)
        if csv_path is None:
            print(f"  [SKIP] No CSV found in hinglish_multimodal/{subfolder}/")
            continue

        pairs = load_pairs(csv_path, pred_col)
        if not pairs:
            print(f"  [SKIP metrics] 0 valid pairs")
            continue

        m = compute_metrics(pairs)
        all_results.append({
            "model":           display_name,
            "n_samples":       m["n_samples"],
            "macro_f1":        m["macro_f1"],
            "weighted_f1":     m["weighted_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall":    m["macro_recall"],
        })
        for pl in m["per_label"]:
            all_per_label.append({"model": display_name, **pl})

    if not all_results:
        print("\nNo results to save.")
        return

    # ---- Summary CSV ----
    summary_path = OUTPUT_DIR / "hinglish_multimodal_metrics_summary.csv"
    cols = ["model", "n_samples", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {summary_path}")

    # ---- Per-label CSV ----
    per_label_path = OUTPUT_DIR / "hinglish_multimodal_metrics_per_label.csv"
    pl_cols = ["model", "symptom", "precision", "recall", "f1", "support"]
    with open(per_label_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pl_cols)
        writer.writeheader()
        writer.writerows(all_per_label)
    print(f"Saved: {per_label_path}")

    # ---- Text report ----
    report_path = OUTPUT_DIR / "hinglish_multimodal_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HINGLISH MEMES — METRICS SUMMARY (Multimodal Models)\n")
        f.write("=" * 80 + "\n")
        f.write("44 Hinglish memes, evaluated against human labels (39 labelled).\n")
        f.write("Metrics: multilabel, averaged across 7 PHQ-9 depression symptoms.\n\n")

        col_w = [24, 10, 12, 14, 18, 14]
        header = ["Model", "N Samples", "Macro-F1", "Weighted-F1", "Macro-Precision", "Macro-Recall"]
        f.write("  ".join(h.ljust(w) for h, w in zip(header, col_w)) + "\n")
        f.write("-" * 100 + "\n")
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
            col_w2 = [26, 12, 12, 10, 10]
            f.write("  ".join(h.ljust(w) for h, w in zip(
                ["Symptom", "Support", "Precision", "Recall", "F1"], col_w2)) + "\n")
            for pl in [x for x in all_per_label if x["model"] == mname]:
                row = [pl["symptom"], str(pl["support"]),
                       str(pl["precision"]), str(pl["recall"]), str(pl["f1"])]
                f.write("  ".join(v.ljust(w) for v, w in zip(row, col_w2)) + "\n")

    print(f"Saved: {report_path}\n")

    # Print the report to stdout
    with open(report_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
