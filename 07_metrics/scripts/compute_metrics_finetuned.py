#!/usr/bin/env python3
"""
compute_metrics_finetuned.py

Computes metrics for all fine-tuned unimodal text and vision models
(translated train → translated test predictions only).

Steps:
1. Reads human labels from test.json (category strings -> 7-D binary vector).
2. Fills the human_label column in every prediction CSV.
3. Computes macro-F1, weighted-F1, macro-Precision, macro-Recall per model.
4. Saves metrics under ``--metrics-out-dir`` (default: metrics/finetuned/).

CLI:
  python 03_scripts/compute_metrics_finetuned.py \\
    --prediction-root runs/my_exp \\
    --metrics-out-dir runs/my_exp/metrics/finetuned

Use ``--no-mutate-prediction-csvs`` to score without rewriting prediction CSVs.
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults (overridden by CLI)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_VALID_SAMPLES = 50

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

# Paths relative to repo ROOT
MODEL_CONFIGS = [
    # Unimodal text — MAMA Table 2 + Hindi extensions (run_unimodal_finetuned.py --preset all)
    ("unimodal/finetuned/BERT_ft_predictions.csv",            "BERT_ft_pred",            "BERT-uncased (FT, MAMA)"),
    ("unimodal/finetuned/MentalBERT_ft_predictions.csv",      "MentalBERT_ft_pred",      "MentalBERT (FT)"),
    ("unimodal/finetuned/BART_ft_predictions.csv",            "BART_ft_pred",            "BART-Base (FT)"),
    ("unimodal/finetuned/MentalBART_ft_predictions.csv",      "MentalBART_ft_pred",      "MentalBART (FT)"),
    ("unimodal/finetuned/BERT_mBERT_ft_predictions.csv",      "BERT_mBERT_ft_pred",      "BERT-multilingual (FT)"),
    ("unimodal/finetuned/IndicBERTv2_SS_ft_predictions.csv",  "IndicBERTv2_SS_ft_pred",  "IndicBERTv2-SS (FT)"),
    ("unimodal/finetuned/IndicBERTv2_Sam_ft_predictions.csv", "IndicBERTv2_Sam_ft_pred", "IndicBERTv2-Sam (FT)"),
    ("unimodal/finetuned/MuRIL_ft_predictions.csv",           "MuRIL_ft_pred",           "MuRIL (FT)"),
    ("unimodal/finetuned/MentalRoBERTa_ft_predictions.csv",   "MentalRoBERTa_ft_pred",   "MentalRoBERTa (FT)"),
    ("unimodal/finetuned/BART_Large_ft_predictions.csv",      "BART_Large_ft_pred",      "BART-Large (FT)"),
    # Unimodal image (vision/run_vision.py; default EfficientNet-B0)
    ("vision/ViT_predictions.csv",              "ViT_pred",              "ViT-B/16 (FT)"),
    ("vision/ResNet_predictions.csv",           "ResNet_pred",           "ResNet-50 (FT)"),
    ("vision/EfficientNet_B0_predictions.csv",  "EfficientNet_B0_pred",  "EfficientNet-B0 (FT)"),
    ("vision/EfficientNet_B7_predictions.csv",  "EfficientNet_B7_pred",  "EfficientNet-B7 (FT)"),
    # Multimodal FT — MAMA-style (run_mama_multimodal_finetune.py)
    ("multimodal/finetuned_mama/clip_predictions.csv",        "clip_pred",        "CLIP ViT-B/32 (FT)"),
    ("multimodal/finetuned_mama/vit_bert_predictions.csv",    "vit_bert_pred",    "ViT+BERT fusion (FT)"),
    ("multimodal/finetuned_mama/visualbert_predictions.csv",   "visualbert_pred",  "VisualBERT + ResNet regions (FT)"),
    # Generative VLMs — supervised linear/projector head (run_vlm_multilabel_finetune.py)
    ("multimodal/finetuned_vlm/LLaVA15_ft_predictions.csv",     "LLaVA15_ft_pred",     "LLaVA-1.5-7B (multilabel head FT)"),
    ("multimodal/finetuned_vlm/LLaVA_NEXT_ft_predictions.csv", "LLaVA_NEXT_ft_pred",  "LLaVA-NeXT-7B (multilabel head FT)"),
    ("multimodal/finetuned_vlm/BLIP2_ft_predictions.csv",      "BLIP2_ft_pred",       "BLIP-2 (multilabel head FT)"),
    ("multimodal/finetuned_vlm/PALO_ft_predictions.csv",       "PALO_ft_pred",        "PALO-7B (multilabel head FT)"),
    ("multimodal/finetuned_vlm/MiniCPM_V_ft_predictions.csv",   "MiniCPM_V_ft_pred",   "MiniCPM-V (multilabel head FT)"),
    ("multimodal/finetuned_vlm/IDEFICS_ft_predictions.csv",     "IDEFICS_ft_pred",     "IDEFICS-9B (multilabel head FT)"),
    ("multimodal/finetuned_vlm/InstructBLIP_ft_predictions.csv", "InstructBLIP_ft_pred", "InstructBLIP-Vicuna-7B (multilabel head FT)"),
    ("multimodal/finetuned_vlm/mBLIP_ft_predictions.csv",       "mBLIP_ft_pred",       "mBLIP-mT0-XL (multilabel head FT)"),
    ("multimodal/finetuned_vlm/BLIP_ft_predictions.csv",        "BLIP_ft_pred",        "BLIP-Base (multilabel head FT)"),
    ("multimodal/finetuned_vlm/Chitrarth_ft_predictions.csv",   "Chitrarth_ft_pred",   "Chitrarth (multilabel head FT)"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_label_dict(json_path):
    """Returns dict: sample_id -> binary list [0/1 x 7]."""
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
    """'TE-8.jpg' -> 'TE-8'"""
    return re.sub(r'\.(jpg|jpeg|png)$', '', image_id, flags=re.IGNORECASE)


def vec_to_str(vec):
    return "[" + ",".join(str(v) for v in vec) + "]"


def update_and_collect(csv_path, pred_col, label_dict, csv_root: Path, no_mutate: bool):
    """
    Reads csv, fills human_label in memory, optionally writes back, returns valid pairs.
    """
    path = csv_root / csv_path
    if not path.exists():
        print(f"  [SKIP] File not found: {csv_path}")
        return []

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames[:]
        rows = list(reader)

    if pred_col not in fieldnames:
        print(f"  [SKIP] Column '{pred_col}' not found in {csv_path}")
        return []

    valid_pairs = []
    filled = skipped_no_label = skipped_bad_pred = 0

    for row in rows:
        sample_id = image_id_to_sample_id(row.get("image_id", ""))
        if sample_id in label_dict:
            label_vec = label_dict[sample_id]
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
                except Exception:
                    skipped_bad_pred += 1
        else:
            skipped_no_label += 1

    if not no_mutate:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"  {csv_path}")
    print(f"    filled={filled}  valid_pairs={len(valid_pairs)}"
          f"  no_match={skipped_no_label}  bad_pred={skipped_bad_pred}")
    return valid_pairs


def compute_metrics(pairs):
    """Multi-label metrics from (y_true, y_pred) pairs."""
    per_label = []
    support_total = weighted_f1_sum = 0.0

    for j in range(7):
        tp = fp = fn = support = 0
        for yt, yp in pairs:
            t = int(yt[j])
            p = min(1, max(0, int(yp[j])))
            if t == 1:
                support += 1
                tp += 1 if p == 1 else 0
                fn += 0 if p == 1 else 1
            else:
                fp += 1 if p == 1 else 0

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_label.append({
            "symptom":   SYMPTOMS[j],
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   support,
        })
        support_total    += support
        weighted_f1_sum  += f1 * support

    macro_f1        = round(sum(pl["f1"]        for pl in per_label) / 7, 4)
    macro_precision = round(sum(pl["precision"] for pl in per_label) / 7, 4)
    macro_recall    = round(sum(pl["recall"]    for pl in per_label) / 7, 4)
    weighted_f1     = round(weighted_f1_sum / support_total if support_total > 0 else 0.0, 4)

    return {
        "n_samples":        len(pairs),
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "macro_precision":  macro_precision,
        "macro_recall":     macro_recall,
        "per_label":        per_label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Metrics for fine-tuned models; fills human_label on prediction CSVs unless --no-mutate.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root (contains test.json). Default: parent of 03_scripts/.",
    )
    p.add_argument(
        "--prediction-root",
        type=Path,
        default=None,
        help="Root for prediction CSV paths (same layout as repo: unimodal/finetuned/..., vision/...). "
        "Default: same as --repo-root. Use an isolated run directory to avoid touching prior CSVs.",
    )
    p.add_argument(
        "--metrics-out-dir",
        type=Path,
        default=None,
        help="Where to write finetuned_metrics_*.csv / .txt. Default: <repo-root>/metrics/finetuned",
    )
    p.add_argument(
        "--test-json",
        type=Path,
        default=None,
        help="Gold labels JSON. Default: <repo-root>/test.json",
    )
    p.add_argument(
        "--no-mutate-prediction-csvs",
        action="store_true",
        help="Do not write human_label back into prediction CSVs (metrics only).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    csv_root = (args.prediction_root or repo_root).resolve()
    out_dir = (args.metrics_out_dir or (repo_root / "metrics" / "finetuned")).resolve()
    json_path = (args.test_json or (repo_root / "test.json")).resolve()
    no_mutate = args.no_mutate_prediction_csvs

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Building human label dict from test.json")
    print("=" * 60)
    label_dict = build_label_dict(json_path)
    print(f"  Loaded {len(label_dict)} labeled samples\n")

    print("=" * 60)
    print("STEP 2: Filling human_label + collecting valid pairs")
    print("=" * 60)
    print(f"  prediction CSV root: {csv_root}")
    print(f"  metrics output dir:  {out_dir}")
    print(f"  mutate prediction CSVs: {not no_mutate}\n")

    all_results   = []
    all_per_label = []

    for csv_rel, pred_col, model_name in MODEL_CONFIGS:
        print(f"\n[{model_name}]")
        pairs = update_and_collect(csv_rel, pred_col, label_dict, csv_root, no_mutate)
        if len(pairs) < MIN_VALID_SAMPLES:
            print(f"  [SKIP metrics] Only {len(pairs)} valid pairs (< {MIN_VALID_SAMPLES})")
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

    # --- Save summary CSV ---
    summary_path = out_dir / "finetuned_metrics_summary.csv"
    cols = ["model","n_samples","macro_f1","weighted_f1","macro_precision","macro_recall"]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(all_results)
    print(f"\nSaved: {summary_path}")

    # --- Save per-label CSV ---
    per_label_path = out_dir / "finetuned_metrics_per_label.csv"
    cols2 = ["model","symptom","precision","recall","f1","support"]
    with open(per_label_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols2)
        w.writeheader(); w.writerows(all_per_label)
    print(f"Saved: {per_label_path}")

    # --- Save + print human-readable report ---
    report_path = out_dir / "finetuned_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FINE-TUNED MODELS — METRICS SUMMARY\n")
        f.write("=" * 85 + "\n")
        f.write(f"Ground truth: {json_path.name} ({len(label_dict)} samples)\n")
        f.write(f"Metrics: multilabel macro/weighted across 7 PHQ-9 symptoms\n\n")

        cw = [28, 10, 12, 14, 18, 14]
        hdr = ["Model","N Samples","Macro-F1","Weighted-F1","Macro-Precision","Macro-Recall"]
        f.write("  ".join(h.ljust(w) for h, w in zip(hdr, cw)) + "\n")
        f.write("-" * 100 + "\n")
        for r in sorted(all_results, key=lambda x: -x["macro_f1"]):
            row = [r["model"], str(r["n_samples"]), str(r["macro_f1"]),
                   str(r["weighted_f1"]), str(r["macro_precision"]), str(r["macro_recall"])]
            f.write("  ".join(v.ljust(w) for v, w in zip(row, cw)) + "\n")

        f.write("\n\nPER-SYMPTOM BREAKDOWN\n")
        f.write("=" * 85 + "\n")
        cw2 = [26, 10, 12, 10, 10]
        h2  = ["Symptom","Support","Precision","Recall","F1"]
        for r in all_results:
            f.write(f"\n{r['model']}\n")
            f.write("-" * 60 + "\n")
            f.write("  ".join(h.ljust(w) for h, w in zip(h2, cw2)) + "\n")
            for pl in [x for x in all_per_label if x["model"] == r["model"]]:
                row = [pl["symptom"], str(pl["support"]), str(pl["precision"]),
                       str(pl["recall"]), str(pl["f1"])]
                f.write("  ".join(v.ljust(w) for v, w in zip(row, cw2)) + "\n")

    print(f"Saved: {report_path}\n")
    with open(report_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
