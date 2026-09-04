"""
Evaluation helpers for 7-label multilabel classification.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _parse_one(pred) -> list[int]:
    if isinstance(pred, list) and len(pred) == 7:
        return [int(x) for x in pred]
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return [0] * 7

    s = str(pred).strip()
    # First try bracketed vector
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) == 7:
            return [int(x) for x in v]
    except Exception:
        pass

    # Try PREDICTION: [..]
    m = re.search(r"PREDICTION:\s*\[([01,\s]+)\]", s, flags=re.IGNORECASE)
    if m:
        arr = [x.strip() for x in m.group(1).split(",")]
        if len(arr) == 7 and all(x in ("0", "1") for x in arr):
            return [int(x) for x in arr]

    # Fallback: first 7 binary digits in text
    bits = re.findall(r"\b[01]\b", s)
    if len(bits) >= 7:
        return [int(x) for x in bits[:7]]
    return [0] * 7


def parse_predictions(predictions_list) -> np.ndarray:
    return np.array([_parse_one(x) for x in predictions_list], dtype=int)


def calculate_metrics(y_true, y_pred, symptom_names):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch y_true={y_true.shape}, y_pred={y_pred.shape}")

    per = []
    for i, name in enumerate(symptom_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        per.append(
            {
                "symptom": name,
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "support": int(yt.sum()),
                "predicted_positive": int(yp.sum()),
            }
        )

    out = {
        "n_samples": int(y_true.shape[0]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_symptom": per,
    }
    return out


def print_results_table(metrics_dict):
    print(f"n={metrics_dict['n_samples']}")
    print(
        f"Macro-F1={metrics_dict['macro_f1']:.4f} | "
        f"Weighted-F1={metrics_dict['weighted_f1']:.4f} | "
        f"Macro-P={metrics_dict['macro_precision']:.4f} | "
        f"Macro-R={metrics_dict['macro_recall']:.4f}"
    )
    print("-" * 80)
    print(f"{'Symptom':30} {'P':>7} {'R':>7} {'F1':>7} {'Gold+':>8} {'Pred+':>8}")
    print("-" * 80)
    for r in metrics_dict["per_symptom"]:
        print(
            f"{r['symptom'][:30]:30} "
            f"{r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f} "
            f"{r['support']:8d} {r['predicted_positive']:8d}"
        )


def save_metrics_csv(metrics_dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_dict["per_symptom"]).to_csv(output_path, index=False)
