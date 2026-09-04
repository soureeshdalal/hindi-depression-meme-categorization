#!/usr/bin/env python3
"""
Zero-shot (non-fine-tuned) unimodal image-only baseline runner.

Models:
1) ViT-B/16      -> CLIP ViT-B/16 zero-shot scoring
2) ResNet-50     -> CLIP RN50 zero-shot scoring
3) EfficientNet-B0 -> ImageNet classifier + keyword heuristic projection to 7 symptoms

Outputs (append-only / no-overwrite):
- per-model predictions CSV
- per-model metrics JSON
- run summary CSV
- optional report markdown append
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
)

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

CLIP_MODEL_CONFIGS = [
    ("ViT-B/16", "openai/clip-vit-base-patch16"),
    ("ResNet-50", "openai/clip-rn50"),
]

EFF_NAME = "EfficientNet-B0"
EFF_MODEL_ID = "google/efficientnet-b0"

# Paired prompts for binary per-symptom decision (positive vs negative).
CLIP_PROMPT_PAIRS = {
    "Feeling Down": (
        "A meme expressing sadness, depression, hopeless mood, emotional pain.",
        "A meme not expressing sadness or depression; neutral or positive mood.",
    ),
    "Lack of Interest": (
        "A meme expressing apathy, no interest, no motivation, emotional numbness.",
        "A meme showing normal interest, engagement, motivation, or enthusiasm.",
    ),
    "Self-Harm": (
        "A meme expressing self-harm, suicide thoughts, self-injury, or wanting to die.",
        "A meme not expressing self-harm or suicide thoughts.",
    ),
    "Eating Disorder": (
        "A meme about disordered eating, starvation, bingeing, body-image eating issues.",
        "A meme not about disordered eating behavior.",
    ),
    "Low Self-Esteem": (
        "A meme expressing low self-worth, insecurity, self-hate, or worthlessness.",
        "A meme expressing confidence, self-worth, or neutral self-image.",
    ),
    "Concentration Problem": (
        "A meme expressing poor focus, overthinking, mental distraction, concentration issues.",
        "A meme not expressing concentration or focus problems.",
    ),
    "Sleeping Disorder": (
        "A meme expressing insomnia, poor sleep, sleep problems, disturbed sleep.",
        "A meme not expressing sleep problems.",
    ),
}

# Heuristic mapping from ImageNet label text to symptom evidence.
EFFICIENTNET_KEYWORDS = {
    "Feeling Down": {"cry", "tear", "gloom", "sad", "grave", "coffin"},
    "Lack of Interest": {"sloth", "couch", "sleeping", "lazy"},
    "Self-Harm": {"knife", "blade", "gun", "bullet", "blood", "syringe"},
    "Eating Disorder": {"food", "plate", "restaurant", "pizza", "burger", "spoon", "fork"},
    "Low Self-Esteem": {"mirror", "mask", "costume", "rag", "beggar"},
    "Concentration Problem": {"maze", "labyrinth", "puzzle", "book", "screen", "keyboard"},
    "Sleeping Disorder": {"bed", "pajama", "night", "pillow", "clock", "sleep"},
}


def safe_write_json(path: Path, data: dict) -> bool:
    if path.exists():
        print(f"[SKIP WRITE] Exists: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


def safe_write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> bool:
    if path.exists():
        print(f"[SKIP WRITE] Exists: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def load_ground_truth(test_json: Path) -> Dict[str, List[int]]:
    with open(test_json, encoding="utf-8") as f:
        data = json.load(f)
    label_dict = {}
    for entry in data:
        sid = str(entry["sample_id"])
        vec = [0] * len(SYMPTOMS)
        for cat in entry.get("meme_depressive_categories", []):
            idx = LABEL_TO_IDX.get(cat)
            if idx is not None:
                vec[idx] = 1
        label_dict[sid] = vec
    return label_dict


def collect_test_images(data_dir: Path) -> List[dict]:
    rows = []
    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        test_dir = cat_dir / "test"
        if not test_dir.exists():
            continue
        for img in sorted(test_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            sid = img.stem
            rows.append(
                {
                    "sample_id": sid,
                    "image_id": img.name,
                    "category": cat_dir.name,
                    "image_path": str(img),
                }
            )
    return rows


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, mode: str) -> Tuple[float, np.ndarray]:
    if mode == "fixed":
        thr = 0.5
        y_pred = (y_score >= thr).astype(int)
        return thr, y_pred
    best = (-1.0, 0.5, None)
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        y_pred = (y_score >= thr).astype(int)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if mf1 > best[0]:
            best = (mf1, thr, y_pred)
    return float(best[1]), best[2]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_label = []
    for i, name in enumerate(SYMPTOMS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        per_label.append(
            {
                "label": name,
                "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
                "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
                "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
                "support": int(np.sum(yt)),
            }
        )
    return {
        "n": int(y_true.shape[0]),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_label": per_label,
    }


def clip_scores_for_image(model, processor, image: Image.Image, device: torch.device) -> List[float]:
    scores = []
    for symptom in SYMPTOMS:
        pos, neg = CLIP_PROMPT_PAIRS[symptom]
        inputs = processor(text=[pos, neg], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits_per_image[0]
            probs = torch.softmax(logits, dim=0)
            scores.append(float(probs[0].detach().cpu().item()))
    return scores


def efficientnet_scores_for_image(model, processor, image: Image.Image, id2label: Dict[int, str], device: torch.device) -> List[float]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    top_idx = np.argsort(-probs)[:30]
    top_items = [(id2label[int(i)].lower(), float(probs[int(i)])) for i in top_idx]
    out = []
    for symptom in SYMPTOMS:
        keys = EFFICIENTNET_KEYWORDS[symptom]
        score = 0.0
        for label_txt, p in top_items:
            if any(k in label_txt for k in keys):
                score += p
        out.append(float(min(1.0, score)))
    return out


def append_report(report_path: Path, run_name: str, rows: List[dict]) -> None:
    lines = []
    lines.append("")
    lines.append(f"#### Unimodal Image Zero-shot Baseline ({run_name})")
    lines.append("")
    lines.append("| Model | n | Threshold | Macro-F1 | Micro-F1 | Macro-Precision | Macro-Recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['n']} | {r['threshold']:.2f} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} | {r['macro_precision']:.4f} | {r['macro_recall']:.4f} |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run non-fine-tuned unimodal image baseline on test set")
    parser.add_argument("--data_dir", default="translated_categorized_memes")
    parser.add_argument("--test_json", default="test.json")
    parser.add_argument("--output_dir", default="zeroshot_unimodal_image")
    parser.add_argument("--report_md", default="01_documentation/UNIMODAL_MULTIMODAL_RESULTS_REPORT.md")
    parser.add_argument("--threshold_mode", choices=["fixed", "best_on_test"], default="best_on_test")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", default="all", choices=["all", "vit", "resnet", "efficientnet"])
    args = parser.parse_args()

    root = Path(".").resolve()
    data_dir = root / args.data_dir
    test_json = root / args.test_json
    out_dir = root / args.output_dir
    report_md = root / args.report_md

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gt = load_ground_truth(test_json)
    images = collect_test_images(data_dir)

    # Keep only test images in the ground-truth file.
    images = [x for x in images if x["sample_id"] in gt]
    if len(images) != 650:
        print(f"[WARN] Expected 650 matched test images, found {len(images)}")
    else:
        print("[OK] Found 650 matched test images.")

    y_true = np.array([gt[x["sample_id"]] for x in images], dtype=int)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    def run_one_clip(display_name: str, model_id: str, key: str):
        pred_csv = run_dir / f"{key}_predictions.csv"
        metrics_json = run_dir / f"{key}_metrics.json"
        if pred_csv.exists() or metrics_json.exists():
            print(f"[SKIP] Existing files for {display_name} in {run_dir}")
            return
        print(f"\n=== {display_name} ({model_id}) ===")
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        score_rows = []
        score_matrix = []
        for item in tqdm(images, desc=f"{display_name} infer"):
            img = Image.open(item["image_path"]).convert("RGB")
            scores = clip_scores_for_image(model, processor, img, device)
            score_matrix.append(scores)
            row = {**item}
            for i, s in enumerate(scores):
                row[f"score_{i}"] = round(float(s), 6)
            score_rows.append(row)
        y_score = np.array(score_matrix, dtype=float)
        thr, y_pred = choose_threshold(y_true, y_score, "fixed" if args.threshold_mode == "fixed" else "best_on_test")
        for r, yp in zip(score_rows, y_pred.tolist()):
            for i, b in enumerate(yp):
                r[f"label_{i}"] = int(b)
        pred_fields = ["sample_id", "image_id", "category", "image_path"] + [f"score_{i}" for i in range(7)] + [f"label_{i}" for i in range(7)]
        safe_write_csv(pred_csv, score_rows, pred_fields)
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = display_name
        metrics["model_id"] = model_id
        metrics["threshold"] = thr
        metrics["threshold_mode"] = args.threshold_mode
        safe_write_json(metrics_json, metrics)
        summary_rows.append(
            {
                "model": display_name,
                "n": metrics["n"],
                "threshold": thr,
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "predictions_csv": str(pred_csv),
                "metrics_json": str(metrics_json),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_one_efficientnet():
        pred_csv = run_dir / "efficientnet_b0_predictions.csv"
        metrics_json = run_dir / "efficientnet_b0_metrics.json"
        if pred_csv.exists() or metrics_json.exists():
            print(f"[SKIP] Existing files for {EFF_NAME} in {run_dir}")
            return
        print(f"\n=== {EFF_NAME} ({EFF_MODEL_ID}) ===")
        model = AutoModelForImageClassification.from_pretrained(EFF_MODEL_ID).to(device)
        processor = AutoImageProcessor.from_pretrained(EFF_MODEL_ID)
        id2label = model.config.id2label
        score_rows = []
        score_matrix = []
        for item in tqdm(images, desc=f"{EFF_NAME} infer"):
            img = Image.open(item["image_path"]).convert("RGB")
            scores = efficientnet_scores_for_image(model, processor, img, id2label, device)
            score_matrix.append(scores)
            row = {**item}
            for i, s in enumerate(scores):
                row[f"score_{i}"] = round(float(s), 6)
            score_rows.append(row)
        y_score = np.array(score_matrix, dtype=float)
        thr, y_pred = choose_threshold(y_true, y_score, "fixed" if args.threshold_mode == "fixed" else "best_on_test")
        for r, yp in zip(score_rows, y_pred.tolist()):
            for i, b in enumerate(yp):
                r[f"label_{i}"] = int(b)
        pred_fields = ["sample_id", "image_id", "category", "image_path"] + [f"score_{i}" for i in range(7)] + [f"label_{i}" for i in range(7)]
        safe_write_csv(pred_csv, score_rows, pred_fields)
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = EFF_NAME
        metrics["model_id"] = EFF_MODEL_ID
        metrics["threshold"] = thr
        metrics["threshold_mode"] = args.threshold_mode
        metrics["note"] = "ImageNet logits projected to symptoms by keyword heuristic."
        safe_write_json(metrics_json, metrics)
        summary_rows.append(
            {
                "model": EFF_NAME,
                "n": metrics["n"],
                "threshold": thr,
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "predictions_csv": str(pred_csv),
                "metrics_json": str(metrics_json),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model in {"all", "vit"}:
        run_one_clip(CLIP_MODEL_CONFIGS[0][0], CLIP_MODEL_CONFIGS[0][1], "vit_b16_clip")
    if args.model in {"all", "resnet"}:
        run_one_clip(CLIP_MODEL_CONFIGS[1][0], CLIP_MODEL_CONFIGS[1][1], "resnet50_clip")
    if args.model in {"all", "efficientnet"}:
        run_one_efficientnet()

    if not summary_rows:
        print("\nNo model outputs generated.")
        return

    # Save run summary (append-safe: run dir is timestamped unique).
    summary_csv = run_dir / "summary.csv"
    safe_write_csv(
        summary_csv,
        summary_rows,
        [
            "model",
            "n",
            "threshold",
            "macro_f1",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "predictions_csv",
            "metrics_json",
        ],
    )
    append_report(report_md, run_name, summary_rows)
    print(f"\nDone. Outputs in: {run_dir}")
    print(f"Report appended: {report_md}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
