#!/usr/bin/env python3
# === COLAB SETUP ===
# 1. Mount Google Drive
# 2. Set BASE_DIR to your project root on Drive
# 3. pip install transformers scikit-learn pandas torch easyocr tqdm sentencepiece protobuf
"""
Fine-tune 8 text encoders on EasyOCR text from translated_train/, evaluate on the 581-sample
test subset (ocr / explanation / ocr+explanation). Does not use train.json ocr_text (English).

Outputs: unimodal/train_ocr_cache.json, unimodal/finetuned_581/*_ft.pt, *_predictions.csv, metrics_summary.csv
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import ssl
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from multimodal.mama_finetune_common import assert_train_json_not_test_gold

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_JSON = BASE_DIR / "train.json"
TRAIN_IMAGE_DIR = BASE_DIR / "translated_train"
OCR_CACHE_PATH = BASE_DIR / "unimodal" / "train_ocr_cache.json"
EXPL_CSV = BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv"
METADATA_CSV = BASE_DIR / "colab_experiments" / "data" / "metadata.csv"
OUT_DIR = BASE_DIR / "unimodal" / "finetuned_581"

HF_TOKEN = os.environ.get("HF_TOKEN")

TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_LABELS = 7
TRAIN_MAX_LENGTH = 128
INFER_MAX_LENGTH = 512
SEED = 42
EXPECTED_TEST_N = 581

SYMPTOM_NAMES = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

# (hf_id, folder_name, display_name, fixed_threshold, use_adaptive) — mirror run_unimodal_581.py
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

INPUT_MODES_EVAL = (
    ("ocr", "ocr_text_input"),
    ("explanation", "explanation_input"),
    ("ocr+explanation", "ocr_plus_explanation"),
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class FinetuneTextClassifier(nn.Module):
    """
    Same encoder + pooling as run_unimodal_581.UnimodalClassifier; returns logits for
    BCEWithLogitsLoss (no sigmoid in forward).
    """

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

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_hs = getattr(out, "encoder_last_hidden_state", None)
        cls = enc_hs[:, 0, :] if enc_hs is not None else out.last_hidden_state[:, 0, :]
        if cls.dtype != self.classifier.weight.dtype:
            cls = cls.to(dtype=self.classifier.weight.dtype)
        return self.classifier(cls)


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


def run_inference_batched_probs(
    model, tokenizer, texts, batch_size, fixed_threshold, use_adaptive
):
    """Sigmoid(logits) then same fixed + adaptive thresholding as run_unimodal_581.py."""
    model.eval()
    all_probs = []
    t0 = time.time()
    n = len(texts)

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        try:
            enc = tokenizer(
                batch,
                max_length=INFER_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                )
                probs = torch.sigmoid(logits).cpu().float().numpy()
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


def find_train_image(base: Path, sid: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = base / f"{sid}{ext}"
        if p.exists():
            return p
    return None


def build_or_load_ocr_cache(train_json_path: Path, image_dir: Path, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"[OCR cache] loading existing (no EasyOCR) → {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    assert_train_json_not_test_gold(train_json_path)
    with open(train_json_path) as f:
        data = json.load(f)

    seen = set()
    jobs = []
    for item in data:
        sid = (item.get("sample_id") or "").strip()
        if not sid or sid in seen:
            continue
        img = find_train_image(image_dir, sid)
        if img is None:
            continue
        seen.add(sid)
        jobs.append((sid, str(img)))

    print(f"[OCR cache] {len(jobs)} images to OCR (EasyOCR hi,en) …")
    reader = easyocr.Reader(["hi", "en"], gpu=(str(DEVICE).startswith("cuda")))
    cache = {}
    for sid, path in tqdm(jobs, desc="OCR train"):
        try:
            results = reader.readtext(path, detail=0)
            text = " ".join(results).strip()
            cache[sid] = text if text else "[NO_TEXT]"
        except Exception as e:
            print(f"  [OCR train error] {path}: {e}", file=sys.stderr)
            cache[sid] = "[OCR_ERROR]"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"[OCR cache] wrote {len(cache)} entries → {cache_path}")
    return cache


def aggregate_train_labels(train_json_path: Path) -> dict[str, list[float]]:
    assert_train_json_not_test_gold(train_json_path)
    with open(train_json_path) as f:
        data = json.load(f)
    agg: dict[str, list[float]] = {}
    for item in data:
        sid = (item.get("sample_id") or "").strip()
        if not sid:
            continue
        vec = [0.0] * NUM_LABELS
        for cat in item.get("meme_depressive_categories", []):
            if cat in SYMPTOM_NAMES:
                vec[SYMPTOM_NAMES.index(cat)] = 1.0
        if sid not in agg:
            agg[sid] = [0.0] * NUM_LABELS
        for i in range(NUM_LABELS):
            agg[sid][i] = max(agg[sid][i], vec[i])
    return agg


def build_training_arrays(ocr_cache: dict, label_map: dict[str, list[float]]):
    sids = sorted(set(ocr_cache.keys()) & set(label_map.keys()))
    texts = [ocr_cache[s] for s in sids]
    labels = np.array([label_map[s] for s in sids], dtype=np.float32)
    return sids, texts, labels


def print_label_stats(labels: np.ndarray):
    print(f"  Total training samples: {len(labels)}")
    print("  Positive count per symptom:")
    for i, name in enumerate(SYMPTOM_NAMES):
        c = int(labels[:, i].sum())
        print(f"    {name}: {c}")


def split_train_val(texts: list[str], labels: np.ndarray, seed: int = SEED):
    n = len(texts)
    val_n = max(1, int(0.1 * n)) if n >= 20 else 0
    if val_n == 0 or val_n >= n:
        return texts, labels, [], np.zeros((0, NUM_LABELS), dtype=np.float32)

    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]
    tr_texts = [texts[i] for i in tr_idx]
    tr_y = labels[tr_idx]
    va_texts = [texts[i] for i in val_idx]
    va_y = labels[val_idx]
    return tr_texts, tr_y, va_texts, va_y


def eval_bce_loss(model, tokenizer, texts, labels, device) -> float:
    if len(texts) == 0:
        return float("inf")
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total, cnt = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(texts), TRAIN_BATCH_SIZE):
            batch_texts = texts[start : start + TRAIN_BATCH_SIZE]
            batch_y = torch.tensor(labels[start : start + TRAIN_BATCH_SIZE], dtype=torch.float32).to(
                device
            )
            enc = tokenizer(
                batch_texts,
                max_length=TRAIN_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_fn(logits, batch_y)
            total += loss.item() * len(batch_texts)
            cnt += len(batch_texts)
    model.train()
    return total / max(cnt, 1)


def fine_tune_model(model, tokenizer, train_texts, train_labels, val_texts, val_labels, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train_texts)
    use_val_best = len(val_texts) > 0
    best_state = None
    best_val = float("inf")

    for epoch in range(EPOCHS):
        indices = list(range(n))
        random.shuffle(indices)
        bar = tqdm(
            range(0, n, TRAIN_BATCH_SIZE),
            desc=f"  Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
        )
        epoch_loss = 0.0
        nb = 0
        for start in bar:
            batch_idx = indices[start : start + TRAIN_BATCH_SIZE]
            batch_texts = [train_texts[i] for i in batch_idx]
            batch_y = torch.tensor(train_labels[batch_idx], dtype=torch.float32).to(device)

            try:
                enc = tokenizer(
                    batch_texts,
                    max_length=TRAIN_MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                optimizer.zero_grad()
                logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                nb += 1
                bar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                print(f"\n  [Train batch error] {e}", file=sys.stderr)

        avg_tr = epoch_loss / max(nb, 1)
        print(f"  Epoch {epoch + 1}/{EPOCHS} — avg train loss: {avg_tr:.4f}")

        if use_val_best:
            vloss = eval_bce_loss(model, tokenizer, val_texts, val_labels, device)
            print(f"  Val BCE: {vloss:.4f}")
            if vloss < best_val:
                best_val = vloss
                best_state = copy.deepcopy(model.state_dict())

    if use_val_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Loaded best val checkpoint (val BCE={best_val:.4f})")
    model.eval()


def load_test_dataframe() -> pd.DataFrame:
    expl = pd.read_csv(EXPL_CSV)
    valid = expl["explanation"].notna() & (expl["explanation"].astype(str).str.strip() != "")
    expl = expl.loc[valid].copy()
    meta = pd.read_csv(METADATA_CSV)[["image_id", "ocr_text"]]
    df = expl.merge(meta, on="image_id", how="inner")
    if len(df) != EXPECTED_TEST_N:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_N} test rows after join; got {len(df)}"
        )
    df["ocr_text_input"] = df["ocr_text"].fillna("").astype(str).str.strip()
    df["explanation_input"] = df["explanation"].fillna("").astype(str).str.strip()
    df["ocr_plus_explanation"] = (
        df["ocr_text_input"] + " [SEP] " + df["explanation_input"]
    )
    return df


def predictions_path(out_dir: Path, folder: str, input_mode: str) -> Path:
    return out_dir / f"{folder}_{input_mode}_predictions.csv"


def checkpoint_path(out_dir: Path, folder: str) -> Path:
    return out_dir / f"{folder}_ft.pt"


def run_eval_for_mode(
    model,
    tokenizer,
    df: pd.DataFrame,
    input_mode: str,
    col_name: str,
    folder: str,
    display_name: str,
    fixed_threshold: float,
    use_adaptive: bool,
    out_dir: Path,
):
    out_csv = predictions_path(out_dir, folder, input_mode)
    if out_csv.exists():
        print(f"  Skip predictions (exists): {out_csv.name}")
        return

    texts = df[col_name].tolist()
    gold_col = df["gold_labels"].tolist()
    preds_fixed, _, raw_scores = run_inference_batched_probs(
        model,
        tokenizer,
        texts,
        INFER_BATCH_SIZE,
        fixed_threshold,
        use_adaptive,
    )

    out_df = pd.DataFrame(
        {
            "image_id": df["image_id"].tolist(),
            "category": df["category"].tolist(),
            "gold_labels": gold_col,
            "ocr_text": df["ocr_text_input"].tolist(),
            "explanation": df["explanation_input"].tolist(),
            "predicted_labels": preds_fixed,
            "raw_scores": raw_scores,
        }
    )
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv.name}")

    m = compute_metrics(gold_col, preds_fixed)
    if m:
        print(
            f"    [{input_mode}] macro_f1={m['macro_f1']} weighted_f1={m['weighted_f1']} "
            f"P={m['macro_precision']} R={m['macro_recall']}"
        )


def build_metrics_summary(out_dir: Path) -> pd.DataFrame:
    rows = []
    for model_id, folder, display_name, _, _ in MODEL_CONFIGS:
        for input_mode, _ in INPUT_MODES_EVAL:
            p = predictions_path(out_dir, folder, input_mode)
            if not p.exists():
                continue
            sub = pd.read_csv(p)
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
    p = argparse.ArgumentParser(description="Fine-tune 8 text models; eval on 581 subset (3 input modes).")
    p.add_argument("--base-dir", type=Path, default=None, help="Project root (overrides default BASE_DIR).")
    p.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="Path to train.json (default: BASE_DIR/train.json).",
    )
    p.add_argument(
        "--train-image-dir",
        type=Path,
        default=None,
        help="Translated train images (default: BASE_DIR/translated_train).",
    )
    return p.parse_args()


def main():
    global BASE_DIR, TRAIN_JSON, TRAIN_IMAGE_DIR, OCR_CACHE_PATH, EXPL_CSV, METADATA_CSV, OUT_DIR

    args = parse_args()
    if args.base_dir is not None:
        BASE_DIR = args.base_dir.resolve()
        TRAIN_JSON = BASE_DIR / "train.json"
        TRAIN_IMAGE_DIR = BASE_DIR / "translated_train"
        OCR_CACHE_PATH = BASE_DIR / "unimodal" / "train_ocr_cache.json"
        EXPL_CSV = BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv"
        METADATA_CSV = BASE_DIR / "colab_experiments" / "data" / "metadata.csv"
        OUT_DIR = BASE_DIR / "unimodal" / "finetuned_581"

    if args.train_json is not None:
        TRAIN_JSON = args.train_json.resolve()
    if args.train_image_dir is not None:
        TRAIN_IMAGE_DIR = args.train_image_dir.resolve()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("UNIMODAL FINE-TUNE (train OCR) + EVAL (581 × 3 modes)")
    print(f"  BASE_DIR      : {BASE_DIR}")
    print(f"  Device        : {DEVICE}")
    print(f"  Train JSON    : {TRAIN_JSON}")
    print(f"  Train images  : {TRAIN_IMAGE_DIR}")
    print(f"  OCR cache     : {OCR_CACHE_PATH}")
    print(f"  Checkpoints   : {OUT_DIR}")
    print("=" * 70)

    ocr_cache = build_or_load_ocr_cache(TRAIN_JSON, TRAIN_IMAGE_DIR, OCR_CACHE_PATH)

    label_map = aggregate_train_labels(TRAIN_JSON)
    _, train_texts, train_labels = build_training_arrays(ocr_cache, label_map)
    print("\n[Training set]")
    print_label_stats(train_labels)

    test_df = load_test_dataframe()
    print(f"\n[Test subset] {len(test_df)} rows\n")

    assert len(MODEL_CONFIGS) == 8
    assert len(SYMPTOM_NAMES) == 7

    for model_id, folder, display_name, fixed_threshold, use_adaptive in MODEL_CONFIGS:
        ckpt = checkpoint_path(OUT_DIR, folder)
        print(f"\n{'='*60}\n[MODEL] {display_name} ({model_id})\n{'='*60}")

        model = None
        tokenizer = None

        if ckpt.exists():
            print(f"  Skip fine-tuning (checkpoint exists): {ckpt.name}")
        else:
            tr_tx, tr_y, va_tx, va_y = split_train_val(train_texts, train_labels)
            print(
                f"  Train/val split: {len(tr_tx)} train, {len(va_tx)} val "
                f"(val best={'on' if len(va_tx) else 'off'})"
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, token=HF_TOKEN
            )
            model = FinetuneTextClassifier(model_id, NUM_LABELS, HF_TOKEN).to(DEVICE)

            fine_tune_model(model, tokenizer, tr_tx, tr_y, va_tx, va_y, DEVICE)

            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint → {ckpt}")

        for input_mode, col in INPUT_MODES_EVAL:
            pred_p = predictions_path(OUT_DIR, folder, input_mode)
            if pred_p.exists():
                continue

            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True, token=HF_TOKEN
                )
                model = FinetuneTextClassifier(model_id, NUM_LABELS, HF_TOKEN).to(DEVICE)
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                model.eval()

            print(f"\n  [Eval] {input_mode}")
            run_eval_for_mode(
                model,
                tokenizer,
                test_df,
                input_mode,
                col,
                folder,
                display_name,
                fixed_threshold,
                use_adaptive,
                OUT_DIR,
            )

        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = OUT_DIR / "metrics_summary.csv"
    if summary_path.exists():
        print(
            f"\n{summary_path.name} already exists — not overwriting.",
            file=sys.stderr,
        )
    else:
        sdf = build_metrics_summary(OUT_DIR)
        if sdf.empty:
            print("No prediction files found for metrics summary.", file=sys.stderr)
        else:
            sdf.to_csv(summary_path, index=False)
            print(f"\nWrote {summary_path}")
            print(sdf.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
