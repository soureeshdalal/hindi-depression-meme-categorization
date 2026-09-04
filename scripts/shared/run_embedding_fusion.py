#!/usr/bin/env python3
"""
Embedding Fusion pipeline for Hindi mental health meme classification.

Architecture:
  ResNet-50 (frozen, 2048-dim) + MuRIL (frozen, 768-dim)
  → concat (2816-dim) → MLP fusion head → 7-symptom binary predictions

Training:
  5-fold MultilabelStratifiedKFold cross-validation
  Only the fusion MLP trains; both encoders stay frozen

Output:
  /mnt/user-data/outputs/embedding_fusion/
  ├── embeddings/   (cached .npy)
  ├── models/       (per-fold checkpoints)
  ├── results/      (CSVs + confusion matrices)
  └── logs/         (training_log.txt)
"""

import json
import logging
import os
import random
import ssl
import sys
import warnings
from pathlib import Path

# Allow model downloads on environments with strict/missing SSL certs
if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False
    warnings.warn(
        "iterstrat not found — falling back to KFold.\n"
        "Install with: pip install iterative-stratification"
    )
    from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Paths — set RUN_ENV = "local" or "aws"
# ---------------------------------------------------------------------------
RUN_ENV = os.environ.get("RUN_ENV", "local")

if RUN_ENV == "aws":
    CSV_PATH   = "/mnt/user-data/uploads/BART_Large_predictions.csv"
    IMAGE_BASE = "/mnt/user-data/uploads/translated_categorized_memes"
    OUTPUT_DIR = Path("/mnt/user-data/outputs/embedding_fusion")
else:
    # Local Mac paths (relative to project root)
    _ROOT      = Path(__file__).parent
    CSV_PATH   = str(_ROOT / "unimodal" / "BART_Large_predictions.csv")
    IMAGE_BASE = str(_ROOT / "translated_categorized_memes")
    OUTPUT_DIR = _ROOT / "embedding_fusion_output"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMPTOMS = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

SYMPTOM_SLUGS = [
    "feeling_down", "lack_of_interest", "self_harm",
    "eating_disorder", "low_self_esteem", "concentration_problem",
    "sleeping_disorder",
]

IMAGE_DIM  = 2048
TEXT_DIM   = 768
NUM_LABELS = 7
SEED       = 42

# Training hypers
EPOCHS      = 50
PATIENCE    = 10
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY= 1e-5
N_FOLDS     = 5
THRESHOLD   = 0.5

# Embedding extraction batch sizes
# Smaller on 8 GB unified RAM (M2 MacBook Air); larger on AWS A10G
IMG_BATCH  = 16 if RUN_ENV == "local" else 64
TEXT_BATCH = 4  if RUN_ENV == "local" else 32


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("embedding_fusion")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_label(label_str):
    """Parse '[1,0,0,0,0,0,0]' → numpy array of length 7."""
    label_str = str(label_str).strip()
    try:
        vals = json.loads(label_str)
        arr = np.array(vals, dtype=np.float32)
        if len(arr) == NUM_LABELS:
            return arr
    except Exception:
        pass
    # fallback: extract digits
    digits = [int(c) for c in label_str if c in "01"]
    if len(digits) == NUM_LABELS:
        return np.array(digits, dtype=np.float32)
    return np.zeros(NUM_LABELS, dtype=np.float32)


def load_data(csv_path: str, image_base: str, logger) -> pd.DataFrame:
    """Load CSV, parse labels, build image paths, verify files exist."""
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Build image paths
    df["image_path"] = df.apply(
        lambda r: str(Path(image_base) / r["category"] / "test" / r["image_id"]),
        axis=1,
    )

    # Parse labels
    df["label_array"] = df["human_label"].apply(parse_label)

    # Validate
    missing = df[~df["image_path"].apply(os.path.exists)]
    if len(missing):
        logger.warning(f"  {len(missing)} images not found — will use zero embedding")

    no_label = df[df["label_array"].apply(lambda x: x.sum() == 0 and
                  str(df.loc[df["label_array"].apply(id) == id(x), "human_label"].values[0]).count("1") == 0)]

    df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
    df["ocr_text"] = df["ocr_text"].replace("", "[PAD]")

    logger.info(f"  {len(df)} samples | {len(missing)} missing images")
    logger.info(f"  Categories: {sorted(df['category'].unique())}")

    # Label stats
    label_matrix = np.vstack(df["label_array"].values)
    for i, s in enumerate(SYMPTOMS):
        pos = int(label_matrix[:, i].sum())
        logger.info(f"    {s}: {pos} positive ({100*pos/len(df):.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_resnet_embeddings(df: pd.DataFrame, device: torch.device,
                               batch_size: int = IMG_BATCH) -> np.ndarray:
    """Extract frozen ResNet-50 avgpool features → [N, 2048]."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Drop final FC, keep avgpool output [batch, 2048, 1, 1]
    extractor = nn.Sequential(*list(backbone.children())[:-1])
    extractor.eval().to(device)
    for p in extractor.parameters():
        p.requires_grad = False

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_embs = []
    n = len(df)

    for start in tqdm(range(0, n, batch_size), desc="ResNet-50 embeddings"):
        batch_rows = df.iloc[start: start + batch_size]
        tensors = []
        for _, row in batch_rows.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                tensors.append(tfm(img))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = extractor(batch)          # [B, 2048, 1, 1]
            feats = feats.flatten(1)          # [B, 2048]
        all_embs.append(feats.cpu().numpy())

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return np.vstack(all_embs).astype(np.float32)


def extract_muril_embeddings(df: pd.DataFrame, device: torch.device,
                              batch_size: int = TEXT_BATCH) -> np.ndarray:
    """Extract frozen MuRIL CLS token embeddings → [N, 768]."""
    model_name = "google/muril-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    all_embs = []
    n = len(df)

    for start in tqdm(range(0, n, batch_size), desc="MuRIL embeddings"):
        batch_rows = df.iloc[start: start + batch_size]
        texts = []
        for _, row in batch_rows.iterrows():
            t = str(row.get("ocr_text", "") or "").strip()
            texts.append(t if t else "[PAD]")

        enc = tokenizer(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]    # [B, 768]
        all_embs.append(cls.cpu().numpy())

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return np.vstack(all_embs).astype(np.float32)


def extract_embeddings(df: pd.DataFrame, emb_dir: Path,
                        device: torch.device, logger) -> tuple:
    """
    Extract (or reload cached) ResNet-50 + MuRIL embeddings.
    Returns (image_embs [N,2048], text_embs [N,768]).
    """
    emb_dir.mkdir(parents=True, exist_ok=True)
    img_path  = emb_dir / "resnet50_embeddings.npy"
    txt_path  = emb_dir / "muril_embeddings.npy"
    meta_path = emb_dir / "metadata.json"

    if img_path.exists() and txt_path.exists() and meta_path.exists():
        logger.info("Reusing cached embeddings from disk.")
        image_embs = np.load(img_path)
        text_embs  = np.load(txt_path)
        return image_embs, text_embs

    logger.info("Extracting ResNet-50 image embeddings...")
    image_embs = extract_resnet_embeddings(df, device)
    np.save(img_path, image_embs)
    logger.info(f"  Saved → {img_path}  shape={image_embs.shape}")

    logger.info("Extracting MuRIL text embeddings...")
    text_embs = extract_muril_embeddings(df, device)
    np.save(txt_path, text_embs)
    logger.info(f"  Saved → {txt_path}  shape={text_embs.shape}")

    meta = {img_id: idx for idx, img_id in enumerate(df["image_id"].tolist())}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata saved → {meta_path}")

    return image_embs, text_embs


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """
    3-layer MLP fusion head.
    Input:  concat(ResNet-50, MuRIL) = 2816-dim
    Output: 7 sigmoid probabilities
    """

    def __init__(self, image_dim=IMAGE_DIM, text_dim=TEXT_DIM,
                 num_labels=NUM_LABELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim + text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-symptom + overall precision / recall / F1."""
    metrics = {}
    for i, sym in enumerate(SYMPTOMS):
        metrics[sym] = {
            "precision": float(precision_score(y_true[:, i], y_pred[:, i],
                                               zero_division=0)),
            "recall":    float(recall_score(y_true[:, i], y_pred[:, i],
                                            zero_division=0)),
            "f1":        float(f1_score(y_true[:, i], y_pred[:, i],
                                        zero_division=0)),
        }
    # Overall macro
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                               zero_division=0))
    macro_p  = float(precision_score(y_true, y_pred, average="macro",
                                     zero_division=0))
    macro_r  = float(recall_score(y_true, y_pred, average="macro",
                                  zero_division=0))
    metrics["OVERALL"] = {"precision": macro_p, "recall": macro_r, "f1": macro_f1}
    return metrics


def train_fold(fold_num: int, train_idx, val_idx,
               image_embs: np.ndarray, text_embs: np.ndarray,
               labels: np.ndarray, device: torch.device,
               models_dir: Path, logger,
               epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE):
    """Train fusion MLP for one fold. Returns (metrics_dict, val_preds, val_idx)."""

    X_all = np.hstack([image_embs, text_embs]).astype(np.float32)

    X_train = torch.tensor(X_all[train_idx])
    X_val   = torch.tensor(X_all[val_idx]).to(device)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
    y_val   = torch.tensor(labels[val_idx], dtype=torch.float32).to(device)

    model     = FusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.BCELoss()

    best_f1      = -1.0
    best_epoch   = 0
    patience_ctr = 0
    best_path    = models_dir / f"fold_{fold_num}_best.pth"

    n = len(X_train)
    logger.info(f"  Fold {fold_num}/{N_FOLDS}: Train={len(train_idx)}, "
                f"Val={len(val_idx)}")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx   = perm[start: start + batch_size]
            xb    = X_train[idx].to(device)
            yb    = y_train[idx].to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss  = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            val_probs  = model(X_val).cpu().numpy()
        val_binary = (val_probs >= THRESHOLD).astype(int)
        val_f1     = f1_score(labels[val_idx], val_binary,
                              average="macro", zero_division=0)
        avg_loss   = epoch_loss / max(n_batches, 1)

        arrow = " [↑ saved]" if val_f1 > best_f1 else ""
        logger.info(
            f"    Epoch {epoch:2d}/{epochs}: "
            f"Loss={avg_loss:.4f}, Val-F1={val_f1:.4f}{arrow}"
        )

        if val_f1 > best_f1:
            best_f1      = val_f1
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(
                    f"    Early stopping at epoch {epoch} "
                    f"(best Val-F1={best_f1:.4f} @ epoch {best_epoch})"
                )
                break

    # ---- final eval with best checkpoint ----
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val).cpu().numpy()
    val_binary = (val_probs >= THRESHOLD).astype(int)
    metrics    = compute_metrics(labels[val_idx], val_binary)

    logger.info(f"  Fold {fold_num} best Val-F1: {best_f1:.4f}")
    return metrics, val_binary, val_probs


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def save_confusion_matrices(all_true: np.ndarray, all_pred: np.ndarray,
                              cm_dir: Path, logger):
    """Save one 2×2 confusion matrix image per symptom."""
    cm_dir.mkdir(parents=True, exist_ok=True)
    for i, (sym, slug) in enumerate(zip(SYMPTOMS, SYMPTOM_SLUGS)):
        cm = confusion_matrix(all_true[:, i], all_pred[:, i], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"], ax=ax)
        ax.set_title(f"{sym}\n(Macro over all folds)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        out = cm_dir / f"symptom_{i}_{slug}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
    logger.info(f"  Confusion matrices saved → {cm_dir}")


def save_per_fold_metrics(fold_metrics: list, results_dir: Path, logger):
    """Save per_fold_metrics.csv."""
    rows = []
    for fold_num, metrics in enumerate(fold_metrics, 1):
        for sym in SYMPTOMS:
            m = metrics[sym]
            rows.append({
                "fold":      fold_num,
                "symptom":   sym,
                "precision": round(m["precision"], 4),
                "recall":    round(m["recall"], 4),
                "f1":        round(m["f1"], 4),
            })
    df = pd.DataFrame(rows)
    path = results_dir / "per_fold_metrics.csv"
    df.to_csv(path, index=False)
    logger.info(f"  per_fold_metrics.csv saved → {path}")
    return df


def save_averaged_symptom_f1(fold_metrics: list, results_dir: Path, logger):
    """Save averaged_symptom_f1.csv (primary output)."""
    rows = []
    for sym in SYMPTOMS + ["OVERALL"]:
        prec_vals = [m[sym]["precision"] for m in fold_metrics]
        rec_vals  = [m[sym]["recall"]    for m in fold_metrics]
        f1_vals   = [m[sym]["f1"]        for m in fold_metrics]
        rows.append({
            "symptom":        sym,
            "macro_precision": round(float(np.mean(prec_vals)), 4),
            "macro_recall":    round(float(np.mean(rec_vals)),  4),
            "macro_f1":        round(float(np.mean(f1_vals)),   4),
            "std_f1":          round(float(np.std(f1_vals)),    4),
        })
    df = pd.DataFrame(rows)
    path = results_dir / "averaged_symptom_f1.csv"
    df.to_csv(path, index=False)
    logger.info(f"  averaged_symptom_f1.csv saved → {path}")
    return df


def save_per_category_performance(data_df: pd.DataFrame,
                                   all_true: np.ndarray,
                                   all_pred: np.ndarray,
                                   all_indices: np.ndarray,
                                   results_dir: Path, logger):
    """
    Compute macro + weighted F1 per meme category using aggregated
    val predictions from all folds.
    """
    rows = []
    categories = sorted(data_df["category"].unique())

    for cat in categories:
        cat_mask  = (data_df.iloc[all_indices]["category"].values == cat)
        if cat_mask.sum() == 0:
            continue
        yt = all_true[cat_mask]
        yp = all_pred[cat_mask]
        rows.append({
            "category":    cat,
            "num_samples": int(cat_mask.sum()),
            "macro_f1":    round(float(f1_score(yt, yp, average="macro",
                                                zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(yt, yp, average="weighted",
                                                zero_division=0)), 4),
        })

    df = pd.DataFrame(rows)
    path = results_dir / "per_category_performance.csv"
    df.to_csv(path, index=False)
    logger.info(f"  per_category_performance.csv saved → {path}")
    return df


# ---------------------------------------------------------------------------
# Cross-validation splitter
# ---------------------------------------------------------------------------

def get_cv_splits(labels: np.ndarray):
    """
    Return list of (train_idx, val_idx) tuples.
    Uses MultilabelStratifiedKFold when available, falls back to KFold.
    """
    if HAS_ITERSTRAT:
        kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                        random_state=SEED)
        splits = list(kf.split(np.zeros(len(labels)), labels))
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(np.zeros(len(labels))))
    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- seeds ----
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- directory structure ----
    emb_dir     = OUTPUT_DIR / "embeddings"
    models_dir  = OUTPUT_DIR / "models"
    results_dir = OUTPUT_DIR / "results"
    cm_dir      = results_dir / "confusion_matrices"
    logs_dir    = OUTPUT_DIR / "logs"
    for d in [emb_dir, models_dir, results_dir, cm_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_dir / "training_log.txt")
    device = get_device()

    logger.info("=" * 60)
    logger.info("EMBEDDING FUSION — Hindi Mental Health Meme Classification")
    logger.info("=" * 60)
    logger.info(f"Device : {device}")
    logger.info(f"Output : {OUTPUT_DIR}")

    # ---- load data ----
    df = load_data(CSV_PATH, IMAGE_BASE, logger)
    labels = np.vstack(df["label_array"].values).astype(np.float32)

    logger.info(f"\nDataset : {len(df)} memes across "
                f"{df['category'].nunique()} categories")
    logger.info(f"Labels  : {NUM_LABELS} PHQ-9 symptoms (multi-label)")
    logger.info(f"Embed   : ResNet-50 ({IMAGE_DIM}-dim) + MuRIL ({TEXT_DIM}-dim)")
    logger.info(f"Training: {N_FOLDS}-fold stratified CV\n")

    # ---- extract / load cached embeddings ----
    logger.info("─" * 40)
    logger.info("STEP 1 — Embeddings")
    logger.info("─" * 40)
    image_embs, text_embs = extract_embeddings(df, emb_dir, device, logger)
    logger.info(f"Image embeddings : {image_embs.shape}")
    logger.info(f"Text  embeddings : {text_embs.shape}")

    # ---- cross-validation ----
    logger.info("\n" + "─" * 40)
    logger.info("STEP 2 — 5-Fold Cross-Validation")
    logger.info("─" * 40)
    strat_label = "MultilabelStratifiedKFold" if HAS_ITERSTRAT else "KFold (fallback)"
    logger.info(f"Stratification : {strat_label}")

    splits       = get_cv_splits(labels)
    fold_metrics = []

    # Accumulators for aggregated confusion matrices + category performance
    all_true_list  = []
    all_pred_list  = []
    all_idx_list   = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
        logger.info(f"\nFold {fold_num}/{N_FOLDS}")
        logger.info("─" * 30)

        metrics, val_pred_binary, val_pred_probs = train_fold(
            fold_num, train_idx, val_idx,
            image_embs, text_embs, labels,
            device, models_dir, logger,
        )
        fold_metrics.append(metrics)

        all_true_list.append(labels[val_idx])
        all_pred_list.append(val_pred_binary)
        all_idx_list.append(val_idx)

        # Per-fold summary
        sym_f1s = [metrics[s]["f1"] for s in SYMPTOMS]
        logger.info(
            f"  Fold {fold_num} Macro-F1 = {np.mean(sym_f1s):.4f}  "
            f"| per-symptom: "
            + "  ".join(f"{s[:4]}={metrics[s]['f1']:.2f}" for s in SYMPTOMS)
        )

    # ---- aggregate ----
    all_true    = np.vstack(all_true_list)
    all_pred    = np.vstack(all_pred_list)
    all_indices = np.concatenate(all_idx_list)

    # ---- save outputs ----
    logger.info("\n" + "─" * 40)
    logger.info("STEP 3 — Saving Results")
    logger.info("─" * 40)

    save_per_fold_metrics(fold_metrics, results_dir, logger)
    avg_df = save_averaged_symptom_f1(fold_metrics, results_dir, logger)
    save_per_category_performance(df, all_true, all_pred,
                                   all_indices, results_dir, logger)
    save_confusion_matrices(all_true, all_pred, cm_dir, logger)

    # ---- final summary ----
    logger.info("\n" + "=" * 60)
    logger.info("5-Fold Cross-Validation Results")
    logger.info("=" * 60)

    overall_row = avg_df[avg_df["symptom"] == "OVERALL"].iloc[0]
    overall_f1  = overall_row["macro_f1"]
    overall_std = overall_row["std_f1"]

    logger.info(f"\nOverall Macro-F1 : {overall_f1:.4f} ± {overall_std:.4f}\n")
    logger.info("Per-Symptom F1 (mean ± std across folds):")

    for _, row in avg_df[avg_df["symptom"] != "OVERALL"].iterrows():
        f1_vals = [fold_metrics[k][row["symptom"]]["f1"]
                   for k in range(N_FOLDS)]
        logger.info(
            f"  {row['symptom']:<28s}: "
            f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}"
        )

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")

    # ---- success criteria ----
    logger.info("\n" + "─" * 40)
    if overall_f1 > 0.55:
        logger.info(f"✅ Stretch goal achieved: Macro-F1 = {overall_f1:.4f} > 0.55")
    elif overall_f1 > 0.45:
        logger.info(f"✅ Success: Macro-F1 = {overall_f1:.4f} > 0.45 "
                    f"(beats 32% zero-shot baseline)")
    else:
        logger.info(f"⚠️  Macro-F1 = {overall_f1:.4f} — below 0.45 threshold")


if __name__ == "__main__":
    main()
