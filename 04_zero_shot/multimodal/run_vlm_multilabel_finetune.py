#!/usr/bin/env python3
"""
Fine-tune **all VLMs listed in** ``run_multimodal.MODEL_CONFIGS`` with the same
**supervised objective** as MAMA-Memeia-style multimodal baselines: **multi-label BCE** on 7
symptom logits — not autoregressive generation + parsing.

How it works
------------
1. Build inputs the same way as `run_multimodal.py` (image + prompt; optional OCR appended).
2. Run a **single forward** through the VLM backbone to obtain **last hidden states**.
3. **Mean-pool** over the text sequence (attention mask) → fixed vector → **Linear(D → 7)**.
4. Train with **BCEWithLogitsLoss** on **train.json** (TR-* train split only) + `translated_train/`;
   run inference on test images **without** reading `test.json`. Use `compute_metrics_finetuned.py` + **test.json** for metrics.

Modes
-----
* `--mode linear_probe` (default): backbone **frozen**, only the **7-logit head** trains (fast, small GPU memory).
* `--mode tune_projector`: also unfreezes **LLaVA-style `multi_modal_projector`** (when present) for light adaptation.

This is the closest analogue to “fine-tune the multimodal model for the classification task” without
full LoRA SFT on the entire 7B LM (which is a separate, heavier recipe).

Usage (repo root, CUDA recommended, batch 1 for 7B):

  python multimodal/run_vlm_multilabel_finetune.py \\
    --data_dir translated_categorized_memes \\
    --train_json train.json \\
    --train_image_dir translated_train \\
    --model_type llava \\
    --model_id llava-hf/llava-1.5-7b-hf \\
    --output_dir multimodal/finetuned_vlm

  python multimodal/run_vlm_multilabel_finetune.py ... --model_type blip2 --model_id Salesforce/blip2-flan-t5-xl
  python multimodal/run_vlm_multilabel_finetune.py ... --model_type palo --model_id MBZUAI/PALO-7B
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from multimodal.mama_finetune_common import (
    NUM_LABELS,
    SEED,
    SYMPTOM_NAMES,
    collect_test_images,
    get_device,
    load_pil,
    load_training_paths_and_labels,
    ocr_paths,
)

# Load run_multimodal (loaders + prompts) without running main
_rm_path = Path(__file__).resolve().parent / "run_multimodal.py"
_spec = importlib.util.spec_from_file_location("multimodal_run_multimodal", _rm_path)
_rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rm)
load_model = _rm.load_model
MODEL_CONFIGS_ALL = _rm.MODEL_CONFIGS

from multimodal.vlm_pooled_extractors import extract_pooled

HF_TOKEN = os.environ.get("HF_TOKEN")

DEFAULT_EPOCHS = 3
DEFAULT_LR_HEAD = 1e-3
DEFAULT_LR_PROJECTOR = 1e-5
BATCH_SIZE = 1  # safe for 7B + long sequences
THRESHOLD = 0.5

MODEL_TYPE_CHOICES = (
    "llava",
    "llava_next",
    "blip2",
    "instructblip",
    "blip",
    "palo",
    "idefics",
    "minicpm_v",
    "chitrarth",
)


def resolve_output_short(model_id: str, model_type: str, tag: str | None) -> str:
    """CSV basename stem: e.g. LLaVA15_ft -> LLaVA15_ft_predictions.csv"""
    if tag:
        return tag
    # Legacy stems (match existing finetuned_vlm/*.csv and compute_metrics_finetuned.py)
    if model_id == "llava-hf/llava-1.5-7b-hf" and model_type == "llava":
        return "LLaVA15_ft"
    if model_type == "llava_next":
        return "LLaVA_NEXT_ft"
    if model_id == "Salesforce/blip2-flan-t5-xl" and model_type == "blip2":
        return "BLIP2_ft"
    for mid, sname, _, _, mtype in MODEL_CONFIGS_ALL:
        if mid == model_id and mtype == model_type:
            return f"{sname}_ft"
    if model_type == "palo":
        return "PALO_ft"
    if model_type == "blip2" and "mblip" in model_id.lower():
        return "mBLIP_ft"
    if model_type == "blip2":
        return "BLIP2_ft"
    slug = model_id.split("/")[-1].replace("-", "_")
    return (slug + "_ft")[:52]


def load_labels_matrix(train_json_path: str | Path):
    """sample_id -> 7-d label (for aligning with image path stems)."""
    with open(train_json_path) as f:
        data = json.load(f)
    by_id = {}
    for item in data:
        sid = (item.get("sample_id") or "").strip()
        if not sid:
            continue
        vec = [0.0] * NUM_LABELS
        for cat in item.get("meme_depressive_categories", []):
            if cat in SYMPTOM_NAMES:
                vec[SYMPTOM_NAMES.index(cat)] = 1.0
        by_id[sid] = vec
    return by_id


def align_train(paths_labels, by_id):
    paths, labels = [], []
    for p, _ in paths_labels:
        stem = Path(p).stem
        if stem in by_id:
            paths.append(p)
            labels.append(by_id[stem])
    return paths, np.array(labels, dtype=np.float32)


def build_prompt_base(model_type: str, use_hindi: bool, ocr_text: str, model_id: str = "") -> str:
    """Same templates as run_multimodal; append OCR for text-conditioned pooling."""
    if model_type in ("llava", "palo"):
        base = _rm.LLaVA_PROMPT_HINDI if use_hindi else _rm.LLaVA_PROMPT
    elif model_type == "llava_next":
        base = _rm.LLAVA_NEXT_PROMPT
    elif model_type == "blip2":
        if "mblip" in (model_id or "").lower():
            base = _rm.MBLIP_PROMPT
        else:
            base = _rm.BLIP2_PROMPT_HINDI if use_hindi else _rm.BLIP2_PROMPT
    elif model_type in ("instructblip", "blip"):
        base = _rm.BLIP2_PROMPT_HINDI if use_hindi else _rm.BLIP2_PROMPT
    elif model_type == "minicpm_v":
        base = _rm.MINICPM_V_QUESTION
    elif model_type == "idefics":
        base = _rm.IDEFICS_QUESTION
    elif model_type == "chitrarth":
        base = _rm.CHITRARTH_QUESTION
    else:
        base = _rm.LLaVA_PROMPT
    ocr = (ocr_text or "").strip()
    if ocr and ocr not in ("[NO_TEXT]", "[OCR_ERROR]"):
        base = base.rstrip() + f"\nOCR: {ocr[:500]}\n"
    return base


def set_trainable_modules(model, model_type: str, mode: str):
    model.requires_grad_(False)
    if mode != "tune_projector":
        return
    if model_type in (
        "blip2",
        "instructblip",
        "blip",
        "idefics",
        "minicpm_v",
        "chitrarth",
        "palo",
    ):
        print(f"  [warn] tune_projector not supported for {model_type}; using frozen backbone.")
        return
    if model_type in ("llava", "llava_next"):
        inner = getattr(model, "model", None)
        proj = getattr(inner, "multi_modal_projector", None) if inner is not None else None
        if proj is not None:
            proj.requires_grad_(True)
            print("  Unfrozen: multi_modal_projector")


def train_vlm(
    model,
    processor,
    model_type: str,
    model_id: str,
    train_paths: list[str],
    train_ocrs: list[str],
    train_y: np.ndarray,
    device: torch.device,
    mode: str,
    epochs: int,
    lr_head: float,
    lr_proj: float,
    use_hindi: bool,
):
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    # Probe hidden size
    model.eval()
    with torch.no_grad():
        p0 = train_paths[0]
        img = load_pil(p0)
        if img is None:
            raise RuntimeError("No valid image for hidden-size probe")
        pr = build_prompt_base(model_type, use_hindi, train_ocrs[0], model_id)
        ph = extract_pooled(model, processor, model_type, img, pr, dev, dtype)
        if ph is None:
            raise RuntimeError(f"Could not extract pooled features for model_type={model_type}")
        hidden = ph.shape[-1]
    print(f"  Pooled feature dim: {hidden}")

    # Align head + labels with backbone (device_map="auto" may place weights on cuda:N, not get_device())
    head = nn.Linear(hidden, NUM_LABELS).to(dev)
    set_trainable_modules(model, model_type, mode)

    proj = getattr(getattr(model, "model", None), "multi_modal_projector", None)
    if (
        mode == "tune_projector"
        and model_type in ("llava", "llava_next")
        and proj is not None
    ):
        opt = torch.optim.AdamW(
            [
                {"params": head.parameters(), "lr": lr_head},
                {"params": proj.parameters(), "lr": lr_proj},
            ],
            weight_decay=0.01,
        )
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=lr_head, weight_decay=0.01)

    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train_paths)
    idxs = list(range(n))
    use_grad_backbone = mode == "tune_projector" and model_type in ("llava", "llava_next")

    for ep in range(epochs):
        random.shuffle(idxs)
        model.train()
        head.train()
        total, nb = 0.0, 0
        for i in tqdm(idxs, desc=f"Epoch {ep+1}/{epochs}", leave=False):
            path = train_paths[i]
            img = load_pil(path)
            if img is None:
                continue
            pr = build_prompt_base(model_type, use_hindi, train_ocrs[i], model_id)
            y = torch.tensor(train_y[i : i + 1], dtype=torch.float32, device=dev)
            try:
                if model_type == "palo":
                    with torch.no_grad():
                        pooled = extract_pooled(model, processor, model_type, img, pr, dev, dtype)
                elif use_grad_backbone:
                    pooled = extract_pooled(model, processor, model_type, img, pr, dev, dtype)
                else:
                    with torch.no_grad():
                        pooled = extract_pooled(model, processor, model_type, img, pr, dev, dtype)
                if pooled is None:
                    continue
                if not use_grad_backbone:
                    pooled = pooled.detach()
                logits = head(pooled.float())
                opt.zero_grad()
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            except Exception as e:
                print(f"  [train skip] {path}: {e}", file=sys.stderr)
        print(f"  Epoch {ep+1} avg loss {total/max(nb,1):.4f}")

    model.eval()
    head.eval()
    return head, hidden


@torch.no_grad()
def infer_vlm(
    model,
    processor,
    head: nn.Module,
    model_type: str,
    model_id: str,
    test_images: list[dict],
    test_ocrs: list[str],
    device: torch.device,
    use_hindi: bool,
):
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    preds, raw = [], []
    model.eval()
    head.eval()
    head = head.to(dev)
    for j, info in enumerate(tqdm(test_images, desc="Inference")):
        img = load_pil(info["image_path"])
        ocr = test_ocrs[j] if j < len(test_ocrs) else ""
        pr = build_prompt_base(model_type, use_hindi, ocr, model_id)
        if img is None:
            preds.append("[INFERENCE_ERROR]")
            raw.append("[INFERENCE_ERROR]")
            continue
        try:
            pooled = extract_pooled(model, processor, model_type, img, pr, dev, dtype)
            if pooled is None:
                preds.append("[INFERENCE_ERROR]")
                raw.append("[INFERENCE_ERROR]")
                continue
            logits = head(pooled.float().to(dev))
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            b = (prob >= THRESHOLD).astype(int)
            preds.append("[" + ",".join(map(str, b.tolist())) + "]")
            raw.append("[" + ",".join(f"{x:.2f}" for x in prob.tolist()) + "]")
        except Exception as e:
            print(f"  [infer] {info['image_path']}: {e}", file=sys.stderr)
            preds.append("[INFERENCE_ERROR]")
            raw.append("[INFERENCE_ERROR]")
    return preds, raw


def run_one(
    *,
    data_dir: str,
    train_json: str,
    train_image_dir: str,
    output_dir: str,
    model_id: str,
    model_type: str,
    mode: str,
    epochs: int,
    lr_head: float,
    lr_proj: float,
    use_hindi: bool,
    max_images: int | None,
    tag: str | None,
    reader,
    train_paths: list[str],
    train_y: np.ndarray,
    train_ocrs: list[str],
    test_images: list[dict],
    test_ocrs: list[str],
):
    device = get_device()
    short = resolve_output_short(model_id, model_type, tag)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = Path(output_dir) / f"{short}_predictions.csv"
    if out_csv.exists():
        print(f"[SKIP] {out_csv} exists")
        return

    print(f"Loading VLM {model_id} ({model_type})...")
    try:
        model, processor = load_model(model_id, model_type)
    except Exception as e:
        print(f"  [FAIL] load_model: {e}", file=sys.stderr)
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    try:
        head, _ = train_vlm(
            model,
            processor,
            model_type,
            model_id,
            train_paths,
            train_ocrs,
            train_y,
            device,
            mode,
            epochs,
            lr_head,
            lr_proj,
            use_hindi,
        )
    except Exception as e:
        print(f"  [FAIL] train_vlm: {e}", file=sys.stderr)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    torch.save(head.state_dict(), Path(output_dir) / f"{short}_head.pt")

    pred_col = f"{short}_pred"
    raw_col = f"{short}_raw"
    preds, raws = infer_vlm(
        model,
        processor,
        head,
        model_type,
        model_id,
        test_images,
        test_ocrs,
        device,
        use_hindi,
    )
    df = pd.DataFrame(
        {
            "image_id": [x["filename"] for x in test_images],
            "category": [x["category"] for x in test_images],
            "ocr_text": test_ocrs,
            pred_col: preds,
            raw_col: raws,
            "human_label": [""] * len(test_images),
        }
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    del model, head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--train_image_dir", default="translated_train")
    ap.add_argument("--output_dir", default="multimodal/finetuned_vlm")
    ap.add_argument("--model_id", default=None, help="HF model id (required unless --run_all)")
    ap.add_argument(
        "--model_type",
        default=None,
        choices=MODEL_TYPE_CHOICES,
        help="Must match load_model in run_multimodal.py (required unless --run_all)",
    )
    ap.add_argument(
        "--run_all",
        action="store_true",
        help="Run MAMA-style multilabel head FT for every entry in run_multimodal.MODEL_CONFIGS",
    )
    ap.add_argument("--mode", choices=("linear_probe", "tune_projector"), default="linear_probe")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr_head", type=float, default=DEFAULT_LR_HEAD)
    ap.add_argument("--lr_proj", type=float, default=DEFAULT_LR_PROJECTOR)
    ap.add_argument("--use_hindi", action="store_true", help="Use Hindi prompt variant where available")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Override output CSV / column prefix (default: {MODEL_CONFIG short_name}_ft)",
    )
    args = ap.parse_args()

    if args.run_all:
        pass
    elif not args.model_id or not args.model_type:
        ap.error("--model_id and --model_type are required unless --run_all")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Device: {device} | mode={args.mode}")

    import easyocr

    reader = easyocr.Reader(["hi", "en"], gpu=(str(device) != "cpu"))

    raw_samples = load_training_paths_and_labels(args.train_json, args.train_image_dir)
    by_id = load_labels_matrix(args.train_json)
    train_paths, train_y = align_train(raw_samples, by_id)
    train_ocrs = ocr_paths(reader, train_paths, desc="OCR train")
    print(f"Train: {len(train_paths)} samples")

    test_images = collect_test_images(args.data_dir)
    if args.max_images:
        test_images = test_images[: args.max_images]
    test_ocrs = ocr_paths(reader, [x["image_path"] for x in test_images], desc="OCR test")
    print(f"Test: {len(test_images)} images")

    if args.run_all:
        for model_id, short_name, display_name, cfg_hindi, model_type in MODEL_CONFIGS_ALL:
            use_hi = args.use_hindi or cfg_hindi
            print(f"\n=== {display_name} ({model_type}) ===")
            if args.tag:
                print(
                    "  [warn] --tag is ignored with --run_all (use resolve_output_short per model)",
                    file=sys.stderr,
                )
            run_one(
                data_dir=args.data_dir,
                train_json=args.train_json,
                train_image_dir=args.train_image_dir,
                output_dir=args.output_dir,
                model_id=model_id,
                model_type=model_type,
                mode=args.mode,
                epochs=args.epochs,
                lr_head=args.lr_head,
                lr_proj=args.lr_proj,
                use_hindi=use_hi,
                max_images=args.max_images,
                tag=None,
                reader=reader,
                train_paths=train_paths,
                train_y=train_y,
                train_ocrs=train_ocrs,
                test_images=test_images,
                test_ocrs=test_ocrs,
            )
        print("Done (--run_all).")
        return

    print(f"model_type={args.model_type} | model_id={args.model_id}")
    run_one(
        data_dir=args.data_dir,
        train_json=args.train_json,
        train_image_dir=args.train_image_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_type=args.model_type,
        mode=args.mode,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_proj=args.lr_proj,
        use_hindi=args.use_hindi,
        max_images=args.max_images,
        tag=args.tag,
        reader=reader,
        train_paths=train_paths,
        train_y=train_y,
        train_ocrs=train_ocrs,
        test_images=test_images,
        test_ocrs=test_ocrs,
    )
    print("Done.")


if __name__ == "__main__":
    main()
