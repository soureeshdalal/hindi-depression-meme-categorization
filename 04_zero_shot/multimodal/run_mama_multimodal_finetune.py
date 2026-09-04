#!/usr/bin/env python3
"""
MAMA-Memeia Table 2 style **fine-tuned multimodal** baselines (translated data):
  - CLIP (image + OCR text embeddings fused → 7 logits)
  - ViT + BERT (HF ViT-B/16 + multilingual BERT [CLS] concat → 7 logits)
  - VisualBERT (BERT + ResNet-derived region embeddings; standard HF VisualBertModel)

These replace generative VLM prompting with **supervised multi-label fine-tuning**, matching the paper's
baseline *family* (not the MAMA multi-agent method itself).

Usage (from repo root):
  pip install transformers torch torchvision pillow easyocr tqdm pandas numpy accelerate

  python multimodal/run_mama_multimodal_finetune.py \\
      --data_dir translated_categorized_memes \\
      --train_json train.json \\
      --train_image_dir translated_train \\
      --baseline all

  python multimodal/run_mama_multimodal_finetune.py ... --baseline clip
"""

from __future__ import annotations

import argparse
import os
import random
import ssl
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torchvision.models as tv_models

from multimodal.mama_finetune_common import (
    NUM_LABELS,
    SEED,
    collect_test_images,
    get_device,
    labels_to_tensor,
    load_pil,
    load_training_paths_and_labels,
    ocr_paths,
)

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# Hyperparameters (similar to typical RESTORE / meme FT setups)
# ---------------------------------------------------------------------------
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
INFER_BATCH = 8
THRESHOLD = 0.5


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# =============================================================================
# 1) CLIP multi-label (image_embed ⊕ text_embed → head)
# =============================================================================
class CLIPMultiLabel(nn.Module):
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPModel

        self.clip = CLIPModel.from_pretrained(model_id, token=HF_TOKEN)
        d = self.clip.config.projection_dim
        self.head = nn.Linear(d * 2, NUM_LABELS)

    def forward(self, pixel_values, input_ids, attention_mask):
        out = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h = torch.cat([out.image_embeds, out.text_embeds], dim=-1)
        return self.head(h)


def train_loop_clip(
    model,
    processor,
    train_paths,
    train_texts,
    train_labels,
    device,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train_paths)
    idxs = list(range(n))

    for ep in range(epochs):
        random.shuffle(idxs)
        total, nb = 0.0, 0
        for start in tqdm(range(0, n, batch_size), desc=f"CLIP ep{ep+1}/{epochs}", leave=False):
            bi = idxs[start : start + batch_size]
            imgs, texts, lbls = [], [], []
            for i in bi:
                im = load_pil(train_paths[i])
                if im is None:
                    continue
                imgs.append(im)
                texts.append(train_texts[i] if train_texts[i] else " ")
                lbls.append(train_labels[i])
            if not imgs:
                continue
            try:
                batch = processor(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True)
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                y = labels_to_tensor(lbls).to(device)
                opt.zero_grad()
                logits = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            except Exception as e:
                print(f"  [CLIP train batch] {e}", file=sys.stderr)
        print(f"  CLIP epoch {ep+1} avg loss {total/max(nb,1):.4f}")
    model.eval()


def infer_clip(model, processor, test_images, ocr_texts, device, batch_size=INFER_BATCH):
    model.eval()
    preds, raw = [], []
    for start in tqdm(range(0, len(test_images), batch_size), desc="CLIP infer"):
        batch = test_images[start : start + batch_size]
        texts = [ocr_texts[start + j] if start + j < len(ocr_texts) else " " for j in range(len(batch))]
        imgs = []
        ok = []
        for j, info in enumerate(batch):
            im = load_pil(info["image_path"])
            if im is not None:
                imgs.append(im)
                ok.append(True)
            else:
                ok.append(False)
        if not imgs:
            for _ in batch:
                preds.append("[INFERENCE_ERROR]")
                raw.append("[INFERENCE_ERROR]")
            continue
        try:
            proc_texts = [texts[j] if ok[j] else " " for j in range(len(batch)) if ok[j]]
            batch_p = processor(text=proc_texts, images=imgs, return_tensors="pt", padding=True, truncation=True)
            batch_p = {k: v.to(device) for k, v in batch_p.items()}
            with torch.no_grad():
                logits = model(
                    pixel_values=batch_p["pixel_values"],
                    input_ids=batch_p["input_ids"],
                    attention_mask=batch_p["attention_mask"],
                )
                probs = torch.sigmoid(logits).cpu().numpy()
            pi = 0
            for j, info in enumerate(batch):
                if not ok[j]:
                    preds.append("[INFERENCE_ERROR]")
                    raw.append("[INFERENCE_ERROR]")
                else:
                    pr = probs[pi]
                    pi += 1
                    b = (pr >= THRESHOLD).astype(int)
                    preds.append("[" + ",".join(map(str, b.tolist())) + "]")
                    raw.append("[" + ",".join(f"{x:.2f}" for x in pr.tolist()) + "]")
        except Exception as e:
            print(f"  [CLIP infer] {e}", file=sys.stderr)
            for _ in batch:
                preds.append("[INFERENCE_ERROR]")
                raw.append("[INFERENCE_ERROR]")
    return preds, raw


# =============================================================================
# 2) ViT + BERT (late fusion)
# =============================================================================
class ViTBERTFusion(nn.Module):
    def __init__(self, vit_id: str = "google/vit-base-patch16-224", bert_id: str = "google-bert/bert-base-multilingual-cased"):
        super().__init__()
        from transformers import AutoModel, ViTModel

        self.vit = ViTModel.from_pretrained(vit_id, token=HF_TOKEN)
        self.bert = AutoModel.from_pretrained(bert_id, token=HF_TOKEN)
        self.head = nn.Linear(768 + 768, NUM_LABELS)

    def forward(self, pixel_values, input_ids, attention_mask):
        v = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0]
        b = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        if v.dtype != self.head.weight.dtype:
            v = v.to(self.head.weight.dtype)
            b = b.to(self.head.weight.dtype)
        return self.head(torch.cat([v, b], dim=-1))


def train_loop_vit_bert(
    model, processor_vit, tokenizer, train_paths, train_texts, train_labels, device, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train_paths)
    idxs = list(range(n))

    for ep in range(epochs):
        random.shuffle(idxs)
        total, nb = 0.0, 0
        for start in tqdm(range(0, n, batch_size), desc=f"ViT+BERT ep{ep+1}/{epochs}", leave=False):
            bi = idxs[start : start + batch_size]
            imgs, texts, lbls = [], [], []
            for i in bi:
                im = load_pil(train_paths[i])
                if im is None:
                    continue
                imgs.append(im)
                texts.append(train_texts[i] if train_texts[i] else " ")
                lbls.append(train_labels[i])
            if not imgs:
                continue
            try:
                pv = processor_vit(images=imgs, return_tensors="pt")["pixel_values"].to(device)
                enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                y = labels_to_tensor(lbls).to(device)
                opt.zero_grad()
                logits = model(pixel_values=pv, input_ids=input_ids, attention_mask=attn)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            except Exception as e:
                print(f"  [ViT+BERT train] {e}", file=sys.stderr)
        print(f"  ViT+BERT epoch {ep+1} avg loss {total/max(nb,1):.4f}")
    model.eval()


def infer_vit_bert(model, processor_vit, tokenizer, test_images, ocr_texts, device, batch_size=INFER_BATCH):
    model.eval()
    preds, raw = [], []
    for start in tqdm(range(0, len(test_images), batch_size), desc="ViT+BERT infer"):
        batch = test_images[start : start + batch_size]
        texts = [ocr_texts[start + j] if start + j < len(ocr_texts) else " " for j in range(len(batch))]
        imgs, ok_idx = [], []
        for j, info in enumerate(batch):
            im = load_pil(info["image_path"])
            if im is not None:
                imgs.append(im)
                ok_idx.append(j)
        if not imgs:
            for _ in batch:
                preds.append("[INFERENCE_ERROR]")
                raw.append("[INFERENCE_ERROR]")
            continue
        try:
            pv = processor_vit(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            proc_texts = [texts[j] for j in ok_idx]
            enc = tokenizer(proc_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = model(
                    pixel_values=pv,
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                probs = torch.sigmoid(logits).cpu().numpy()
            pr_i = 0
            ok_set = set(ok_idx)
            for j in range(len(batch)):
                if j not in ok_set:
                    preds.append("[INFERENCE_ERROR]")
                    raw.append("[INFERENCE_ERROR]")
                else:
                    pr = probs[pr_i]
                    pr_i += 1
                    b = (pr >= THRESHOLD).astype(int)
                    preds.append("[" + ",".join(map(str, b.tolist())) + "]")
                    raw.append("[" + ",".join(f"{x:.2f}" for x in pr.tolist()) + "]")
        except Exception as e:
            print(f"  [ViT+BERT infer] {e}", file=sys.stderr)
            for _ in batch:
                preds.append("[INFERENCE_ERROR]")
                raw.append("[INFERENCE_ERROR]")
    return preds, raw


# =============================================================================
# 3) VisualBERT + ResNet region features (approximation of bottom-up features)
# =============================================================================
class ResNetVisualEmbedder(nn.Module):
    """Maps image → sequence of 512-dim 'region' embeddings for VisualBERT."""

    def __init__(self):
        super().__init__()
        r = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        children = list(r.children())
        self.stem = nn.Sequential(*children[:-2])  # up to layer4: N,2048,h,w
        self.pool = nn.AdaptiveAvgPool2d((6, 6))  # 36 regions
        self.proj = nn.Linear(2048, 512)

    def forward(self, x):
        # x: ImageNet normalized tensor
        f = self.stem(x)  # N,2048,h,w
        f = self.pool(f)  # N,2048,6,6
        n, c, h, w = f.shape
        f = f.flatten(2).transpose(1, 2)  # N, 36, 2048
        return self.proj(f)  # N, 36, 512


class VisualBERTClassifier(nn.Module):
    def __init__(self, vb_name: str = "uclanlp/visualbert-vcr-coco-pre"):
        super().__init__()
        from transformers import VisualBertModel

        self.visual_proj = ResNetVisualEmbedder()
        self.vb = VisualBertModel.from_pretrained(vb_name, token=HF_TOKEN)
        hidden = self.vb.config.hidden_size
        self.head = nn.Linear(hidden, NUM_LABELS)

    def forward(self, pixel_values_resnet, input_ids, attention_mask, visual_attention_mask):
        # pixel_values_resnet: standard ResNet normalized (same as torchvision)
        ve = self.visual_proj(pixel_values_resnet)
        out = self.vb(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=ve,
            visual_attention_mask=visual_attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
        )
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


_RESNET_TF = tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms()


def train_loop_visualbert(
    model, tokenizer, train_paths, train_texts, train_labels, device, epochs=EPOCHS, batch_size=4, lr=LR
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train_paths)
    idxs = list(range(n))

    for ep in range(epochs):
        random.shuffle(idxs)
        total, nb = 0.0, 0
        for start in tqdm(range(0, n, batch_size), desc=f"VisualBERT ep{ep+1}/{epochs}", leave=False):
            bi = idxs[start : start + batch_size]
            tensors, texts, lbls = [], [], []
            for i in bi:
                im = load_pil(train_paths[i])
                if im is None:
                    continue
                tensors.append(_RESNET_TF(im))
                texts.append(train_texts[i] if train_texts[i] else " ")
                lbls.append(train_labels[i])
            if not tensors:
                continue
            try:
                pv = torch.stack(tensors).to(device)
                enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                bsz, nvis = pv.size(0), 36
                vmask = torch.ones(bsz, nvis, device=device)
                y = labels_to_tensor(lbls).to(device)
                opt.zero_grad()
                logits = model(pv, input_ids, attn, vmask)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            except Exception as e:
                print(f"  [VisualBERT train] {e}", file=sys.stderr)
        print(f"  VisualBERT epoch {ep+1} avg loss {total/max(nb,1):.4f}")
    model.eval()


def infer_visualbert(model, tokenizer, test_images, ocr_texts, device, batch_size=4):
    model.eval()
    preds, raw = [], []
    for start in tqdm(range(0, len(test_images), batch_size), desc="VisualBERT infer"):
        batch = test_images[start : start + batch_size]
        texts = [ocr_texts[start + j] if start + j < len(ocr_texts) else " " for j in range(len(batch))]
        tensors, ok_idx = [], []
        for j, info in enumerate(batch):
            im = load_pil(info["image_path"])
            if im is not None:
                tensors.append(_RESNET_TF(im))
                ok_idx.append(j)
        if not tensors:
            for _ in batch:
                preds.append("[INFERENCE_ERROR]")
                raw.append("[INFERENCE_ERROR]")
            continue
        try:
            pv = torch.stack(tensors).to(device)
            proc_texts = [texts[j] for j in ok_idx]
            enc = tokenizer(proc_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            bsz = pv.size(0)
            vmask = torch.ones(bsz, 36, device=device)
            with torch.no_grad():
                logits = model(
                    pv,
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    vmask,
                )
                probs = torch.sigmoid(logits).cpu().numpy()
            pr_i = 0
            ok_set = set(ok_idx)
            for j in range(len(batch)):
                if j not in ok_set:
                    preds.append("[INFERENCE_ERROR]")
                    raw.append("[INFERENCE_ERROR]")
                else:
                    pr = probs[pr_i]
                    pr_i += 1
                    b = (pr >= THRESHOLD).astype(int)
                    preds.append("[" + ",".join(map(str, b.tolist())) + "]")
                    raw.append("[" + ",".join(f"{x:.2f}" for x in pr.tolist()) + "]")
        except Exception as e:
            print(f"  [VisualBERT infer] {e}", file=sys.stderr)
            for _ in batch:
                preds.append("[INFERENCE_ERROR]")
                raw.append("[INFERENCE_ERROR]")
    return preds, raw


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MAMA-style fine-tuned multimodal baselines")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--train_image_dir", default="translated_train")
    parser.add_argument("--output_dir", default="multimodal/finetuned_mama")
    parser.add_argument("--baseline", choices=("clip", "vit_bert", "visualbert", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()
    set_seed()
    device = get_device()
    print(f"Device: {device}")

    import easyocr

    reader = easyocr.Reader(["hi", "en"], gpu=(str(device) != "cpu"))

    train_samples = load_training_paths_and_labels(args.train_json, args.train_image_dir)
    train_paths = [p for p, _ in train_samples]
    train_labels = [y for _, y in train_samples]
    print(f"Train samples: {len(train_paths)}")
    train_texts = ocr_paths(reader, train_paths, desc="OCR train")

    test_images = collect_test_images(args.data_dir)
    if args.max_images:
        test_images = test_images[: args.max_images]
    print(f"Test images: {len(test_images)}")
    test_paths = [x["image_path"] for x in test_images]
    test_ocr = ocr_paths(reader, test_paths, desc="OCR test")

    os.makedirs(args.output_dir, exist_ok=True)

    baselines = ["clip", "vit_bert", "visualbert"] if args.baseline == "all" else [args.baseline]

    for name in baselines:
        out_csv = Path(args.output_dir) / f"{name}_predictions.csv"
        if out_csv.exists():
            print(f"[SKIP] {out_csv} exists")
            continue

        print(f"\n{'='*60}\n[{name}]\n{'='*60}")

        try:
            if name == "clip":
                from transformers import CLIPProcessor

                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=HF_TOKEN)
                model = CLIPMultiLabel().to(device)
                train_loop_clip(
                    model, processor, train_paths, train_texts, train_labels, device, epochs=args.epochs
                )
                preds, raw = infer_clip(model, processor, test_images, test_ocr, device)

            elif name == "vit_bert":
                from transformers import AutoTokenizer, ViTImageProcessor

                processor_vit = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", token=HF_TOKEN)
                tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", token=HF_TOKEN)
                model = ViTBERTFusion().to(device)
                train_loop_vit_bert(
                    model,
                    processor_vit,
                    tokenizer,
                    train_paths,
                    train_texts,
                    train_labels,
                    device,
                    epochs=args.epochs,
                )
                preds, raw = infer_vit_bert(model, processor_vit, tokenizer, test_images, test_ocr, device)

            elif name == "visualbert":
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=HF_TOKEN)
                model = VisualBERTClassifier().to(device)
                train_loop_visualbert(
                    model, tokenizer, train_paths, train_texts, train_labels, device, epochs=args.epochs
                )
                preds, raw = infer_visualbert(model, tokenizer, test_images, test_ocr, device)

            df = pd.DataFrame(
                {
                    "image_id": [x["filename"] for x in test_images],
                    "category": [x["category"] for x in test_images],
                    "ocr_text": test_ocr,
                    f"{name}_pred": preds,
                    f"{name}_raw": raw,
                    "human_label": [""] * len(test_images),
                }
            )
            df.to_csv(out_csv, index=False)
            print(f"Saved {out_csv}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
        finally:
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
