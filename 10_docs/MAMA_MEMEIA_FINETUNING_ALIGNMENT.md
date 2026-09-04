# Aligning fine-tuning with MAMA-Memeia (same setup, your dataset)

This document maps **your Hindi / translated RESTORE-style benchmark** to the **supervised baselines** described in *MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes* (AAAI 2026; dataset **RESTOREx**, task from Yadav et al. 2023).  
**Goal:** Same *experimental design* as the paper’s Table 2 baselines; **only the corpus** differs (translated memes + your splits).

---

## 1. Task (must match)

| Aspect | MAMA-Memeia / RESTOREx | This repo |
|--------|-------------------------|-----------|
| Formulation | **Multi-label** binary classification, **7** PHQ-9–style symptoms | Same (`NUM_LABELS = 7`, `BCEWithLogitsLoss`) |
| Metrics | **Macro-F1**, **Weighted-F1** (primary in Table 2) | `03_scripts/compute_metrics_finetuned.py` (same) |
| Text for text baselines | **OCR** from meme images (Table 2 “OCR:” rows); separate rows use **LLM explanations** | **OCR (EasyOCR)** on **translated** train/test images (domain-matched). Optional future row: explanations CSV → same heads |
| Image for vision baselines | Meme **pixels** | **Translated** train images → translated **test** images |

### Symptom labels (same names; index order in code)

RESTOREx Table 1 uses abbreviations **LOI, FD, ED, SD, LSE, CP, SH**.  
Our `SYMPTOM_NAMES` index → abbreviation:

| Index | `SYMPTOM_NAMES` | MAMA / RESTOREx |
|------:|-----------------|-----------------|
| 0 | Feeling Down | **FD** |
| 1 | Lack of Interest | **LOI** |
| 2 | Self-Harm | **SH** |
| 3 | Eating Disorder | **ED** |
| 4 | Low Self-Esteem | **LSE** |
| 5 | Concentration Problem | **CP** |
| 6 | Sleeping Disorder | **SD** |

Use this table when comparing **per-label** F1 to paper tables.

---

## 2. Data splits (analogous roles)

| Role | RESTOREx (paper) | Your setup |
|------|------------------|------------|
| Train | Large train split (~7k label positions in Table 1) | **`translated_train/`** + `train.json` labels (full train, no carve-out) |
| Validation | Dedicated val split (310 samples in Table 1) | **`translated_categorized_memes/*/validation/`** (use when you add **early stopping** or dev metrics; labels must come from your annotation files) |
| Test | Held-out test (520 in Table 1) | **`translated_categorized_memes/*/test/`** + `test.json` gold labels |

**Important:** Numbers of examples differ from RESTOREx; that is expected. Roles (train / val / test) should stay **disjoint** and **domain-consistent** (all translated for your Hindi benchmark).

---

## 3. Unimodal text baselines

**In this repo, fine-tuning uses the same seven checkpoints and order as `unimodal/run_unimodal.py`** (non–fine-tuned OCR baselines): IndicBERTv2-SS, IndicBERTv2-Sam-TLM, MuRIL, MentalBERT, MentalRoBERTa, BART-base, BART-large. That way **no-FT vs FT** comparisons are apples-to-apples on architecture.

**MAMA-Memeia Table 2** also includes plain **BERT** and **MentalBART**; we do **not** add those automatically so the FT suite stays aligned with *your* published zero-shot text baselines. You can add them back in `MODEL_CONFIGS` if you need explicit MAMA table replication.

| Your baseline (`run_unimodal.py`) | Fine-tuned (`run_unimodal_finetuned.py`) |
|-----------------------------------|------------------------------------------|
| Same `model_id` | Same `model_id`, `_ft` output CSV suffix |

**Second text condition in the paper:** Table 2 “Explanation:” rows replace OCR with **model-generated explanations**. To replicate that row you need a column or file of explanation text **per train/test id**, then point the dataloader at that text instead of OCR (same classifier stack).

### Hyperparameters (text)

Aligned with common BERT-family practice used in meme/depression FT work and your existing code:

- **Epochs:** 3  
- **Learning rate:** 2e-5  
- **Optimizer:** AdamW, weight decay 0.01  
- **Batch size:** 16 (train), 64 (inference)  
- **Max tokens:** 128  
- **Threshold:** 0.5 on sigmoid(logits) for binary preds  

If you later obtain **exact** values from Yadav et al. (2023) appendix, override these in one config block and cite them in the paper.

---

## 4. Unimodal image baselines (Table 2, “Unimodal Image”)

Paper: **ViT**, **ResNet**, **EfficientNet**.

| Paper | Your script | Implementation |
|-------|-------------|----------------|
| ViT (Dosovitskiy et al. 2021) | `vision/run_vision.py` | **ViT-B/16**, ImageNet weights, linear head on CLS |
| ResNet (He et al. 2015) | same | **ResNet-50** |
| EfficientNet (Tan & Le 2019) | same | **EfficientNet-B0** default in `vision/run_vision.py`; use `--efficientnet b7` for B7 |

**Hyperparameters (vision):** AdamW, 10 epochs, lr 1e-4, batch 16 — standard torchvision FT. Tune or match Yadav et al. if you publish their numbers.

**Train images:** `--train_dir translated_train` (default).

---

## 5. Multimodal baselines (Table 2 — supervised FT)

Paper lists **CLIP**, **VisualBERT**, **ViT + BERT** (image + OCR), plus Yadav et al. (2023) fusion SOTA.

- **Implemented:** `multimodal/run_mama_multimodal_finetune.py` — fine-tuned **CLIP** (ViT-B/32 + text tower), **ViT+BERT** late fusion (HF ViT-B/16 + multilingual BERT), **VisualBERT** + ResNet-50 grid “regions” (approximation of bottom-up features). See **`multimodal/MAMA_FINETUNING_README.md`**.
- **Still separate:** Your **generative** VLM pipeline (`run_multimodal.py` etc.) — complementary, not a replacement for these FT baselines.
- **Not reimplemented:** Yadav et al. **conditional adaptive gating** fusion — add only if you need exact SOTA row replication.

---

## 6. What to report

1. **Setup paragraph:** “We follow the MAMA-Memeia / RESTOREx **multi-label** PHQ-9 symptom setup and **supervised fine-tuning** baselines (Table 2), adapted to our **Hindi translated** corpus; train and test are **domain-matched** (translated OCR / translated pixels).”  
2. **Table:** Macro-F1, Weighted-F1 for each baseline you run.  
3. **Extensions:** Clearly label IndicBERT / MuRIL / MentalRoBERTa as **Hindi-focused additions**, not in the original MAMA table.

---

## 7. Commands (project root)

**Full stack (text FT + image FT + MAMA multimodal + generative-VLM BCE head + metrics + explanation benchmarks if CSV exists):**

```bash
bash run_full_pipeline.sh
# See 01_documentation/FULL_PIPELINE.md
```

**Original four-step replication only** (no generative-VLM head FT, no explanations):

```bash
bash run_mama_replication.sh
```

**Step-by-step:**

```bash
# Text — `--preset mama` (Table 2 only) or `all` (MAMA + Hindi extensions)
python unimodal/run_unimodal_finetuned.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --output_dir unimodal/finetuned \
  --preset all

# Vision — default EfficientNet-B0; `--efficientnet b7` optional
python vision/run_vision.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_dir translated_train \
  --output_dir vision

# Multimodal FT (CLIP, ViT+BERT, VisualBERT) — see multimodal/MAMA_FINETUNING_README.md
python multimodal/run_mama_multimodal_finetune.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --output_dir multimodal/finetuned_mama \
  --baseline all

python 03_scripts/compute_metrics_finetuned.py
```

---

## 8. Citation

```bibtex
@inproceedings{mama_memeia_2026,
  title={MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes},
  author={Agarwal, Siddhant and others},
  booktitle={AAAI},
  year={2026}
}
```

Also cite **RESTORE / RESTOREx** and **Yadav et al. (2023)** as in the paper.
