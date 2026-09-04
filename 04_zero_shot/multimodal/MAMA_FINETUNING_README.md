# MAMA-Memeia–style **fine-tuned** multimodal baselines

Your existing `run_multimodal.py` path uses **generative VLMs** (prompt → text → parse).  
MAMA-Memeia **Table 2** multimodal rows (**CLIP**, **VisualBERT**, **ViT + BERT**) are **supervised classifiers** trained with **BCE** on 7 PHQ-style labels.

This folder adds **`run_mama_multimodal_finetune.py`**, which fine-tunes:

| Baseline | Implementation |
|----------|----------------|
| **CLIP** | `openai/clip-vit-base-patch32` — image + OCR text embeddings concatenated → linear → 7 logits |
| **ViT + BERT** | `google/vit-base-patch16-224` + `bert-base-multilingual-cased` — [CLS] concat → linear → 7 logits |
| **VisualBERT** | `uclanlp/visualbert-vcr-coco-pre` + **ResNet-50** grid features → 36×512 “regions” (approximation of bottom-up features) |

**Data:** `translated_train/` + `train.json` labels; **OCR** via EasyOCR; **test** = `translated_categorized_memes/*/test/`.

## Run (GPU recommended)

```bash
# From repository root
export HF_TOKEN=   # if needed for gated checkpoints

python multimodal/run_mama_multimodal_finetune.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --output_dir multimodal/finetuned_mama \
  --baseline all

# One baseline only
python multimodal/run_mama_multimodal_finetune.py ... --baseline clip
```

Outputs: `multimodal/finetuned_mama/clip_predictions.csv`, `vit_bert_predictions.csv`, `visualbert_predictions.csv`.

## Full pipeline (everything in one command)

```bash
bash run_full_pipeline.sh
```

Optional: `--no-explanations`, `--no-vlm-ft`, `--explanations-csv PATH`, `--with-no-ft-metrics`.  
Details: `01_documentation/FULL_PIPELINE.md`.

**Shorter run** (text + vision + CLIP/ViT+BERT/VisualBERT FT + metrics only):

```bash
bash run_mama_replication.sh
```

## Your generative VLMs (LLaVA, PALO, BLIP-2, …) — same **classification objective**

To apply the **same supervised multi-label loss** (BCE on 7 logits) to the **same models** you run in `run_multimodal.py` (without switching to CLIP/VisualBERT), use:

```bash
python multimodal/run_vlm_multilabel_finetune.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --model_type llava \
  --model_id llava-hf/llava-1.5-7b-hf \
  --output_dir multimodal/finetuned_vlm
```

**Run every VLM in `run_multimodal.MODEL_CONFIGS`** (skips a model if its `*_ft_predictions.csv` already exists; continues on load/train failure):

```bash
python multimodal/run_vlm_multilabel_finetune.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --output_dir multimodal/finetuned_vlm \
  --run_all
```

Supported `--model_type` values match `load_model` in `run_multimodal.py`: `llava`, `llava_next`, `minicpm_v`, `idefics`, `instructblip`, `blip2`, `blip`, `palo`, `chitrarth`. Pooling is implemented in `multimodal/vlm_pooled_extractors.py`.

- **Default `linear_probe`:** frozen VLM, train only a linear head on **mean-pooled last hidden states** (same spirit as supervised adaptation, minimal VRAM).
- **`--mode tune_projector`:** also trains LLaVA’s **`multi_modal_projector`** (LLaVA / LLaVA-NeXT only).

Outputs: e.g. `LLaVA15_ft_predictions.csv`, `MiniCPM_V_ft_predictions.csv`, … (or `--tag MyRun_ft` for a single-model run).  
Metrics: `03_scripts/compute_metrics_finetuned.py` includes rows for the standard `*_ft` stems; add a row if you use a custom `--tag`.

---

## Notes

- **VisualBERT** in the original paper uses Faster R-CNN region features; we use a **standard approximation** (ResNet spatial grid) so the code runs without extra detectors. Cite this limitation in write-ups.
- **Yadav et al. (2023) adaptive gating SOTA** is not reimplemented here; add separately if required.
- Dependencies: `torch`, `torchvision`, `transformers`, `easyocr`, `pandas`, `tqdm`, `pillow`, `accelerate` (see root `requirements.txt`).
