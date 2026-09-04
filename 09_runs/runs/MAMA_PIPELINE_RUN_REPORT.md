# MAMA-aligned full pipeline — execution report

**Primary isolated run:** `runs/pipeline_20260323_181358/`  
**Console log:** `runs/pipeline_nohup.log`  
**Detailed run log:** `runs/PIPELINE_RUN_LOG.md`

---

## Phase 0 — Environment

| Item | Value |
|------|--------|
| Repo root | `/home/ubuntu/projects/categorization` |
| Python | 3.13 (`/opt/pytorch/bin/python`) |
| PyTorch | 2.10.0+cu130, CUDA available |
| GPUs | 4× NVIDIA A10G |

---

## Phase 1 — Required paths

All present: `train.json`, `test.json`, `translated_train/`, `translated_categorized_memes/`, `gemini_runs/run_v2/phase1_explanations_v2.csv`.

---

## Phase 2 — `HF_TOKEN`

- Automated check showed **`HF_TOKEN` unset** in the shell that launched `runs/pipeline_nohup.log`.
- **Impact:** `mental/mental-roberta-base` is **gated** on Hugging Face → **unimodal FT** skipped `MentalRoBERTa_ft_predictions.csv` with 401; **explanation unimodal** skipped MentalRoBERTa for the same reason.
- **Action (no secrets in logs):**  
  `export HF_TOKEN=…`  
  Accept the model on the Hub, then re-run only the missing steps (commands below).

---

## Phase 3 — Isolated pipeline

Launched with:

```bash
cd /home/ubuntu/projects/categorization
nohup bash run_full_pipeline.sh > runs/pipeline_nohup.log 2>&1 &
```

Default layout: **`runs/pipeline_20260323_181358/`** (timestamped; legacy `unimodal/finetuned` etc. not used).

---

## What completed in `pipeline_20260323_181358`

| Step | Status | Notes |
|------|--------|--------|
| 1 Unimodal text FT (OCR) | **OK** (9/10 CSVs) | Missing: `MentalRoBERTa_ft_predictions.csv` (gated repo + no token). |
| 2 Vision FT | **OK** | ViT, ResNet-50, EfficientNet-B0 CSVs present. |
| 3 MAMA multimodal FT | **OK** | `clip_predictions.csv`, `vit_bert_predictions.csv`, `visualbert_predictions.csv`. |
| 4 VLM multilabel BCE FT | **Failed (no CSVs)** | See “Root causes” below; **code fixes applied** — re-run step 4. |
| 5 Metrics (`test.json` only) | **Ran** | `metrics/finetuned/finetuned_metrics_*.csv`; VLM rows skipped (files missing). |
| 6 Explanation unimodal | **Mostly OK** | MentalRoBERTa skipped (gated); others under `explanation_runs/unimodal/`. |
| 7 Explanation multimodal | **In progress / long** | BLIP-2, LLaVA-1.5, LLaVA-NeXT done; IDEFICS + InstructBLIP heavy. Check `runs/pipeline_nohup.log` tail. |

---

## Step 4 (VLM FT) — root causes (from `pipeline_nohup.log`)

1. **Device mismatch:** `device_map="auto"` spread weights across GPUs; linear head + labels were on `cuda:0` → `Expected all tensors to be on the same device, cuda:3 and cuda:0`.
2. **LLaVA-NeXT:** follow-on failure (`NoneType` not iterable) after the first model error path.
3. **MiniCPM:** `cannot import name 'is_torch_fx_available'` — Transformers / model stack mismatch.
4. **BLIP-2 / InstructBLIP / BLIP / IDEFICS / PALO:** `Could not extract pooled features` — pooling path vs. current `transformers` internals.
5. **Chitrarth:** optional package not installed (`pip install git+https://github.com/ola-krutrim/Chitrarth.git`).

---

## Fixes applied in-repo (for next run)

1. **`multimodal/run_vlm_multilabel_finetune.py`**  
   - BCE head and label tensor use **`dev = next(model.parameters()).device`** (same device as sharded backbone).  
   - Inference moves pooled activations to backbone device before the head.

2. **`run_full_pipeline.sh`**  
   - **`export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"`** so `device_map="auto"` sees a single GPU by default (override if you need another GPU index).

3. **`explanation_runs/scripts/run_unimodal.py` / `run_multimodal.py`**  
   - **`errors.log`** written under **`OUT_ROOT.parent`** (isolated run: `…/explanation_runs/errors.log` inside the run tree).

---

## Duplicate pipelines / GPU contention

Two `run_full_pipeline.sh` instances were observed (different logs: `pipeline_run.log` vs `runs/pipeline_nohup.log`), and **two** `run_multimodal.py` processes writing to **`pipeline_20260323_174116`** vs **`pipeline_20260323_181358`**.  

**Recommendation:** Keep **one** full pipeline; stop the older job so VRAM is free for IDEFICS / InstructBLIP.

---

## No test leakage

- Fine-tuning scripts take **`--train_json train.json`** only; `multimodal/mama_finetune_common.py` refuses `test.json` as training labels.
- Metrics use **`test.json`** via `03_scripts/compute_metrics_finetuned.py` after predictions exist.

---

## Re-run commands (after setting `HF_TOKEN` if needed)

**MentalRoBERTa unimodal FT only (into same isolated run):**

```bash
cd /home/ubuntu/projects/categorization
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_TOKEN="…"   # required for mental/mental-roberta-base
python unimodal/run_unimodal_finetuned.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --output_dir runs/pipeline_20260323_181358/unimodal/finetuned \
  --preset all \
  --only_model MentalRoBERTa_ft
```

**VLM multilabel FT only (same run tree; single GPU recommended):**

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_TOKEN="…"   # if any model is gated
python multimodal/run_vlm_multilabel_finetune.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --output_dir runs/pipeline_20260323_181358/multimodal/finetuned_vlm \
  --run_all
```

**Recompute finetuned metrics only:**

```bash
python 03_scripts/compute_metrics_finetuned.py \
  --repo-root /home/ubuntu/projects/categorization \
  --prediction-root /home/ubuntu/projects/categorization/runs/pipeline_20260323_181358 \
  --metrics-out-dir /home/ubuntu/projects/categorization/runs/pipeline_20260323_181358/metrics/finetuned
```

**Explanation steps only (isolated outputs):**

```bash
python explanation_runs/scripts/run_unimodal.py \
  --repo-root /home/ubuntu/projects/categorization \
  --explanations-csv gemini_runs/run_v2/phase1_explanations_v2.csv \
  --output-dir runs/pipeline_20260323_181358/explanation_runs/unimodal

python explanation_runs/scripts/run_multimodal.py \
  --repo-root /home/ubuntu/projects/categorization \
  --explanations-csv gemini_runs/run_v2/phase1_explanations_v2.csv \
  --output-dir runs/pipeline_20260323_181358/explanation_runs/multimodal
```

---

## Post-run checklist (target paths under `runs/pipeline_20260323_181358/`)

| Artifact | Expected |
|----------|----------|
| `unimodal/finetuned/*.csv` | Yes (except MentalRoBERTa until token + re-run) |
| `vision/*.csv` | Yes |
| `multimodal/finetuned_mama/*.csv` | Yes |
| `multimodal/finetuned_vlm/*.csv` | **Re-run step 4** after fixes |
| `metrics/finetuned/finetuned_metrics_*.csv` | Yes (VLM lines missing until CSVs exist) |
| `explanation_runs/unimodal/metrics_summary.csv` | Yes |
| `explanation_runs/multimodal/metrics_summary.csv` | After step 7 completes |
| `explanation_runs/errors.log` | Yes (isolated) |

---

## Suggested next move

1. **Stop duplicate** pipeline / duplicate `run_multimodal.py` if still running.  
2. **`export HF_TOKEN`** (and Hub access for Mental RoBERTa).  
3. **Re-run** MentalRoBERTa FT + full **`--run_all` VLM FT** with **`CUDA_VISIBLE_DEVICES=0`**.  
4. **Re-run** `compute_metrics_finetuned.py`.  
5. Let **explanation multimodal** finish (or restart step 7 alone if the process died).  
6. **Sync** `runs/pipeline_20260323_181358/` (or newer `runs/pipeline_*`) to your laptop for paper tables.
