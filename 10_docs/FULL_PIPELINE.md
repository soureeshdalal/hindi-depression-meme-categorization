# One-command full pipeline

## Gold labels (important)

- **`train.json`** — used **only** as **training** supervision (with `translated_train/`, TR-* ids). Required for supervised FT; **not** used to evaluate the test set during training.
- **`test.json`** — used **only** in `03_scripts/compute_metrics*.py` after predictions exist. **No fine-tuning script reads `test.json`.**

See **`GOLD_LABELS_POLICY.md`**.

## What you get

| Step | Script | Input | What it is |
|------|--------|-------|------------|
| 1 | `unimodal/run_unimodal_finetuned.py` | OCR on **translated_train** | **Unimodal text (fine-tuned)** — MAMA-style BCE on 7 labels |
| 2 | `vision/run_vision.py` | **Pixels** on translated train | **Unimodal image (fine-tuned)** — ViT, ResNet, EfficientNet-B0 |
| 3 | `multimodal/run_mama_multimodal_finetune.py` | Image + **OCR** | **Multimodal image + OCR** — CLIP, ViT+BERT, VisualBERT |
| 4 | `multimodal/run_vlm_multilabel_finetune.py --run_all` | Image + **OCR** in prompt | **Multimodal image + OCR** — generative VLMs with **7-logit BCE head** (same `MODEL_CONFIGS` as `run_multimodal.py`) |
| 5 | `03_scripts/compute_metrics_finetuned.py` | `test.json` | Metrics for all FT CSVs |
| 6 | `explanation_runs/scripts/run_unimodal.py` | Gemini **explanation** text | **Unimodal text with explanations** (frozen encoder + linear head on explanations; not the same code path as OCR FT) |
| 7 | `explanation_runs/scripts/run_multimodal.py` | **Image + explanation** | **Multimodal image + explanations** — generative YES/NO parsing |

**Not** in this pipeline: zero-shot **no–fine-tuning** `run_multimodal.py` / `run_unimodal.py` (OCR). Add manually if you need raw generative baselines, then `python 03_scripts/compute_metrics.py`.

## Command

```bash
cd /path/to/categorization
export HF_TOKEN=...   # if needed

bash run_full_pipeline.sh
```

### Where outputs go (clean / no overlap)

**Default:** a **new folder** under `runs/pipeline_YYYYMMDD_HHMMSS/` holds **all** prediction CSVs and finetuned metrics for that run. Existing `unimodal/finetuned/`, `vision/`, `multimodal/finetuned_*`, and `metrics/finetuned/` are **not** overwritten.

- **`--in-place`** — write to those legacy repo paths (models may **skip** if a CSV already exists).
- **`--run-dir my_exp`** — isolated root `runs/my_exp/` (or use an absolute path).

`run_mama_replication.sh` uses **`--in-place`** so behavior matches the old single-location layout.

Metrics script options (advanced): `python 03_scripts/compute_metrics_finetuned.py --help` (`--prediction-root`, `--metrics-out-dir`, `--no-mutate-prediction-csvs`).

### Options

- `--explanations-csv PATH` — phase-1 CSV (`image_id`, `category`, `gold_labels`, `explanation`). Default: `gemini_runs/run_v2/phase1_explanations_v2.csv`.
- `--no-explanations` — skip steps 6–7.
- `--no-vlm-ft` — skip step 4 (generative VLM BCE-head FT); keeps step 3 (CLIP / ViT+BERT / VisualBERT).
- `--with-no-ft-metrics` — run `03_scripts/compute_metrics.py` (still updates **`metrics/no_finetuning/`** — omit if you want zero overlap there too).

### Backward compatibility

`bash run_mama_replication.sh` → `run_full_pipeline.sh --in-place --no-explanations --no-vlm-ft`.

## Alignment notes

- Explanation **multimodal** checkpoints match **`multimodal/run_multimodal.py`** (e.g. LLaVA-NeXT **Vicuna**, IDEFICS **9b-instruct**, InstructBLIP **Vicuna-7B**). Older runs used different checkpoints; new outputs use folder names `LLaVA`, `LLaVA_NeXT`, etc.
- **Explanation** step 6 is *not* full supervised FT like step 1; it trains only a small head on explanation text. For strict “MAMA Table 2 explanation rows,” you would reuse the step-1 trainer with explanation strings instead of OCR (future extension).

See also: `MAMA_MEMEIA_FINETUNING_ALIGNMENT.md`, `multimodal/MAMA_FINETUNING_README.md`.

## AWS

Use **`01_documentation/AWS_RUN.md`**: GPU sizing, `train.json`/`test.json` + image folders, CUDA PyTorch install, `HF_TOKEN`, and syncing `runs/pipeline_*` off the instance.
