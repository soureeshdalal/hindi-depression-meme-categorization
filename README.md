# Hindi & Hinglish Depression Meme Categorization

Research project evaluating vision-language and text models for classifying Hindi mental-health memes using a **PHQ-9–style 7-symptom framework**. This repository contains the full pipeline: dataset construction, translation, Gemini gold labeling, zero-shot benchmarking, MAMA-aligned fine-tuning, explanation-augmented experiments, metrics, and analysis.

**Start here:** Read [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) for the chronological story of every phase and the reasoning behind each decision.

---

## Dataset images (Google Drive)

Image files are **not** in this GitHub repo (~2.9 GB). Download them from Google Drive after cloning:

| Dataset | Link |
|---------|------|
| **All project images** | [Google Drive](https://drive.google.com/drive/folders/1XQgvJPdJCgnWx0Jt7F6IhAik_C5uorhb?usp=sharing) |

See [`docs/DATA_ON_GDRIVE.md`](docs/DATA_ON_GDRIVE.md) for upload/download instructions and folder structure.

```bash
pip install gdown
export GDRIVE_FOLDER_ID=1XQgvJPdJCgnWx0Jt7F6IhAik_C5uorhb
bash scripts/download_images.sh
python3 scripts/verify_images.py   # confirm folders are populated
```

---

## Repository layout (chronological phases)

| Phase | Folder | What it contains |
|-------|--------|------------------|
| **0 — Dataset** | [`00_dataset/`](00_dataset/) | Labels (`train.json`, `test.json`); image folders (images on [Google Drive](docs/DATA_ON_GDRIVE.md)) |
| **1 — Categorization** | [`01_categorization/`](01_categorization/) | Gemini Vision category discovery (13 meme categories) |
| **2 — Translation** | [`02_translation/`](02_translation/) | Per-category and bulk English→Devanagari translation scripts + logs |
| **3 — Labeling** | [`03_labeling/`](03_labeling/) | Gemini API pipelines for gold labels and explanations (v2 canonical) |
| **4 — Zero-shot** | [`04_zero_shot/`](04_zero_shot/) | Unimodal text, multimodal VLM, vision-only, and Hinglish baselines (no fine-tuning) |
| **5 — Fine-tuning** | [`05_fine_tuning/`](05_fine_tuning/) | MAMA-aligned supervised FT outputs (unimodal, vision, multimodal) |
| **6 — Explanations** | [`06_explanations/`](06_explanations/) | Models fed Gemini-generated explanations; Colab experiment bundle |
| **7 — Metrics** | [`07_metrics/`](07_metrics/) | F1/precision/recall summaries for all experiment types |
| **8 — Analysis** | [`08_analysis/`](08_analysis/) | Cross-model HTML reports, paper figures, Colab API results |
| **9 — Runs** | [`09_runs/`](09_runs/) | Timestamped isolated pipeline outputs |
| **10 — Docs** | [`10_docs/`](10_docs/) | Detailed documentation, papers, dataset docx, thesis (when added) |
| **Archive** | [`archive/`](archive/) | Superseded experiments (local only — not on GitHub) |

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_multimodal.txt   # for VLM experiments
```

### Run the full fine-tuning pipeline

```bash
export HF_TOKEN=...   # if needed for gated Hugging Face models

bash run_full_pipeline.sh
```

Outputs go to `runs/pipeline_YYYYMMDD_HHMMSS/` by default. See [`10_docs/FULL_PIPELINE.md`](10_docs/FULL_PIPELINE.md) for all options.

### Compute metrics (after predictions exist)

```bash
python 07_metrics/scripts/compute_metrics.py              # zero-shot
python 07_metrics/scripts/compute_metrics_finetuned.py    # fine-tuned
```

---

## Key data files

| File | Role |
|------|------|
| `train.json` | Training gold labels (`TR-*` ids) — used **only** for fine-tuning |
| `test.json` | Test gold labels (`TE-*` ids) — used **only** for evaluation metrics |
| `translated_train/` | Devanagari training images paired with `train.json` |
| `translated_categorized_memes/` | 650 test images in 13 category folders |
| `03_labeling/gemini_runs/run_v2/phase1_explanations_v2.csv` | Canonical Gemini explanations (default for explanation experiments) |

See [`10_docs/GOLD_LABELS_POLICY.md`](10_docs/GOLD_LABELS_POLICY.md) for the train/test separation policy.

---

## Task definition

**Input:** Hindi (Devanagari) mental-health meme — text (OCR), image, or both.

**Output:** 7-dimensional binary vector indicating which PHQ-9–style symptoms are present:

1. Feeling down · 2. Lack of interest · 3. Self-harm · 4. Eating disorder · 5. Low self-esteem · 6. Concentration problem · 7. Sleeping disorder

Category definitions and translation methodology: [`10_docs/Category_definitions.md`](10_docs/Category_definitions.md)

---

---

## After upload — local cleanup

Once GitHub and Drive are verified, delete local copies to free ~7+ GB:

```bash
bash scripts/cleanup_after_upload.sh --archive-only      # ~4.4 GB
bash scripts/cleanup_after_upload.sh --include-images    # + ~2.9 GB images
```

See [`docs/LOCAL_CLEANUP.md`](docs/LOCAL_CLEANUP.md).

---

Symlinks at the repo root (e.g. `unimodal/`, `train.json`, `gemini_runs/`) point to the new phase folders so existing scripts and docs continue to work without path changes.

---

## Thesis

**[Soureesh_Dalal_Thesis.pdf](10_docs/thesis/Soureesh_Dalal_Thesis.pdf)** — university thesis documenting this research.

---

## Further reading

- [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) — full chronological narrative
- [`10_docs/EXPERIMENT_MODELS_AND_SETUP.md`](10_docs/EXPERIMENT_MODELS_AND_SETUP.md) — models run and outcomes
- [`10_docs/UNIMODAL_MULTIMODAL_RESULTS_REPORT.md`](10_docs/UNIMODAL_MULTIMODAL_RESULTS_REPORT.md) — results synthesis
- [`10_docs/MAMA_MEMEIA_FINETUNING_ALIGNMENT.md`](10_docs/MAMA_MEMEIA_FINETUNING_ALIGNMENT.md) — alignment with MAMA-MEMEIA baseline
- [`10_docs/AWS_RUN.md`](10_docs/AWS_RUN.md) — GPU instance setup
- [`docs/DATA_ON_GDRIVE.md`](docs/DATA_ON_GDRIVE.md) — image upload/download (Google Drive)
- [`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md) — push to private GitHub
