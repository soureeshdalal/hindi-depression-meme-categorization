# Project Timeline

Chronological narrative of the Hindi & Hinglish Depression Meme Categorization project. Read top-to-bottom to understand **what was done, in what order, and why**.

---

## Phase 0 — Raw dataset & gold labels (Jul 2023)

**What:** Original English mental-health meme dataset with human annotations.

**Artifacts:**
- `00_dataset/labels/train.json` — 8,814 training samples (`TR-*` ids)
- `00_dataset/labels/test.json` — 650 test samples (`TE-*` ids)
- `00_dataset/raw/train/` — original English training images

**Why:** Establish a supervised learning split before any Hindi adaptation. Training labels are kept strictly separate from test labels to prevent leakage ([`10_docs/GOLD_LABELS_POLICY.md`](../10_docs/GOLD_LABELS_POLICY.md)).

---

## Phase 1 — Category discovery (Feb 2026)

**What:** Used Gemini Vision to analyze ~1,021 memes and discover 13 cultural-dependency categories (e.g. UNIVERSAL_HUMAN, POP_CULTURE_DEPENDENT, LANGUAGE_DEPENDENT).

**Artifacts:**
- `01_categorization/gemini_categorization/analysis_output/` — JSON results, category report
- `10_docs/Category_definitions.md` — consolidated definitions with translation methodology per category

**Why:** Not all memes translate the same way. A meme referencing American pop culture needs different handling than one depicting universal sadness. Category-aware translation (Phase 2) depends on this taxonomy.

**Key finding:** 81% of memes are UNIVERSAL_HUMAN (low cultural dependency); the remaining 19% require category-specific adaptation strategies.

---

## Phase 2 — Translation (Feb–Mar 2026)

**What:** Two translation tracks:

1. **Per-category test translation** — 13 category-specific scripts in `02_translation/scripts/` translate the 650 test memes into Devanagari Hindi, preserving visuals where appropriate.
2. **Bulk train translation** — `scripts/shared/translate_memes.py` translates training images from `00_dataset/raw/train/` → `00_dataset/translated/translated_train/` (2,077 completed of 8,814).

**Artifacts:**
- `00_dataset/translated/translated_categorized_memes/` — **650 Devanagari test images** (`TE-*.jpg`) in 13 category folders, each with a `test/` subfolder
- `00_dataset/translated/translated_train/` — **2,077 Devanagari training images** (`TR-*.jpg`)
- `02_translation/logs/translation_log.csv`, `translation_progress.json`
- `00_dataset/adaptation/csvs/` — human/LLM adaptation notes per category

**Image locations (canonical):** All translated images live under `00_dataset/translated/`. Symlinks at the repo root (`translated_train/`, `translated_categorized_memes/`) point here. A flat copy of the 650 test images also exists in `06_explanations/colab_experiments/data/test_images/` for Colab portability.

**Why:** Models must evaluate on Hindi text and images, not English originals. Category-specific scripts apply different strategies (direct translation vs. reconceptualization) based on Phase 1 dependency scores.

---

## Phase 3 — Gemini gold labeling (Mar–Apr 2026)

**What:** Gemini API pipelines to generate gold labels and natural-language explanations for test memes.

**Iterations:**
| Version | Location | Status |
|---------|----------|--------|
| v1 | `archive/gemini_runs_superseded/run_v1/` | Superseded — simpler two-phase pipeline, Macro-F1 ≈ 0.60 |
| **v2** | `03_labeling/gemini_runs/run_v2/` | **Canonical** — structured prompts, repair scripts, `phase1_explanations_v2.csv` |
| v2.1 | `03_labeling/gemini_runs/run_v2.1/` | Recommended labeling refinement |
| v3 pro preview | `03_labeling/gemini_runs/run_v3_pro_preview/` | Experimental pro-model family |
| v3 test | `archive/gemini_runs_superseded/run_v3_test/` | Small test run, archived |

**Also:** `03_labeling/gemini_labels/` holds v1 outputs still referenced by v2 phase-1 image lists. `03_labeling/gemini_v2_advisor/` contains advisor-style experiments with standardized CSVs.

**Why:** Human annotation at scale is expensive. Gemini provides consistent gold labels and explanations that downstream models can use for evaluation and as input features (Phase 6).

---

## Phase 4 — Zero-shot benchmarking (Feb–Mar 2026)

**What:** Evaluate models **without fine-tuning** on 650 translated test memes.

### Unimodal (text-only, OCR input)
8 models: IndicBERT v2 (×2), MuRIL, MentalBERT, MentalRoBERTa, BART (base + large), MentalBART.

→ `04_zero_shot/unimodal/*_predictions.csv`

### Multimodal (image + prompt)
9 VLMs: LLaVA 1.5, LLaVA-NeXT, MiniCPM-V, IDEFICS, BLIP-2, BLIP, PALO-7B, mBLIP (poor utility), InstructBLIP.

→ `04_zero_shot/multimodal/{Model}/*_predictions.csv`

### Vision-only
ViT, ResNet, EfficientNet zero-shot baselines.

→ `04_zero_shot/vision/`

### Hinglish side experiment (Feb 23, 2026)
44 Roman-script (Hinglish) variants of test memes to test script-mixing effects.

→ `04_zero_shot/hinglish/`

**Why:** Zero-shot establishes baselines before investing in fine-tuning. Reveals which models handle Hindi OCR text vs. which need visual understanding. Hinglish subset tests whether Devanagari translation matters.

**Documented in:** [`10_docs/EXPERIMENT_MODELS_AND_SETUP.md`](../10_docs/EXPERIMENT_MODELS_AND_SETUP.md)

---

## Phase 5 — Fine-tuning (Mar 23–24, 2026)

**What:** MAMA-MEMEIA–aligned supervised fine-tuning on `train.json` + `translated_train/`.

| Track | Script | Output |
|-------|--------|--------|
| Unimodal text | `04_zero_shot/unimodal/run_unimodal_finetuned.py` | `05_fine_tuning/unimodal/finetuned/` |
| Vision-only | `04_zero_shot/vision/run_vision.py` | `04_zero_shot/vision/` (ViT, ResNet, EfficientNet CSVs) |
| Multimodal (CLIP/ViT+BERT/VisualBERT) | `04_zero_shot/multimodal/run_mama_multimodal_finetune.py` | `05_fine_tuning/multimodal/finetuned_mama/` |
| Generative VLM + BCE head | `04_zero_shot/multimodal/run_vlm_multilabel_finetune.py` | `05_fine_tuning/multimodal/finetuned_vlm/` |

**Orchestrator:** `scripts/run_full_pipeline.sh` runs all steps; `scripts/run_mama_replication.sh` is a legacy wrapper.

**Why:** Zero-shot models underperform on domain-specific Hindi memes. Fine-tuning on translated training data with PHQ-9 labels aligns with the MAMA-MEMEIA benchmark for fair comparison.

**Alignment details:** [`10_docs/MAMA_MEMEIA_FINETUNING_ALIGNMENT.md`](../10_docs/MAMA_MEMEIA_FINETUNING_ALIGNMENT.md)

---

## Phase 6 — Explanation-augmented experiments (Mar–Apr 2026)

**What:** Instead of raw OCR, feed models Gemini-generated **explanations** of meme content.

| Track | Script | Input |
|-------|--------|-------|
| Unimodal + explanations | `06_explanations/explanation_runs/scripts/run_unimodal.py` | Explanation text only |
| Multimodal + explanations | `06_explanations/explanation_runs/scripts/run_multimodal.py` | Image + explanation |
| Colab bundle | `06_explanations/colab_experiments/` | Self-contained Colab package with 650 test images |

**Canonical explanations:** `03_labeling/gemini_runs/run_v2/phase1_explanations_v2.csv`

**Why:** OCR misses visual context; explanations capture both text and visual semantics. Tests whether structured natural-language descriptions improve classification over raw OCR.

**Legacy:** `archive/explanation_results/` (superseded by `explanation_runs/`).

---

## Phase 7 — Metrics (ongoing)

**What:** Compute Macro-F1, Weighted-F1, precision, recall per model and per symptom.

| Script | Evaluates |
|--------|-----------|
| `07_metrics/scripts/compute_metrics.py` | Zero-shot unimodal + multimodal |
| `07_metrics/scripts/compute_metrics_finetuned.py` | All fine-tuned models |
| `07_metrics/scripts/compute_metrics_hinglish.py` | Hinglish unimodal |
| `07_metrics/scripts/compute_metrics_hinglish_multimodal.py` | Hinglish multimodal |

**Outputs:** `07_metrics/results/` (subdirs: `no_finetuning/`, `finetuned/`, `hinglish/`, `colab/`)

**Why:** Standardized metrics enable cross-model comparison and alignment with published baselines (MAMA-MEMEIA Table 2).

---

## Phase 8 — Analysis & paper artifacts (Jun–Jul 2026)

**What:** Synthesis, visualization, and deep-dive analysis.

| Artifact | Purpose |
|----------|---------|
| `08_analysis/analysis_all_labels/` | Interactive HTML deep-dive across all 7 labels and all models |
| `08_analysis/analysis_loi/` | Focused analysis on "Lack of Interest" symptom |
| `08_analysis/paper_figures/` | LaTeX tables and figures for publication |
| `08_analysis/colab_results/` | Canonical Colab Pro API model results |
| `08_analysis/colab_test_results/` | GPT-4o, Mistral, Llama API experiment outputs |
| `10_docs/UNIMODAL_MULTIMODAL_RESULTS_REPORT.md` | Written results synthesis |

**Why:** Raw metrics CSVs aren't enough for a thesis or paper. Interactive HTML reports and LaTeX figures support narrative analysis and publication.

**Thesis:** [`10_docs/thesis/Soureesh_Dalal_Thesis.pdf`](../10_docs/thesis/Soureesh_Dalal_Thesis.pdf)

---

## Phase 9 — Pipeline runs

**What:** Timestamped isolated outputs from full pipeline executions.

→ `09_runs/runs/pipeline_YYYYMMDD_HHMMSS/`

**Why:** Each full run produces a self-contained snapshot (predictions + metrics) without overwriting previous results.

---

## What was archived and why

See [`archive/README.md`](../archive/README.md) for the full list. Summary:

- **`garbage/`** — failed runs, duplicate scripts, intermediate artifacts
- **`drive_downloads/`, `Colab_files/`** — duplicate Google Drive/Colab exports (~4 GB)
- **`gemini_runs/run_v1`, `run_v3_test`** — superseded labeling pipelines
- **`explanation_results/`, `Blip_inference/`, `embedding_fusion_output/`** — one-off or failed experiments

---

## Decision log (key reasoning)

| Decision | Reasoning |
|----------|-----------|
| 13 category taxonomy before translation | Cultural dependency varies wildly; one-size-fits-all translation loses meaning |
| Separate `train.json` / `test.json` | Prevent test-set leakage during fine-tuning |
| Zero-shot before fine-tuning | Establish baselines; identify which model families are worth FT investment |
| MAMA alignment for fine-tuning | Enable direct comparison with published multimodal meme classification benchmark |
| Gemini explanations as input | OCR alone misses visual semantics; explanations encode both modalities in text |
| Hinglish subset (44 memes) | Test whether Devanagari script matters vs. Roman Hindi |
| Archive duplicates, don't delete | Reproducibility and audit trail for thesis defense |

---

## Next steps (for repository maintainers)

- [x] Add thesis PDF to `10_docs/thesis/`
- [x] Upload images to Google Drive — [link](https://drive.google.com/drive/folders/1XQgvJPdJCgnWx0Jt7F6IhAik_C5uorhb?usp=sharing)
- [x] Paste Drive link in `README.md`
- [ ] Push to private GitHub — [`GITHUB_SETUP.md`](GITHUB_SETUP.md)
- [ ] Complete remaining train translation (2,077 / 8,814 done)
