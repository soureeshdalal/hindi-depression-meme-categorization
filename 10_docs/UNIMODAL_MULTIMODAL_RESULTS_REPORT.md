# Unimodal and Multimodal Results Report (with Explanation Runs)

This document compiles results for:

1. Unimodal text  
2. Unimodal image  
3. Multimodal  
4. Gemini API (image-only, explanation-only, and joint image+explanation where artifacts exist)

Each section includes (where available):
- baseline (no fine-tuning),
- explanation-based results,
- fine-tuned results,
- MAMA-Memeia model-family view.

## Data sources used

- Baseline/no-FT: `metrics/no_finetuning/metrics_summary.csv`
- Colab multimodal supplement (paths + notes): `metrics/colab/colab_multimodal_metrics.csv`, predictions under `colab_results/results/`
- BLIP Colab diagnostic (no scored outputs): `blip_inference/`, index `metrics/blip_inference/README.md`
- Fine-tuned (latest isolated run):  
  `runs/pipeline_20260323_181358/metrics/finetuned/finetuned_metrics_summary.csv`
- Explanation runs (legacy root):
  - `explanation_runs/unimodal/metrics_summary.csv`
  - `explanation_runs/multimodal/metrics_summary.csv`
- Explanation runs (isolated pipelines):
  - `runs/pipeline_20260323_181358/explanation_runs/unimodal/metrics_summary.csv`
  - `runs/pipeline_20260323_181358/explanation_runs/multimodal/metrics_summary.csv`
  - `runs/pipeline_20260323_174116/explanation_runs/unimodal/metrics_summary.csv`
  - `runs/pipeline_20260323_174116/explanation_runs/multimodal/metrics_summary.csv`
- MAMA alignment reference: `01_documentation/MAMA_MEMEIA_FINETUNING_ALIGNMENT.md`
- Gemini (Google API) runs: `gemini_runs/run_v1/`, `gemini_runs/run_v2/` + `run_v2.1/`, `gemini_runs/run_v3_test/`, `gemini_runs/run_v3_pro_preview/run_pro_family_650.py`

## Comparability notes

- Baseline/no-FT multimodal rows are mostly `n=509`; **InstructBLIP-Vicuna-7B** is **`n=650`** (Colab, different prompt protocol—see §3.2).
- Most fine-tuned and explanation runs use `n=581` or `n=650`.
- Because `n` differs, compare trends with care.
- Unimodal image explanation-only is **not applicable** by design.
- MAMA multimodal no-FT baselines (CLIP/ViT+BERT/VisualBERT) are not in baseline summary; their FT rows are available.

---

## 1) Unimodal Text

### 1.1 Models run

- Baseline/no-FT (`unimodal/run_unimodal.py`): IndicBERTv2-SS, IndicBERTv2-Sam, MuRIL, MentalBERT, MentalRoBERTa, BART-Base, BART-Large, MentalBART
- Explanation-only (`explanation_runs/unimodal`): same text model family (input switched from OCR text to explanation text)
- FT (`unimodal/run_unimodal_finetuned.py --preset all`):
  - MAMA subset: BERT-uncased, MentalBERT, BART-Base, MentalBART
  - Extended: mBERT, IndicBERTv2-SS, IndicBERTv2-Sam, MuRIL, MentalRoBERTa, BART-Large

### 1.2 Baseline (no FT) results

| Model | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---:|---:|---:|---:|---:|
| IndicBERTv2-SS | 509 | 0.1741 | 0.1517 | 0.2025 | 0.3605 |
| IndicBERTv2-Sam | 509 | 0.2031 | 0.1873 | 0.1920 | 0.3891 |
| MuRIL | 509 | 0.1061 | 0.0766 | 0.0606 | 0.4286 |
| MentalBERT | 509 | 0.3166 | 0.3723 | 0.1980 | 0.9588 |
| MentalRoBERTa | 509 | 0.1807 | 0.1549 | 0.1156 | 0.5895 |
| BART-Base | 509 | 0.1921 | 0.1621 | 0.1167 | 0.6154 |
| BART-Large | 509 | 0.1577 | 0.2423 | 0.1681 | 0.3029 |
| MentalBART | 509 | 0.3195 | 0.3769 | 0.1969 | 0.9980 |

### 1.3 Explanation-only results (A)

#### Latest isolated run (pipeline_20260323_181358)

| Model | Input | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---:|---:|---:|---:|---:|
| IndicBERTv2-SS | explanation | 581 | 0.2294 | 0.2819 | 0.1470 | 0.6388 |
| IndicBERTv2-Sam | explanation | 581 | 0.1678 | 0.1413 | 0.1759 | 0.5774 |
| MuRIL | explanation | 581 | 0.1280 | 0.1089 | 0.0723 | 0.5714 |
| MentalBERT | explanation | 581 | 0.2314 | 0.2846 | 0.1429 | 0.8432 |
| BART-Base | explanation | 581 | 0.2573 | 0.3042 | 0.1517 | 1.0000 |
| BART-Large | explanation | 581 | 0.1129 | 0.0968 | 0.1012 | 0.3210 |
| MentalBART | explanation | 581 | 0.2573 | 0.3042 | 0.1517 | 1.0000 |

#### Legacy root explanation run (`explanation_runs/unimodal`)

| Model | Input | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---:|---:|---:|---:|---:|
| IndicBERTv2-SS | explanation | 581 | 0.1960 | 0.2504 | 0.1275 | 0.5641 |
| IndicBERTv2-Sam | explanation | 581 | 0.2380 | 0.2593 | 0.1647 | 0.7071 |
| MuRIL | explanation | 581 | 0.0295 | 0.0225 | 0.0165 | 0.1429 |
| MentalBERT | explanation | 581 | 0.2558 | 0.3017 | 0.1516 | 0.9817 |
| MentalRoBERTa | explanation | 581 | 0.1974 | 0.2580 | 0.1183 | 0.7143 |
| BART-Base | explanation | 581 | 0.1864 | 0.1531 | 0.1056 | 0.8230 |
| BART-Large | explanation | 581 | 0.1895 | 0.2043 | 0.1442 | 0.5634 |
| MentalBART | explanation | 581 | 0.2573 | 0.3042 | 0.1517 | 1.0000 |

### 1.3.1 mBERT (`bert-base-multilingual-cased`) — `n=581`

Multi-input comparison on the **581-sample** test subset (aligned with explanation/OCR+explanation runs). **Zero-shot:** frozen encoder + classification head. **OCR fine-tuned:** supervised on **2077** training images (`translated_train/` + `train.json`); metrics below are still on the **`n=581`** eval slice for comparability with the zero-shot rows.

| Mode | Setting | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---:|---:|---:|---:|
| OCR | Zero-shot | 0.2211 | 0.2922 | 0.2315 | 0.5637 |
| Explanation | Zero-shot | 0.1145 | 0.0950 | 0.2253 | 0.3322 |
| OCR+Explanation | Zero-shot | 0.1236 | 0.1128 | 0.2079 | 0.3462 |
| OCR | Fine-tuned (2077 train) | 0.1912 | 0.2343 | 0.2427 | 0.2208 |
| Explanation | Fine-tuned | N/A | N/A | N/A | N/A |
| OCR+Explanation | Fine-tuned | N/A | N/A | N/A | N/A |

### 1.4 Fine-tuned results

| Model | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---:|---:|---:|---:|---:|
| BERT-uncased (FT, MAMA) | 650 | 0.1466 | 0.1645 | 0.1932 | 0.1420 |
| MentalBERT (FT) | 650 | 0.0677 | 0.0669 | 0.1644 | 0.0623 |
| BART-Base (FT) | 650 | 0.0474 | 0.0371 | 0.0674 | 0.1025 |
| MentalBART (FT) | 650 | 0.0814 | 0.0710 | 0.2239 | 0.1293 |
| BERT-multilingual (FT) | 650 | 0.1747 | 0.2073 | 0.2130 | 0.1920 |
| IndicBERTv2-SS (FT) | 650 | 0.2562 | 0.3235 | 0.2390 | 0.2791 |
| IndicBERTv2-Sam (FT) | 650 | 0.2506 | 0.3182 | 0.2236 | 0.2906 |
| MuRIL (FT) | 650 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| MentalRoBERTa (FT) | 650 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| BART-Large (FT) | 650 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 1.5 MAMA-Memeia text models (separate)

| MAMA text model | Baseline Macro-F1 | Explanation-only Macro-F1 (latest isolated) | FT Macro-F1 |
|---|---:|---:|---:|
| BERT-uncased | N/A (no explicit no-FT row) | N/A (not in explanation runner set) | 0.1466 |
| MentalBERT | 0.3166 | 0.2314 | 0.0677 |
| BART-Base | 0.1921 | 0.2573 | 0.0474 |
| MentalBART | 0.3195 | 0.2573 | 0.0814 |

### 1.6 Model reference — unimodal text (research notes)

**Task setup (all text models):** Input is either **OCR text** from the meme (baseline / FT) or **Gemini-structured explanation text** (explanation-only track). Each model uses a **7-way multi-label head** (one logit per PHQ-9-style symptom). **Decoding:** `sigmoid(logits)` → **binary vector** with threshold **0.5** (FT and standard eval); some no-FT runs also record **adaptive per-symptom thresholds** (median on held predictions) in addition to 0.5—see `unimodal/run_unimodal.py`. **Metrics** are macro-/weighted-F1 on the 7 labels.

**How to read “at par / better / worse”:** Compared to the **best baseline Macro-F1 in §1.2** (MentalBART ≈ 0.32) and **best FT Macro-F1 in §1.4** (IndicBERTv2-SS ≈ 0.26). “Full potential” here means: *not instruction-tuned for this meme task*; zero-shot / light heads rarely match supervised FT on a different objective.

#### IndicBERTv2-SS & IndicBERTv2-Sam (AI4Bharat)

| | |
|---|---|
| **Release / lineage** | IndicBERTv2 family (2021–2022); SS = sentence-segmented pretraining variant; Sam = additional TLM objective. |
| **Scale** | ~**110M** parameters (BERT-base–scale encoder). |
| **Strengths** | Hindi / multilingual tokenization and representation; good for short social text. |
| **Prompting** | N/A (encoder + classification head, not generative). OCR or explanation string fed to tokenizer. |
| **This project** | Baseline moderate; **FT improves** to best text-FT Macro-F1; explanation track mixed vs OCR. |
| **7-dim output** | Sigmoid + threshold 0.5 on 7 logits. |

#### MuRIL (Google)

| | |
|---|---|
| **Release** | 2021; multilingual representations for Indian languages. |
| **Scale** | ~**110M** (BERT-base class). |
| **Strengths** | Multilingual / transliterated text. |
| **This project** | Weak on this meme OCR baseline; **FT collapsed to zero** in the reported run—investigate training stability / class imbalance before interpreting as model limit. |

#### MentalBERT & MentalRoBERTa (mental-health–adapted Transformers)

| | |
|---|---|
| **Release** | MentalBERT ~2019–2020; MentalRoBERTa follows same idea on RoBERTa. |
| **Scale** | ~**110M** / **125M**. |
| **Strengths** | Language of mental health in **English** pretraining; may not transfer cleanly to **Hindi Devanagari** memes. |
| **This project** | **MentalBERT baseline very high recall / moderate precision** (typical of head calibration on easy positives); **FT underperformed** baseline—domain / size / hyperparams; not “maxed out.” |
| **7-dim output** | Same head + sigmoid thresholding. |

#### BART-Base, BART-Large, MentalBART

| | |
|---|---|
| **Release** | BART (2019); MentalBART = BART continued-pretrain on mental-health text. |
| **Scale** | **~140M** (base), **~400M** (large). |
| **Strengths** | Seq2seq denoising; MentalBART adds domain bias similar to MentalBERT. |
| **This project** | Baseline competitive; explanation tie for best isolated Macro-F1; **FT BARTs reported as 0** need debugging or are collapsed training—do not treat as intrinsic BART limit. |

#### BERT-uncased & BERT-multilingual (FT / MAMA alignment)

| | |
|---|---|
| **Release** | BERT (2018); mBERT multilingual. |
| **Scale** | ~**110M** each. |
| **Strengths** | Standard baselines; mBERT for non-English coverage. |
| **This project** | MAMA-aligned **BERT-uncased FT** included for table compatibility; **no dedicated no-FT row** in §1.2. |

---

## 2) Unimodal Image

### 2.1 Models run

From `vision/run_vision.py`:
- ViT-B/16
- ResNet-50
- EfficientNet-B0 (B7 optional)

### 2.2 Baseline (no FT) results

| Model | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 (no FT; frozen encoder + random head) | 650 | 0.2557 | 0.2829 | 0.1928 | 0.4901 |
| ResNet-50 (no FT; frozen encoder + random head) | 650 | 0.2833 | 0.3418 | 0.1927 | 0.6848 |
| EfficientNet-B0 (no FT; frozen encoder + random head) | 650 | 0.2201 | 0.2317 | 0.2004 | 0.3294 |

Observations:
- All three no-FT image-only models cluster in a narrow Macro-F1 band (~0.22-0.28), indicating weak PHQ-9 signal from visual features alone without adaptation.
- ResNet-50 shows the highest recall but low precision, indicating over-prediction behavior.
- EfficientNet-B0 is more conservative (higher precision than the others, lower recall).
- Compared with image-only FT rows in Section 2.4, the gap is small, suggesting only marginal vision-only gains from fine-tuning on this dataset.

### 2.3 Explanation-only / explanation+image

- Explanation-only for **unimodal image**: **not applicable**.
- Explanation+image is tracked under multimodal explanation runs (see Section 3.3).

### 2.4 Fine-tuned results

| Model | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 (FT) | 650 | 0.1264 | 0.1770 | 0.1605 | 0.1327 |
| ResNet-50 (FT) | 650 | 0.1534 | 0.2021 | 0.1861 | 0.1676 |
| EfficientNet-B0 (FT) | 650 | 0.1275 | 0.1718 | 0.1784 | 0.1436 |

### 2.5 MAMA-Memeia image models (separate)

MAMA image families (ViT, ResNet, EfficientNet) are represented in FT results above.

### 2.6 Model reference — unimodal image (research notes)

**Task setup:** **Image-only** encoder (ImageNet-pretrained backbone) → **linear layer** on pooled features → **7 sigmoid logits** → threshold **0.5** → binary vector. **Loss:** BCE-with-logits on training split (`vision/run_vision.py` style). **No OCR** in this pathway.

#### ViT-B/16 (Vision Transformer)

| | |
|---|---|
| **Release / lineage** | ViT (2020); B/16 patch variant standard in torchvision / HF. |
| **Scale** | ~**86M** parameters. |
| **Strengths** | Global context; strong generic visual features. |
| **This project** | Moderate FT scores; memes need text—**image-only is intentionally weak** vs multimodal. |

#### ResNet-50

| | |
|---|---|
| **Release** | 2015; still a strong CNN baseline. |
| **Scale** | ~**25M** parameters. |
| **Strengths** | Local textures, layout; fast to train. |
| **This project** | Slightly better Macro-F1 than ViT in FT table—task-dependent. |

#### EfficientNet-B0 (optional B7)

| | |
|---|---|
| **Release** | EfficientNet family (2019). |
| **Scale** | **~5.3M** (B0); B7 much larger if used. |
| **Strengths** | Parameter-efficient scaling. |
| **This project** | Competitive FT weighted-F1; good speed/accuracy tradeoff. |

---

## 3) Multimodal

### 3.1 Models run

- Baseline/no-FT generative VLMs (`multimodal/run_multimodal.py` + `compute_metrics.py`):
  - PALO-7B, LLaVA-1.5, LLaVA-NeXT, BLIP-2, IDEFICS, BLIP
- Baseline/no-FT Colab row: **InstructBLIP-Vicuna-7B** (`n=650`, natural-language prompt; see `colab_results/` + `metrics/colab/`)
- Explanation+image (B):
  - BLIP-2, LLaVA-1.5, LLaVA-NeXT, IDEFICS, InstructBLIP
- Fine-tuned multimodal:
  - MAMA-style supervised: CLIP, ViT+BERT, VisualBERT
  - VLM multilabel-head FT: LLaVA-1.5 (available in latest FT summary)

### 3.2 Baseline (no FT) results

| Model | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---:|---:|---:|---:|---:|
| PALO-7B | 509 | 0.1550 | 0.2440 | 0.1293 | 0.2473 |
| LLaVA-1.5 | 509 | 0.2336 | 0.3128 | 0.3435 | 0.4012 |
| LLaVA-NeXT | 509 | 0.1692 | 0.2621 | 0.2027 | 0.2765 |
| BLIP-2 | 509 | 0.2793 | 0.3272 | 0.2143 | 0.4346 |
| IDEFICS | 509 | 0.2025 | 0.2485 | 0.2046 | 0.2262 |
| BLIP | 509 | 0.0536 | 0.0699 | 0.3262 | 0.0298 |
| InstructBLIP-Vicuna-7B | 650 | 0.2500 | 0.3021 | 0.1621 | 0.5607 |

*InstructBLIP row:* Google Colab Pro (A100), **natural-language** zero-shot prompt on the **full translated test set** (`n=650`); not the same prompt/protocol as `n=509` rows above. Per-symptom metrics: `colab_results/results/metrics/instructblip_natural_language_metrics.json`.

### 3.3 Explanation+image results (B)

#### Latest isolated run (pipeline_20260323_181358)

| Model | Input | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---:|---:|---:|---:|---:|
| BLIP-2-Flan-T5-XL | explanation+image | 581 | 0.3279 | 0.3648 | 0.2916 | 0.5757 |
| LLaVA-1.5-7B | explanation+image | 581 | 0.3085 | 0.2675 | 0.4614 | 0.2582 |
| LLaVA-NeXT-7B | explanation+image | 581 | 0.3039 | 0.3251 | 0.4756 | 0.3111 |
| IDEFICS-9B-Instruct | explanation+image | 345 | 0.3165 | 0.3559 | 0.4133 | 0.3491 |
| InstructBLIP-Vicuna-7B | explanation+image | 581 | 0.2573 | 0.3042 | 0.1517 | 1.0000 |

#### Legacy root explanation run (`explanation_runs/multimodal`)

| Model | Input | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---:|---:|---:|---:|---:|
| BLIP-2 | explanation+image | 581 | 0.3279 | 0.3648 | 0.2916 | 0.5757 |
| LLaVA-1.5 | explanation+image | 581 | 0.3085 | 0.2675 | 0.4614 | 0.2582 |
| LLaVA-NeXT | explanation+image | 578 | 0.2527 | 0.2992 | 0.1638 | 0.6320 |
| IDEFICS | explanation+image | 64 | 0.2476 | 0.3212 | 0.1496 | 0.9054 |
| InstructBLIP | explanation+image | 581 | 0.2807 | 0.3244 | 0.1768 | 0.7623 |

#### Colab supplement (not pipeline `explanation_runs`)

| Model | Input | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| BLIP-2-Flan-T5-XL | explanation+image (Colab) | 650 | 0.0126 | 0.0270 | 0.0449 | 0.0073 | Diagnostic run only—model did not follow task; see `colab_results/results/summaries/experiment_3_blip2_failure_summary.txt`. **Do not** substitute for the pipeline BLIP-2 row above (`n=581`). |
| BLIP-VQA-capfilt-large | explanation+image (Colab) | 0 | — | — | — | — | **No scored run**—prompt echo only; no predictions file. See `blip_inference/experiment_4_blip_failure_summary.txt` and `metrics/colab/colab_multimodal_metrics.csv` (`explanation_image_colab_diagnostic_failed`). For a scored BLIP-Base + explanation path aligned with the repo, use `explanation_runs/scripts/run_multimodal.py --only BLIP`. |

### 3.4 Fine-tuned results

| Model | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-B/32 (FT) | 650 | 0.0055 | 0.0043 | 0.1429 | 0.0028 |
| ViT+BERT fusion (FT) | 650 | 0.2706 | 0.3347 | 0.1922 | 0.5184 |
| VisualBERT + ResNet regions (FT) | 650 | 0.1505 | 0.1891 | 0.1549 | 0.1944 |
| LLaVA-1.5-7B (multilabel head FT) | 650 | 0.1176 | 0.1775 | 0.1745 | 0.1614 |

### 3.5 MAMA-Memeia multimodal models (separate)

| MAMA multimodal model | Baseline Macro-F1 | Explanation+image Macro-F1 | FT Macro-F1 |
|---|---:|---:|---:|
| CLIP | N/A (not in baseline summary) | N/A (not in explanation runner set) | 0.0055 |
| ViT+BERT | N/A (not in baseline summary) | N/A (not in explanation runner set) | 0.2706 |
| VisualBERT | N/A (not in baseline summary) | N/A (not in explanation runner set) | 0.1505 |

### 3.6 Model reference — multimodal (research notes)

**Global decoding (generative no-FT track, `n≈509`):** Models are asked for a **compact answer** (symptom indices **1–7** or **None** / natural language). `multimodal/run_multimodal.py` **`parse_natural_language_response`** maps text → **7 binary** scores with heuristics (explicit digits, keywords, negation). **This is not the models’ native output format**; performance is **ceiling-limited** by how well each architecture follows instructions in English on Hindi-heavy memes.

**Explanation+image pipeline (`explanation_runs/scripts/run_multimodal.py`, `n≈581` where explanations exist):** Shared **YES/NO** template over 7 symptoms (plus fallbacks / narrative parsing). **BLIP-2 & InstructBLIP** use **7 separate VQA-style YES/NO calls** (one per symptom) with short explanation context—direct **7-bit** construction, usually more stable than one-shot free generation.

**Supervised multimodal FT:** **CLIP / ViT+BERT / VisualBERT** = fused features → **7 BCE logits** (threshold 0.5). **LLaVA FT** = small multilabel head on pooled VLM representations (see `multimodal/run_vlm_multilabel_finetune.py`).

#### PALO-7B (`MBZUAI/PALO-7B`)

| | |
|---|---|
| **Release / lineage** | 2024-era research VLM (vicuna-7B + CLIP ViT-L/14–336 style stack in repo config). |
| **Scale** | ~**7B** language-side parameters + vision tower (order-of-magnitude **10–15B total** interactive). |
| **Strengths** | Grounded LMM; built for image+text reasoning when custom code loads. |
| **No-FT prompting** | LLaVA-style **USER: `<image>` … ASSISTANT:** + **1–7 / None** symptom question (`CHITRARTH`-style block in repo). |
| **This project** | Baseline modest Macro-F1; **Colab explanation+image attempt failed at load** (custom `PaloForCausalLM` not in stock Transformers). **Full potential not reached** on consumer stack without PALO codebase. |

#### LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)

| | |
|---|---|
| **Release** | LLaVA-1.5 (2023–2024); HF checkpoint for easy inference. |
| **Scale** | ~**7B**. |
| **Strengths** | Instruction-style chat over image; strong general VQA. |
| **No-FT prompting** | `USER: <image>` + **numbered symptom list**, ask for **only applicable numbers**. |
| **Explanation+image** | Same YES/NO block as other VLMs in `explanation_runs` (text includes Gemini explanation). |
| **This project** | Solid no-FT and explanation+image; **FT head** lower—**frozen backbone + small head** is a different operating point than prompting. |

#### LLaVA-NeXT-7B (`llava-hf/llava-v1.6-vicuna-7b-hf`)

| | |
|---|---|
| **Release** | LLaVA-1.6 / NeXT (2024); higher-res / tiling. |
| **Scale** | ~**7B**. |
| **Strengths** | Better fine detail; longer visual context. |
| **No-FT prompting** | Same pattern as 1.5; **no truncation** of image tokens in explanation script (text-only truncation avoided). |
| **This project** | Competitive; **legacy explanation run** had slightly lower `n`—coverage artifact, not model definition. |

#### BLIP-2-Flan-T5-XL (`Salesforce/blip2-flan-t5-xl`)

| | |
|---|---|
| **Release** | BLIP-2 (2023); Flan-T5-XL language model. |
| **Scale** | ~**3–4B** class (ViT + Q-Former + Flan-T5-XL—component counts vary by source). |
| **Strengths** | Strong **VQA / captioning**; follows short questions better than open chat. |
| **No-FT prompting** | Short **“which symptoms 1–7”** English prompt (`BLIP2_PROMPT`). |
| **Explanation+image** | **Per-symptom YES/NO** with explanation snippet—**best-aligned** with getting a 7-vector. |
| **This project** | **Pipeline explanation+image best Macro-F1** in §3.3. **Colab single-prompt explanation run** failed (gibberish)—**not comparable** to pipeline decoding. |

#### IDEFICS-9B-Instruct (`HuggingFaceM4/idefics-9b-instruct`)

| | |
|---|---|
| **Release** | IDEFICS (2023); instruct-tuned multimodal. |
| **Scale** | ~**9B**. |
| **Strengths** | Interleaved image+text; instruction following. |
| **No-FT / explanation** | Chat-style **User / image / Assistant** wrapper around same **1–7** question. |
| **This project** | **Lower `n`** in some rows = partial run / filtering—check artifacts before cross-model ranking. |

#### BLIP-Base (`Salesforce/blip-image-captioning-base`)

| | |
|---|---|
| **Release** | Original BLIP (2021). |
| **Scale** | ~**224M** (order-of-magnitude). |
| **Strengths** | Captioning + VQA variants; **not** instruction-tuned. |
| **No-FT in repo** | **Image → caption only** in `predict_symptoms` (prompt **ignored**); symptoms inferred by **keyword parser on caption** → **weak alignment** to task. |
| **This project** | **Lowest no-FT Macro-F1** among main VLMs—expected. **Colab `blip-vqa-capfilt-large`** experiment: **prompt echo**, no scored file—documented as diagnostic only. **Scored explanation+image** for BLIP-Base: use `explanation_runs/.../run_multimodal.py --only BLIP` (per-symptom path). |

#### InstructBLIP-Vicuna-7B (`Salesforce/instructblip-vicuna-7b`)

| | |
|---|---|
| **Release** | InstructBLIP (2023); Vicuna-7B LLM backend. |
| **Scale** | ~**7–8B** effective. |
| **Strengths** | **Instruction tuning** on BLIP-2 stack—best BLIP-family fit for **“reply with numbers only”** style. |
| **No-FT (`n=509` track)** | Not in original six-model summary; **Colab `n=650`** row uses **natural-language symptom list** (depression, sleep problems, …) → **your script maps phrases → 7 binary** (see `colab_results` metrics JSON). **Different from** strict **1–7** prompt in `run_multimodal.py`. |
| **Explanation+image** | **Seven YES/NO queries** like BLIP-2. |
| **This project** | Colab NL baseline **~0.25 Macro-F1** with **high recall**—prompt choice matters more than for BLIP-2 pipeline. |

#### Colab / diagnostic experiments (tracked in `metrics/colab/colab_multimodal_metrics.csv`)

| Experiment | What it tests | Outcome |
|------------|----------------|---------|
| **1 — InstructBLIP NL zero-shot** | Full **650** test, **natural-language** symptom query | **Usable metrics**; integrated in §3.2 |
| **2 — PALO Colab** | Explanation + image | **Load failure** — see `colab_results` / PALO summary |
| **3 — BLIP-2 Colab** | Single forward **explanation+image** | **Failed** (~1.3% Macro-F1); conversational drift |
| **4 — BLIP-VQA** | Alternate BLIP checkpoint | **Prompt echo**; **no predictions** — `blip_inference/` |

#### CLIP ViT-B/32 (supervised FT)

| | |
|---|---|
| **Release** | CLIP (2021). |
| **Scale** | ~**150M** image+text towers (ViT-B/32). |
| **Strengths** | Alignment embedding space; **not** generative. |
| **This project** | **Very low FT Macro-F1** here—linear probe on frozen CLIP may be insufficient for fine-grained symptom multilabel on memes. |

#### ViT+BERT fusion & VisualBERT + ResNet regions (supervised FT)

| | |
|---|---|
| **Lineage** | MAMA / RESTORE-style late fusion (ViT CLS + BERT [CLS]); VisualBERT = BERT + detected region features (here approximated with ResNet grids). |
| **Strengths** | **Explicit OCR path** (multilingual BERT on meme text) + vision—matches paper Table-2 spirit. |
| **This project** | **ViT+BERT best supervised multimodal Macro-F1** in §3.4. |
| **7-dim output** | BCE logits, threshold 0.5. |

#### LLaVA-1.5-7B multilabel head (FT)

| | |
|---|---|
| **Setup** | Pooled VLM features → small MLP head; **BCE** training (`run_vlm_multilabel_finetune.py`). |
| **This project** | Moderate scores—**prompt-free inference** at test; different tradeoff than zero-shot generation. |

---

## 4) Gemini (Google Generative Language API)

These rows are **seven-label multilabel** PHQ-9 symptom prediction on the **translated test memes** (same task as elsewhere in this report), but produced with Gemini instead of the local Hugging Face VLMs. They are **not** directly comparable to `n=509` baseline rows without checking each run’s `n`.

### Gemini-3.1-Pro-Preview — zero-shot classification (consolidated)

| Experiment | Input | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---:|---:|---:|---:|---:|
| Image only | Image | 650 | 0.6776 | 0.6749 | 0.6898 | 0.6793 |
| Image only | Image | 581 | 0.6862 | 0.6816 | 0.7152 | 0.6740 |
| Explanation only | Text explanation | 581 | 0.6022 | 0.6096 | 0.6251 | 0.6184 |
| Image + explanation | Image + text | 581 | 0.6588 | 0.6574 | 0.6324 | 0.7112 |

**Model:** `gemini-3.1-pro-preview` (zero-shot). **`n=581`** rows use the subset aligned with valid Gemini explanations; **`n=650`** is the full translated test set.

**Observations**

- **Image-only** is the strongest single modality (Macro-F1 **0.6776–0.6862**) — strong vision understanding of meme content without relying on separate text input.
- **Explanation-only** underperforms image-only on the same **`n=581`** (**0.6022** vs **0.6862**) — suggests explanations can abstract or diverge from the visual signal the model would use from pixels alone.
- **Image + explanation** does not beat image-only on Macro-F1 (**0.6588** vs **0.6862**); joint input may introduce conflicting cues or heavier prompt load.
- **Image + explanation** yields the **highest recall** (**0.7112**) at lower precision (**0.6324**) — more liberal symptom predictions when text is added.
- All four rows sit **well above** the best local VLM rows in this report for comparable tracks (e.g. BLIP-2 explanation+image **0.3279** in the latest isolated multimodal summary).

*Legacy Phase 1/2 pipelines (§4.1–4.3) use different prompts, parsers, and often different `n`; treat this table as the primary zero-shot Gemini snapshot for paper tables unless you explicitly compare to V1/V2.1 artifacts.*

### 4.1 Explanation-only (test set)

Phase 2 takes **structured explanation text** as input (no image in the label call). Explanations were generated earlier from images in Run V2 Phase 1 (`gemini-3-pro-image-preview`).

| Run | Phase-2 model | n | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Artifact |
|-----|----------------|---:|---:|---:|---:|---:|---|
| **V2.1** (recommended) | `gemini-3.1-pro-preview` | 576 | 0.4014 | 0.4197 | 0.5727 | 0.3283 | `gemini_runs/run_v2.1/metrics_v2.1.json` |
| V1 (legacy) | `gemini-3.1-pro-preview` | 549 | 0.6011 | 0.6031 | 0.5684 | 0.6543 | `gemini_runs/run_v1/metrics_v1.json` |

V1 used **free-form** Phase 1 explanations (`gemini-3-pro-image-preview`) and a **simpler** Phase 2 prompt (no few-shot block). V2.1 uses **structured** Phase 1 text (Run V2) and the **V2.1** Phase 2 prompt; lower aggregate F1 partly reflects stricter calibration and different explanation format—not a like-for-like model upgrade comparison.

### 4.2 Image-only (pilot, not full test)

**Goal:** Predict the **7 binary** symptom vector **from the meme image alone** (no OCR pipeline, no prior explanation).

**Run:** `gemini_runs/run_v3_test/run_v3_test.py` — **`gemini-3.1-pro-preview`**, **50** random images from the translated test set (`sample_manifest.csv`), two prompt regimes:

| Condition | Prompt style | Parse success (`n`) | Macro-F1 | Artifact |
|-----------|----------------|---------------------|----------|----------|
| **A — Zero-shot** | Minimal YES/NO list + `PREDICTION: [0/1,…]` | **6** / 50 | 0.4000 | `metrics_comparison_postprocessed.json` → `zeroshot` |
| **B — Detailed** | V2.1-style definitions + few-shot + calibration, adapted for **image** input | **13** / 50 | **0.7039** | same file → `detailed` |

**Interpretation:** The headline **0.7039** Macro-F1 is on **`n=13` successfully parsed rows** out of 50—**not** a full-set score. Many rows failed strict `PREDICTION:` parsing or API/format issues; treat this as a **feasibility pilot**, not comparable to `n=576` explanation-only or `n=509` HF baselines.

**Full test (image-only):** Consolidated **`n=650` / `n=581`** image-only Macro-F1 for **`gemini-3.1-pro-preview`** is recorded in the **Gemini-3.1-Pro-Preview — zero-shot classification** table at the start of §4.

**Full-test path (alternate / script artifacts):** `gemini_runs/run_v3_pro_preview/run_pro_family_650.py` writes `predictions_image_only_pro.csv` and aggregates **`task2_image_only`** in `metrics_pro_family.json` (default model `gemini-3-pro-preview`; override with `--model`).

### 4.3 Joint image + explanation (full test pending)

**Goal:** Same **7-dim** labels using **both** the image and a **structured analysis** string (typically the Phase-1 explanation for that meme).

**Defined in code:** `run_pro_family_650.py` — prompt **`IMAGE_PLUS_EXPLANATION_PROMPT`**; output CSV **`predictions_image_plus_explanation_pro.csv`**; metric bucket **`task4_image_plus_explanation`** in `metrics_pro_family.json`.

**Status:** For **`gemini-3.1-pro-preview`** joint **image + explanation** on **`n=581`**, see the consolidated table at the start of §4. The **`run_pro_family_650.py`** path remains the in-repo runner; paste `task4_image_plus_explanation` from `metrics_pro_family.json` here if you need script-native artifacts for `gemini-3-pro-preview` or other IDs.

### 4.4 Model reference — Gemini (research notes)

**Models used in this report**

| Model ID (as in scripts) | Role | Notes |
|--------------------------|------|--------|
| `gemini-3-pro-image-preview` | **Phase 1 (V1, V2)** | **Multimodal** — image in, **structured or free-form explanation** out. Good at long, grounded descriptions; **not** natively a 7-bit classifier. |
| `gemini-3.1-pro-preview` | **Phase 2 V2 / V2.1**, **V3 image pilot** | **Text-in** for Phase 2 (explanation → labels); **image+text** in V3 test. Strong instruction following; **low temperature** (≈0.1) for structured answers. |
| `gemini-3-pro-preview` | **`run_pro_family_650` default** | Single-ID run for tasks 1–4 when you want one model name for the whole sweep. |

**Release / positioning:** These are **Google Generative Language API** “Pro” family **preview** IDs; availability and behavior can shift. For papers, record **exact model string + run date** from your JSON/CSV metadata.

**Strengths vs this task:** Gemini is strong at **(a)** producing **rich English rationales** from Hindi/English memes and **(b)** following a **fixed line** format when the prompt includes few-shot `PREDICTION: […]` examples. Weaknesses include **safety blocks** on some images (see `run_v2/blocked_snapshot.csv` / repair scripts), **parse fragility** if the model chatters beyond the template, and **non-comparability** across runs that change Phase 1 structure (V1 vs V2).

**Prompting used**

- **Phase 1 (V2):** Seven **observational** sections (mood, motivation, …) + context — elicits text that **aligns** with PHQ dimensions without forcing 7 bits yet (`PHASE1_PROMPT_V2` in `phase1_v2.py` / `run_pro_family_650.py`).
- **Phase 2 (V2.1):** Long **rubric + few-shot + per-indicator YES/NO reasons** + final **`PREDICTION: [x,…,x]`** — best match to **research-grade calibration** for text-only labeling.
- **Image-only (pro family):** Short **`IMAGE_ONLY_PROMPT`** — single line `PREDICTION: […]` only.
- **Image + explanation (pro family):** **`IMAGE_PLUS_EXPLANATION_PROMPT`** — same 7-bit line after showing analysis text.

**How model output becomes the 7-dim binary**

- **Phase 2 / pro-family / V3 test:** Regex (and fallbacks) extract **`PREDICTION:`** followed by **seven `0`/`1`** tokens; failed parse → often **all-zero** or drop from metrics depending on script (`parse_prediction` in `phase2_v2.1.py`; `parse_bits` in `run_pro_family_650.py`). This is **explicit discretization**, not sigmoid thresholds like HF encoders.
- **Compared to local VLMs:** Gemini path avoids **keyword mining from captions** (unlike BLIP-Base no-FT) but **depends** on format compliance more than **per-symptom VQA** (unlike BLIP-2 in `explanation_runs`).

**Performance vs rest of report**

- **Gemini-3.1-Pro-Preview zero-shot (§4 consolidated table):** Image-only and joint modalities reach **~0.60–0.69** Macro-F1 on **`n=581`/`n=650`** — **above** local HF VLMs in this report for the same broad task; still record exact prompt, parse rules, and subset definition for reproducibility.
- **Explanation-only Phase 2 (legacy):** **V2.1** Macro-F1 **0.4014** (`n=576`) — **above** typical unimodal OCR baselines in §1.2; **not** the same protocol as the §4 zero-shot explanation-only row.
- **Legacy V1 Phase 2** higher F1 with **different** Phase 1 and prompt — cite as **separate experimental condition**, not “better model.”
- **Image-only V3 pilot:** High F1 on **tiny parsed `n`** — supplementary only; prefer the §4 **`n=650`/`n=581`** image-only rows for full-test claims.

**Did we get “full potential”?** No single number captures it: **preview APIs**, **prompt/version drift**, and **blocklist/repair** cycles mean reported scores are **lower bounds** on what a frozen pipeline could achieve. The **V2.1** prompt is closer to **maximum** careful calibration for **text-from-explanation** than the minimal image-only prompts.

---

## Quick takeaways (with explanations included)

- **Unimodal text (latest isolated explanation run):** BART-Base and MentalBART tie best at Macro-F1 = 0.2573.
- **Unimodal text baseline best:** MentalBART (0.3195).
- **Unimodal text FT best:** IndicBERTv2-SS (0.2562).
- **Multimodal explanation+image best (latest isolated):** BLIP-2-Flan-T5-XL (0.3279).
- **Multimodal baseline no-FT best (n=509 track):** BLIP-2 (0.2793).
- **InstructBLIP no-FT (Colab, n=650, NL prompt):** Macro-F1 = 0.2500 (`colab_results/`).
- **MAMA multimodal FT best:** ViT+BERT fusion (0.2706).
- **Gemini-3.1-Pro-Preview (zero-shot, consolidated §4):** Image-only **0.6776** (`n=650`) / **0.6862** (`n=581`); explanation-only **0.6022** (`n=581`); image+explanation **0.6588** (`n=581`).
- **Gemini explanation-only (legacy Phase 2):** V2.1 Macro-F1 = 0.4014 (`n=576`); legacy V1 Phase 2 **0.6011** (`n=549`) — different Phase 1/2 setup than the §4 zero-shot table (see §4.1).
- **Gemini image-only (V3 pilot):** `n=13` parsed / 50 sampled Macro-F1 = 0.7039 — not full test; full `n=650`/`n=581` scores in §4 consolidated table (see §4.2).

## Suggested next reporting cleanup

- Recompute all tracks on a unified sample size (`n=650`) for strict comparability.
- Add no-FT baselines for MAMA image and MAMA multimodal families if direct before/after is required.
- Keep one locked "reporting snapshot" folder with all three tables (baseline, explanation, FT) to avoid drift.
- Optionally run `run_pro_family_650.py` and paste `task2_image_only` / `task4_image_plus_explanation` from `metrics_pro_family.json` into §4.2–4.3 if you need script-native artifacts (e.g. `gemini-3-pro-preview`); the §4 consolidated **`gemini-3.1-pro-preview`** zero-shot table already records full `n=650`/`n=581` image and joint rows.

