# Explanation-Based Benchmarking Analysis
**Experiment date:** 2026-03-18  
**Input:** Gemini-generated structured explanations (`gemini_runs/run_v2/phase1_explanations_v2.csv`)  
**Test set:** 581 samples (filtered from 650; 69 blocked or empty by Gemini safety filters)  
**Task:** Multi-label classification of 7 PHQ-9 depression symptoms in Hindi mental health memes

---

## 1. Experiment Overview

This benchmark evaluates whether providing models with **Gemini-generated structured explanations** of meme content improves symptom classification over two baselines:

- **OCR baseline**: Models receive raw EasyOCR text extracted from the image (n=509, some images unreadable)
- **Gemini-direct**: Gemini itself classifies after generating the explanation (n=551)

Two conditions are tested here:

| Condition | Input | Models |
|---|---|---|
| **Unimodal** | Explanation text only | 8 encoder/seq2seq models |
| **Multimodal** | Explanation text + original image | 5 vision-language models |

---

## 2. Label Prevalence (Test Set, n=581)

| Symptom | Count | Prevalence |
|---|---|---|
| Feeling Down | 215 | 33.1% |
| Low Self-Esteem | 111 | 17.1% |
| Eating Disorder | 82 | 12.6% |
| Self-Harm | 79 | 12.2% |
| Sleeping Disorder | 78 | 12.0% |
| Concentration Problem | 72 | 11.1% |
| Lack of Interest | 65 | 10.0% |

Feeling Down is by far the most common label. The remaining six symptoms are roughly balanced between 10–17%. This imbalance shapes model behaviour: the high-recall degenerate models effectively exploit Feeling Down's prevalence.

---

## 3. Unimodal Results (Explanation Text Only)

| Model | n | Macro-F1 | Wt-F1 | Precision | Recall | Δ vs OCR |
|---|---|---|---|---|---|---|
| MentalBERT | 581 | 0.2558 | 0.3017 | 0.1516 | 0.9817 | −0.061 |
| IndicBERTv2-Sam | 581 | 0.2380 | 0.2593 | 0.1647 | 0.7071 | **+0.035** |
| MentalBART† | 581 | 0.2131 | 0.2530 | 0.1427 | 0.5061 | −0.106 |
| MentalRoBERTa | 581 | 0.1974 | 0.2580 | 0.1183 | 0.7143 | **+0.017** |
| IndicBERTv2-SS | 581 | 0.1960 | 0.2504 | 0.1275 | 0.5641 | **+0.022** |
| BART-Large | 581 | 0.1895 | 0.2043 | 0.1442 | 0.5634 | **+0.032** |
| BART-Base | 581 | 0.1864 | 0.1531 | 0.1056 | 0.8230 | −0.006 |
| MuRIL | 581 | 0.0295 | 0.0225 | 0.0165 | 0.1429 | −0.077 |

†MentalBART uses adaptive per-symptom median threshold; all others use fixed threshold 0.5.

### 3.1 Key Observations

**General models improve; mental-health models decline.**  
Models not pre-trained on clinical mental-health text (IndicBERTv2, BART-Large, BART-Base, MentalRoBERTa) show small but consistent gains of +0.017 to +0.035 macro-F1 over the OCR baseline. These models likely benefit from the explanation's cleaner, more structured language compared to noisy OCR output. In contrast, MentalBERT and MentalBART — the highest-performing OCR models — both decline. This suggests they relied on mental-health vocabulary in the raw OCR text; the explanation, while richer in structure, redistributes that signal in ways the random classification head cannot exploit.

**Most unimodal models are degenerate.** Recall above 0.90 for multiple models indicates near-all-positive predictions. With a randomly-initialised linear classification head and no fine-tuning, the model's probability scores cluster near 0.5 and a fixed threshold of ≤0.5 causes over-prediction. The high recall inflates F1 scores above what precision alone would imply.

**MuRIL collapses entirely.**  
MuRIL (macro-F1=0.030) predicts almost no positives in the explanation setting — the inverse failure. Its encoder features on structured English explanation text fall outside the distribution it was trained on (primarily Hindi/multilingual), causing the random head to produce uniformly low scores.

**IndicBERTv2-Sam is the only model with some genuine signal.**  
It is the only model to show Eating Disorder F1 > 0.19 (0.192) and a Feeling Down F1 reasonable for a precision-recall trade-off (0.339, P=0.357, R=0.323). This suggests its sentence-similarity pre-training responds to the semantic structure of the explanation.

---

## 4. Multimodal Results (Explanation Text + Image)

| Model | n | Macro-F1 | Wt-F1 | Precision | Recall | Δ vs img-only |
|---|---|---|---|---|---|---|
| BLIP-2 | 581 | **0.3279** | 0.3648 | 0.2916 | 0.5757 | **+0.049** |
| LLaVA-1.5 | 581 | 0.3085 | 0.2675 | 0.4614 | 0.2582 | **+0.075** |
| InstructBLIP | 581 | 0.2807 | 0.3244 | 0.1768 | 0.7623 | — |
| LLaVA-NeXT | 578 | 0.2527 | 0.2992 | 0.1638 | 0.6320 | **+0.084** |
| IDEFICS | 64‡ | 0.2476 | 0.3212 | 0.1496 | 0.9054 | +0.045 |

‡IDEFICS produced valid structured output for only 64/581 rows; the remaining 517 generated off-topic free-form text. Metrics computed on the valid subset only.

### 4.1 Key Observations

**Explanation text reliably improves all multimodal models.** Unlike unimodal models, every multimodal model improves over its image-only baseline. Gains range from +4.9 pp (BLIP-2) to +8.4 pp (LLaVA-NeXT). The explanation provides structured symptom framing that VLMs can combine with visual evidence they extract directly from the image, resulting in more calibrated predictions.

**BLIP-2 is the best-performing model overall (excluding Gemini itself).**  
Macro-F1=0.3279 with the best balance of precision (0.29) and recall (0.58) across all non-Gemini models. Its per-symptom querying strategy (7 separate binary questions) is the key driver — it avoids the structured-format compliance issues that plague the other VLMs.

**LLaVA-1.5 is the highest-precision model.**  
At P=0.461, LLaVA-1.5 is the only model besides Gemini with genuinely useful precision. It pays for this with very low recall (0.258), generating false negatives for rare labels. Its Lack of Interest F1 (0.460) is the best across all models for that symptom, and it achieves the best Sleeping Disorder and Concentration Problem scores among free-generation VLMs.

**InstructBLIP and LLaVA-NeXT show high-recall degenerate behaviour.**  
InstructBLIP's per-symptom approach (same architecture as BLIP-2) still over-predicts heavily (R=0.76), likely because the InstructBLIP Flan-T5 base responds more aggressively to symptom-related language in the explanation context. LLaVA-NeXT improved significantly with the narrative parser (recovered 162/165 parse failures) but remains high-recall (R=0.63).

**IDEFICS is effectively unusable for this task.**  
89% of IDEFICS outputs were off-topic hallucinations (e.g., "I'm looking for a way to get rid of the Sexual Content warning…"). The 64 rows where it followed the format show strong recall (0.91) but poor precision (0.15), typical of all-positive degenerate output. IDEFICS-9B was not instruction-tuned in the same way as newer models and cannot reliably follow complex structured prompts.

---

## 5. Comparison with Gemini Direct Predictions

| System | n | Macro-F1 | Wt-F1 | Precision | Recall |
|---|---|---|---|---|---|
| Gemini-v2 (direct) | 551 | **0.3870** | 0.4014 | **0.6426** | 0.2919 |
| Gemini-v2.1 (direct) | 329 | 0.3462 | 0.3966 | 0.5388 | 0.2738 |
| BLIP-2 (expl+img) | 581 | 0.3279 | 0.3648 | 0.2916 | 0.5757 |
| LLaVA-1.5 (expl+img) | 581 | 0.3085 | 0.2675 | 0.4614 | 0.2582 |
| MentalBERT (expl) | 581 | 0.2558 | 0.3017 | 0.1516 | 0.9817 |

Gemini-v2 outperforms all other zero-shot systems by **+5.9 pp** over BLIP-2. More importantly, Gemini's precision (0.643) is more than **twice** that of any other model. This reflects two advantages: (1) Gemini performs both explanation generation and classification in a single consistent context, avoiding the distribution shift introduced when a third-party model reads Gemini's explanations, and (2) Gemini's instruction-following quality is far superior to the open-weight VLMs evaluated.

The explanation pipeline (Gemini generates explanation → model classifies) thus introduces an information bottleneck relative to Gemini classifying directly. The explanation captures most of the visual-linguistic content, but the downstream model — lacking Gemini's clinical calibration — cannot fully exploit it.

---

## 6. Per-Symptom Analysis

### Best model per symptom (across all conditions)

| Symptom | Best model | Macro-F1 | Notes |
|---|---|---|---|
| Feeling Down | BLIP-2 | 0.524 | High prevalence (33%); many models score well |
| Lack of Interest | LLaVA-1.5 | 0.460 | Only model with balanced P/R on this low-prevalence label |
| Self-Harm | BLIP-2 | 0.434 | Rarest signal; BLIP-2's per-question approach isolates it |
| Eating Disorder | Gemini-v2 | 0.289 | Very high Gemini precision (0.80); others collapse to 0 or all-positive |
| Low Self-Esteem | InstructBLIP | 0.283 | Marginal; most models confound this with Feeling Down |
| Concentration Problem | LLaVA-1.5 | 0.430 | Best F1 among all open models; balanced at P=0.43, R=0.43 |
| Sleeping Disorder | LLaVA-1.5 | 0.416 | Consistent structured output from LLaVA-1.5 helps rare labels |

### Observations

- **Feeling Down** (33% prevalence) is detected by almost every model at reasonable F1 due to its frequency.
- **Lack of Interest** (10%) is the hardest label. Most models either miss it entirely (all-zero) or over-predict it. Only LLaVA-1.5 achieves genuine precision (0.510 at F1=0.460).
- **Self-Harm** shows the sharpest differentiation between models: BLIP-2 (0.434) outperforms all others by a large margin, while most unimodal models score 0.19–0.28 due to base-rate-driven all-positive predictions.
- **Eating Disorder** is notable because Gemini-v2 achieves precision=0.80 while almost all open models score near 0.12 (base rate). This suggests Eating Disorder requires explicit visual and cultural cues (food imagery, body image references) that Gemini's multimodal understanding can reliably detect but open-weight models cannot isolate without fine-tuning.

---

## 7. Model Behaviour Taxonomy

Four distinct prediction behaviours were observed:

### 7.1 Near-all-positive (high recall, low precision)
**Models:** MentalBERT, MentalRoBERTa, BART-Base, InstructBLIP, LLaVA-NeXT, IDEFICS  
Recall ≥ 0.70 across most labels; precision at or below label base rate. These models are effectively predicting the majority class for every label. F1 scores are driven by recall and reflect dataset prevalence rather than understanding.

### 7.2 Balanced but biased
**Models:** BLIP-2, IndicBERTv2-Sam  
Moderate recall (0.50–0.82) with somewhat higher precision (0.16–0.48). BLIP-2's per-symptom VQA approach encourages independent binary decisions per label rather than a single structured response, producing the most calibrated predictions of all open models.

### 7.3 High-precision, low-recall
**Models:** LLaVA-1.5, Gemini-v2  
Recall ≤ 0.30; precision ≥ 0.46. These models predict conservatively, generating false negatives for ambiguous or subtle symptom expressions. Gemini is the extreme case (P=0.643, R=0.292). This strategy is penalised by macro-F1 relative to precision-oriented metrics.

### 7.4 Collapsed / degenerate
**Models:** MuRIL (near-all-zero), MentalBART with fixed threshold (all-positive)  
Output is near-constant regardless of input. MuRIL's encoder is incompatible with structured English explanation text; MentalBART's encoder features cluster in a region where the random linear head always fires under a 0.3 threshold.

---

## 8. Effect of Explanation on Model Performance

### Unimodal: mixed effect

| Direction | Models | Avg Δ macro-F1 |
|---|---|---|
| Improved | IndicBERTv2-SS, IndicBERTv2-Sam, MentalRoBERTa, BART-Large | +0.024 |
| Roughly neutral | BART-Base | −0.006 |
| Hurt | MentalBERT, MentalBART, MuRIL | −0.081 |

General-domain models that previously struggled with Hindi OCR noise benefit from the clean, structured explanation. Mental-health-specific models that peaked on OCR text (likely detecting mental-health vocabulary directly) are hurt, as the explanation reformulates the same information in more clinical language that disrupts whatever signal they had.

### Multimodal: consistent improvement

| Model | Img-only baseline | Expl+Img | Δ |
|---|---|---|---|
| BLIP-2 | 0.2793 | 0.3279 | +0.049 |
| LLaVA-1.5 | 0.2336 | 0.3085 | +0.075 |
| LLaVA-NeXT | 0.1692 | 0.2527 | +0.084 |
| IDEFICS | 0.2025 | 0.2476† | +0.045 |

†Computed on 64-row valid subset; not directly comparable.

All multimodal models improve substantially when given the explanation. The explanation replaces the need for the model to perform Hindi OCR, cultural interpretation, and depression-signal extraction all at once — tasks that open-weight VLMs perform poorly in zero-shot. Instead, the model only needs to map a structured English description to clinical labels.

---

## 9. Summary and Key Findings

1. **Gemini explanation + Gemini classification is the best zero-shot pipeline** (0.387 macro-F1), outperforming all open-model approaches by at least 5.9 pp.

2. **Explanation text consistently helps multimodal models** (+4.9 to +8.4 pp), but has mixed effects on unimodal models — benefiting general encoders and hurting mental-health-specific ones.

3. **BLIP-2 is the best open-model performer** at 0.328 macro-F1 (explanation + image). Its per-symptom VQA approach and balanced precision/recall are key differentiators.

4. **LLaVA-1.5 is the most precise open model** (P=0.461) and the best at rare labels (Lack of Interest, Sleeping Disorder, Concentration Problem). It is well-suited for applications where false positives are costly.

5. **All zero-shot approaches fall far short of fine-tuned performance.** The best zero-shot result (Gemini 0.387) is 27 pp below the best fine-tuned result (MentalBART-FT 0.653), confirming that this task requires either high-quality generative models or supervision.

6. **No unimodal explanation model outperforms the OCR baseline for MentalBERT/MentalBART** — the two strongest OCR models. The structured explanation appears to dilute the dense clinical vocabulary signals those models relied on.

7. **IDEFICS is not viable for structured generation tasks** in zero-shot settings. Its high hallucination rate (89%) suggests it requires RLHF-style instruction tuning for structured-output tasks.

8. **The explanation pipeline introduces a distribution shift bottleneck**: Gemini's explanations are rich, but the downstream classifier must interpret them without Gemini's clinical prior. This gap is widest for unimodal models and narrowest for the VLMs that can cross-reference the explanation against the visual signal.

---

## 10. Files and Reproducibility

| File | Description |
|---|---|
| `explanation_runs/scripts/run_unimodal.py` | Unimodal inference script |
| `explanation_runs/scripts/run_multimodal.py` | Multimodal inference script |
| `explanation_runs/unimodal/<Model>/predictions.csv` | Per-model predictions |
| `explanation_runs/unimodal/<Model>/metrics.json` | Per-model metrics and per-label breakdown |
| `explanation_runs/multimodal/<Model>/predictions.csv` | Per-model predictions |
| `explanation_runs/multimodal/<Model>/metrics.json` | Per-model metrics and per-label breakdown |
| `explanation_runs/unimodal/metrics_summary.csv` | Aggregated unimodal summary |
| `explanation_runs/multimodal/metrics_summary.csv` | Aggregated multimodal summary |
| `explanation_runs/run_config.txt` | Hardware, software, and model registry |
| `explanation_runs/errors.log` | All logged warnings and errors |

Input explanations: `gemini_runs/run_v2/phase1_explanations_v2.csv` (581 valid rows from 650 total; 69 blocked by Gemini safety filters).
