# Experiment summary

**Task:** seven PHQ-9-style symptoms, multi-label, threshold 0.5. **Numbers in tables:** macro-F1, then weighted F1 in parentheses.

**Test set size depends on the experiment.** OCR-only zero-shot: **n = 509**. Supervised fine-tuning on full test: **n = 650**. Explanation-conditioned runs: **n = 581** (IDEFICS multimodal **n = 345** after drops). Comparisons across columns are approximate when **n** differs.

---

## Versus MAMA-Memeia

**Same idea:** multi-label BCE, macro/weighted F1, text on **OCR**, vision on **pixels**, classic **image + OCR** fusion (CLIP, ViT+BERT, VisualBERT) trained supervised.

**Different here:** Hindi **translated** memes and your splits; extra encoders (IndicBERT, MuRIL, Mental*, BART); **generative VLMs** (zero-shot and a small BCE head on top) as an add-on; **Gemini explanations** as a separate input type. Explanation **unimodal** here is **frozen pretrained encoder + randomly initialized linear head with no training step** (not full-sequence BCE fine-tune on explanations). **Supervised fine-tune on explanation text with the OCR trainer** was **not** run.

---

## Unimodal text (OCR vs explanation vs fine-tune on OCR)

| Model | OCR, zero-shot | Explanation, frozen encoder + untrained head | OCR, supervised FT |
|-------|----------------|---------------------------------------------|-------------------|
| | n=509 | n=581 | n=650 |
| BERT-uncased (MAMA) | not computed | not computed | 0.147 (0.165) |
| BERT-multilingual | not computed | not computed | 0.175 (0.207) |
| IndicBERTv2-SS | 0.174 (0.152) | 0.229 (0.282) | 0.256 (0.324) |
| IndicBERTv2-Sam | 0.203 (0.187) | 0.168 (0.141) | 0.251 (0.318) |
| MuRIL | 0.106 (0.077) | 0.128 (0.109) | 0.000 (0.000) |
| MentalBERT | 0.317 (0.372) | 0.231 (0.285) | 0.068 (0.067) |
| MentalRoBERTa | 0.181 (0.155) | not run | 0.000 (0.000) |
| BART-Base | 0.192 (0.162) | 0.257 (0.304) | 0.047 (0.037) |
| BART-Large | 0.158 (0.242) | 0.113 (0.097) | 0.000 (0.000) |
| MentalBART | 0.320 (0.377) | 0.257 (0.304) | 0.081 (0.071) |

*PALO-7B OCR zero-shot only:* 0.155 (0.244), n=509.

*BERT-uncased* and *BERT-multilingual* appear only in the OCR fine-tune column; the legacy OCR zero-shot aggregate and the explanation unimodal script do not include those checkpoints.

*MuRIL, BART-Large, MentalRoBERTa FT* report 0.0 in the aggregate (failed or degenerate at metric time). MentalRoBERTa explanation column skipped (gated model access).

---

## Unimodal image (supervised FT only)

No vision zero-shot numbers were computed for this stack.

| Model | Macro F1 | Weighted F1 |
|-------|----------|-------------|
| ViT-B/16 | 0.116 | 0.148 |
| ResNet-50 | 0.150 | 0.198 |
| EfficientNet-B0 | 0.149 | 0.208 |

n = 650.

---

## Multimodal: image + OCR (zero-shot vs MAMA-style FT vs VLM head FT)

**Zero-shot generative** (image + OCR in the prompt, no FT on this task):

| Model | Macro F1 (weighted F1) |
|-------|----------------------|
| BLIP-2 | 0.279 (0.327) |
| LLaVA-1.5 | 0.234 (0.313) |
| IDEFICS | 0.203 (0.249) |
| LLaVA-NeXT | 0.169 (0.262) |
| BLIP | 0.054 (0.070) |

n = 509.

**Supervised fine-tune, image + OCR (MAMA-style):**

| Model | Macro F1 (weighted F1) |
|-------|----------------------|
| ViT+BERT | 0.271 (0.335) |
| VisualBERT + ResNet | 0.151 (0.189) |
| CLIP ViT-B/32 | 0.006 (0.004) |

n = 650.

**VLM with multilabel BCE head fine-tuned** (only one model completed in this run): LLaVA-1.5-7B **0.118 (0.178)**, n = 650.

---

## Multimodal: image + explanation (zero-shot generative)

| Model | Macro F1 (weighted F1) | n |
|-------|------------------------|---|
| BLIP-2-Flan-T5-XL | 0.328 (0.365) | 581 |
| IDEFICS-9B-Instruct | 0.317 (0.356) | 345 |
| LLaVA-1.5-7B | 0.309 (0.268) | 581 |
| LLaVA-NeXT-7B | 0.304 (0.325) | 581 |
| InstructBLIP-Vicuna-7B | 0.257 (0.304) | 581 |

---

## Quick read

Strongest **text** macro-F1: MentalBART OCR zero-shot (0.320); best **OCR fine-tune** among listed encoders IndicBERTv2-SS (0.256); among MAMA-style BERT FT rows, BERT-multilingual (0.175) beats BERT-uncased (0.147). Strongest **vision** FT: ResNet-50 (0.150). Strongest **multimodal** macro-F1 in the tables: BLIP-2 with **image + explanation** (0.328); best **MAMA-style** fusion: ViT+BERT (0.271). CLIP FT and the single VLM-head FT row are weak in this snapshot.
