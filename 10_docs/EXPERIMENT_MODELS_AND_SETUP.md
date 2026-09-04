# Experiment: Models and Setup

## Task and data

- **Task:** Predict which of 7 depression-related symptoms (PHQ-9 style) are present in Hindi mental-health memes. Output is a 7-dimensional binary vector plus optional raw scores/text.
- **Symptoms:** 1=Feeling down, 2=Lack of interest, 3=Self-harm, 4=Eating disorder, 5=Low self-esteem, 6=Concentration problem, 7=Sleeping disorder.
- **Data:** 650 test images in `translated_categorized_memes`, 13 categories. Text for unimodal models comes from OCR (EasyOCR) on the same images. Three images are corrupt or unreadable (TE-421, TE-618, TE-86); pipelines log an error and fill a default row for those.

---

## Unimodal (text-only) models

**Input:** OCR text from meme images. **Output:** 7-D binary prediction and, where applicable, confidence scores or raw model output.

### Run and succeeded

- **IndicBERT v2 (SS).** Sentence-similarity / embedding-based. Full run on 650 samples. Output: `unimodal/IndicBERTv2_SS_predictions.csv`.
- **IndicBERT v2 (Sam-TLM).** Same model, different head/config. Full run. Output: `unimodal/IndicBERTv2_Sam_predictions.csv`.
- **MuRIL.** Multilingual (Hindi/English). Full run. Output: `unimodal/MuRIL_predictions.csv`. Literature notes low diversity in outputs.
- **MentalBERT.** Mental-health-tuned BERT. Full run. Output: `unimodal/MentalBERT_predictions.csv`.
- **MentalRoBERTa.** Mental-health-tuned RoBERTa. Full run. Output: `unimodal/MentalRoBERTa_predictions.csv`.
- **BART (base).** Seq2seq/classification-style. Full run. Output: `unimodal/BART_predictions.csv`.
- **BART (large).** Same as above, larger. Full run. Output: `unimodal/BART_Large_predictions.csv`.
- **MentalBART.** Generative (Tianlin668/MentalBART). Separate script `run_mentalbart.py`. Prompt asks for symptom numbers 1–7; output is parsed. Full run. Output: `unimodal/MentalBART_predictions.csv`.

### Run and failed

None. All eight unimodal models completed without pipeline failure.

### Considered but not run

- **VisualBERT, CLIP, ViT, ResNet, EfficientNet (image-only or fusion).** Not run. They need a different pipeline (e.g. image-only encoder or a 7-way classification head). Our setup is text-only for unimodal and VLM free-form generation for multimodal.
- **GPT-4o / Claude / Gemini (API VLMs).** Not run. Requirement was self-hosted only; no API-based models.

---

## Multimodal (vision–language) models

**Input:** Image plus a short prompt asking which of the 7 symptoms appear. **Output:** 7-D binary derived from the model’s free-form text (parsed for numbers 1–7 or symptom keywords).

### Run and succeeded

- **LLaVA 1.5** (llava-hf/llava-1.5-7b-hf). USER/ASSISTANT format with image token. Full run. Output: `multimodal/LLaVA/LLaVA_predictions.csv`.
- **LLaVA-NeXT** (llava-hf/llava-v1.6-vicuna-7b-hf). Same style as LLaVA 1.5. Full run. Output: `multimodal/LLaVA_NeXT/LLaVA_NeXT_predictions.csv`.
- **MiniCPM-V** (openbmb/MiniCPM-V). Custom chat API (image + messages). Adapter in pipeline. Full run. Output: `multimodal/MiniCPM_V/MiniCPM_V_predictions.csv`.
- **IDEFICS** (HuggingFaceM4/idefics-9b-instruct). Prompt format: User:, image, question. Full run. Output: `multimodal/IDEFICS/IDEFICS_predictions.csv`.
- **BLIP-2** (Salesforce/blip2-flan-t5-xl). Standard BLIP-2 generate. Full run. Output: `multimodal/BLIP2/BLIP2_predictions.csv`.
- **BLIP** (Salesforce/blip-image-captioning-base). Captioning model; no instruction. Caption is generated then parsed for symptom keywords/numbers. Full run. Output: `multimodal/BLIP/BLIP_predictions.csv`.
- **PALO-7B** (MBZUAI/PALO-7B). LLaVA-style. Initial load/version issues; fixed. Full 650-image run; three image-file errors as above. Output: `multimodal/PALO/PALO_predictions.csv`.

### Run but poor utility

- **mBLIP** (Gregor/mblip-mt0-xl). Pipeline runs; full 650-image run completed (again with three image-file errors). Almost all predictions are [0,0,0,0,0,0,0]. Model typically responds with “None” or “Yes, it does.” and does not list symptom numbers. Prompt variants (stricter “numbers only”, etc.) did not fix this. Output: `multimodal/mBLIP/mBLIP_predictions.csv` (not useful for evaluation).
- **InstructBLIP** (Salesforce/instructblip-vicuna-7b). In pipeline and run; code comment notes possible device-side assert. Output: `multimodal/InstructBLIP/InstructBLIP_predictions.csv`.

### Run and failed (skipped)

- **Chitrarth-7.5B** (krutrim-ai-labs/Chitrarth). Multiple runs attempted; inference failed or was unstable. Causes: model built for older Hugging Face transformers; running with transformers 5 required patches (RoPE compat, tie_weights signature, GenerationMixin, DynamicCache vs list-of-tuples cache, forward signature, attention_mask in prepare_inputs_for_generation). Decided to skip; not production-ready in this stack.

### Considered but not run

- **VisualBERT, CLIP (multimodal fusion).** Would need a 7-way classification head and different training or zero-shot setup; not a drop-in for the current VLM free-form generation loop.
- **ViT / ResNet / EfficientNet (image-only).** Image-only encoders; would require a separate image-only branch and classifier.
- **API VLMs (GPT-4o, Claude, Gemini).** Not self-hosted; excluded by design.

---

## Parameters and experiment setup

### Unimodal

- **Text source:** EasyOCR on the same 650 test images; one text string per image.
- **Batch size:** 64 (run_unimodal.py).
- **Max token length (encoder models):** 128 for tokenizer.
- **Threshold:** Fixed 0.5 for binarizing scores where applicable; some models also have adaptive (per-symptom median) thresholds for analysis.
- **MentalBART:** Max input length 400 tokens (prompt + OCR), max new tokens 80, do_sample=False. Prompt: “Analyze this text for depression symptoms. List ONLY the numbers (1-7) that clearly apply…”

### Multimodal

- **Images:** Same 650 test images (or subset if --max_images is set).
- **Max new tokens:** 256 (default); Chitrarth branch caps at 64 for speed.
- **Generation:** do_sample=False (deterministic); temperature 0.0 where applicable.
- **Prompt:** Short instruction asking which of the 7 symptoms the meme shows and to reply with numbers (e.g. “1, 5”) or “None”. Symptom definitions (1–7) are in the prompt. mBLIP has a separate, stricter prompt (numbers or None only); it did not improve useful output.

### Hardware

- Single GPU (e.g. one of 4× A10G on g5.12xlarge). Multimodal runs use --only_model when needed to avoid OOM.

---

## Why models failed or performed poorly

- **Chitrarth.** Failure is technical: library and transformers version mismatch. Not a task-fit issue; the model was not brought to a stable run long enough to judge quality.
- **mBLIP.** Runs but is not useful: it rarely produces symptom numbers. Likely causes include (1) model not tuned for short, structured “list numbers” answers, (2) prompt not aligned with how the model was trained to respond, (3) conservative or generic answers (“None”, “Yes, it does.”). Bottleneck: model output format and possibly task alignment, not pipeline or parsing.
- **InstructBLIP.** May hit device-side assert in some runs; if it runs, outputs are usable. Bottleneck: stability on this hardware/driver stack.
- **Three image-file errors.** Corrupt or unreadable files (TE-421, TE-618, TE-86). Bottleneck: data quality; pipeline correctly logs and fills a default row.

---

## Bottlenecks (summary)

1. **Multimodal:** VRAM and runtime. Large VLMs (7B+) are run one model at a time (--only_model). Autoregressive generation (e.g. 256 max new tokens) is slow per image.
2. **Unimodal:** OCR quality and language. Text is Hindi/mixed; model performance depends on how well they handle this and how well OCR preserves meaning.
3. **Evaluation:** No human labels in the CSVs yet; human_label column is empty. So “success” here means pipeline ran and produced a 7-D prediction per image, not that predictions are validated against gold labels.
4. **Parsing:** VLMs output free-form text; we parse numbers 1–7 and keywords. Models that do not output numbers (e.g. mBLIP) get default [0,0,0,0,0,0,0] and cannot be evaluated meaningfully.
