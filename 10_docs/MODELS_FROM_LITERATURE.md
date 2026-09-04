# Models from Literature Survey – Fit for This Task

**Task:** Meme image (+ optional text) → 7 PHQ-9 style symptoms (Feeling Down, Lack of Interest, Self-Harm, Eating Disorder, Low Self-Esteem, Concentration Problem, Sleeping Disorder).

**MAMA-Memeia parity (fine-tuning):** See **`MAMA_MEMEIA_FINETUNING_ALIGNMENT.md`** — maps Table 2 baselines (text OCR, vision, metrics, splits) to this repo; dataset is your translated benchmark.

**Pipeline:** HuggingFace, self-hosted (no API). Unimodal = `run_unimodal.py` (text from OCR/captions). Multimodal = `run_multimodal.py` (image + prompt → VLM response → parsed symptoms).

---

## Summary table

| Model | Paper / section | Type | In our pipeline? | Would work? | Notes |
|-------|------------------|------|-------------------|-------------|--------|
| **LLaVA 1.5** | MAMA-Memeia, Zero-Shot VLMs | Multimodal VLM | ✅ Yes | ✅ | Working; USER/ASSISTANT + \<image\>. |
| **MentalBERT** | MAMA-Memeia | Unimodal text | ✅ Yes | ✅ | In run_unimodal. |
| **BART** | MAMA-Memeia | Unimodal text | ✅ Yes | ✅ | BART-base, BART-large in run_unimodal. |
| **MentalRoBERTa** | (mental health) | Unimodal text | ✅ Yes | ✅ | In run_unimodal. |
| **IndicBERT v2** | IndicBERT v2 section | Unimodal text | ✅ Yes | ✅ | SS + Sam-TLM in run_unimodal. |
| **MuRIL** | Multilingual section, CM-Off-Meme | Unimodal text | ✅ Yes | ✅ | Low diversity in our runs. |
| **MentalBART** | MAMA-Memeia | Unimodal text | ❌ No | ⚠️ Maybe | If on HuggingFace; same interface as BART. |
| **LLaVA-NeXT** | MAMA-Memeia | Multimodal VLM | ❌ No | ⚠️ Maybe | 59% macro-F1 in paper; check HF for `llava-next`. |
| **MiniCPM-V** | MAMA-Memeia | Multimodal VLM | ❌ No | ✅ Likely | On HF (e.g. openbmb/MiniCPM-V). Different API; needs adapter. |
| **InstructBLIP** | Zero-Shot VLMs | Multimodal VLM | ❌ No | ✅ Likely | HF: Salesforce/instructblip-vicuna-7b. BLIP-2 style. |
| **IDEFICS** | Zero-Shot VLMs | Multimodal VLM | ❌ No | ✅ Likely | On HF; needs prompt/parse adapter. |
| **PALO** (7B/13B) | PALO section | Multimodal VLM | ❌ No | ⚠️ Blocked | LLaVA-style; we had load errors (safetensors/torch). |
| **mBLIP** | mBLIP section | Multimodal VLM | ❌ Dropped | ❌ No | Echoed prompt; no useful generation. |
| **Chitrarth** | Chitrarth section | Multimodal VLM | ❌ No | ⚠️ Maybe | LLaVA-style; we had arch mismatches; could retry. |
| **VisualBERT** | MAMA-Memeia, Multi-Granular | Multimodal fusion | ❌ No | ⚠️ Different | Classification head for 7 classes; not free-form VLM. |
| **CLIP** | MAMA-Memeia, Memotion, etc. | Multimodal (image+text) | ❌ No | ⚠️ Different | Would need 7-way classifier on top; different pipeline. |
| **ViT / ResNet / EfficientNet** | MAMA-Memeia | Unimodal image | ❌ No | ⚠️ Different | Image-only; we’d need an image-only branch. |
| **GPT-4o / Claude / Gemini** | MAMA-Memeia | API VLMs | ❌ No | N/A | Not self-hosted; user said no API. |

---

## Recommended next steps (would work in this task)

1. **Unimodal**
   - **MentalBART:** If you find a HuggingFace MentalBART (MAMA-Memeia reports 61–65% macro-F1), add it to `run_unimodal.py` with the same BART-style config.

2. **Multimodal (VLMs) – same style as LLaVA**
   - **MiniCPM-V** (e.g. `openbmb/MiniCPM-V` or MiniCPM-V-2): Add to `run_multimodal.py` with a small adapter (load/generate/parse); prompt format may differ.
   - **InstructBLIP** (`Salesforce/instructblip-vicuna-7b`): BLIP-2-based; add with InstructBLIP processor + same symptom prompt/parse.
   - **Chitrarth:** Retry in pipeline with LLaVA-style prompt; fix arch mismatch if still present.
   - **PALO:** Fix loading (safetensors / torch version) then add as LLaVA-style model.

3. **Multimodal (fusion) – different pipeline**
   - **VisualBERT / CLIP:** Need a 7-class classification head and training or zero-shot setup; not drop-in for current VLM loop.

4. **Leave as-is**
   - **mBLIP:** Keep dropped (echo, no useful output).
   - **LLaVA 1.5:** Keep as-is per your choice.

---

## References in document

- **MAMA-Memeia:** Same 7 symptoms; benchmarks BERT, MentalBERT, BART, MentalBART, CLIP, VisualBERT, LLaVA 1.5, LLaVA-NeXT, MiniCPM-V, API models.
- **Multilingual section:** PALO, mBLIP, Chitrarth, MuRIL, IndicBERT, M-CLIP.
- **Zero-Shot VLMs (Hate Meme):** IDEFICS, LLaVA-1.5, InstructBLIP.
