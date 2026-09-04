# Multimodal models: performance, issues, and how to improve

This note summarizes the vision-language models (VLMs) run for Hindi mental health meme symptom classification. It covers what worked, what failed or was not useful, and how you can use this when designing your own pipeline.

---

## Task reminder

Input: the meme image (no OCR). The pipeline sends the image plus a short prompt asking which of the 7 PHQ-9 style symptoms the meme shows. Output: a 7-dimensional binary vector. The model generates free-form text; the pipeline parses that text for numbers 1–7 and for symptom keywords (in English and Hindi) to produce the binary vector. All models use the same 650 test images (with three known bad images that cause errors and get a default row). No human labels were used for evaluation yet; the notes below are based on whether each model runs, what it outputs, and how well that parses.

---

## Model-by-model summary

LLaVA 1.5 (7B)

LLaVA-style model with an image token and a user/assistant style prompt. No Hindi in the prompt.

Pros: Runs reliably. Follows the instruction to list symptom numbers. The parser often gets clear answers (e.g. "1, 5" becomes [1,0,0,0,1,0,0]). Raw responses are parsed into a confidence-style list for the CSV.

Cons: Limited Hindi support; the model is mainly English. In the runs, predictions are strongly biased toward symptom 1 (feeling down) and symptom 5 (low self-esteem). Many rows are [1,0,0,0,1,0,0] or [1,0,0,0,1,1,1]. So you get little spread across the other symptoms. Good for a baseline; not good for balanced multi-label behaviour without calibration or different prompting.

LLaVA-NeXT (7B)

Same family as LLaVA 1.5, newer variant. Same pipeline and prompt style.

Pros: Same as LLaVA 1.5: stable run, follows instructions, parses cleanly. Slightly more variation in the predictions (e.g. some [1,0,0,0,0,0,0] in addition to [1,0,0,0,1,0,0]).

Cons: Same language and bias limits. Still mostly symptoms 1 and 5; limited Hindi. Literature (e.g. MAMA-MEMEIA) reports better macro-F1 than LLaVA 1.5, but in this setup the output pattern is similar.

PALO-7B

Polyglot LLaVA-style model (MBZUAI). Supports Hindi and other languages. In the pipeline it uses the same English prompt as the others (Hindi prompt option exists in code but was not used in the reported runs).

Pros: Best Hindi capability among the models that ran successfully. Often returns short, parseable answers like "1, 5" or "1 , 5" (SentencePiece-style spacing is normalized by the parser). When it gives longer prose, the parser still extracts symptoms from keywords and numbered lists. Prediction vectors show more variety than LLaVA (e.g. [1,0,0,0,1,0,0], [1,1,0,0,0,1,1], [0,0,0,0,0,0,0]). Full 650-row run completed aside from the three known bad images.

Cons: Sometimes describes all seven symptoms in prose, so the parsed vector becomes all 1s for that sample. Load and version issues were fixed for this pipeline but may reappear on different environments (safetensors, torch, or custom PALO repo).

MiniCPM-V (3B)

Small VLM with a custom chat API (image plus messages). Adapter is implemented in the pipeline.

Pros: Smaller than the 7B models; lower VRAM. Different API is wrapped so it fits the same predict/parse flow.

Cons: In the available run, inference failed for the sampled images (INFERENCE_ERROR in the CSV). The prediction file in the repo has only one or two data rows. So in this setup MiniCPM-V did not produce a usable full run. Failures may be environment-specific (driver, CUDA, or API mismatch); worth retrying on a clean machine or with a different MiniCPM-V checkpoint.

IDEFICS (9B instruct)

BLIP-2-style model. Prompt: user block with image and the symptom question. Response is parsed with the same natural-language and number rules.

Pros: Runs and produces 650 rows. Parser handles both numeric answers and keyword-heavy prose. You see a mix of all zeros, all ones, and some mixed vectors (e.g. [1,1,0,1,0,0,1]). So there is real variation.

Cons: Output is bipolar: either the model says nothing relevant (parsed as [0,0,0,0,0,0,0]) or it lists many symptoms (often all seven). Few rows have a small, sparse set of symptoms. So the model tends to be either very conservative or over-inclusive. Raw responses are converted to a confidence list; the underlying text is not always a clean "1, 5" style list.

BLIP-2 (Flan-T5-XL)

Standard BLIP-2 with a Flan-T5-XL language model. Same prompt and parse logic as the other instruction VLMs.

Pros: Runs on the full set. Often returns short numeric answers that parse well (e.g. "1" or "1, 2, 3, 4, 5, 6"). Predictions vary: some [1,0,0,0,0,0,0], some [0,1,0,0,0,0,0], some [1,1,1,1,1,1,0] or [1,1,1,1,1,1,1]. So you get more spread across symptoms than LLaVA.

Cons: No Hindi; English-only. When the model lists many numbers, you again get an all-1s or nearly all-1s vector. No calibration, so the choice of "which symptoms to list" is inconsistent.

BLIP (base, image captioning)

Captioning model only; it does not take an instruction. The pipeline generates one caption per image and then runs the same keyword/number parser on that caption.

Pros: Lightweight. Runs and produces 650 rows. When the caption happens to contain symptom-related words (e.g. "eating", "depression", "anxiety"), the parser sets the corresponding symptom bits. No need for instruction-following.

Cons: Most captions are generic (e.g. "a man in white shirt", "hindi video chat"). So most rows are [0,0,0,0,0,0,0]. Only a small fraction of images get any symptom predicted. The model was not trained for mental health or for following a question; it just describes the scene. Useful only as a weak baseline or for keyword-only signal.

mBLIP (mT0-XL)

Multilingual BLIP-2-style model. The pipeline uses a stricter prompt (numbers or "None" only) to try to get list-style answers.

Pros: Runs without crashing. Literature mentions good multilingual support.

Cons: In practice the model almost never returns symptom numbers. Typical raw responses are "None." or "Yes, it does." or short non-list answers. The parser then outputs [0,0,0,0,0,0,0] for almost every sample. Stricter prompts were tried and did not fix this. So for this task the output has no useful diversity; the bottleneck is model behaviour (not trained to answer with a short list of numbers), not the pipeline. The prediction file in the repo may be from a short or aborted run; a full run would still be expected to be almost all zeros.

InstructBLIP (Vicuna-7B)

Instruction-following BLIP-2 variant. Same prompt style as the other VLMs.

Pros: In principle fits the task (image + question, then parse). Listed in the pipeline config.

Cons: In the run recorded in the repo, every row is INFERENCE_ERROR. Code comments note that this model may hit a device-side assert (e.g. CUDA or driver). So the pipeline completes (650 rows) but no image gets a valid prediction. Not usable in this environment until the assert is resolved or a different build is used.

Chitrarth (7.5B)

Hindi-capable LLaVA-style model (Krutrim). Would be a good fit for language.

Pros: Designed for Indian languages; same LLaVA-style interface.

Cons: Not production-ready in this stack. Multiple runs hit inference errors (e.g. missing attributes, wrong signature for inputs_embeds, attention_mask, or generation cache). The model was built for an older Hugging Face transformers version; running it with transformers 5 required many patches and was still unstable. The pipeline was told to skip Chitrarth. The Chitrarth CSV in the repo has only a couple of rows, with errors in the raw column. Do not rely on it until the model is updated or the stack is pinned to a compatible library set.

---

## Summary: who did well, who did poorly

Relatively better (for this setup, without human F1):

- PALO-7B: best balance of Hindi support and parseable, varied outputs; often short "1, 5" style answers; sometimes over-lists in prose.
- LLaVA 1.5 and LLaVA-NeXT: stable, follow instructions, but strong bias toward symptoms 1 and 5 and limited Hindi.
- BLIP-2: good variety in which symptoms are predicted; no Hindi; sometimes over-lists.
- IDEFICS: mix of zeros and ones; bipolar (either nothing or many symptoms).

Weak or not useful:

- BLIP: caption-only; most rows zeros; only occasional keyword hits.
- mBLIP: runs but almost never gives symptom numbers; effectively all zeros.
- InstructBLIP: full run failed with inference errors on every image.
- MiniCPM-V: inference errors in the available run; no full usable CSV.
- Chitrarth: skipped due to library and inference errors; not stable in this stack.

---

## Issues to overcome in your own framework

1. Output format: VLMs generate free-form text. The pipeline parses numbers 1–7 and symptom keywords (English and Hindi). Models that do not output numbers (mBLIP) or that output prose listing every symptom (PALO, IDEFICS, MentalBART in unimodal) lead to useless or biased vectors. You need either models that follow a strict format (e.g. "1, 5" or "None") or a parsing and confidence strategy that can handle prose without marking everything as present.

2. Language: Memes are Hindi or code-mixed. LLaVA, LLaVA-NeXT, BLIP-2, IDEFICS, BLIP are English-centric. PALO is the only one in this set with strong Hindi support that ran successfully. For a Hindi-first benchmark, prefer Hindi-capable VLMs or add a Hindi prompt/prefix and validate that answers stay parseable.

3. Bias: Even when the model follows instructions, predictions can be skewed (e.g. LLaVA toward 1 and 5). Without human labels you cannot tune thresholds or recalibrate. With labels, you can add per-symptom or per-model calibration and report F1.

4. Stability: InstructBLIP (device assert), MiniCPM-V (inference errors), and Chitrarth (library/API mismatch) show that not every VLM runs reliably on every machine. Plan for fallbacks (skip model, retry with different batch size or precision) and log errors so you know which model failed where.

5. Caption-only models: BLIP is a reminder that captioning alone is a poor fit for "which of these 7 symptoms?" unless you add a second stage (e.g. a classifier on the caption). Prefer instruction-following VLMs for this task.

6. Resource use: Models run one at a time to avoid OOM (e.g. on a single 24 GB GPU). PALO-7B and LLaVA-7B need about 16 GB VRAM each. IDEFICS-9B and similar need more. If you add larger models (e.g. PALO-13B), plan for multi-GPU or sequential runs with model clear between them.

7. Bad images: Three test images (TE-421, TE-618, TE-86) are corrupt or unreadable. The pipeline logs an error and fills a default row. Any downstream analysis should exclude or flag these if you care about per-image metrics.

---

## How to use this when building your own framework

- Prefer VLMs that both support your language (e.g. PALO for Hindi) and follow instructions well enough to return short lists (e.g. "1, 5" or "None"). If a model often returns long prose that lists all symptoms, either change the prompt to force sparsity (e.g. "at most two numbers") or treat that model as over-inclusive and calibrate with labels.

- Do not assume every model in the config will run. Include health checks: run a few images per model, check for INFERENCE_ERROR or all-zero outputs, and drop or retry models that always fail. Keep an error log and document which environment (CUDA, drivers, transformers version) was used.

- For parsing: keep a single parse function (numbers 1–7 plus keyword fallback) so all VLMs are compared on the same rules. Add Hindi keywords if your data and prompts are Hindi. If you add new models that output in a different format (e.g. JSON), extend the parser but keep the same 7-D binary output.

- Use BLIP-style caption-only models only as a weak baseline or for ablations (e.g. "image caption + text classifier" vs "VLM with instruction"). Do not rely on them for the main symptom predictions.

- If you have human labels, compute macro-F1 and weighted-F1 per model and per symptom. Then you can rank models, tune prompts, and decide which VLMs are worth keeping. Without labels, use output diversity (e.g. fraction of rows that are not all zeros and not all ones) and language fit as a rough filter.

- Document VRAM and runtime per model so others can reproduce. Run models sequentially and clear GPU memory between them to avoid OOM; use --only_model when debugging a single VLM.

---

## File references

Prediction CSVs: under `multimodal/`, one folder per model (e.g. `multimodal/PALO/PALO_predictions.csv`, `multimodal/LLaVA/LLaVA_predictions.csv`). Models with full 650-row runs: PALO, LLaVA, LLaVA_NeXT, BLIP2, IDEFICS, BLIP; InstructBLIP has 650 rows but all INFERENCE_ERROR. Short or failed runs: MiniCPM_V, mBLIP, Chitrarth.

Pipeline: `multimodal/run_multimodal.py`. Model list and types in MODEL_CONFIGS; parsing in `parse_natural_language_response`. Single-model run: `--only_model PALO` (or LLaVA, BLIP2, etc.).

Errors: `multimodal_errors.log` (if used); bad images TE-421, TE-618, TE-86 are documented in EXPERIMENT_MODELS_AND_SETUP.md and MODELS_TRIED_SUMMARY.md.
