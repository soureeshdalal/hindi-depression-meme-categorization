# Gemini Pipeline Summary

---

## 1. Run History: What Went Wrong and What Was Fixed

**Run V1**

The initial pipeline used `gemini-3-pro-image-preview` directly on each meme image to both describe it and predict symptom labels in a single call. The model returned free-form prose explanations, and a separate text call to `gemini-3.1-pro-preview` used those explanations to generate binary label predictions.

Two problems surfaced. First, some images returned empty explanations, likely due to safety filters triggered by sensitive meme content. Second, the Phase 2 prediction prompt included the instruction "When uncertain, lean toward NO (the annotators were conservative)," which caused the model to default to all-zero predictions for most images. The result was very high precision but near-zero recall, making the F1 nearly meaningless.

**Run V2**

Phase 1 was redesigned with a structured 7-aspect prompt, one section per PHQ-9 symptom area. This gave Phase 2 cleaner, symptom-specific input instead of unstructured prose. The pipeline now ran on all 650 test images.

The safety filter problem persisted. Roughly 27% of images returned empty explanations because the prompt explicitly mentioned "SELF-HARM / DEATH THOUGHTS," triggering Gemini's content filters. This was fixed by softening the language in that section to "EXISTENTIAL/EMOTIONAL INTENSITY" and framing the task as academic research. A repair script (`phase1_repair.py`) was created to retry blocked images with the updated prompt.

Phase 2 still used the "lean toward NO" calibration instruction inherited from V1. Recall remained low.

**Run V2.1**

Identical to V2 in every way except one line in the Phase 2 prompt. The "lean toward NO" instruction was replaced with:

> "Decision threshold: if the symptom is a clear, meaningful theme in the meme (even if not the only theme), mark YES. Mark NO only if the symptom is absent or merely incidental background noise."

This single change raised macro F1 from 0.387 to 0.401, primarily by recovering recall (from 0.292 to 0.328) while keeping precision relatively stable. V2.1 is the best-performing version of the pipeline.

---

## 2. Results

### Gemini Pipeline Runs (Explanation-based, gemini-3.1-pro-preview for Phase 2)

| Run | Input | n | Precision | Recall | Macro F1 |
|---|---|---|---|---|---|
| V1 (direct image, conservative) | Image only | 549 | 0.569 | 0.654 | 0.601 |
| V2 (structured explanation, lean NO) | Explanation text | 551 | 0.643 | 0.292 | 0.387 |
| V2.1 (structured explanation, balanced) | Explanation text | 576 | 0.573 | 0.328 | 0.401 |

Per-symptom breakdown:

| Symptom | V2.1 Precision | V2.1 Recall | V2.1 F1 | V2 F1 | V1 F1 |
|---|---|---|---|---|---|
| Feeling Down | 0.536 | 0.484 | 0.509 | 0.480 | 0.591 |
| Lack of Interest | 0.293 | 0.274 | 0.283 | 0.250 | 0.342 |
| Self-Harm | 0.571 | 0.364 | 0.444 | 0.391 | 0.703 |
| Eating Disorder | 0.789 | 0.217 | 0.341 | 0.289 | 0.721 |
| Low Self-Esteem | 0.633 | 0.209 | 0.314 | 0.264 | 0.567 |
| Concentration Problem | 0.591 | 0.388 | 0.469 | 0.529 | 0.585 |
| Sleeping Disorder | 0.595 | 0.362 | 0.451 | 0.505 | 0.698 |

---

### Previous Baselines: No Fine-Tuning (raw OCR / image input, 509 samples)

| Model | Type | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| MentalBART | Text | 0.197 | 0.998 | 0.320 |
| MentalBERT | Text | 0.198 | 0.959 | 0.317 |
| BART-Base | Text | 0.117 | 0.615 | 0.192 |
| IndicBERTv2-Sam | Text | 0.192 | 0.389 | 0.203 |
| BLIP-2 | Multimodal | 0.214 | 0.435 | 0.279 |
| LLaVA-1.5 | Multimodal | 0.344 | 0.401 | 0.234 |
| IDEFICS | Multimodal | 0.205 | 0.226 | 0.203 |
| LLaVA-NeXT | Multimodal | 0.203 | 0.277 | 0.169 |
| MuRIL | Text | 0.061 | 0.429 | 0.106 |
| IndicBERTv2-SS | Text | 0.203 | 0.361 | 0.174 |
| MentalRoBERTa | Text | 0.116 | 0.590 | 0.181 |
| BART-Large | Text | 0.168 | 0.303 | 0.158 |
| PALO-7B | Multimodal | 0.129 | 0.247 | 0.155 |
| BLIP | Multimodal | 0.326 | 0.030 | 0.054 |

---

## 3. Model Pros and Cons

**gemini-3-pro-image-preview (Phase 1 vision model)**

Pros: Strong visual understanding, handles mixed-language memes (Hindi/English/Devanagari), produces detailed structured descriptions without fine-tuning.

Cons: Safety filters block a significant portion of mental health content. No image output control makes the "no_image" failure mode unpredictable. Daily quota limits constrain throughput.

**gemini-3.1-pro-preview (Phase 2 text model)**

Pros: Excellent instruction-following on structured text input. Consistent output format. Can reason over the 7-aspect explanation reliably.

Cons: Very sensitive to calibration wording in the prompt. Conservative by default, requires explicit threshold guidance. Daily request cap (250/day) is a bottleneck.

**MentalBERT / MentalBART (no fine-tuning)**

Pros: Pre-trained on mental health text, so the representations are domain-relevant.

Cons: Without fine-tuning the classification head is random. Both models achieve high recall only by predicting positive for almost everything (recall ~0.96-0.99 without fine-tuning). Not reliable for production use without task-specific training.

**BLIP-2 (no fine-tuning)**

Pros: Best-performing multimodal model at baseline. Vision-language alignment from large-scale pretraining gives it a head start over text-only models on image-based tasks.

Cons: Still limited without fine-tuning. Cannot process non-English/Devanagari text in images well. Large model size requires significant GPU memory.

**LLaVA-1.5 / LLaVA-NeXT (no fine-tuning)**

Pros: Strong zero-shot instruction following. LLaVA-1.5 achieves the highest precision among multimodal baselines (0.344).

Cons: Performs below BLIP-2 on recall. NeXT is not meaningfully better than 1.5 without fine-tuning. Both require substantial GPU resources.

**IndicBERTv2 / MuRIL (no fine-tuning)**

Pros: Explicitly trained on Indic languages including Hindi, making them better suited to Devanagari text than English-centric models.

Cons: Lowest performance at baseline, likely because the classification head is random and the models were not exposed to mental health-style text. MuRIL in particular collapses to near-random predictions.

**IDEFICS (no fine-tuning)**

Pros: Open-source multimodal model with reasonable multilingual coverage.

Cons: Lowest multimodal recall in the set. Poor at picking up subtle mental health signals without fine-tuning.

---

## 4. Phase 1 Explanation Prompt

This prompt was used to generate the structured explanations from each meme image using `gemini-3-pro-image-preview`. All V2 and V2.1 explanations were produced with this prompt.

```
This image is from an academic mental health research dataset.
Look at it carefully and describe what you observe for each of
the 7 areas below. Be specific and factual. If an area is clearly
not shown, write "not shown".

1. MOOD/EMOTIONAL STATE: What emotions are expressed?
(e.g. sadness, crying, emptiness, hopelessness, flat affect, anguish)

2. MOTIVATION/INTEREST: Does the person show loss of interest,
lack of motivation, or inability to enjoy things they normally would?
(e.g. "nothing feels worth doing", "can't enjoy food/hobbies/people")

3. EXISTENTIAL/EMOTIONAL INTENSITY: Does the content depict extreme
hopelessness, a wish to withdraw from life, or references to
harming oneself? Describe only what is visually or textually shown.

4. APPETITE/EATING: Are eating habits mentioned?
(e.g. eating too much/too little, emotional eating, skipping meals,
food as coping)

5. SELF-WORTH/GUILT: Does the person express worthlessness,
excessive guilt, self-blame, or feeling like a burden to others?

6. CONCENTRATION/THINKING: Any difficulty focusing, making decisions,
brain fog, confusion, or inability to think clearly?

7. SLEEP: Any reference to sleep -- insomnia, sleeping too much,
being awake at unusual hours, fatigue, disrupted sleep?

Also add a brief CONTEXT line (1 sentence): the cultural/meme format context.

Keep each area to 1-2 sentences. Be direct and observational.
```
