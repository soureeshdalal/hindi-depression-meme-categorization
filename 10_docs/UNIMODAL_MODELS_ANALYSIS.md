# Unimodal models: performance, issues, and how to improve

This note summarizes what worked and what did not among the eight unimodal (text-only) models run for Hindi mental health meme symptom classification. It then suggests a simple way to use this when designing your own setup.

---

## Task reminder

Input: OCR text from meme images (Hindi/mixed). Output: a 7-dimensional binary vector (one bit per PHQ-9 style symptom). All models use the same OCR and the same 650 test images. No human labels were used for evaluation yet; the notes below are based on how each model behaves (output spread, bias, and known limits).

---

## Model-by-model summary

IndicBERTv2 (SS)

This is the sentence-similarity style IndicBERTv2 model. In the pipeline it is used with a classifier head on top (same as the other encoder models).

Pros: Good Hindi support (Indic). Raw scores spread in a reasonable range (around 0.46–0.53). Predictions change across samples; you see a mix of all-zero and mixed 0/1 vectors, so the model is not stuck on one output.

Cons: Scores are still close to 0.5, so the added classifier head is only weakly discriminative. Without task-specific training, it is only a modest baseline.

IndicBERTv2 (Sam-TLM)

Same family as above, different pretrained variant (MLM-Sam-TLM).

Pros: Same good Hindi support. Prediction vectors vary across samples.

Cons: Raw scores are again clustered near 0.5 (about 0.46–0.53). The model does not strongly separate “symptom present” from “absent”; small changes in score cross the 0.5 threshold and flip labels. So behaviour is similar to SS: weakly discriminative without fine-tuning.

MuRIL

Multilingual (Hindi/English) encoder with a classifier head on top.

Pros: Runs without errors. Handles Hindi and English text.

Cons: Very low output diversity. Raw scores are almost the same on every sample (around 0.49–0.51 for all seven symptoms). So the model effectively outputs the same prediction for almost every input. This was also noted in prior work (e.g. CM-Off-Meme). For this task, MuRIL in the current setup is not useful.

MentalBERT

Mental-health-tuned BERT (large) with a classifier head. The pipeline uses a fixed threshold of 0.4 and an optional adaptive (per-symptom median) threshold.

Pros: Domain fit (mental health). Raw scores show real spread (roughly 0.38–0.67). The model does react to the text.

Cons: It tends to predict “symptom present” too often. Many rows are [1,1,1,1,1,0,1] or [1,1,1,1,1,1,1]. So you get high recall but likely low precision; the model is biased toward seeing depression in most inputs. Useful if you want to avoid missing cases; not useful if you need balanced or conservative predictions without tuning.

MentalRoBERTa

Mental-health-tuned RoBERTa (base) with a classifier head. Fixed threshold 0.5.

Pros: Same mental health domain. Raw scores vary (about 0.43–0.59). More variety in prediction vectors than MentalBERT (e.g. many [0,1,0,1,1,0,1] instead of almost all 1s).

Cons: Scores still cluster in the middle. Many predictions look similar across samples (e.g. symptom 2, 4, 5, 7 often 1). So there is some bias toward certain symptoms and limited discrimination without calibration or training.

BART (base)

Standard BART with a classifier head. Pipeline uses fixed threshold 0.3 and an optional adaptive threshold.

Pros: Predictions differ across samples. Raw scores have a clear spread; the model is not stuck at 0.5.

Cons: Strong symptom bias. In the runs, symptom 1 (feeling down) is almost always predicted 0 (scores around 0.03), while symptoms 4–7 (eating, self-esteem, concentration, sleep) are often above threshold. So you get a skewed pattern (e.g. [0,0,0,1,1,1,1] or [0,1,0,1,1,1,1]) rather than balanced behaviour. This likely comes from using a generic BART with a randomly initialized head and no task training.

BART (large)

Larger BART, same pipeline idea.

Pros: More capacity. Raw scores show clear variation.

Cons: Different but still strong bias. In the runs, symptom 1 (feeling down) and symptom 7 (sleep) are often high (e.g. 0.65–0.77), while symptoms 2–6 are often below 0.5. So the model often predicts [1,0,0,0,0,0,1]. Again, the head is not trained for this task, so you get systematic bias rather than balanced multi-label behaviour.

MentalBART

Generative model (Tianlin668/MentalBART). Prompt: “Analyze this text for depression symptoms. List ONLY the numbers (1-7) that clearly apply. If none, say None.” Output is parsed from generated text (numbers and keywords).

Pros: Mental health oriented. Runs and produces a 7-D vector per sample.

Cons: Output is not useful for discrimination. The model usually generates long prose that lists all seven symptoms (e.g. “feeling down, lack of interest, self-harm, eating disorder…”). The parser then sets all seven bits to 1. So almost every sample gets [1,1,1,1,1,1,1]. No diversity and no way to tell which memes actually show which symptoms. The bottleneck is the model’s tendency to describe all symptoms in the prompt rather than only the ones that apply.

---

## Summary: who did well, who did poorly

Relatively better (for this setup, without human F1):

- IndicBERTv2 SS: best balance of Hindi support and some prediction diversity; scores near 0.5 but not collapsed to one vector.
- MentalRoBERTa: domain fit and more varied vectors than MentalBERT; still biased toward certain symptoms.
- BART base / BART large: varied scores and predictions, but with strong symptom-specific bias (different for base vs large).

Poor or not useful:

- MuRIL: no real diversity; same scores and effectively same prediction everywhere.
- MentalBART: almost always all 1s; parsing cannot recover which symptoms actually apply.
- MentalBERT: if you need precision, the “predict almost everything” behaviour is a con; only useful as a high-recall option.

---

## Issues to overcome in your own framework

From the above, the main issues are:

1. Models that were not trained for this task (IndicBERTv2, MuRIL, BART) get a randomly initialized classifier head. Their scores cluster near 0.5 (IndicBERTv2, MuRIL) or show strong bias (BART). So zero-shot classification with a fresh head is unreliable.

2. Mental-health-tuned encoders (MentalBERT, MentalRoBERTa) help with domain but still need a head trained for your labels and your data; otherwise you get bias (e.g. MentalBERT toward “all present”, MentalRoBERTa toward a subset of symptoms).

3. Relying on a generative model (MentalBART) to output only symptom numbers fails when the model instead outputs prose that mentions every symptom; the parser then cannot discriminate.

4. OCR is shared by all unimodal models. Any noise, script mix, or loss of meaning in OCR affects every text-based model. Improving OCR or adding a cleaning step helps all of them.

5. Thresholds matter. The pipeline uses fixed (and sometimes adaptive) thresholds. Models with scores clustered near 0.5 are very sensitive to the choice of threshold; biased models need per-symptom or per-model calibration if you want balanced behaviour.

---

## How to use this when building your own framework

- Prefer encoders that both support your language (e.g. IndicBERTv2 for Hindi) and, if possible, match the domain (e.g. mental health). Combine language fit with a head trained on your task (fine-tune on labeled data) instead of using a zero-shot head only.

- Do not use a single “best model” from this list without validation. Run multiple models and compare prediction distributions (e.g. how many unique vectors, how often each symptom is 1). Drop or downweight models that collapse to one vector (MuRIL-like) or always predict all 1s (MentalBART-like).

- For generative setups, avoid prompts that ask for “all symptoms that might apply” in a way that encourages the model to list all seven. Prefer formats that force a sparse answer (e.g. “at most two numbers” or “only the single strongest symptom”) and parse strictly, or avoid generative models for the final 7-D label if they do not follow the format.

- Treat mental-health-tuned models (MentalBERT, MentalRoBERTa) as candidates for fine-tuning, not as ready-made classifiers. Use adaptive or learned thresholds per symptom if you have labels, to correct for “all present” or “only 1 and 7” type bias.

- Use BART (or similar) only if you are willing to train or at least calibrate the head; otherwise the strong symptom bias (base vs large differing) will dominate. If you have labels, train the head; if not, consider ensemble or rules to balance symptom-specific bias.

- Add an explicit OCR-quality or text-quality step (e.g. filter or flag low-confidence OCR, or use only samples with sufficient Hindi text). That improves inputs for every unimodal model.

- Where possible, collect human labels for at least a subset of images. Then you can compute F1 (macro/weighted) and per-symptom metrics, and decide which models and thresholds are worth keeping in your framework. Without labels, use diversity and bias checks (as above) to avoid clearly bad models (no diversity or all 1s).

---

## File references

Unimodal prediction CSVs: `unimodal/IndicBERTv2_SS_predictions.csv`, `IndicBERTv2_Sam_predictions.csv`, `MuRIL_predictions.csv`, `MentalBERT_predictions.csv`, `MentalRoBERTa_predictions.csv`, `BART_predictions.csv`, `BART_Large_predictions.csv`, `MentalBART_predictions.csv`.

Pipeline: `unimodal/run_unimodal.py` (all except MentalBART), `unimodal/run_mentalbart.py` (MentalBART). Model configs and thresholds: see MODEL_CONFIGS in `run_unimodal.py`.
