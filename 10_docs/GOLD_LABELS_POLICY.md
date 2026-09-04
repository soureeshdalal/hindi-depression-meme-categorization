# Gold labels: `train.json` vs `test.json`

This repo keeps **human annotations** separate from **evaluation** so there is no test-set leakage during fine-tuning.

## `test.json`

- **Role:** Held-out **evaluation** labels only (e.g. `TE-*` `sample_id` values).
- **Used by:** `03_scripts/compute_metrics.py`, `compute_metrics_finetuned.py`, `compute_metrics_hinglish.py` — to fill `human_label` on prediction CSVs and compute F1 / precision / recall.
- **Not used by:** Any fine-tuning script. None of the `*_finetuned*.py` or `run_*multilabel*` trainers open `test.json`.

## `train.json`

- **Role:** **Training-split** supervision only (e.g. `TR-*` ids), paired with images under `translated_train/`.
- **Used by:** Supervised fine-tuning scripts (unimodal text FT, vision FT, MAMA multimodal FT, VLM BCE-head FT). Labels drive the loss **only** on training images.
- **Not used for:** Scoring the test set during training. Test memes are only run through the model to produce predictions; gold for test comes from `test.json` **later**, in metric scripts.

## Accidental misuse

If you pass `test.json` as `--train_json`, training will **abort** with an error (`assert_train_json_not_test_gold` in `multimodal/mama_finetune_common.py`).

## Explanation benchmarks (`explanation_runs/`)

- `gold_labels` in the Gemini CSV are used to **compute metrics after inference**, not to fine-tune the backbone in those scripts (the explanation unimodal path does not run a supervised training loop on labels).

## If you forbid *all* supervised fine-tuning

MAMA-style **Table 2** baselines are **supervised** and expect **training** labels (`train.json` + train images). If your protocol disallows using any human gold for training, skip the `*_finetuned*` steps and use zero-shot / no-FT runners only; `test.json` remains for metrics when you have predictions.
