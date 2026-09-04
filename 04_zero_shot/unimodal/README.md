# Unimodal (text-only) pipeline

- **run_unimodal.py** – Runs IndicBERT v2, MuRIL, MentalBERT, MentalRoBERTa, BART, BART-Large on OCR text. Input: `--data_dir` with test images; uses OCR or existing text.
- **run_unimodal_finetuned.py** – MAMA Table 2 text baselines + Hindi extensions: `--preset mama` (BERT-uncased, MentalBERT, BART, MentalBART) or `--preset all` (adds mBERT, IndicBERT×2, MuRIL, MentalRoBERTa, BART-large). OCR on `translated_train/`. See `MAMA_MEMEIA_FINETUNING_ALIGNMENT.md` and `bash run_mama_replication.sh`.
- **run_mentalbart.py** – MentalBART generative model; outputs `mentalbart_predictions.csv`.

**Canonical outputs:** One CSV per model, explicitly named: `IndicBERTv2_SS_predictions.csv`, `IndicBERTv2_Sam_predictions.csv`, `MuRIL_predictions.csv`, `MentalBERT_predictions.csv`, `MentalRoBERTa_predictions.csv`, `BART_predictions.csv`, `BART_Large_predictions.csv`, `MentalBART_predictions.csv`.

Run from project root, e.g.:
```bash
python unimodal/run_unimodal.py --data_dir translated_categorized_memes
python unimodal/run_mentalbart.py --data_dir translated_categorized_memes

# Fine-tuned text (translated train OCR → translated test)
python unimodal/run_unimodal_finetuned.py \
  --data_dir translated_categorized_memes \
  --train_json train.json \
  --train_image_dir translated_train \
  --output_dir unimodal/finetuned \
  --preset all
```
