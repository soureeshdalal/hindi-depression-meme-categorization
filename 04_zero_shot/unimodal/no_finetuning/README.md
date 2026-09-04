# Unimodal (text-only) pipeline

- **run_unimodal.py** – Runs IndicBERT v2, MuRIL, MentalBERT, MentalRoBERTa, BART, BART-Large on OCR text. Input: `--data_dir` with test images; uses OCR or existing text.
- **run_mentalbart.py** – MentalBART generative model; outputs `mentalbart_predictions.csv`.

**Canonical outputs:** One CSV per model, explicitly named: `IndicBERTv2_SS_predictions.csv`, `IndicBERTv2_Sam_predictions.csv`, `MuRIL_predictions.csv`, `MentalBERT_predictions.csv`, `MentalRoBERTa_predictions.csv`, `BART_predictions.csv`, `BART_Large_predictions.csv`, `MentalBART_predictions.csv`.

Run from project root, e.g.:
```bash
python unimodal/run_unimodal.py --data_dir translated_categorized_memes
python unimodal/run_mentalbart.py --data_dir translated_categorized_memes
```
