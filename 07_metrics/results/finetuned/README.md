# Fine-tuned model metrics (translated train → translated test)

**Setup vs. MAMA-Memeia:** See `01_documentation/MAMA_MEMEIA_FINETUNING_ALIGNMENT.md` (same task + baseline family; your translated dataset).

Previous **English-train / translated-test** runs were removed (invalid domain mismatch).

After you re-run:

- `python unimodal/run_unimodal_finetuned.py ...` → saves `*_ft_predictions.csv` under `unimodal/finetuned/` (use `--output_dir unimodal/finetuned`).
- `python vision/run_vision.py ...` with `--train_dir translated_train` → saves `*_predictions.csv` under `vision/`.

Regenerate summaries:

```bash
python 03_scripts/compute_metrics_finetuned.py
```

**Isolated runs:** if predictions live under `runs/<id>/`, point the script there and pick a metrics folder under the same run:

```bash
python 03_scripts/compute_metrics_finetuned.py \
  --prediction-root runs/<id> \
  --metrics-out-dir runs/<id>/metrics/finetuned
```

(`bash run_full_pipeline.sh` does this automatically for default isolated runs.)
