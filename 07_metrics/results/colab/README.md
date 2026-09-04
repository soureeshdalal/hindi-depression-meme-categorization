# Colab multimodal metrics (supplement)

- **`colab_multimodal_metrics.csv`** — Rows from `colab_results/` that are **not** produced by `multimodal/run_multimodal.py` + `compute_metrics.py`.

**InstructBLIP (no-FT):** The primary headline number is also appended to `metrics/no_finetuning/metrics_summary.csv` as `InstructBLIP-Vicuna-7B` with `n=650` (full test), distinct from the older `n=509` generative-VLM rows.

**BLIP-2:** The Colab explanation+image run is recorded here for traceability only; do not treat it as comparable to `explanation_runs/multimodal` pipeline results.

**BLIP-VQA (Experiment 4):** Diagnostic attempt with `Salesforce/blip-vqa-capfilt-large` under `blip_inference/` — prompt echo, no prediction CSV; see `metrics/blip_inference/README.md`.
