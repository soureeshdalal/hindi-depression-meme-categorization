# BLIP explanation+image diagnostic (Experiment 4)

This folder mirrors **`blip_inference/`** at the repo root for metrics bookkeeping.

- **Summary:** `../../blip_inference/experiment_4_blip_failure_summary.txt`
- **Run artifact:** `../../blip_inference/04_blip_inference` (Colab export / notebook payload)
- **Scored metrics:** none — run stopped after prompt-echo failure (`n_scored = 0`)

The same row is recorded in **`metrics/colab/colab_multimodal_metrics.csv`** (`track=explanation_image_colab_diagnostic_failed`).

For a **pipeline-compatible** BLIP-Base explanation+image run, use  
`python explanation_runs/scripts/run_multimodal.py --repo-root . --only BLIP`  
(`Salesforce/blip-image-captioning-base`, per-symptom YES/NO path).
