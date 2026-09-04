# Isolated pipeline outputs

`bash run_full_pipeline.sh` (without `--in-place`) creates a timestamped folder here, e.g. `pipeline_20260225_153045/`, containing a full mirror of prediction CSVs and metrics for that run only. Older results under `unimodal/finetuned/`, `vision/`, etc. are left unchanged.

Use `--in-place` or `run_mama_replication.sh` to write to the legacy repo paths instead.
