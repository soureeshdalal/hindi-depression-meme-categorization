#!/usr/bin/env bash
# Back-compat: MAMA text + vision + CLIP/ViT+BERT/VisualBERT FT + finetuned metrics only.
# Writes to default repo folders (--in-place). No generative-VLM BCE-head FT; no explanations.
# For a clean isolated run:  bash run_full_pipeline.sh   (omit --in-place)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/run_full_pipeline.sh" --in-place --no-explanations --no-vlm-ft "$@"
