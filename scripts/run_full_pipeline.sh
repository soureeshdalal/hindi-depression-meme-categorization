#!/usr/bin/env bash
# =============================================================================
# One-command pipeline (MAMA-aligned FT + explanation-conditioned benchmarks)
# =============================================================================
# By default this run is ISOLATED under runs/pipeline_YYYYMMDD_HHMMSS/ so new
# CSVs and metrics do not overwrite or mix with previous work. Use --in-place
# to write to the legacy repo paths (unimodal/finetuned, vision/, …).
#
# Gold labels: train.json = training only; test.json = metrics only (see docs).
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Single visible GPU avoids device_map="auto" spreading a VLM across cuda:0–N while the
# BCE head lived on cuda:0 (see multimodal/run_vlm_multilabel_finetune.py). Override if needed:
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SKIP_EXPL=0
SKIP_VLM=0
WITH_NOFT_METRICS=0
EXPL_CSV_CLI=""
IN_PLACE=0
RUN_DIR=""

usage() {
  cat <<'EOF'
run_full_pipeline.sh — MAMA-aligned FT + explanation benchmarks

Output layout (default: isolated — does not touch prior CSVs in unimodal/finetuned, etc.):
  runs/pipeline_YYYYMMDD_HHMMSS/
    unimodal/finetuned/*.csv
    vision/*.csv
    multimodal/finetuned_mama/*.csv
    multimodal/finetuned_vlm/*.csv
    metrics/finetuned/finetuned_metrics_*.csv
    explanation_runs/unimodal/  and  explanation_runs/multimodal/

Options:
  --in-place                Write to repo default dirs (legacy; may skip if CSVs exist)
  --run-dir PATH            Isolated root: PATH relative to repo, or absolute (implies not --in-place)
  --explanations-csv PATH   Phase-1 explanations CSV
  --no-explanations         Skip explanation steps
  --no-vlm-ft               Skip generative VLM BCE-head FT
  --with-no-ft-metrics      Run compute_metrics.py (still uses metrics/no_finetuning/)
  -h, --help

Environment: HF_TOKEN, EXPLANATIONS_CSV
Docs: 01_documentation/FULL_PIPELINE.md
AWS:  01_documentation/AWS_RUN.md
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-explanations) SKIP_EXPL=1; shift ;;
    --no-vlm-ft)       SKIP_VLM=1; shift ;;
    --with-no-ft-metrics) WITH_NOFT_METRICS=1; shift ;;
    --in-place)        IN_PLACE=1; shift ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --explanations-csv)
      EXPL_CSV_CLI="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

EXPL_CSV="${EXPL_CSV_CLI:-${EXPLANATIONS_CSV:-$ROOT/gemini_runs/run_v2/phase1_explanations_v2.csv}}"

# --- Resolve output root: isolated timestamp by default ---
if [[ "$IN_PLACE" -eq 1 ]]; then
  BASE_OUT=""
elif [[ -n "$RUN_DIR" ]]; then
  if [[ "$RUN_DIR" = /* ]]; then
    BASE_OUT="$RUN_DIR"
  else
    BASE_OUT="$ROOT/$RUN_DIR"
  fi
else
  BASE_OUT="$ROOT/runs/pipeline_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -n "$BASE_OUT" ]]; then
  mkdir -p \
    "$BASE_OUT/unimodal/finetuned" \
    "$BASE_OUT/vision" \
    "$BASE_OUT/multimodal/finetuned_mama" \
    "$BASE_OUT/multimodal/finetuned_vlm" \
    "$BASE_OUT/metrics/finetuned" \
    "$BASE_OUT/explanation_runs/unimodal" \
    "$BASE_OUT/explanation_runs/multimodal"
  OUT_UNIMODAL_FT="$BASE_OUT/unimodal/finetuned"
  OUT_VISION="$BASE_OUT/vision"
  OUT_MAMA="$BASE_OUT/multimodal/finetuned_mama"
  OUT_VLM="$BASE_OUT/multimodal/finetuned_vlm"
  OUT_MET_FIN="$BASE_OUT/metrics/finetuned"
  OUT_EXPL_U="$BASE_OUT/explanation_runs/unimodal"
  OUT_EXPL_M="$BASE_OUT/explanation_runs/multimodal"
  MET_CMD=(python 03_scripts/compute_metrics_finetuned.py \
    --repo-root "$ROOT" \
    --prediction-root "$BASE_OUT" \
    --metrics-out-dir "$OUT_MET_FIN")
  EXPL_U_ARGS=(--output-dir "$OUT_EXPL_U")
  EXPL_M_ARGS=(--output-dir "$OUT_EXPL_M")
  echo ""
  echo ">>> Isolated run — all new artifacts under:"
  echo "    $BASE_OUT"
  echo ""
else
  OUT_UNIMODAL_FT="$ROOT/unimodal/finetuned"
  OUT_VISION="$ROOT/vision"
  OUT_MAMA="$ROOT/multimodal/finetuned_mama"
  OUT_VLM="$ROOT/multimodal/finetuned_vlm"
  OUT_MET_FIN="$ROOT/metrics/finetuned"
  MET_CMD=(python 03_scripts/compute_metrics_finetuned.py --repo-root "$ROOT")
  EXPL_U_ARGS=()
  EXPL_M_ARGS=()
  echo ""
  echo ">>> In-place run — outputs go to default repo folders (existing CSVs may be skipped)."
  echo ""
fi

run_step() {
  local title="$1"
  shift
  echo ""
  echo "================================================================================"
  echo "  $title"
  echo "================================================================================"
  if ! "$@"; then
    echo "[WARNING] Step failed (exit $?): $title — continuing with pipeline." >&2
  fi
}

# -----------------------------------------------------------------------------
# 1–4: Supervised fine-tuning
# -----------------------------------------------------------------------------
run_step "1/7 Unimodal text FT (OCR, preset=all)" \
  python unimodal/run_unimodal_finetuned.py \
    --data_dir translated_categorized_memes \
    --train_json train.json \
    --train_image_dir translated_train \
    --output_dir "$OUT_UNIMODAL_FT" \
    --preset all

run_step "2/7 Unimodal image FT (ViT, ResNet, EfficientNet-B0)" \
  python vision/run_vision.py \
    --data_dir translated_categorized_memes \
    --train_json train.json \
    --train_dir translated_train \
    --output_dir "$OUT_VISION" \
    --efficientnet b0

run_step "3/7 Multimodal MAMA FT — image + OCR (CLIP, ViT+BERT, VisualBERT)" \
  python multimodal/run_mama_multimodal_finetune.py \
    --data_dir translated_categorized_memes \
    --train_json train.json \
    --train_image_dir translated_train \
    --output_dir "$OUT_MAMA" \
    --baseline all

if [[ "$SKIP_VLM" -eq 0 ]]; then
  run_step "4/7 Generative VLM FT — image + OCR, multilabel BCE head (all MODEL_CONFIGS)" \
    python multimodal/run_vlm_multilabel_finetune.py \
      --data_dir translated_categorized_memes \
      --train_json train.json \
      --train_image_dir translated_train \
      --output_dir "$OUT_VLM" \
      --run_all
else
  echo ""
  echo ">>> Skipping step 4 (--no-vlm-ft): generative VLM multilabel-head FT"
fi

# -----------------------------------------------------------------------------
# 5: Metrics (read prediction CSVs from same tree as this run)
# -----------------------------------------------------------------------------
run_step "5/7 Metrics — finetuned models (test.json)" \
  "${MET_CMD[@]}"

if [[ "$WITH_NOFT_METRICS" -eq 1 ]]; then
  run_step "Optional: Metrics — no-FT OCR baselines (for explanation Δ table)" \
    python 03_scripts/compute_metrics.py
fi

# -----------------------------------------------------------------------------
# 6–7: Explanation benchmarks (separate output dirs when isolated)
# -----------------------------------------------------------------------------
if [[ "$SKIP_EXPL" -eq 0 ]]; then
  if [[ -f "$EXPL_CSV" ]]; then
    run_step "6/7 Unimodal — explanation text only (not OCR)" \
      python explanation_runs/scripts/run_unimodal.py \
        --repo-root "$ROOT" \
        --explanations-csv "$EXPL_CSV" \
        "${EXPL_U_ARGS[@]}"

    run_step "7/7 Multimodal — image + explanation text" \
      python explanation_runs/scripts/run_multimodal.py \
        --repo-root "$ROOT" \
        --explanations-csv "$EXPL_CSV" \
        "${EXPL_M_ARGS[@]}"
  else
    echo ""
    echo ">>> Skipping explanation steps 6–7: file not found:"
    echo "    $EXPL_CSV"
  fi
else
  echo ""
  echo ">>> Skipping explanation steps 6–7 (--no-explanations)"
fi

echo ""
echo "================================================================================"
echo "  Pipeline finished."
if [[ -n "$BASE_OUT" ]]; then
  echo "  Run folder:          $BASE_OUT"
  echo "  Finetuned metrics:   $OUT_MET_FIN/"
  echo "  Explanation outputs: $OUT_EXPL_U/  and  $OUT_EXPL_M/"
else
  echo "  Finetuned metrics:   $OUT_MET_FIN/"
  echo "  Explanation outputs: explanation_runs/unimodal/  and  explanation_runs/multimodal/"
fi
echo "================================================================================"
