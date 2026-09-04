#!/bin/bash
# Run all 10 multimodal models on 44 Hinglish memes, one at a time.
# Each model saves its own CSV to hinglish_multimodal/{ModelName}/.
# Never overwrites — every run appends a timestamp to the filename.
# Usage: bash run_hinglish_all_models.sh [--test]
#   --test  Run only first 3 images per model to verify setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/multimodal/run_hinglish_multimodal.py"

EXTRA_ARGS=""
if [[ "$1" == "--test" ]]; then
    EXTRA_ARGS="--max_images 3"
    echo "=== TEST MODE: only 3 images per model ==="
fi

MODELS=(
    LLaVA
    LLaVA_NeXT
    MiniCPM_V
    IDEFICS
    InstructBLIP
    BLIP2
    mBLIP
    BLIP
    PALO
    Chitrarth
)

echo "=== Starting Hinglish multimodal pipeline ==="
echo "Models: ${MODELS[*]}"
echo "Start time: $(date)"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "  Running: $MODEL   ($(date))"
    echo "============================================================"
    python3 "$RUNNER" --only_model "$MODEL" $EXTRA_ARGS
    echo ""
done

echo "=== All models complete at $(date) ==="
echo "Results in: $SCRIPT_DIR/hinglish_multimodal/"
