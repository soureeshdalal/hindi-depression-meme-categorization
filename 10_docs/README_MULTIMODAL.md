# Multimodal Benchmarking Pipeline for Hindi Mental Health Memes

Complete pipeline for evaluating vision-language models on Hindi mental health meme classification using the PHQ-9 depression symptom framework.

## Overview

This pipeline processes meme images directly through vision-language models (no OCR step) to predict 7 mental health symptoms based on the PHQ-9 questionnaire. It runs 5 multimodal models on the same test set and saves predictions in a standardized CSV format for easy comparison with unimodal results.

## Hardware Requirements

- **Recommended**: AWS g5.12xlarge with 4× NVIDIA A10G GPUs (24 GB each = 96 GB total VRAM)
- **Minimum**: Single GPU with 24+ GB VRAM (models run sequentially)
- All models run in full FP16 precision (no quantization needed)
- Uses `device_map="auto"` for optimal GPU allocation

## Installation

```bash
# Install multimodal dependencies
pip install -r requirements_multimodal.txt

# Optional: Install flash-attention for faster inference (may require compilation)
pip install flash-attn --no-build-isolation
```

## Models

The pipeline evaluates 5 vision-language models:

| Model | HuggingFace ID | Parameters | VRAM (FP16) | Hindi Support |
|-------|---------------|------------|-------------|---------------|
| PALO-7B | MBZUAI/PALO-7B | 7B | ~16 GB | ✅ Excellent |
| PALO-13B | MBZUAI/PALO-13B | 13B | ~28 GB | ✅ Excellent |
| Chitrarth-1 | krutrim-ai-labs/Chitrarth | 7.5B | ~16 GB | ✅ Excellent |
| mBLIP-BLOOMZ-7B | Gregor/mblip-bloomz-7b | 8.2B | ~18 GB | ✅ Good |
| LLaVA-1.5-7B | llava-hf/llava-1.5-7b-hf | 7B | ~16 GB | ⚠️ Limited (baseline) |

## Classification Task

Each meme is classified for 7 binary labels (PHQ-9 symptoms):

0. **Feeling Down** (उदास महसूस करना) - Feeling sad, depressed, or hopeless
1. **Lack of Interest** (रुचि की कमी) - Little interest or pleasure in doing things
2. **Self-Harm** (आत्म-हानि) - Thoughts of self-harm or suicide
3. **Eating Disorder** (खाने का विकार) - Poor appetite or overeating
4. **Low Self-Esteem** (कम आत्म-सम्मान) - Feeling bad about yourself
5. **Concentration Problem** (एकाग्रता की समस्या) - Trouble concentrating
6. **Sleeping Disorder** (नींद का विकार) - Sleep problems

## Usage

### 1. Run Multimodal Benchmarking

```bash
python run_multimodal.py \
    --data_dir translated_categorized_memes \
    --output multimodal_predictions.csv \
    --error_log multimodal_errors.log
```

**Options:**
- `--data_dir`: Path to categorized memes folder (required)
- `--output`: Output CSV path (default: `multimodal_predictions.csv`)
- `--max_images`: Limit number of images for testing (optional)
- `--error_log`: Error log file path (default: `multimodal_errors.log`)

**Expected runtime**: ~1.5-2 hours for all 5 models on 623 images (on g5.12xlarge)

### 2. Calculate F1 Scores (After Human Annotation)

```bash
# For multimodal predictions only
python calculate_f1_multimodal.py \
    --predictions multimodal_predictions.csv \
    --output f1_results_multimodal.csv

# For unified analysis (unimodal + multimodal)
python calculate_f1_unified.py \
    --predictions merged_predictions.csv \
    --output f1_results_unified.csv
```

### 3. Merge Unimodal and Multimodal Results (Optional)

```bash
source venv/bin/activate
python merge_predictions.py \
    --unimodal unimodal/MentalBERT_predictions.csv \
    --multimodal multimodal/PALO/PALO_predictions.csv \
    --output merged_predictions.csv
```

## Output Format

### multimodal_predictions.csv

Columns:
- `image_id`: Filename (e.g., TR-001.jpg)
- `category`: One of 15 meme categories
- `PALO_7B_pred`: Binary predictions `[1,0,0,0,1,0,0]`
- `PALO_7B_raw`: Confidence scores `[0.82,0.11,0.03,0.09,0.76,0.14,0.22]`
- `PALO_13B_pred`, `PALO_13B_raw`: Same for PALO-13B
- `Chitrarth_pred`, `Chitrarth_raw`: Same for Chitrarth
- `mBLIP_pred`, `mBLIP_raw`: Same for mBLIP
- `LLaVA_pred`, `LLaVA_raw`: Same for LLaVA
- `human_label`: Empty (to be filled during annotation)

### f1_results_multimodal.csv

Two tables:
1. **Overall results**: Macro-F1 and Weighted-F1 for each model
2. **Per-category results**: F1 scores broken down by meme category

## Folder Structure

**Three translated splits (no need to carve validation out of training):**

| Split | Location | Role |
|-------|----------|------|
| **Train** | `translated_train/` (+ `train.json` labels) | Full translated training set — use entirely for fine-tuning; do not hold out a slice for val. |
| **Validation** | `translated_categorized_memes/<CATEGORY>/validation/` | Separate translated dev set (e.g. early stopping, hyperparameter checks during FT). |
| **Test** | `translated_categorized_memes/<CATEGORY>/test/` | Separate translated held-out test for final benchmarks. |

```
translated_train/                    # flat: all TR-*.jpg (Devanagari), pairs with train.json
translated_categorized_memes/
├── BRAND_DEPENDENT/
│   ├── validation/
│   └── test/
├── CELEBRITY_DEPENDENT/
│   ├── validation/
│   └── test/
├── … (other categories, same pattern)
└── VISUAL_SYMBOL_DEPENDENT/
    ├── validation/
    └── test/
```

**Note**: This **multimodal inference** script only walks `test/`; it does not read `validation/` or `translated_train/`. Fine-tuning scripts should use **full** `translated_train` + labels for training and the dedicated `validation/` trees when you need a dev set — not a random split of train.

## Prompt Engineering

The pipeline uses carefully crafted prompts to elicit structured JSON responses from vision-language models:

```python
CLASSIFICATION_PROMPT = """You are a mental health expert analyzing memes...
Respond with ONLY a JSON object in this exact format:
{
    "symptoms": [symptom numbers that apply],
    "confidence": [confidence scores for each of 7 symptoms]
}
"""
```

For Hindi-capable models (PALO, Chitrarth, mBLIP), a Hindi translation suffix is added to improve performance.

## Error Handling

- Model loading failures: Skip model, continue with others
- Inference failures: Log error, use `[0,0,0,0,0,0,0]` prediction
- All errors logged to `multimodal_errors.log`
- Pipeline never crashes for single failures

## Memory Management

- Models run sequentially (one at a time)
- VRAM cleared between models using `torch.cuda.empty_cache()`
- Uses `device_map="auto"` for optimal multi-GPU allocation

## Comparison to MAMA-MEMEIA

Results are directly comparable to MAMA-Memeia (ACL 2024):
- Same 7 symptoms (PHQ-9 framework)
- Same metrics (Macro-F1, Weighted-F1)
- Similar model types (LLaVA baseline included)
- **Your contribution**: First Hindi benchmark + per-category analysis

**Expected F1 range**: 35-55% for zero-shot models (similar to MAMA-Memeia's non-agent baselines)

## Troubleshooting

### Chitrarth-specific issues

If you encounter errors with Chitrarth:

```bash
pip install torch==2.5.1 transformers==4.37.0
pip install flash-attn --no-build-isolation  # optional
```

The script includes fallback loading without flash-attention if installation fails.

### CUDA out of memory

If you run out of VRAM:
1. Ensure models run sequentially (not in parallel)
2. Reduce `max_new_tokens` in `predict_symptoms()` function
3. Skip larger models (PALO-13B) if needed

### JSON parsing failures

Models sometimes output malformed JSON. The script:
- Uses regex to extract JSON from response
- Falls back to all-zeros prediction if parsing fails
- Logs all failures to error log

## Files Created

- `run_multimodal.py` - Main inference script for 5 VLMs
- `calculate_f1_multimodal.py` - F1 calculation for multimodal results
- `calculate_f1_unified.py` - Unified F1 calculation (auto-detects models)
- `merge_predictions.py` - Merge unimodal + multimodal predictions
- `requirements_multimodal.txt` - Python dependencies
- `multimodal_errors.log` - Error log (auto-created during run)

## Citation

If you use this pipeline, please cite:

```bibtex
@inproceedings{sharma2024mama,
  title={MAMA-MeMIA: A Multi-Agent Multi-Modal Mental Health Conversational Framework},
  author={Sharma, Ayush and others},
  booktitle={Proceedings of ACL 2024},
  year={2024}
}
```

## License

This pipeline is provided for research purposes. Model licenses:
- PALO: Apache 2.0
- Chitrarth: Check model card
- mBLIP: Apache 2.0
- LLaVA: Apache 2.0
