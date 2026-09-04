# AWS Quick Start Guide

> **Deprecated for the current repo layout.** Use **`AWS_RUN.md`** for `run_full_pipeline.sh`, isolated `runs/`, and real requirements paths.

## Pre-Transfer Checklist ✅

Run this on your local machine before transferring to AWS:

```bash
python test_pipeline_simple.py
```

Expected output: "✅ ALL TESTS PASSED - Ready for AWS!"

## Files to Transfer

Transfer these files to AWS:

```bash
# Core scripts (6 files)
run_multimodal.py
calculate_f1_multimodal.py
calculate_f1_unified.py
merge_predictions.py
annotation_helper.py
test_multimodal_setup.py

# Documentation (3 files)
README_MULTIMODAL.md
EXECUTION_GUIDE.md
requirements_multimodal.txt

# Data directory
translated_categorized_memes/
```

## AWS Setup (10 minutes)

### 1. Connect to AWS Instance

```bash
ssh -i your-key.pem ubuntu@your-aws-instance
```

### 2. Create Working Directory

```bash
mkdir -p ~/meme-pipeline
cd ~/meme-pipeline
```

### 3. Transfer Files

From your local machine:

```bash
# Transfer scripts
scp -i your-key.pem *.py *.txt *.md ubuntu@your-aws-instance:~/meme-pipeline/

# Transfer data (this may take a while - 650 images)
scp -i your-key.pem -r translated_categorized_memes ubuntu@your-aws-instance:~/meme-pipeline/
```

### 4. Install Dependencies

On AWS:

```bash
# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements_multimodal.txt

# Verify installation
python test_multimodal_setup.py
```

Expected output: Should show CUDA available with 4 GPUs and 96GB VRAM.

## Quick Test (5 minutes)

Test with 3 images to verify everything works:

```bash
python run_multimodal.py \
    --data_dir translated_categorized_memes \
    --max_images 3 \
    --output test_predictions.csv
```

### What to Check:

1. **Model downloads successfully** (first run only, ~5 min)
2. **Inference completes** without errors
3. **Output file created**: `test_predictions.csv`
4. **Check predictions**:
   ```bash
   head -5 test_predictions.csv
   ```
   Should show: image_id, category, PALO_7B_pred, PALO_7B_raw, etc.

5. **Check error log**:
   ```bash
   cat multimodal_errors.log
   ```
   Should be empty or have minimal errors.

### Expected Output:

```
=== MULTIMODAL BENCHMARKING PIPELINE ===
Hardware: 4× NVIDIA GPUs (96.0 GB total VRAM)
Found 3 test images across categories

[MODEL 1/5] PALO-7B (MBZUAI/PALO-7B)
Loading model...
Done (15.8 GB VRAM)
Running inference on 3 images...
100%|████████████████| 3/3 [00:05<00:00]
Average time per image: 1.8s

[MODEL 2/5] PALO-13B (MBZUAI/PALO-13B)
...
```

## Full Run (2 hours)

If quick test passes, run the full pipeline:

```bash
# Run in screen/tmux so it continues if you disconnect
screen -S meme-pipeline

# Run full pipeline
python run_multimodal.py \
    --data_dir translated_categorized_memes \
    --output multimodal_predictions.csv \
    --error_log multimodal_errors.log

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r meme-pipeline
```

### Monitor Progress

In another terminal:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check progress (count completed rows)
wc -l multimodal_predictions.csv

# Check errors
tail -f multimodal_errors.log
```

### Expected Timeline:

- Model 1 (PALO-7B): ~20 minutes
- Model 2 (PALO-13B): ~25 minutes  
- Model 3 (Chitrarth): ~20 minutes
- Model 4 (mBLIP): ~22 minutes
- Model 5 (LLaVA): ~18 minutes
- **Total: ~1.5-2 hours**

## Verify Results

After completion:

```bash
# Check output file
wc -l multimodal_predictions.csv
# Should show: 651 (650 images + 1 header)

# Check for errors
grep -c ERROR multimodal_predictions.csv
# Should be < 50 (less than 10% error rate)

# View sample predictions
head -5 multimodal_predictions.csv

# Check error log
wc -l multimodal_errors.log
```

## Download Results

From your local machine:

```bash
scp -i your-key.pem ubuntu@your-aws-instance:~/meme-pipeline/multimodal_predictions.csv .
scp -i your-key.pem ubuntu@your-aws-instance:~/meme-pipeline/multimodal_errors.log .
```

## Troubleshooting on AWS

### Issue: CUDA Out of Memory

```bash
# Check GPU memory
nvidia-smi

# If stuck, kill process and restart
pkill -f run_multimodal.py

# Skip PALO-13B (largest model) by editing MODEL_CONFIGS
nano run_multimodal.py
# Comment out the PALO-13B line
```

### Issue: Model Download Fails

```bash
# Check internet connection
ping huggingface.co

# Pre-download models
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('MBZUAI/PALO-7B')"
```

### Issue: Slow Inference

```bash
# Verify CUDA is being used
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check GPU utilization
nvidia-smi
# Should show high GPU usage (80-100%) during inference
```

## Cost Optimization

### g5.12xlarge Pricing (us-east-1):
- On-Demand: ~$5.67/hour
- Spot Instance: ~$1.70/hour (70% savings)

### Estimated Costs:
- Setup + Quick Test: 15 min = $1.42 (on-demand) or $0.43 (spot)
- Full Pipeline: 2 hours = $11.34 (on-demand) or $3.40 (spot)
- **Total: ~$13 (on-demand) or ~$4 (spot)**

### Recommendation:
Use Spot Instances if comfortable with potential interruptions (unlikely for 2-hour job).

## After Completion

1. **Download results** to local machine
2. **Terminate AWS instance** to stop charges
3. **Annotate results** locally using `annotation_helper.py`
4. **Calculate F1 scores** using `calculate_f1_multimodal.py`

## Quick Commands Reference

```bash
# Setup
pip install -r requirements_multimodal.txt
python test_multimodal_setup.py

# Quick test (3 images)
python run_multimodal.py --data_dir translated_categorized_memes --max_images 3

# Full run (650 images)
python run_multimodal.py --data_dir translated_categorized_memes

# Monitor
watch -n 1 nvidia-smi
tail -f multimodal_errors.log

# Check results
wc -l multimodal_predictions.csv
grep -c ERROR multimodal_predictions.csv
```

## Success Criteria

✅ Quick test completes without errors  
✅ All 5 models load successfully  
✅ Inference runs at ~2s per image  
✅ Output CSV has 651 rows (650 + header)  
✅ Error rate < 10%  
✅ GPU utilization 80-100% during inference  

If all criteria met, the pipeline is working correctly! 🎉
