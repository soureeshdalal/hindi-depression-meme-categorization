# Running on AWS (current pipeline)

The project is **Linux- and CUDA-ready**: scripts prefer **CUDA** when available; Apple **MPS** is only a fallback and is unused on AWS.

## 1. Instance

| Workload | Suggestion |
|----------|------------|
| Full `run_full_pipeline.sh` (incl. 7B VLMs, `--run_all`) | **g5.2xlarge** (24 GB) minimum per big model; **g5.12xlarge** (4×24 GB) if you want parallelism or less sequential loading |
| MAMA + text + vision only (`run_mama_replication.sh` / `--no-vlm-ft`) | **g5.xlarge** or **g5.2xlarge** often enough |

Use an **NVIDIA GPU** AMI or **Deep Learning AMI** with CUDA drivers, or install drivers + use a **PyTorch** base image.

## 2. Sync the repo (not individual loose scripts)

```bash
git clone <your-repo-url> categorization
cd categorization
```

## 3. Data on the instance

At repo root you need:

- `translated_categorized_memes/` (with `*/test/` images)
- `translated_train/` (TR-* images aligned with `train.json`)
- `train.json`, `test.json`
- Optional: `gemini_runs/.../phase1_*.csv` if you run explanation steps

## 4. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install GPU PyTorch first (CUDA version must match your driver / AMI)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements_multimodal.txt
```

## 5. Hugging Face

```bash
export HF_TOKEN=hf_...   # gated models (LLaVA, some checkpoints)
```

Model weights cache under `~/.cache/huggingface/` — **tens of GB** for a full VLM sweep; size the EBS volume accordingly.

## 6. Run

```bash
cd ~/categorization
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Recommended: isolated outputs under runs/pipeline_*
bash run_full_pipeline.sh

# Or legacy paths on the instance:
# bash run_full_pipeline.sh --in-place
```

Logs: watch stdout/stderr; GPU memory errors → use a larger instance or `--no-vlm-ft` / run one model at a time.

## 7. Pull results back

```bash
# Example: copy latest isolated run
scp -r -i key.pem ubuntu@host:~/categorization/runs/pipeline_* ./
```

## Not automated here

- **IAM / security groups / SSH keys**
- **Spot interruption** handling (checkpointing is per-model CSV skip-if-exists, not mid-epoch)
- **Docker** — you can wrap the same steps in a container with NVIDIA runtime

---

For an older, file-list-style guide see `AWS_QUICK_START.md` (**deprecated** vs this doc).
