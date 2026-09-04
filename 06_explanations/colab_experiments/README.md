# Google Colab Inference Experiments

This folder packages 3 multimodal inference experiments for Colab Pro:

1. `InstructBLIP-Vicuna-7B` (image-only, zero-shot)
2. `PALO-7B` (image + explanation)
3. `BLIP-2-Flan-T5-XL` (image + explanation)

## 1) Prepare data locally

From repo root:

```bash
python colab_experiments/data/prepare_data.py
```

This creates:

- `colab_experiments/data/test_images/` (flattened test images)
- `colab_experiments/data/metadata.csv`
- `colab_experiments/data/explanations.csv` (if source exists)

If explanations are not found, create a template:

```bash
python colab_experiments/data/generate_explanations_template.py
```

## 2) Upload to Drive (recommended)

Upload the full `colab_experiments/` folder to:

`MyDrive/categorization/colab_experiments`

You can also zip it:

```bash
cd colab_experiments && zip -r colab_experiments_bundle.zip data utils notebooks results
```

## 3) Run notebooks (Colab)

Open and run one notebook at a time:

- `notebooks/01_instructblip_inference.ipynb`
- `notebooks/02_palo7b_inference.ipynb`
- `notebooks/03_blip2_inference.ipynb`

Each notebook:

- mounts Drive,
- loads `data/metadata.csv` (+ `data/explanations.csv` where needed),
- checkpoints every 50 images to `results/*_final.csv`,
- resumes from existing output if interrupted,
- exports metrics CSV + JSON to `results/`.

## 4) Output files

- `results/instructblip_final.csv`
- `results/palo7b_final.csv`
- `results/blip2_final.csv`
- `results/instructblip_metrics.csv` / `.json`
- `results/palo7b_metrics.csv` / `.json`
- `results/blip2_metrics.csv` / `.json`

## Notes

- Colab model availability can vary by environment and package versions.
- Keep `transformers`/`torch` versions fixed once a run starts.
- If runtime disconnects, relaunch and rerun notebook; it resumes from saved CSV.
