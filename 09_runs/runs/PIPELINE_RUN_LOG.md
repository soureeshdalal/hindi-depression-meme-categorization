# MAMA-aligned full pipeline — run log

**Started:** 2026-03-23  
**Repo:** `/home/ubuntu/projects/categorization`  
**Mode:** Isolated (`bash run_full_pipeline.sh`, no `--in-place`)  
**Run folder:** `runs/pipeline_20260323_181358/`  
**Full report:** `runs/MAMA_PIPELINE_RUN_REPORT.md`

## Phase 0 — Environment

| Check | Result |
|-------|--------|
| `pwd` | `/home/ubuntu/projects/categorization` |
| Python | 3.13.12 (`/opt/pytorch/bin/python`) |
| CUDA | 4× NVIDIA A10G |
| PyTorch | 2.10.0+cu130, `torch.cuda.is_available()` True |

## Phase 1 — Required paths

| Path | OK |
|------|-----|
| train.json | YES |
| test.json | YES |
| translated_train/ | YES |
| translated_categorized_memes/ | YES |
| gemini_runs/run_v2/phase1_explanations_v2.csv | YES |

## Phase 2 — HF_TOKEN

- Shell check: **missing** at automation start (no value logged).
- **Action:** If Hub downloads return 401, run: `export HF_TOKEN='…'` (never commit or print token).

## Patches applied (before run)

1. **`explanation_runs/scripts/run_unimodal.py`** — `ERRORS_LOG` → `OUT_ROOT.parent / "errors.log"` so isolated runs don’t append to legacy `explanation_runs/errors.log`.
2. **`explanation_runs/scripts/run_multimodal.py`** — same.

## Commands

```bash
cd /home/ubuntu/projects/categorization
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Optional if models are gated:
# export HF_TOKEN='…'
nohup bash run_full_pipeline.sh > runs/pipeline_nohup.log 2>&1 &
```

## Run folder (filled after start)

- **Isolated root:** `runs/pipeline_YYYYMMDD_HHMMSS/` (see `runs/pipeline_nohup.log` or `ls -dt runs/pipeline_* | head -1`)

## Failures / fixes

- (append as they occur)

## Post-run validation

- (checkbox after completion)
