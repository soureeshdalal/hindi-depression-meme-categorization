# Colab inference results

Outputs from Google Colab Pro runs (650 translated test memes). Layout is **canonical under `results/`** only—no duplicate copies at the repo root.

## Layout

```
colab_results/
├── README.md                 # This file
├── notebooks/                # Notebook snapshots used on Colab
│   ├── 01_instructblip_inference.ipynb
│   ├── 02_palo7b_inference.ipynb
│   └── 03_blip2_inference.ipynb
└── results/
    ├── predictions/          # Per-image CSVs
    ├── metrics/              # Aggregate F1 / JSON
    ├── summaries/            # Human-readable run notes
    └── archive/              # Superseded or diagnostic artifacts
```

## Predictions (`results/predictions/`)

| File | Experiment | Notes |
|------|------------|--------|
| `instructblip_natural_language.csv` | **Exp 1 — primary** | InstructBLIP zero-shot with natural-language symptom list; **use this** for reporting. |
| `instructblip_structured_prompt_attempt.csv` | Exp 1 — ablated | Same model with rigid `PREDICTION: [..]` style prompt; **parse failures** (`parse_ok=False`); kept for documentation only. |
| `blip2_explanation_augmented.csv` | Exp 3 | BLIP-2 + explanation; run completed but **poor task fit** (see summary). |

## Metrics (`results/metrics/`)

| File | Matches |
|------|---------|
| `instructblip_natural_language_metrics.{csv,json}` | `instructblip_natural_language.csv` |
| `blip2_explanation_augmented_metrics.{csv,json}` | `blip2_explanation_augmented.csv` |

## Summaries (`results/summaries/`)

- `experiment_1_summary.txt` — InstructBLIP (natural language) completion notes.
- `palo7b_attempt_summary.txt` — PALO-7B load failure; **no prediction file**.
- `experiment_3_blip2_failure_summary.txt` — BLIP-2 explanation run postmortem.

## Archive (`results/archive/`)

- `superseded_instructblip_structured_attempt.json` — Early Track A note (structured output); **superseded** by the natural-language prompt run. Do not cite as final metrics.

## Gaps vs original plan

- **PALO-7B:** not run (model code not loadable in Colab as documented).
- **BLIP-2:** artifacts present; interpret with the failure summary, not as a strong baseline.

## Syncing with the main project

For paper/report tables, prefer merging these prediction CSVs into your existing `metrics/` or `runs/` pipeline with explicit **model name**, **prompt type**, and **row counts** (`n=650` unless filtered).
