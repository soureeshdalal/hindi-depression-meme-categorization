# Multimodal (vision–language) pipeline

- **run_multimodal.py** – Runs VLMs (LLaVA, PALO, mBLIP, etc.) on images + prompt. Uses `chitrarth_rope_compat.py` and **palo_repo/** when loading Chitrarth/PALO.
- **Per-model CSVs** (one per model, explicitly named): **PALO/PALO_predictions.csv**, **mBLIP/mBLIP_predictions.csv**, **LLaVA/LLaVA_predictions.csv**, **InstructBLIP/InstructBLIP_predictions.csv**, **BLIP/BLIP_predictions.csv**, **BLIP2/BLIP2_predictions.csv**, **Chitrarth/Chitrarth_predictions.csv**, **IDEFICS/IDEFICS_predictions.csv**, **LLaVA_NeXT/LLaVA_NeXT_predictions.csv**, **MiniCPM_V/MiniCPM_V_predictions.csv**.

Run from project root, e.g.:
```bash
python multimodal/run_multimodal.py --data_dir translated_categorized_memes --only_model PALO
python multimodal/run_multimodal.py --data_dir translated_categorized_memes --only_model mBLIP
```
Output CSVs are written to the current working directory (or `--output`); move them into the matching model subfolder if you want them filed here.
