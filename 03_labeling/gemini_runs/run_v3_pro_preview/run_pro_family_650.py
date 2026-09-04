"""
run_pro_family_650.py
---------------------
Run all 4 advisor-requested tasks on the full 650-image test set using one
Gemini Pro-family model ID:

1) Phase-1 explanations (same V2 structured prompt)
2) 7-symptom classification from image only
3) 7-symptom classification from explanation only
4) 7-symptom classification from image + explanation

Outputs are written under gemini_runs/run_v3_pro_preview/:
  - phase1_explanations_pro.csv
  - predictions_image_only_pro.csv
  - predictions_explanation_only_pro.csv
  - predictions_image_plus_explanation_pro.csv
  - metrics_pro_family.json

Usage:
  GEMINI_API_KEY=... python gemini_runs/run_v3_pro_preview/run_pro_family_650.py
  GEMINI_API_KEY=... python gemini_runs/run_v3_pro_preview/run_pro_family_650.py --model gemini-3-pro-image-preview
"""

import argparse
import json
import os
import re
import ssl
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import google.generativeai as genai


DEFAULT_MODEL = "gemini-3-pro-preview"
RETRIES = 3
DEFAULT_RPM = 10
DEFAULT_MAX_OUTPUT_TOKENS = 512

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_BASE = REPO_ROOT / "translated_categorized_memes"
SRC_CSV = REPO_ROOT / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv"
OUT_DIR = REPO_ROOT / "gemini_runs" / "run_v3_pro_preview"

OUT_EXPL = OUT_DIR / "phase1_explanations_pro.csv"
OUT_IMG = OUT_DIR / "predictions_image_only_pro.csv"
OUT_TXT = OUT_DIR / "predictions_explanation_only_pro.csv"
OUT_MM = OUT_DIR / "predictions_image_plus_explanation_pro.csv"
OUT_METRICS = OUT_DIR / "metrics_pro_family.json"

SYMPTOM_NAMES = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

# Task 1 prompt: same Phase-1 prompt used in run_v2/phase1_v2.py
PHASE1_PROMPT_V2 = """\
This image is from an academic mental health research dataset. \
Look at it carefully and describe what you observe for each of \
the 7 areas below. Be specific and factual. If an area is clearly \
not shown, write "not shown".

1. MOOD/EMOTIONAL STATE: What emotions are expressed? \
(e.g. sadness, crying, emptiness, hopelessness, flat affect, anguish)

2. MOTIVATION/INTEREST: Does the person show loss of interest, \
lack of motivation, or inability to enjoy things they normally would? \
(e.g. "nothing feels worth doing", "can't enjoy food/hobbies/people")

3. EXISTENTIAL/EMOTIONAL INTENSITY: Does the content depict extreme \
hopelessness, a wish to withdraw from life, or references to \
harming oneself? Describe only what is visually or textually shown.

4. APPETITE/EATING: Are eating habits mentioned? \
(e.g. eating too much/too little, emotional eating, skipping meals, \
food as coping)

5. SELF-WORTH/GUILT: Does the person express worthlessness, \
excessive guilt, self-blame, or feeling like a burden to others?

6. CONCENTRATION/THINKING: Any difficulty focusing, making decisions, \
brain fog, confusion, or inability to think clearly?

7. SLEEP: Any reference to sleep — insomnia, sleeping too much, \
being awake at unusual hours, fatigue, disrupted sleep?

Also add a brief CONTEXT line (1 sentence): the cultural/meme format context.

Keep each area to 1-2 sentences. Be direct and observational.\
"""

# Task 2 prompt: image-only classification
IMAGE_ONLY_PROMPT = """\
Look at this meme image and classify 7 symptoms.

Output rules:
- Return exactly one line
- Use only this format:
PREDICTION: [x,x,x,x,x,x,x]

Symptom order:
1. Feeling Down
2. Lack of Interest
3. Self-Harm
4. Eating Disorder
5. Low Self-Esteem
6. Concentration Problem
7. Sleeping Disorder\
"""

# Task 3 prompt: explanation-only classification
EXPLANATION_ONLY_PROMPT = """\
You are a mental health researcher labelling social media memes for a PHQ-9 study.

STRUCTURED MEME DESCRIPTION:
{explanation}

Label the 7 indicators in this exact order:
1. Feeling Down
2. Lack of Interest
3. Self-Harm
4. Eating Disorder
5. Low Self-Esteem
6. Concentration Problem
7. Sleeping Disorder

Output exactly one line:
PREDICTION: [x,x,x,x,x,x,x]\
"""

# Task 4 prompt: explanation + image classification
IMAGE_PLUS_EXPLANATION_PROMPT = """\
Here is a meme image and an analysis already written about it:

ANALYSIS:
{explanation}

Look at the image and the analysis above. Decide which of the 7 indicators
are present (1) or absent (0):
1. Feeling Down
2. Lack of Interest
3. Self-Harm
4. Eating Disorder
5. Low Self-Esteem
6. Concentration Problem
7. Sleeping Disorder

Return exactly one line:
PREDICTION: [x,x,x,x,x,x,x]\
"""


class RateLimiter:
    def __init__(self, rpm: int):
        self.min_gap = 60.0 / max(1, rpm)
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_gap:
            time.sleep(self.min_gap - elapsed)
        self.last_call = time.time()


def _extract_text(response) -> str:
    try:
        parts = []
        if hasattr(response, "candidates") and response.candidates:
            content = getattr(response.candidates[0], "content", None)
            if content and hasattr(content, "parts"):
                parts = content.parts
        elif hasattr(response, "parts"):
            parts = response.parts
        texts = [getattr(p, "text", None) for p in (parts or []) if getattr(p, "text", None)]
        return " ".join(texts).strip()
    except Exception:
        return ""


def parse_bits(value: str):
    if value is None:
        return None
    s = str(value)
    nums = re.findall(r"[01]", s)
    if len(nums) >= 7:
        return [int(n) for n in nums[:7]]
    return None


def parse_prediction(text: str):
    if not text:
        return None
    m = re.search(r"PREDICTION\s*:\s*\[\s*([01](?:\s*,\s*[01]){6,})\s*\]", text, re.I)
    if m:
        bits = re.findall(r"[01]", m.group(1))
        if len(bits) >= 7:
            return [int(x) for x in bits[:7]]
    yes_no = re.findall(r"(?:Indicator\s*\d+[^:\n]*:\s*)?(YES|NO)\b", text, re.I)
    if len(yes_no) >= 7:
        return [1 if x.upper() == "YES" else 0 for x in yes_no[:7]]
    return parse_bits(text)


def call_gemini(model, parts, max_output_tokens: int):
    for attempt in range(1, RETRIES + 1):
        try:
            response = model.generate_content(
                parts,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_output_tokens": max_output_tokens,
                },
            )
            text = _extract_text(response)
            if text:
                return True, text, ""
            return False, "", "Empty response"
        except Exception as e:
            err = str(e)
            is_quota = any(x in err.lower() for x in ("quota", "429", "resource_exhausted"))
            wait = (2 ** (attempt - 1)) * (10 if is_quota else 1)
            if attempt < RETRIES:
                time.sleep(wait)
            else:
                return False, "", err[:300]
    return False, "", "Max retries exceeded"


def compute_metrics(rows, pred_col: str):
    y_true, y_pred = [], []
    for r in rows:
        g = parse_bits(r.get("gold_labels"))
        p = parse_bits(r.get(pred_col))
        if g is None or p is None:
            continue
        y_true.append(g)
        y_pred.append(p)
    if not y_true:
        return {"evaluated_rows": 0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "evaluated_rows": int(len(y_true)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_label_f1": {},
    }
    f1_each = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(SYMPTOM_NAMES):
        metrics["per_label_f1"][name] = float(f1_each[i])
    return metrics


def load_test_rows():
    src = pd.read_csv(SRC_CSV)[["image_id", "category", "gold_labels"]].drop_duplicates()
    rows = []
    for _, r in src.iterrows():
        img_path = IMAGE_BASE / str(r["category"]) / "test" / str(r["image_id"])
        if img_path.exists():
            rows.append(
                {
                    "image_id": str(r["image_id"]),
                    "category": str(r["category"]),
                    "gold_labels": str(r["gold_labels"]),
                    "image_path": str(img_path),
                }
            )
    return rows


def load_existing(path: Path, key_col: str = "image_id"):
    existing = {}
    if not path.exists():
        return existing
    try:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            key = str(r.get(key_col, ""))
            if key:
                existing[key] = r.to_dict()
    except Exception:
        return {}
    return existing


def run_phase1_explanations(rows, model, limiter, overwrite: bool, max_output_tokens: int):
    existing = {} if overwrite else load_existing(OUT_EXPL)
    out_rows = []
    for row in rows:
        if row["image_id"] in existing:
            out_rows.append(existing[row["image_id"]])
            continue
        limiter.wait()
        img = Image.open(row["image_path"]).convert("RGB")
        ok, txt, err = call_gemini(model, [PHASE1_PROMPT_V2, img], max_output_tokens=max_output_tokens)
        out_rows.append(
            {
                "image_id": row["image_id"],
                "category": row["category"],
                "gold_labels": row["gold_labels"],
                "explanation": txt if ok else "",
                "error": "" if ok else err,
            }
        )
        pd.DataFrame(out_rows).to_csv(OUT_EXPL, index=False)
    pd.DataFrame(out_rows).to_csv(OUT_EXPL, index=False)
    return out_rows


def run_image_only(rows, model, limiter, overwrite: bool, max_output_tokens: int):
    existing = {} if overwrite else load_existing(OUT_IMG)
    out_rows = []
    for row in tqdm(rows, desc="Task2 image-only"):
        if row["image_id"] in existing:
            out_rows.append(existing[row["image_id"]])
            continue
        limiter.wait()
        img = Image.open(row["image_path"]).convert("RGB")
        ok, txt, err = call_gemini(model, [IMAGE_ONLY_PROMPT, img], max_output_tokens=max_output_tokens)
        bits = parse_prediction(txt) if ok else None
        out_rows.append(
            {
                "image_id": row["image_id"],
                "category": row["category"],
                "gold_labels": row["gold_labels"],
                "raw_response": txt if ok else "",
                "prediction": f"[{','.join(map(str, bits))}]" if bits is not None else "",
                "parse_ok": int(bits is not None),
                "error": "" if ok else err,
            }
        )
        pd.DataFrame(out_rows).to_csv(OUT_IMG, index=False)
    pd.DataFrame(out_rows).to_csv(OUT_IMG, index=False)
    return out_rows


def run_explanation_only(rows, explanations_by_id, model, limiter, overwrite: bool, max_output_tokens: int):
    existing = {} if overwrite else load_existing(OUT_TXT)
    out_rows = []
    for row in tqdm(rows, desc="Task3 explanation-only"):
        if row["image_id"] in existing:
            out_rows.append(existing[row["image_id"]])
            continue
        exp = explanations_by_id.get(row["image_id"], "").strip()
        if not exp:
            out_rows.append(
                {
                    "image_id": row["image_id"],
                    "category": row["category"],
                    "gold_labels": row["gold_labels"],
                    "explanation": "",
                    "raw_response": "",
                    "prediction": "",
                    "parse_ok": 0,
                    "error": "Missing explanation",
                }
            )
            pd.DataFrame(out_rows).to_csv(OUT_TXT, index=False)
            continue
        limiter.wait()
        prompt = EXPLANATION_ONLY_PROMPT.format(explanation=exp)
        ok, txt, err = call_gemini(model, [prompt], max_output_tokens=max_output_tokens)
        bits = parse_prediction(txt) if ok else None
        out_rows.append(
            {
                "image_id": row["image_id"],
                "category": row["category"],
                "gold_labels": row["gold_labels"],
                "explanation": exp,
                "raw_response": txt if ok else "",
                "prediction": f"[{','.join(map(str, bits))}]" if bits is not None else "",
                "parse_ok": int(bits is not None),
                "error": "" if ok else err,
            }
        )
        pd.DataFrame(out_rows).to_csv(OUT_TXT, index=False)
    pd.DataFrame(out_rows).to_csv(OUT_TXT, index=False)
    return out_rows


def run_image_plus_explanation(rows, explanations_by_id, model, limiter, overwrite: bool, max_output_tokens: int):
    existing = {} if overwrite else load_existing(OUT_MM)
    out_rows = []
    for row in tqdm(rows, desc="Task4 image+explanation"):
        if row["image_id"] in existing:
            out_rows.append(existing[row["image_id"]])
            continue
        exp = explanations_by_id.get(row["image_id"], "").strip()
        if not exp:
            out_rows.append(
                {
                    "image_id": row["image_id"],
                    "category": row["category"],
                    "gold_labels": row["gold_labels"],
                    "explanation": "",
                    "raw_response": "",
                    "prediction": "",
                    "parse_ok": 0,
                    "error": "Missing explanation",
                }
            )
            pd.DataFrame(out_rows).to_csv(OUT_MM, index=False)
            continue
        limiter.wait()
        prompt = IMAGE_PLUS_EXPLANATION_PROMPT.format(explanation=exp)
        img = Image.open(row["image_path"]).convert("RGB")
        ok, txt, err = call_gemini(model, [prompt, img], max_output_tokens=max_output_tokens)
        bits = parse_prediction(txt) if ok else None
        out_rows.append(
            {
                "image_id": row["image_id"],
                "category": row["category"],
                "gold_labels": row["gold_labels"],
                "explanation": exp,
                "raw_response": txt if ok else "",
                "prediction": f"[{','.join(map(str, bits))}]" if bits is not None else "",
                "parse_ok": int(bits is not None),
                "error": "" if ok else err,
            }
        )
        pd.DataFrame(out_rows).to_csv(OUT_MM, index=False)
    pd.DataFrame(out_rows).to_csv(OUT_MM, index=False)
    return out_rows


def main():
    parser = argparse.ArgumentParser(description="Run 4-task Gemini Pro family pipeline on full 650 test set")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id, e.g. gemini-3-pro-preview")
    parser.add_argument("--api_key", default=None, help="API key; defaults to GEMINI_API_KEY / GOOGLE_API_KEY")
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing CSVs and regenerate")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass --api_key.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_test_rows()
    if not rows:
        print(f"No test rows found from source {SRC_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Test rows: {len(rows)}")
    print(f"Output dir: {OUT_DIR}")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)
    limiter = RateLimiter(args.rpm)

    # Task 1
    print("\n[1/4] Generating explanations...")
    phase1_rows = run_phase1_explanations(rows, model, limiter, args.overwrite, args.max_output_tokens)
    explanations_by_id = {r["image_id"]: str(r.get("explanation", "")) for r in phase1_rows}
    explained = sum(1 for x in explanations_by_id.values() if x.strip())
    print(f"Explanations available: {explained}/{len(rows)}")

    # Task 2
    print("\n[2/4] Classifying from image only...")
    image_rows = run_image_only(rows, model, limiter, args.overwrite, args.max_output_tokens)

    # Task 3
    print("\n[3/4] Classifying from explanation only...")
    text_rows = run_explanation_only(rows, explanations_by_id, model, limiter, args.overwrite, args.max_output_tokens)

    # Task 4
    print("\n[4/4] Classifying from explanation + image...")
    mm_rows = run_image_plus_explanation(rows, explanations_by_id, model, limiter, args.overwrite, args.max_output_tokens)

    metrics = {
        "model": args.model,
        "n_rows": len(rows),
        "task1_explanations_non_empty": explained,
        "task2_image_only": compute_metrics(image_rows, "prediction"),
        "task3_explanation_only": compute_metrics(text_rows, "prediction"),
        "task4_image_plus_explanation": compute_metrics(mm_rows, "prediction"),
        "outputs": {
            "task1_explanations": str(OUT_EXPL),
            "task2_image_only": str(OUT_IMG),
            "task3_explanation_only": str(OUT_TXT),
            "task4_image_plus_explanation": str(OUT_MM),
        },
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"- {OUT_EXPL}")
    print(f"- {OUT_IMG}")
    print(f"- {OUT_TXT}")
    print(f"- {OUT_MM}")
    print(f"- {OUT_METRICS}")


if __name__ == "__main__":
    main()
