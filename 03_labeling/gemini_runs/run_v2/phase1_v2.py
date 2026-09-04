"""
phase1_v2.py  —  Structured 7-aspect explanation (Run V2)
----------------------------------------------------------
Improvement over V1:
  - Phase 1 prompt now asks for 7 structured observations (one per PHQ-9 area)
    instead of free-form prose. This gives Phase 2 cleaner, symptom-specific
    input and fixes the context-dependent category weakness.
  - Runs on ALL 650 test images (not just empties). V2 is a fresh run.
  - Outputs to gemini_runs/run_v2/phase1_explanations_v2.csv

Usage:
    GEMINI_API_KEY=... python gemini_runs/run_v2/phase1_v2.py [--rpm 10]
"""

import argparse, os, ssl, sys, time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────
MODEL      = "gemini-3-pro-image-preview"
IMAGE_BASE = Path("translated_categorized_memes")
SRC_CSV    = Path("gemini_labels/phase2_gemini_predictions.csv")   # image list
OUT_DIR    = Path("gemini_runs/run_v2")
OUT_CSV    = OUT_DIR / "phase1_explanations_v2.csv"
RETRIES    = 3
RPM        = 10

# ── V2 Prompt: structured 7-aspect ───────────────────────────────────────────
# Instead of free-form prose, explicitly ask for each of the 7 PHQ-9 areas.
# This prevents Phase 2 from missing symptoms that weren't mentioned in
# an unstructured paragraph.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    try:
        parts = []
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            content = getattr(cand, "content", None)
            if content and hasattr(content, "parts"):
                parts = content.parts
        elif hasattr(response, "parts"):
            parts = response.parts
        texts = [getattr(p, "text", None) for p in (parts or []) if getattr(p, "text", None)]
        return " ".join(texts).strip()
    except Exception:
        return ""


def call_gemini(model, image_path: Path) -> tuple[bool, str, str]:
    for attempt in range(1, RETRIES + 1):
        try:
            img = Image.open(image_path).convert("RGB")
            response = model.generate_content(
                [PHASE1_PROMPT_V2, img],
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_output_tokens": 1024,
                },
            )
            text = _extract_text(response)
            if text:
                return True, text, ""
            return False, "", "Empty response (safety filter)"
        except Exception as e:
            err = str(e)
            is_quota = any(x in err.lower() for x in ("quota", "429", "resource_exhausted"))
            wait = (2 ** (attempt - 1)) * (10 if is_quota else 1)
            if attempt < RETRIES:
                time.sleep(wait)
            else:
                return False, "", err[:300]
    return False, "", "Max retries exceeded"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpm", type=int, default=RPM)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full image list from v1 source
    src = pd.read_csv(SRC_CSV)[["image_id", "category", "gold_labels"]].drop_duplicates()
    print(f"\n{'='*58}")
    print(f"  Phase 1 V2 — Structured 7-aspect explanations")
    print(f"{'='*58}")
    print(f"  Total images : {len(src)}")
    print(f"  Model        : {MODEL}")
    print(f"  Output       : {OUT_CSV}")
    print(f"{'='*58}\n")

    # Resume
    existing = {}
    if OUT_CSV.exists():
        try:
            ex = pd.read_csv(OUT_CSV)
            for _, r in ex.iterrows():
                existing[r["image_id"]] = r["explanation"]
        except Exception:
            pass
    if existing:
        print(f"Resuming: {len(existing)} done → {len(src) - len(existing)} remaining\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    min_gap = 60.0 / args.rpm
    last_call = 0.0

    def rate_limit():
        nonlocal last_call
        elapsed = time.time() - last_call
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        last_call = time.time()

    results = []
    # Seed with existing
    for _, row in src.iterrows():
        if row["image_id"] in existing:
            results.append({
                "image_id"   : row["image_id"],
                "category"   : row["category"],
                "gold_labels": row.get("gold_labels", "[0,0,0,0,0,0,0]"),
                "explanation": existing[row["image_id"]],
            })

    n_fixed = n_fail = n_safety = 0

    for _, row in tqdm(src.iterrows(), total=len(src), desc="Phase 1 V2"):
        if row["image_id"] in existing:
            continue

        img_path = IMAGE_BASE / row["category"] / "test" / row["image_id"]
        if not img_path.exists():
            n_fail += 1
            continue

        rate_limit()
        ok, text, err = call_gemini(model, img_path)

        entry = {
            "image_id"   : row["image_id"],
            "category"   : row["category"],
            "gold_labels": row.get("gold_labels", "[0,0,0,0,0,0,0]"),
            "explanation": text if ok else "",
        }
        results.append(entry)

        if ok:
            n_fixed += 1
        else:
            n_safety += "safety" in err.lower() or "empty" in err.lower()
            n_fail += 1
            if any(x in err.lower() for x in ("quota", "429", "resource_exhausted")):
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)
                print(f"\n⚠️  Quota hit. Progress saved ({n_fixed} done). Re-run to continue.")
                sys.exit(0)

        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    has_expl = sum(1 for r in results if r["explanation"])
    print(f"\n{'='*58}")
    print(f"  Done. Got explanations: {has_expl} / {len(src)}")
    print(f"  Safety-blocked        : {n_safety}")
    print(f"  Output                : {OUT_CSV}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
