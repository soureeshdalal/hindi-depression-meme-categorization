"""
phase1_repair.py  —  Retry safety-blocked images from Phase 1 V2
-----------------------------------------------------------------
Reads phase1_explanations_v2.csv, finds rows with empty/NaN
explanations, retries them with the softened prompt, and fills
the results back into the same CSV (no new files, clean directory).

Run AFTER phase1_v2.py has finished its final batch (all 650 rows
should be present in phase1_explanations_v2.csv before running this).

Usage:
    GEMINI_API_KEY=... python gemini_runs/run_v2/phase1_repair.py [--rpm 10]
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
CSV_PATH   = Path("gemini_runs/run_v2/phase1_explanations_v2.csv")
RETRIES    = 3
RPM        = 10

# ── Softened prompt (avoids safety-filter keywords) ───────────────────────────
REPAIR_PROMPT = """\
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
                [REPAIR_PROMPT, img],
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

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run phase1_v2.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # Find blocked rows (empty or NaN explanation)
    blocked_mask = df["explanation"].isna() | (df["explanation"].str.strip() == "")
    blocked = df[blocked_mask].copy()

    print(f"\n{'='*58}")
    print(f"  Phase 1 Repair — Retry safety-blocked images")
    print(f"{'='*58}")
    print(f"  Total rows in CSV  : {len(df)}")
    print(f"  Blocked/empty      : {len(blocked)}")
    print(f"  Model              : {MODEL}")
    print(f"  Output             : {CSV_PATH} (updated in place)")
    print(f"{'='*58}\n")

    if len(blocked) == 0:
        print("Nothing to repair. All rows have explanations.")
        return

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

    n_fixed = n_still_blocked = 0

    for idx, row in tqdm(blocked.iterrows(), total=len(blocked), desc="Repairing"):
        img_path = IMAGE_BASE / row["category"] / "test" / row["image_id"]
        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            n_still_blocked += 1
            continue

        rate_limit()
        ok, text, err = call_gemini(model, img_path)

        if ok:
            df.at[idx, "explanation"] = text
            n_fixed += 1
        else:
            n_still_blocked += 1
            if any(x in err.lower() for x in ("quota", "429", "resource_exhausted")):
                df.to_csv(CSV_PATH, index=False)
                print(f"\n⚠️  Quota hit. Progress saved ({n_fixed} repaired). Re-run to continue.")
                sys.exit(0)

        # Save after every image
        df.to_csv(CSV_PATH, index=False)

    still_blocked = (df["explanation"].isna() | (df["explanation"].str.strip() == "")).sum()
    print(f"\n{'='*58}")
    print(f"  Repair complete.")
    print(f"  Fixed this run    : {n_fixed}")
    print(f"  Still blocked     : {n_still_blocked}")
    print(f"  Total empty in CSV: {still_blocked}")
    print(f"  Output            : {CSV_PATH}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
