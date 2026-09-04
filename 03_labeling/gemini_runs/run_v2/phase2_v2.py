"""
phase2_v2.py  —  Improved label prediction from structured explanations (Run V2)
----------------------------------------------------------------------------------
Improvements over V1:
  1. Few-shot examples calibrate the model to annotator standards
  2. Sharper FD vs LI definitions (biggest v1 failure mode)
  3. Calibration instructions for over-predicted symptoms (FD, Low Self-Esteem)
  4. Fallback: if PREDICTION line missing, parse YES/NO lines

Input : gemini_runs/run_v2/phase1_explanations_v2.csv
Output: gemini_runs/run_v2/phase2_predictions_v2.csv

Usage:
    GEMINI_API_KEY=... python gemini_runs/run_v2/phase2_v2.py [--rpm 10]
"""

import argparse, json, re, ssl, sys, time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────
MODEL   = "gemini-3.1-pro-preview"
SRC_CSV = Path("gemini_runs/run_v2/phase1_explanations_v2.csv")
OUT_DIR = Path("gemini_runs/run_v2")
OUT_CSV = OUT_DIR / "phase2_predictions_v2.csv"
RETRIES = 3
RPM     = 10

# ── V2 Prompt ─────────────────────────────────────────────────────────────────
# Improvements:
#   A) Sharp FD vs LI distinction (fixes biggest v1 failure)
#   B) Three few-shot examples (calibrates to annotator standard)
#   C) Calibration instructions (reduces FD and Low SE false positives)
PHASE2_PROMPT_V2 = """\
You are a mental health researcher labelling social media memes for a PHQ-9 study.

STRUCTURED MEME DESCRIPTION:
{explanation}

─────────────────────────────────────────────
THE 7 PHQ-9 INDICATORS — READ CAREFULLY:

1. FEELING DOWN (Depressed Mood)
   = Persistent sadness, hopelessness, emptiness, or emotional anguish.
   ⚠ Do NOT mark just for frustration, sarcasm, or mild disappointment.
   ⚠ The meme's CORE message must be about sustained low mood.

2. LACK OF INTEREST (Anhedonia)
   = LOSS OF PLEASURE OR MOTIVATION in activities once enjoyed.
   ⚠ This is BEHAVIOURAL — the person has STOPPED doing things or finds
     nothing enjoyable/worthwhile.
   ⚠ NOT the same as Feeling Down. Mark BOTH only when BOTH the emotional
     sadness AND the behavioural withdrawal are explicitly shown.
   ⚠ Example: "food used to be my joy but now I eat mechanically" = YES.
     "I feel sad" alone = NO for this indicator.

3. SELF-HARM / SUICIDAL IDEATION
   = Explicit or strongly implied thoughts of death, suicide, self-harm,
     or not wanting to exist / "wanting it all to stop".

4. EATING DISORDER (Appetite Change)
   = Significant increase OR decrease in appetite / eating.
     Emotional eating, binge eating, restriction, skipping meals.

5. LOW SELF-ESTEEM (Worthlessness / Guilt)
   = Explicit self-blame, feelings of worthlessness, excessive guilt,
     feeling like a burden, harsh self-criticism.
   ⚠ NOT just shyness or embarrassment. Must be about self-worth.

6. CONCENTRATION PROBLEMS
   = Difficulty focusing, brain fog, indecisiveness, inability to think
     clearly, or easily distracted.

7. SLEEP DISORDER
   = Insomnia, hypersomnia, awake at 3 AM, sleeping all day,
     disrupted or irregular sleep patterns.

─────────────────────────────────────────────
FEW-SHOT EXAMPLES (follow this standard):

EXAMPLE A:
Description:
  MOOD: Deep sadness, crying, emptiness — "nothing helps anymore".
  MOTIVATION: Still eating favourite food but it brings no joy.
  SELF-HARM: Not shown.
  APPETITE: Eating normal amounts.
  SELF-WORTH: Not shown.
  CONCENTRATION: Not shown.
  SLEEP: Not shown.
  CONTEXT: Bollywood reaction meme.
→ Indicator 1 (Feeling Down): YES – Core theme is persistent sadness.
→ Indicator 2 (Lack of Interest): YES – Food no longer brings pleasure (anhedonia).
→ Indicator 3 (Self-Harm): NO
→ Indicator 4 (Eating Disorder): NO – Eating normal amounts.
→ Indicator 5 (Low Self-Esteem): NO
→ Indicator 6 (Concentration): NO
→ Indicator 7 (Sleep Disorder): NO
PREDICTION: [1,1,0,0,0,0,0]

EXAMPLE B:
Description:
  MOOD: Mild frustration / sarcasm about dieting culture.
  MOTIVATION: Not shown.
  SELF-HARM: Not shown.
  APPETITE: Meme about crash dieting and then binge eating.
  SELF-WORTH: Slight self-blame about body image.
  CONCENTRATION: Not shown.
  SLEEP: Not shown.
  CONTEXT: Relatable diet-fail meme.
→ Indicator 1 (Feeling Down): NO – Mood is sarcastic/humorous, not persistently depressed.
→ Indicator 2 (Lack of Interest): NO
→ Indicator 3 (Self-Harm): NO
→ Indicator 4 (Eating Disorder): YES – Explicit binge/restrict cycle.
→ Indicator 5 (Low Self-Esteem): NO – Mild frustration, not worthlessness.
→ Indicator 6 (Concentration): NO
→ Indicator 7 (Sleep Disorder): NO
PREDICTION: [0,0,0,1,0,0,0]

EXAMPLE C:
Description:
  MOOD: Hopelessness, feels like giving up.
  MOTIVATION: Lost interest in everything, scrolling phone mindlessly.
  SELF-HARM: Indirect reference — "just want it all to stop".
  APPETITE: Not shown.
  SELF-WORTH: Calls self a "burden" to everyone.
  CONCENTRATION: Can't focus on work.
  SLEEP: Awake at 3 AM, can't sleep.
  CONTEXT: Text-heavy confession meme.
→ Indicator 1 (Feeling Down): YES – Hopelessness, wanting to give up.
→ Indicator 2 (Lack of Interest): YES – Lost interest in everything.
→ Indicator 3 (Self-Harm): YES – "want it all to stop" implies suicidal ideation.
→ Indicator 4 (Eating Disorder): NO
→ Indicator 5 (Low Self-Esteem): YES – Explicitly calls self a burden.
→ Indicator 6 (Concentration): YES – Can't focus on work.
→ Indicator 7 (Sleep Disorder): YES – Awake at 3 AM.
PREDICTION: [1,1,1,0,1,1,1]

─────────────────────────────────────────────
CALIBRATION REMINDERS:
- Feeling Down: Must be the CORE theme, not incidental sadness or humour.
- Lack of Interest: Requires BEHAVIOURAL evidence, not just mood.
- Low Self-Esteem: Explicit worthlessness/guilt only — not general negativity.
- When uncertain, lean toward NO (the annotators were conservative).

─────────────────────────────────────────────
NOW LABEL THE MEME ABOVE. Use this exact format:

Indicator 1 (Feeling Down): YES/NO – one-sentence reason
Indicator 2 (Lack of Interest): YES/NO – one-sentence reason
Indicator 3 (Self-Harm): YES/NO – one-sentence reason
Indicator 4 (Eating Disorder): YES/NO – one-sentence reason
Indicator 5 (Low Self-Esteem): YES/NO – one-sentence reason
Indicator 6 (Concentration): YES/NO – one-sentence reason
Indicator 7 (Sleep Disorder): YES/NO – one-sentence reason

PREDICTION: [x,x,x,x,x,x,x]\
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


def call_gemini(model, prompt: str) -> tuple[bool, str, str]:
    for attempt in range(1, RETRIES + 1):
        try:
            response = model.generate_content(
                [prompt],
                generation_config={
                    "temperature": 0.1,   # very low — deterministic structured output
                    "top_p": 0.9,
                    "max_output_tokens": 1024,
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


def parse_prediction(text: str) -> list | None:
    # Primary: PREDICTION: [1,0,...]
    m = re.search(r"PREDICTION\s*:\s*\[?\s*([01](?:\s*,\s*[01]){6,})\s*\]?", text, re.I)
    if m:
        nums = re.findall(r"[01]", m.group(1))
        if len(nums) >= 7:
            return [int(n) for n in nums[:7]]
    # Fallback: parse YES/NO per indicator line
    yes_no = re.findall(r"Indicator\s*\d+[^:]*:\s*(YES|NO)", text, re.I)
    if len(yes_no) >= 7:
        return [1 if v.upper() == "YES" else 0 for v in yes_no[:7]]
    # Last resort: standalone 7-bit array
    for m in re.finditer(r"\[\s*([01](?:\s*,\s*[01]){6})\s*\]", text):
        nums = re.findall(r"[01]", m.group(1))
        if len(nums) == 7:
            return [int(n) for n in nums]
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpm", type=int, default=RPM)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(SRC_CSV)
    has_expl = src[src["explanation"].notna() & (src["explanation"].str.strip() != "")]

    print(f"\n{'='*58}")
    print(f"  Phase 2 V2 — Improved label prediction")
    print(f"{'='*58}")
    print(f"  Rows with explanation : {len(has_expl)}")
    print(f"  Model                 : {MODEL}")
    print(f"  Output                : {OUT_CSV}")
    print(f"{'='*58}\n")

    # Resume
    existing = {}
    if OUT_CSV.exists():
        try:
            ex = pd.read_csv(OUT_CSV)
            for _, r in ex.iterrows():
                existing[r["image_id"]] = r.to_dict()
        except Exception:
            pass
    if existing:
        print(f"Resuming: {len(existing)} done → {len(has_expl) - len(existing)} remaining\n")

    # Connectivity test
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)
    try:
        test = model.generate_content(["Reply OK."])
        test_text = _extract_text(test)
        print(f"API OK — replied: {test_text[:30]!r}\n")
    except Exception as e:
        print(f"API failed: {e}", file=sys.stderr)
        sys.exit(1)

    min_gap = 60.0 / args.rpm
    last_call = 0.0

    def rate_limit():
        nonlocal last_call
        elapsed = time.time() - last_call
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        last_call = time.time()

    results = list(existing.values())
    n_ok = n_fail = n_no_pred = 0

    for _, row in tqdm(has_expl.iterrows(), total=len(has_expl), desc="Phase 2 V2"):
        if row["image_id"] in existing:
            continue

        expl = str(row["explanation"]).strip()
        prompt = PHASE2_PROMPT_V2.format(explanation=expl)

        rate_limit()
        ok, text, err = call_gemini(model, prompt)

        if ok:
            pred = parse_prediction(text)
            if pred is None:
                n_no_pred += 1
                pred = [0] * 7
                print(f"  [WARN] No PREDICTION for {row['image_id']}: {text[-100:]!r}")
            else:
                n_ok += 1
        else:
            pred = [0] * 7
            n_fail += 1
            if any(x in err.lower() for x in ("quota", "429", "resource_exhausted")):
                if results:
                    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
                print(f"\n⚠️  Quota hit. Progress saved ({n_ok} labelled). Re-run to continue.")
                sys.exit(0)
            print(f"  [FAIL] {row['image_id']}: {err[:100]}")

        results.append({
            "image_id"        : row["image_id"],
            "category"        : row["category"],
            "gold_labels"     : row.get("gold_labels", "[0,0,0,0,0,0,0]"),
            "predicted_labels": str(pred).replace(" ", ""),
            "explanation"     : expl,
        })
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    import ast, numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score
    out_df = pd.read_csv(OUT_CSV)
    y_true, y_pred = [], []
    for _, r in out_df.iterrows():
        try:
            g = ast.literal_eval(str(r["gold_labels"]))
            p = ast.literal_eval(str(r["predicted_labels"]))
            if len(g) == 7 and len(p) == 7:
                y_true.append([int(x) for x in g])
                y_pred.append([int(x) for x in p])
        except Exception:
            pass

    nz = (out_df["predicted_labels"].str.strip() == "[0,0,0,0,0,0,0]").sum()
    snames = ["Feeling Down","Lack of Interest","Self-Harm","Eating Disorder",
              "Low Self-Esteem","Concentration","Sleep Disorder"]
    yt = np.array(y_true); yp = np.array(y_pred)

    print(f"\n{'='*58}")
    print(f"  Complete.")
    print(f"  Rows labelled        : {len(out_df)}")
    print(f"  Successful parses    : {n_ok}")
    print(f"  No PREDICTION line   : {n_no_pred}")
    print(f"  API failures         : {n_fail}")
    print(f"  All-zero rows        : {nz}")
    if len(yt):
        mf1 = f1_score(yt, yp, average="macro", zero_division=0)
        wf1 = f1_score(yt, yp, average="weighted", zero_division=0)
        mp  = precision_score(yt, yp, average="macro", zero_division=0)
        mr  = recall_score(yt, yp, average="macro", zero_division=0)
        print(f"\n  Macro-F1    : {mf1:.4f}")
        print(f"  Weighted-F1 : {wf1:.4f}")
        print(f"  Macro-P     : {mp:.4f}")
        print(f"  Macro-R     : {mr:.4f}")
        print(f"\n  {'Symptom':<22} {'Gold':>5} {'Pred':>5} {'P':>6} {'R':>6} {'F1':>6}")
        print("  " + "-"*54)
        for i, name in enumerate(snames):
            gt = yt[:,i]; pr = yp[:,i]
            p_ = precision_score(gt, pr, zero_division=0)
            r_ = recall_score(gt, pr, zero_division=0)
            f_ = f1_score(gt, pr, zero_division=0)
            print(f"  {name:<22} {int(gt.sum()):>5} {int(pr.sum()):>5} {p_:>6.3f} {r_:>6.3f} {f_:>6.3f}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
