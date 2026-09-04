#!/usr/bin/env python3
"""
translate_to_hinglish.py

Converts 50 original English memes to Hinglish (Hindi written in Roman/Latin
script — the casual way Indians type Hindi in messages and social media).

Source:  categorized_memes/*/test/*.jpg  (original English memes)
Output:  hinglish_memes/  (generated Hinglish memes)
Model:   gemini-3-pro-image-preview  (text analysis + image generation)
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY   = "AIzaSyCE7Tn19v0m6BpiaESw90uhjDos0HrhB7U"
GEMINI_MODEL     = "gemini-3-pro-image-preview"

SOURCE_DIR       = Path("categorized_memes")
OUTPUT_DIR       = Path("hinglish_memes")
CHECKPOINT_FILE  = Path("hinglish_progress.json")
OUTPUT_CSV       = Path("hinglish_translations.csv")

NUM_SAMPLES      = 50
RANDOM_SEED      = 42          # reproducible sample selection
BATCH_DELAY      = 3           # seconds between API calls
CHECKPOINT_EVERY = 5           # save progress after every N memes

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
HINGLISH_PROMPT = """
You are a Hinglish translation specialist. Hinglish means Hindi written in
Roman (Latin) script — the casual, conversational way young Indians type
on WhatsApp, Instagram, and social media. It is NOT Devanagari script.

Examples of Hinglish style:
  English: "I can't get out of bed"
  Hinglish: "Mujhse bed se uthna hi nahi hota"

  English: "Why won't my brain just work"
  Hinglish: "Yaar mera dimaag kaam q nahi karta"

  English: "Nobody understands me"
  Hinglish: "Koi samajhta hi nahi mujhe"

  English: "I'm so tired of everything"
  Hinglish: "Sach mein sab cheez se thak gaya/gayi hoon"

Hinglish rules:
- Write all Hindi words in Roman letters (NO Devanagari script at all)
- Keep commonly used English words as-is (ok, chill, vibe, mood, etc.)
- Sound natural and conversational, like a real Indian person texting
- Preserve the emotional tone and humor of the original meme
- Short, punchy text works best — match the original length roughly

YOUR TASK:
1. Look at this meme image carefully
2. Read all the English text in it
3. Translate the text to Hinglish (Roman script Hindi)
4. Generate a new meme image IDENTICAL to the original EXCEPT:
   - Replace every English text element with its Hinglish translation
   - Keep exact same position, font style, font size, and color as original
   - Keep all visual elements (people, objects, backgrounds) exactly the same
   - Only the text changes — everything else stays identical

CRITICAL:
- Output image must look like the original meme but with Hinglish text
- Do NOT use any Devanagari characters anywhere in the output image
- Do NOT change the meme layout, images, or non-text elements
- The Hinglish text must fit naturally in the same space as the original text

First provide your translation analysis, then generate the image.

FORMAT:
ORIGINAL TEXT: [all English text from the meme]
HINGLISH TRANSLATION: [Roman script Hindi version]
REASONING: [brief note on translation choices]
[Generate the translated meme image]
"""

# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def load_progress():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed": {},   # meme_id -> {status, output_path, translation, ...}
        "failed":    {},
        "sample_ids": [],  # the 50 chosen IDs (fixed once selected)
    }


def save_progress(prog):
    prog["last_saved"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2, ensure_ascii=False)
    done  = len(prog["completed"])
    total = len(prog["sample_ids"])
    print(f"  Checkpoint saved ({done}/{total} done, {len(prog['failed'])} failed)")


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def collect_all_test_images():
    """Return list of (meme_id, path) for all test images across categories."""
    images = []
    for cat_dir in sorted(SOURCE_DIR.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "UNPROCESSED":
            continue
        test_dir = cat_dir / "test"
        if not test_dir.exists():
            continue
        for img in sorted(test_dir.glob("*.jpg")):
            images.append((img.stem, img, cat_dir.name))
    return images


def choose_samples(all_images, n, seed):
    """Pick n images spread across categories for variety."""
    # Group by category
    by_cat = {}
    for meme_id, path, cat in all_images:
        by_cat.setdefault(cat, []).append((meme_id, path, cat))

    rng = random.Random(seed)

    # Round-robin across categories until we have n samples
    chosen = []
    cats = sorted(by_cat.keys())
    # Shuffle each category's list
    for cat in cats:
        rng.shuffle(by_cat[cat])

    idx = 0
    while len(chosen) < n:
        cat = cats[idx % len(cats)]
        if by_cat[cat]:
            chosen.append(by_cat[cat].pop())
        idx += 1

    return chosen[:n]


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def call_gemini(model, image_path):
    """Send image to Gemini, return (success, translation_text, image_bytes_or_None)."""
    try:
        img = Image.open(image_path)
        response = model.generate_content(
            [HINGLISH_PROMPT, img],
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 4096,
            }
        )

        # Extract text
        text_out = ""
        try:
            text_out = response.text or ""
        except Exception:
            pass

        # Extract generated image
        img_bytes = None
        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_bytes = part.inline_data.data
                    break

        return True, text_out, img_bytes

    except Exception as e:
        err = str(e)
        if any(x in err.lower() for x in ("quota", "429", "resource_exhausted")):
            return False, "QUOTA_EXCEEDED", None
        return False, err, None


def save_image(img_bytes, out_path):
    """Save raw bytes or base64 string to file."""
    import base64
    try:
        if isinstance(img_bytes, bytes):
            out_path.write_bytes(img_bytes)
        elif isinstance(img_bytes, str):
            out_path.write_bytes(base64.b64decode(img_bytes))
        elif hasattr(img_bytes, "save"):
            img_bytes.save(out_path)
        else:
            return False
        return True
    except Exception as e:
        print(f"  Error saving image: {e}")
        return False


def extract_translation(text):
    """Pull the Hinglish translation line from Gemini's text response."""
    import re
    m = re.search(r"HINGLISH TRANSLATION:\s*(.+?)(?:\n|REASONING|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text[:200].strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Hinglish Meme Translation Pipeline")
    print("=" * 60)

    # Setup output folder
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load or init progress
    prog = load_progress()

    # Select 50 samples (fixed — same every run)
    if not prog["sample_ids"]:
        all_images = collect_all_test_images()
        chosen = choose_samples(all_images, NUM_SAMPLES, RANDOM_SEED)
        prog["sample_ids"] = [(mid, str(path), cat) for mid, path, cat in chosen]
        save_progress(prog)
        print(f"Selected {len(prog['sample_ids'])} samples across categories")
        for mid, _, cat in prog["sample_ids"]:
            print(f"  {mid}  ({cat})")
        print()

    # Determine what still needs processing
    done_ids   = set(prog["completed"].keys()) | set(prog["failed"].keys())
    remaining  = [(mid, Path(path), cat)
                  for mid, path, cat in prog["sample_ids"]
                  if mid not in done_ids]

    print(f"Already done: {len(done_ids)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All samples already processed!")
        return

    # Setup Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"Gemini model: {GEMINI_MODEL}\n")

    since_checkpoint = 0

    for i, (meme_id, img_path, category) in enumerate(remaining):
        total_done = len(prog["completed"]) + len(prog["failed"])
        print(f"\n[{total_done + 1}/{NUM_SAMPLES}]  {meme_id}  ({category})")

        if not img_path.exists():
            print(f"  Source image not found: {img_path}")
            prog["failed"][meme_id] = {"error": "source not found", "category": category}
            since_checkpoint += 1
        else:
            success, result, img_bytes = call_gemini(model, img_path)

            if not success:
                if result == "QUOTA_EXCEEDED":
                    print("  QUOTA EXCEEDED — stopping.")
                    save_progress(prog)
                    return
                print(f"  FAILED: {result[:120]}")
                prog["failed"][meme_id] = {"error": result[:200], "category": category}
                since_checkpoint += 1
            else:
                translation = extract_translation(result)
                print(f"  Translation: {translation[:80]}")

                # Save generated image if present
                out_path = OUTPUT_DIR / f"{meme_id}.jpg"
                img_saved = False
                if img_bytes:
                    img_saved = save_image(img_bytes, out_path)
                    print(f"  Image saved: {out_path.name}" if img_saved else "  Image save failed")
                else:
                    print("  No image in response (text only)")

                prog["completed"][meme_id] = {
                    "category":    category,
                    "translation": translation,
                    "full_response": result[:500],
                    "image_saved": img_saved,
                    "output_path": str(out_path) if img_saved else "",
                    "timestamp":   datetime.now().isoformat(),
                }
                since_checkpoint += 1

        if since_checkpoint >= CHECKPOINT_EVERY:
            save_progress(prog)
            since_checkpoint = 0

        time.sleep(BATCH_DELAY)

    # Final save
    save_progress(prog)

    # Write CSV summary
    import csv
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "meme_id", "category", "hinglish_translation", "image_saved", "status"
        ])
        writer.writeheader()
        for mid, data in prog["completed"].items():
            writer.writerow({
                "meme_id":              mid,
                "category":             data.get("category", ""),
                "hinglish_translation": data.get("translation", ""),
                "image_saved":          data.get("image_saved", False),
                "status":               "SUCCESS",
            })
        for mid, data in prog["failed"].items():
            writer.writerow({
                "meme_id":              mid,
                "category":             data.get("category", ""),
                "hinglish_translation": "",
                "image_saved":          False,
                "status":               f"FAILED: {data.get('error', '')}",
            })

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Completed: {len(prog['completed'])}")
    print(f"  Failed:    {len(prog['failed'])}")
    print(f"  Images in: {OUTPUT_DIR}/")
    print(f"  CSV:       {OUTPUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
