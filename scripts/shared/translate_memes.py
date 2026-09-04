#!/usr/bin/env python3
"""
translate_memes.py

Translates English/Romanized-Hinglish mental health meme images from the
training set to Devanagari Hindi using the Gemini image generation model.

Input:   train/          (8,814 training images, TR-1.jpg … TR-8814.jpg)
Output:  translated_train/  (Devanagari versions, same filenames)
Log:     translation_log.csv

Rate:    15 images/minute (configurable)
Retry:   3 attempts with exponential backoff (1s → 2s → 4s)
Resume:  skips images whose output already exists
"""

import csv
import json
import os
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path

# macOS SSL workaround (same fix used across this project)
if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import google.generativeai as genai
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY    = "AIzaSyCE7Tn19v0m6BpiaESw90uhjDos0HrhB7U"
GEMINI_MODEL      = "gemini-3-pro-image-preview"

INPUT_DIR         = Path("train")
OUTPUT_DIR        = Path("translated_train")
LOG_FILE          = Path("translation_log.csv")
PROGRESS_FILE     = Path("translation_progress.json")

MAX_IMAGES        = None        # Process all 8,814
IMAGES_PER_MINUTE = 15          # API rate limit
RETRY_ATTEMPTS    = 3
CHECKPOINT_EVERY  = 10          # Save progress every N images

# Cost estimate (USD) per image
COST_PER_IMAGE    = 0.14        # $0.0015 input + $0.1388 output ≈ $0.14

# ---------------------------------------------------------------------------
# Translation prompt
# ---------------------------------------------------------------------------
DEVANAGARI_PROMPT = """You are converting a mental health meme to Devanagari \
script for an Indian audience.

SCRIPT CONVERSION RULES:
1. Convert ALL text to Devanagari script (देवनागरी लिपि) — no English alphabet anywhere
2. For words with direct Hindi equivalents: Translate to Hindi
   - "happy" → "खुश",  "sad" → "उदास",  "I am" → "मैं हूँ"
3. For English words WITHOUT direct Hindi translation: Transliterate to Devanagari phonetically
   - "depression" → "डिप्रेशन",  "anxiety" → "एंग्जायटी"
   - "Instagram"  → "इंस्टाग्राम",  "meme"  → "मीम"
   - "Netflix"    → "नेटफ्लिक्स",  "WhatsApp" → "व्हाट्सएप"
4. Brand names / proper nouns: Transliterate to Devanagari
   - "Google" → "गूगल",  "YouTube" → "यूट्यूब"
5. Mental health terms: Use commonly understood Devanagari versions
   - "therapy" → "थेरेपी",  "counseling" → "काउंसलिंग"

CRITICAL — DO NOT CREATE HINGLISH:
❌ Do NOT mix Devanagari and English alphabet (e.g., "मैं fine हूँ")
❌ Do NOT use Romanized Hindi spelling (e.g., "main thik hoon")
✅ Everything must be in Devanagari script only
✅ Output is pure Devanagari — either translated Hindi or transliterated English

LAYOUT PRESERVATION:
- Maintain EXACT visual layout: text position, size, font style, and colors identical
- Preserve all visual elements: images, backgrounds, meme templates, symbols, emojis
- Keep text alignment, spacing, and formatting exactly as original
- Maintain the emotional tone and humor of the original meme

CULTURAL ADAPTATION:
- Keep cultural references accessible to Indian audiences
- Adapt Western-specific references if needed while maintaining humor
- Preserve internet slang and meme culture tone

MENTAL HEALTH CONTEXT:
This meme discusses depression, anxiety, or related mental health topics. Preserve \
the supportive, relatable, or humorous tone with sensitivity to Indian cultural context.

OUTPUT: Generate a 2K resolution image with ALL text in Devanagari script \
(translated Hindi OR transliterated English), maintaining the original meme's \
visual structure, layout, and emotional impact.

First provide your analysis, then generate the image.

FORMAT:
ORIGINAL TEXT: [all text from the meme]
DEVANAGARI VERSION: [pure Devanagari output]
REASONING: [brief note on translation/transliteration choices]
[Generate the translated meme image]
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_output_directory():
    """Create output directory and initialise CSV log if needed."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "original_file", "translated_file", "status",
                "timestamp", "error_message", "cost_estimate",
            ])
            writer.writeheader()
    print(f"Output directory : {OUTPUT_DIR}/")
    print(f"Log file         : {LOG_FILE}")


def get_image_list(input_dir: Path, max_images=None) -> list:
    """Return sorted list of image paths from input_dir."""
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if max_images is not None:
        images = images[:max_images]
    return images


def load_progress() -> set:
    """Return set of already-completed filenames (from previous runs)."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_progress(completed: set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(completed),
                   "last_saved": datetime.now().isoformat()}, f, indent=2)


def log_result(original: Path, translated: Path, status: str,
               error: str = "", cost: float = 0.0):
    """Append one row to the CSV log."""
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "original_file", "translated_file", "status",
            "timestamp", "error_message", "cost_estimate",
        ])
        writer.writerow({
            "original_file":  str(original),
            "translated_file": str(translated) if translated else "",
            "status":          status,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_message":   error,
            "cost_estimate":   f"{cost:.4f}",
        })


def save_image_bytes(img_bytes, out_path: Path) -> bool:
    """Save raw bytes or base64 string from Gemini response to disk."""
    import base64
    try:
        if isinstance(img_bytes, bytes):
            out_path.write_bytes(img_bytes)
        elif isinstance(img_bytes, str):
            out_path.write_bytes(base64.b64decode(img_bytes))
        elif hasattr(img_bytes, "save"):   # PIL Image
            img_bytes.save(out_path)
        else:
            return False
        return True
    except Exception as e:
        print(f"    [save error] {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Gemini API call with retry
# ---------------------------------------------------------------------------

def translate_image(model, image_path: Path, retries: int = RETRY_ATTEMPTS):
    """
    Send image to Gemini image model.
    Returns (success: bool, image_bytes_or_None, error_message: str).
    """
    for attempt in range(1, retries + 1):
        try:
            img = Image.open(image_path).convert("RGB")
            response = model.generate_content(
                [DEVANAGARI_PROMPT, img],
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 4096,
                },
            )

            # Extract generated image bytes from response parts
            img_bytes = None
            if hasattr(response, "parts"):
                for part in response.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        img_bytes = part.inline_data.data
                        break

            return True, img_bytes, ""

        except Exception as e:
            err_str = str(e)
            is_quota = any(x in err_str.lower()
                           for x in ("quota", "429", "resource_exhausted",
                                     "rate", "limit"))
            wait = 2 ** (attempt - 1)   # 1s, 2s, 4s

            if attempt < retries:
                print(f"    [attempt {attempt}/{retries}] {err_str[:80]} "
                      f"— retrying in {wait}s")
                time.sleep(wait if not is_quota else wait * 10)
            else:
                return False, None, err_str[:200]

    return False, None, "Max retries exceeded"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Ensures we don't exceed `images_per_minute` API calls."""

    def __init__(self, images_per_minute: int):
        self.min_gap = 60.0 / images_per_minute
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        gap = self.min_gap - elapsed
        if gap > 0:
            time.sleep(gap)
        self.last_call = time.time()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Devanagari Meme Translation Pipeline")
    print("=" * 60)

    # Pre-run checks
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory '{INPUT_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    setup_output_directory()

    # Collect images
    all_images = get_image_list(INPUT_DIR, max_images=MAX_IMAGES)
    if not all_images:
        print(f"No images found in {INPUT_DIR}/")
        return

    # Skip already-completed images (resume support)
    completed = load_progress()
    pending = [p for p in all_images if p.name not in completed]

    print(f"\nImages in {INPUT_DIR}/ : {len(all_images)}")
    print(f"Already completed     : {len(completed)}")
    print(f"To process            : {len(pending)}")
    print(f"Rate limit            : {IMAGES_PER_MINUTE} images/minute")
    est_time = len(pending) / IMAGES_PER_MINUTE
    print(f"Estimated time        : {est_time:.1f} minutes")
    print(f"Estimated cost        : ${len(pending) * COST_PER_IMAGE:.2f}")
    print()

    if not pending:
        print("Nothing to do — all images already translated.")
        return

    # Setup Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"Model: {GEMINI_MODEL}\n")

    # Test API connection with first image
    print("Testing API connection...")
    test_path = pending[0]
    ok, _, err = translate_image(model, test_path, retries=1)
    if not ok:
        print(f"API test failed: {err}", file=sys.stderr)
        print("Check your API key and model name.", file=sys.stderr)
        sys.exit(1)
    print("API connection OK.\n")

    # Process images
    limiter   = RateLimiter(IMAGES_PER_MINUTE)
    n_success = 0
    n_failed  = 0
    total_cost = 0.0
    since_checkpoint = 0

    bar = tqdm(pending, desc="Translating", unit="img",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}]")

    for image_path in bar:
        out_path = OUTPUT_DIR / image_path.name

        # Skip if output already exists (extra safety)
        if out_path.exists():
            completed.add(image_path.name)
            n_success += 1
            bar.set_postfix(success=n_success, failed=n_failed)
            continue

        limiter.wait()
        success, img_bytes, error = translate_image(model, image_path)

        if success and img_bytes:
            saved = save_image_bytes(img_bytes, out_path)
            if saved:
                log_result(image_path, out_path, "success",
                           cost=COST_PER_IMAGE)
                completed.add(image_path.name)
                total_cost += COST_PER_IMAGE
                n_success += 1
            else:
                log_result(image_path, None, "failed",
                           error="image save failed", cost=0.0)
                n_failed += 1
        elif success and not img_bytes:
            # Model returned text but no image — log as partial
            log_result(image_path, None, "no_image",
                       error="API returned no image data", cost=COST_PER_IMAGE)
            total_cost += COST_PER_IMAGE
            n_failed += 1
        else:
            log_result(image_path, None, "failed", error=error, cost=0.0)
            n_failed += 1
            # Stop on API quota (429) — resume later when quota resets
            if any(x in (error or "").lower() for x in ("quota", "429", "resource_exhausted")):
                save_progress(completed)
                print(f"\n\n⚠️  API quota exceeded. Stopping. Progress saved.")
                print(f"   Run again later to resume from {len(completed)} completed images.")
                sys.exit(0)

        since_checkpoint += 1
        bar.set_postfix(success=n_success, failed=n_failed,
                        cost=f"${total_cost:.2f}")

        if since_checkpoint >= CHECKPOINT_EVERY:
            save_progress(completed)
            since_checkpoint = 0

    # Final save
    save_progress(completed)

    # Summary
    print(f"\n{'=' * 60}")
    print("Translation Complete")
    print("=" * 60)
    print(f"Total processed  : {n_success + n_failed}")
    print(f"Successful       : {n_success}")
    print(f"Failed           : {n_failed}")
    print(f"Estimated cost   : ${total_cost:.2f}")
    print(f"Output           : {OUTPUT_DIR}/")
    print(f"Log              : {LOG_FILE}")
    print(f"{'=' * 60}")

    if MAX_IMAGES is not None:
        print(f"\nTest run complete ({MAX_IMAGES} images).")
        print("To process all images: set MAX_IMAGES = None and re-run.")


if __name__ == "__main__":
    main()
