#!/usr/bin/env python3
"""
gemini_script_split.py

Classifies 520 translated test images (those in test (1).json) into:
  pure_devanagari  - all text is in Devanagari script, zero English words
  code_mix         - Devanagari + English words both present
  hinglish         - text written entirely in Latin/Roman script (no Devanagari)

Uses Gemini 3 Pro image model to look directly at each image.

Batches of 10 images. Saves progress after every batch.
Retries each image up to 2 times on failure before marking as 'failed'.
Overwrites the existing script_split/ folder.
"""

import base64
import csv
import json
import os
import shutil
import ssl
import time
import urllib.request
import urllib.error
from pathlib import Path

# macOS does not ship Python with system SSL certs; bypass verification for API calls
SSL_CTX = ssl._create_unverified_context()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE       = Path(__file__).resolve().parent
JSON_PATH  = BASE / "test (1).json"
DATA_DIR   = BASE / "translated_categorized_memes"
OUT_DIR    = BASE / "script_split"
PROGRESS   = BASE / "gemini_script_split_progress.json"
OUT_CSV    = BASE / "script_split_classification.csv"

API_KEY    = "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"
MODEL_ID   = "gpt-4o"
API_URL    = "https://api.openai.com/v1/chat/completions"

BATCH_SIZE      = 10   # GPT-4o handles 10 at a time comfortably
BATCH_PAUSE_S   = 5    # short pause between batches
RETRY_PAUSE_S   = 20   # wait 20s before retry on rate limit
MAX_RETRIES     = 2
REQUEST_TIMEOUT = 60   # GPT-4o responds in seconds, not minutes
MIN_REQUEST_GAP = 3    # minimum seconds between individual requests

VALID_LABELS = {"pure_devanagari", "code_mix", "hinglish"}

PROMPT = "Look at this meme image. Based ONLY on the script the text is written in, reply with ONLY one of these three words: pure_devanagari, code_mix, hinglish"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sample_ids(json_path):
    """Returns list of sample_ids from test (1).json, e.g. ['TE-1', 'TE-2', ...]"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [d["sample_id"] for d in data]


def build_image_map(data_dir):
    """
    Returns dict: image_id (e.g. 'TE-8.jpg') -> full Path of the image file.
    Scans all {CATEGORY}/test/ subfolders.
    """
    img_map = {}
    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        test_dir = cat_dir / "test"
        if not test_dir.exists():
            continue
        for img_file in test_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                img_map[img_file.name] = img_file
    return img_map


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def mime_type(path):
    ext = Path(path).suffix.lower()
    return "image/png" if ext == ".png" else "image/jpeg"


def call_gemini(image_path, retries=MAX_RETRIES):
    """
    Sends image to Gemini, returns the label string or 'failed'.
    Enforces MIN_REQUEST_GAP between calls and RETRY_PAUSE_S on 429.
    """
    b64 = image_to_base64(image_path)
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:{mime_type(image_path)};base64,{b64}",
                "detail": "low"
            }},
            {"type": "text", "text": PROMPT}
        ]}],
        "max_tokens": 20,
        "temperature": 0
    }).encode("utf-8")

    for attempt in range(retries + 1):
        t_start = time.time()
        try:
            req = urllib.request.Request(
                API_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=SSL_CTX) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = (result
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                    .lower())

            # Accept if the full label is in the response
            for label in VALID_LABELS:
                if label in text:
                    # Enforce minimum gap between requests
                    elapsed = time.time() - t_start
                    if elapsed < MIN_REQUEST_GAP:
                        time.sleep(MIN_REQUEST_GAP - elapsed)
                    return label

            # Unambiguous prefix matches (handles thinking-model truncation)
            prefix_map = {
                "pure": "pure_devanagari",
                "code": "code_mix",
                "hindi": "hinglish",
                "hingl": "hinglish",
            }
            for prefix, label in prefix_map.items():
                if text.startswith(prefix):
                    elapsed = time.time() - t_start
                    if elapsed < MIN_REQUEST_GAP:
                        time.sleep(MIN_REQUEST_GAP - elapsed)
                    return label

            print(f"    Unexpected response '{text[:60]}', attempt {attempt+1}/{retries+1}", flush=True)

        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            print(f"    API error: HTTP {e.code} {body}, attempt {attempt+1}/{retries+1}", flush=True)
        except Exception as e:
            print(f"    API error: {e}, attempt {attempt+1}/{retries+1}", flush=True)

        if attempt < retries:
            print(f"    Waiting {RETRY_PAUSE_S}s before retry...", flush=True)
            time.sleep(RETRY_PAUSE_S)

    return "failed"


def load_progress():
    if PROGRESS.exists():
        with open(PROGRESS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(results):
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load the 520 sample IDs
    sample_ids = load_sample_ids(JSON_PATH)
    print(f"Loaded {len(sample_ids)} sample IDs from test (1).json")

    # Build image path map
    img_map = build_image_map(DATA_DIR)
    print(f"Found {len(img_map)} images in translated_categorized_memes/")

    # Match sample IDs to image files
    targets = []
    missing = []
    for sid in sample_ids:
        img_id = sid + ".jpg"
        if img_id in img_map:
            targets.append((sid, img_id, img_map[img_id]))
        else:
            missing.append(sid)
    print(f"Matched: {len(targets)} | No image found: {len(missing)}")
    if missing:
        print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Load previously saved progress (allows resuming if interrupted)
    results = load_progress()
    already_done = set(results.keys())
    remaining = [(sid, iid, path) for sid, iid, path in targets if sid not in already_done]
    print(f"Already classified: {len(already_done)} | Remaining: {len(remaining)}")

    # Clear and recreate output folders
    for label in VALID_LABELS | {"failed"}:
        (OUT_DIR / label).mkdir(parents=True, exist_ok=True)

    # Process in batches
    total = len(remaining)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} images) ...")

        for sid, img_id, img_path in batch:
            label = call_gemini(img_path)
            results[sid] = {"image_id": img_id, "label": label}
            status = label.upper()
            print(f"  {img_id} -> {status}")

        # Save progress after every batch
        save_progress(results)
        print(f"  Progress saved ({len(results)}/{len(targets)} done)")

        if batch_start + BATCH_SIZE < total:
            time.sleep(BATCH_PAUSE_S)

    # ---------------------------------------------------------------------------
    # Copy images to their folders
    # ---------------------------------------------------------------------------
    print("\nCopying images to script_split/ subfolders ...")
    counts = {l: 0 for l in VALID_LABELS | {"failed"}}
    for sid, img_id, img_path in targets:
        entry = results.get(sid)
        if not entry:
            continue
        label = entry["label"]
        dest_folder = OUT_DIR / label
        dest_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_folder / img_id)
        counts[label] += 1

    # ---------------------------------------------------------------------------
    # Save classification CSV
    # ---------------------------------------------------------------------------
    # Get category for each image from img_map (reverse lookup)
    cat_map = {}
    for cat_dir in sorted(DATA_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        test_dir = cat_dir / "test"
        if not test_dir.exists():
            continue
        for img_file in test_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                cat_map[img_file.name] = cat_dir.name

    rows = []
    for sid, img_id, _ in targets:
        entry = results.get(sid, {})
        rows.append({
            "sample_id":    sid,
            "image_id":     img_id,
            "category":     cat_map.get(img_id, ""),
            "script_class": entry.get("label", "not_processed"),
        })

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_id", "category", "script_class"])
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    for label in sorted(VALID_LABELS) + ["failed"]:
        print(f"  {label:<20}: {counts[label]}")
    print(f"  {'TOTAL':<20}: {sum(counts.values())}")
    print()
    print(f"Images in  : {OUT_DIR}/")
    print(f"CSV saved  : {OUT_CSV.name}")
    print(f"Progress   : {PROGRESS.name}")

    failed_list = [sid for sid, d in results.items() if d.get("label") == "failed"]
    if failed_list:
        print(f"\nFailed images ({len(failed_list)}) — check manually:")
        for sid in failed_list:
            print(f"  {sid}")


if __name__ == "__main__":
    main()
