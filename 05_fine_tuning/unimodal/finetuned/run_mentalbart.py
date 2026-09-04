#!/usr/bin/env python3
"""
Run MentalBART (Tianlin668/MentalBART) for depression symptom prediction using generation.
Prompt: OCR text → "Which symptoms (1-7) does this suggest?" → parse generated response.
Uses same data/OCR as run_unimodal.py; output format matches for merging.
Output: by default mentalbart_predictions_YYYYMMDD_HHMMSS.csv (never overwrites). Pass --output to write a specific file.
"""

import argparse
import os
import re
import ssl
import sys
from datetime import datetime
from pathlib import Path

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
DEVICE = "cuda:0"
NUM_LABELS = 7
MENTALBART_ID = "Tianlin668/MentalBART"
MAX_INPUT_LENGTH = 400   # tokens for prompt + OCR (BART has 1024 max)
MAX_NEW_TOKENS = 80
TRUNCATE_OCR_CHARS = 600  # chars of OCR to include so prompt fits

# Instruction-style prompt; ask for only clearly applicable symptoms to reduce generic "all 7" responses
MENTALBART_PROMPT_TEMPLATE = """Analyze this text for depression symptoms. List ONLY the numbers (1-7) that clearly apply to the text. If none apply, say None. Do not list all seven.
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating disorder 5=Low self-esteem 6=Concentration 7=Sleep

Text: {text}"""


def collect_test_images(base_dir):
    """Returns list of dicts: {image_path, category, filename}. Only test/ folders."""
    images = []
    base_path = Path(base_dir)
    if not base_path.exists():
        return images
    for category_folder in sorted(base_path.iterdir()):
        if category_folder.is_dir():
            test_folder = category_folder / "test"
            if test_folder.exists():
                for img_file in sorted(test_folder.iterdir()):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        images.append({
                            "image_path": str(img_file),
                            "category": category_folder.name,
                            "filename": img_file.name,
                        })
    return images


def extract_text(reader, image_path):
    """Extract text from image using EasyOCR. Returns placeholder on failure."""
    try:
        results = reader.readtext(image_path, detail=0)
        text = " ".join(results).strip()
        return text if text else "[NO_TEXT]"
    except Exception as e:
        print(f"  [OCR error] {image_path}: {e}", file=sys.stderr)
        return "[OCR_ERROR]"


def parse_generated_response(text):
    """
    Parse MentalBART's generated text to extract which symptoms (1-7) were mentioned.
    Uses both explicit numbers (1-7) and keyword/phrase matching for natural language.
    Returns: (binary_list of length 7, raw_text_for_csv).
    """
    if not text or not isinstance(text, str):
        return [0] * 7, str(text) if text else ""
    text_lower = text.lower().strip()
    binary = [0] * 7

    # "None" or "no symptoms" → all zeros (already)
    if "none" in text_lower and len(text_lower) < 30:
        return binary, text.strip()

    # 1) Prefer explicit numbers 1-7 (comma/space separated at start)
    first_part = text_lower[:80].strip()
    numbers_in_text = re.findall(r'\b([1-7])\b', first_part)
    if numbers_in_text:
        cleaned = re.sub(r'[\s,]', '', first_part)
        if cleaned and all(c in '1234567' for c in cleaned):
            for num_str in numbers_in_text:
                num = int(num_str)
                binary[num - 1] = 1
            return binary, text.strip()

    # 2) "symptoms: 1, 3, 5" style
    number_pattern = re.search(r'symptom[s]?[:\s]+([0-7,\s]+)', text_lower)
    if number_pattern:
        for num_str in re.findall(r'\d+', number_pattern.group(1)):
            num = int(num_str)
            if 1 <= num <= 7:
                binary[num - 1] = 1
        return binary, text.strip()

    # 3) Keyword/phrase hints for each symptom (MentalBART often describes in prose)
    symptom_hints = [
        ["feeling down", "feel down", "sad", "depressed", "hopeless", "depression", "उदास", "निराश"],
        ["lack of interest", "no interest", "pleasure", "enjoy", "motivation", "रुचि"],
        ["self-harm", "self harm", "suicide", "suicidal", "hurt myself", "kill myself", "आत्महत्या"],
        ["eating disorder", "eating", "appetite", "overeating", "not eating", "खाने"],
        ["self-esteem", "self esteem", "worthless", "failure", "useless", "आत्म-सम्मान"],
        ["concentration", "focus", "concentrate", "distracted", "एकाग्रता", "ध्यान"],
        ["sleep", "insomnia", "sleeping", "sleep disturbances", "नींद", "can't sleep"],
    ]
    for i, hints in enumerate(symptom_hints):
        if any(h in text_lower for h in hints):
            binary[i] = 1

    # 4) Any explicit 1-7 as whole tokens (e.g. "symptom 1" or "1, 5")
    for num_str in re.findall(r'\b([1-7])\b', text_lower):
        binary[int(num_str) - 1] = 1
    return binary, text.strip()


def run_generative_inference(model, tokenizer, ocr_texts, device, max_new_tokens=80, debug=False):
    """
    For each OCR text, build prompt, generate, decode, parse.
    Returns: (list of pred strings "[0,1,0,...]", list of raw generated texts).
    """
    model.eval()
    preds = []
    raw_texts = []
    first_done = False

    for text in tqdm(ocr_texts, desc="MentalBART generate"):
        # Truncate OCR so prompt fits
        text_for_prompt = (text[:TRUNCATE_OCR_CHARS] + "…") if len(text) > TRUNCATE_OCR_CHARS else text
        prompt = MENTALBART_PROMPT_TEMPLATE.format(text=text_for_prompt)

        try:
            inputs = tokenizer(
                prompt,
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
            if decoder_start_id is None:
                decoder_start_id = tokenizer.bos_token_id
            if decoder_start_id is None:
                decoder_start_id = tokenizer.eos_token_id
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=5,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    decoder_start_token_id=decoder_start_id,
                )

            # Encoder-decoder generate() returns decoder output only (not encoder input)
            gen_ids = out
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            response = (decoded[0] or "").strip()
            if debug and not first_done:
                first_done = True
                print(f"  [DEBUG] input_ids.shape: {input_ids.shape}, out.shape: {out.shape}, gen_ids.shape: {gen_ids.shape}", file=sys.stderr)
                print(f"  [DEBUG] gen_ids (first 15): {gen_ids[0].tolist()[:15] if gen_ids.shape[1] else []}", file=sys.stderr)
                print(f"  [DEBUG] decoded: {repr(decoded[0] if decoded else '')}", file=sys.stderr)
        except Exception as e:
            print(f"  [Generate error] {e}", file=sys.stderr)
            response = "[GENERATION_ERROR]"

        binary, _ = parse_generated_response(response)
        pred_str = "[" + ",".join(map(str, binary)) + "]"
        preds.append(pred_str)
        raw_texts.append(response if response else "(no output)")

    return preds, raw_texts


def main():
    parser = argparse.ArgumentParser(description="Run MentalBART (generative) for 7 symptoms from OCR text")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to translated_categorized_memes")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: mentalbart_predictions_YYYYMMDD_HHMMSS.csv, never overwrites)",
    )
    parser.add_argument("--max_images", type=int, default=None, help="If set, only process this many images (for testing)")
    parser.add_argument("--debug", action="store_true", help="Print debug info for first generation")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"mentalbart_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print("=== MentalBART generative pipeline ===")
    images = collect_test_images(args.data_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
        print(f"Processing first {len(images)} images (--max_images={args.max_images})")
    print(f"Found {len(images)} test images across categories")

    if not images:
        print("No test images found. Exiting.")
        return

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if str(device) != DEVICE:
        print(f"Note: Using {device} (CUDA not available)")

    print("\n[OCR] Extracting text from all images...")
    reader = easyocr.Reader(["hi", "en"], gpu=(str(device).startswith("cuda")))
    ocr_texts = []
    for img_info in tqdm(images, desc="OCR"):
        text = extract_text(reader, img_info["image_path"])
        ocr_texts.append(text)
    extractable = sum(1 for t in ocr_texts if t not in ("[NO_TEXT]", "[OCR_ERROR]"))
    print(f"OCR complete. {extractable}/{len(images)} images had extractable text.\n")

    print(f"Loading MentalBART from {MENTALBART_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MENTALBART_ID, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MENTALBART_ID,
            use_safetensors=True,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    print("Running generative inference (prompt → generate → parse)...")
    preds, raw_texts = run_generative_inference(
        model, tokenizer, ocr_texts, device, max_new_tokens=MAX_NEW_TOKENS, debug=args.debug
    )

    results = {
        "image_id": [x["filename"] for x in images],
        "category": [x["category"] for x in images],
        "ocr_text": ocr_texts,
        "MentalBART_pred": preds,
        "MentalBART_raw": raw_texts,
        "human_label": [""] * len(images),
    }

    df = pd.DataFrame(results)
    col_order = ["image_id", "category", "ocr_text", "MentalBART_pred", "MentalBART_raw", "human_label"]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
    print("Done. MentalBART_raw contains the model's generated response; MentalBART_pred is parsed binary symptoms.")


if __name__ == "__main__":
    main()
