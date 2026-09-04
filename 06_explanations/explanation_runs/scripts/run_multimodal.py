#!/usr/bin/env python3
"""
Multimodal explanation benchmarking.

Input:  phase-1 explanations CSV + meme images under translated_categorized_memes/
Output: explanation_runs/multimodal/<ModelName>/predictions.csv, metrics_*.csv

Models receive: image + structured explanation as text (generative YES/NO parsing).

Model IDs match ``multimodal/run_multimodal.py`` MODEL_CONFIGS (subset with loaders here).

CLI:
  python explanation_runs/scripts/run_multimodal.py \\
    --repo-root . --explanations-csv gemini_runs/run_v2/phase1_explanations_v2.csv

  # Only BLIP-Base (explanation + image, per-symptom YES/NO like BLIP-2):
  python explanation_runs/scripts/run_multimodal.py --repo-root . --only BLIP
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (set by apply_path_config() in main)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
EXPL_CSV     = BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv"
IMAGES_DIR   = BASE_DIR / "translated_categorized_memes"
BASELINE_CSV = BASE_DIR / "metrics" / "no_finetuning" / "metrics_summary.csv"
OUT_ROOT     = SCRIPT_DIR.parent / "multimodal"
ERRORS_LOG   = SCRIPT_DIR.parent / "errors.log"


def parse_args():
    repo_default = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(
        description="Generative VLMs on image + Gemini explanation text.",
    )
    p.add_argument("--repo-root", type=Path, default=repo_default)
    p.add_argument("--explanations-csv", type=Path, default=None)
    p.add_argument("--baseline-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None, help="Meme images root (default: translated_categorized_memes).")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: explanation_runs/multimodal")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="FOLDER",
        help="Run a single model by output folder name, e.g. BLIP (matches MODEL_CONFIGS second field).",
    )
    return p.parse_args()


def apply_path_config(args: argparse.Namespace) -> None:
    global BASE_DIR, EXPL_CSV, IMAGES_DIR, BASELINE_CSV, OUT_ROOT, ERRORS_LOG
    BASE_DIR = args.repo_root.resolve()
    EXPL_CSV = (
        args.explanations_csv.resolve()
        if args.explanations_csv
        else (BASE_DIR / "gemini_runs" / "run_v2" / "phase1_explanations_v2.csv")
    )
    IMAGES_DIR = (
        args.images_dir.resolve()
        if args.images_dir
        else (BASE_DIR / "translated_categorized_memes")
    )
    BASELINE_CSV = (
        args.baseline_csv.resolve()
        if args.baseline_csv
        else (BASE_DIR / "metrics" / "no_finetuning" / "metrics_summary.csv")
    )
    OUT_ROOT = (
        args.output_dir.resolve()
        if args.output_dir
        else (SCRIPT_DIR.parent / "multimodal")
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ERRORS_LOG = OUT_ROOT.parent / "errors.log"

HF_TOKEN = os.environ.get("HF_TOKEN")

NUM_LABELS = 7
SYMPTOM_NAMES = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

# Prompt template for all multimodal models
PROMPT_TEMPLATE = """Here is a description of this mental health meme:
{explanation}
Based on the image and this description, does the meme express any of the following mental health symptoms?
For each symptom, answer YES or NO.
1. Feeling Down (persistent sadness, hopelessness, emptiness)
2. Lack of Interest (loss of motivation, inability to enjoy things)
3. Self-Harm (references to harming oneself, suicidal thoughts)
4. Eating Disorder (disordered eating habits)
5. Low Self-Esteem (worthlessness, guilt, self-blame)
6. Concentration Problem (difficulty focusing, brain fog)
7. Sleeping Disorder (insomnia, hypersomnia, disrupted sleep)
Respond in this exact format:
Feeling Down: YES/NO
Lack of Interest: YES/NO
Self-Harm: YES/NO
Eating Disorder: YES/NO
Low Self-Esteem: YES/NO
Concentration Problem: YES/NO
Sleeping Disorder: YES/NO"""

# (hf_id, folder_name, display_name, model_type) — IDs aligned with multimodal/run_multimodal.py
MODEL_CONFIGS = [
    ("Salesforce/blip2-flan-t5-xl", "BLIP2", "BLIP-2-Flan-T5-XL", "blip2"),
    ("Salesforce/blip-image-captioning-base", "BLIP", "BLIP-Base", "blip"),
    ("llava-hf/llava-1.5-7b-hf", "LLaVA", "LLaVA-1.5-7B", "llava"),
    ("llava-hf/llava-v1.6-vicuna-7b-hf", "LLaVA_NeXT", "LLaVA-NeXT-7B", "llava_next"),
    ("HuggingFaceM4/idefics-9b-instruct", "IDEFICS", "IDEFICS-9B-Instruct", "idefics"),
    ("Salesforce/instructblip-vicuna-7b", "InstructBLIP", "InstructBLIP-Vicuna-7B", "instructblip"),
]

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_error(model_name: str, msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{model_name}] {msg}\n"
    with open(ERRORS_LOG, "a") as f:
        f.write(line)
    print(f"  ERROR: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------
def find_image(image_id: str, category: str) -> Path | None:
    # Primary: category/test/<image_id>
    p = IMAGES_DIR / category / "test" / image_id
    if p.exists():
        return p
    # Fallback: search all category/test dirs
    for cat_dir in IMAGES_DIR.iterdir():
        if cat_dir.is_dir():
            for sub in ["test", "validation"]:
                pp = cat_dir / sub / image_id
                if pp.exists():
                    return pp
    return None


# ---------------------------------------------------------------------------
# YES/NO parsing
# ---------------------------------------------------------------------------
def parse_yesno_response(text: str) -> list[int] | None:
    """Parse YES/NO lines into binary vector. Returns None if parsing fails."""
    label_keys = [
        ("feeling down",        0),
        ("lack of interest",    1),
        ("self-harm",           2),
        ("self harm",           2),
        ("eating disorder",     3),
        ("low self-esteem",     4),
        ("low self esteem",     4),
        ("concentration problem", 5),
        ("sleeping disorder",   6),
    ]
    vec = [0] * NUM_LABELS
    found = set()

    lines = text.strip().split("\n")
    for line in lines:
        line_l = line.lower().strip()
        for key, idx in label_keys:
            if key in line_l:
                # Match the definitive answer at end of line; avoid "YES/NO" template echoes
                answer = re.search(r":\s*(yes|no)\s*$", line_l)
                if not answer:
                    # Fallback: YES not followed by "/" (template echo guard)
                    answer = re.search(r"\b(yes|no)\b(?!\s*/)", line_l)
                if answer:
                    vec[idx] = 1 if answer.group(1) == "yes" else 0
                    found.add(idx)

    # If we found at least 4 labels, accept it
    if len(found) >= 4:
        return vec

    # Fallback: numbered list 1-7, check YES/NO after each number
    vec2 = [0] * NUM_LABELS
    found2 = set()
    for line in lines:
        m = re.match(r"\s*(\d)\s*[.):-]?\s*(yes|no)\b(?!\s*/)", line.lower())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < NUM_LABELS:
                vec2[idx] = 1 if m.group(2) == "yes" else 0
                found2.add(idx)
    if len(found2) >= 4:
        return vec2

    # Final fallback: narrative extraction
    return parse_narrative_response(text)


def parse_narrative_response(text: str) -> list[int] | None:
    """Fallback parser for narrative responses that don't follow YES/NO format.

    Handles two patterns LLaVA-NeXT produces:
      1. Negation: "does not show / no symptoms / not express" → all zeros
      2. Affirmative narrative: "expresses low self-esteem and feeling down" → keyword match
    Returns None if the text is too ambiguous to extract a signal from.
    """
    t = text.lower().strip()

    # Pattern 1: blanket negation
    negation_phrases = [
        "does not", "doesn't", "do not", "don't",
        "no symptom", "none of the", "not express", "not show",
        "not depict", "not explicitly", "cannot determine",
        "not related to", "not associated",
    ]
    if any(p in t for p in negation_phrases):
        return [0] * NUM_LABELS

    # Pattern 2: affirmative narrative — look for symptom keywords
    keyword_map = [
        (["feeling down", "sadness", "hopeless", "sad", "emotional emptiness", "despair"], 0),
        (["lack of interest", "loss of motivation", "unmotivated", "no motivation",
          "inability to enjoy", "anhedonia", "apathy"], 1),
        (["self-harm", "self harm", "harm oneself", "suicid", "cutting", "self-injur"], 2),
        (["eating disorder", "disordered eating", "binge", "anorexia", "bulimia",
          "food", "calorie", "diet obsess"], 3),
        (["low self-esteem", "self-esteem", "worthless", "guilt", "self-blame",
          "self-doubt", "inadequ", "shame"], 4),
        (["concentration", "focus", "brain fog", "distract", "attention", "cognitive"], 5),
        (["sleeping disorder", "sleep disorder", "insomnia", "hypersomnia",
          "disrupted sleep", "sleep problem", "can't sleep", "cannot sleep"], 6),
    ]
    vec = [0] * NUM_LABELS
    found = set()
    for keywords, idx in keyword_map:
        for kw in keywords:
            if kw in t:
                vec[idx] = 1
                found.add(idx)
                break

    # Only accept if we matched at least one keyword (avoids empty-signal returns)
    if found:
        return vec

    return None


def format_vector(arr: list[int]) -> str:
    return "[" + ",".join(map(str, arr)) + "]"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_blip2(model_id: str):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )
    if getattr(model.config, "image_token_id", None) is None:
        model.config.image_token_id = -1
    processor = Blip2Processor.from_pretrained(model_id, token=HF_TOKEN)
    return model, processor


def load_blip(model_id: str):
    """Original BLIP (captioning / conditional generation), same per-symptom YES/NO path as BLIP-2."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN,
    )
    processor = BlipProcessor.from_pretrained(model_id, token=HF_TOKEN)
    return model, processor


def load_llava(model_id: str):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="flash_attention_2", token=HF_TOKEN
        )
    except Exception:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
        )
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    return model, processor


def load_llava_next(model_id: str):
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="flash_attention_2", token=HF_TOKEN
        )
    except Exception:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
        )
    processor = LlavaNextProcessor.from_pretrained(model_id, token=HF_TOKEN)
    return model, processor


def load_idefics(model_id: str):
    from transformers import IdeficsForVisionText2Text, AutoProcessor
    model = IdeficsForVisionText2Text.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    return model, processor


def load_instructblip(model_id: str):
    from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )
    processor = InstructBlipProcessor.from_pretrained(model_id, token=HF_TOKEN)
    return model, processor


LOADERS = {
    "blip2":        load_blip2,
    "blip":         load_blip,
    "llava":        load_llava,
    "llava_next":   load_llava_next,
    "idefics":      load_idefics,
    "instructblip": load_instructblip,
}


# ---------------------------------------------------------------------------
# Per-model inference
# ---------------------------------------------------------------------------
def build_prompt(model_type: str, explanation: str) -> str | list:
    """Return the prompt string/list expected by each model type."""
    # LLaVA-NeXT: pre-truncate explanation to avoid hitting context limits
    # (dynamic image tiling can use 1000s of tokens for the image alone)
    if model_type == "llava_next":
        explanation = explanation[:400]

    body = PROMPT_TEMPLATE.format(explanation=explanation)

    if model_type in ("llava", "llava_next"):
        return f"USER: <image>\n{body}\nASSISTANT:"
    if model_type == "instructblip":
        return body          # per-symptom querying handles this internally
    if model_type == "idefics":
        return body          # assembled with image list in inference fn
    if model_type in ("blip2", "blip"):
        return body
    return body


BLIP2_SYMPTOM_QUESTIONS = [
    ("Feeling Down",         "Does this meme depict persistent sadness, hopelessness, or emotional emptiness (Feeling Down)?"),
    ("Lack of Interest",     "Does this meme depict loss of motivation or inability to enjoy things (Lack of Interest)?"),
    ("Self-Harm",            "Does this meme reference self-harm, harming oneself, or suicidal thoughts?"),
    ("Eating Disorder",      "Does this meme depict disordered eating habits or an eating disorder?"),
    ("Low Self-Esteem",      "Does this meme depict worthlessness, guilt, or self-blame (Low Self-Esteem)?"),
    ("Concentration Problem","Does this meme depict difficulty focusing or brain fog (Concentration Problem)?"),
    ("Sleeping Disorder",    "Does this meme depict insomnia, hypersomnia, or disrupted sleep (Sleeping Disorder)?"),
]


def infer_blip2_per_symptom(model, processor, image: Image.Image, explanation: str) -> str:
    """BLIP-2 Flan-T5: query each symptom separately.
    Returns a direct binary vector string e.g. '[1,0,1,0,0,0,0]' to avoid CSV newline issues."""
    context = explanation[:300]
    vec = []
    for sym_name, question in BLIP2_SYMPTOM_QUESTIONS:
        prompt = f"{question} Context: {context} Answer YES or NO."
        try:
            inputs = processor(images=image, text=prompt,
                               return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8)
            ans = processor.decode(out[0], skip_special_tokens=True).strip().upper()
            val = 1 if "YES" in ans else 0
        except Exception:
            val = 0
        vec.append(val)
    return "[" + ",".join(map(str, vec)) + "]"


def infer_instructblip_per_symptom(model, processor, image: Image.Image, explanation: str) -> str:
    """InstructBLIP Flan-T5: query each symptom separately, same as BLIP-2.
    Returns a direct binary vector string e.g. '[1,0,1,0,0,0,0]'."""
    context = explanation[:300]
    vec = []
    for sym_name, question in BLIP2_SYMPTOM_QUESTIONS:
        prompt = f"{question} Context: {context} Answer YES or NO."
        try:
            inputs = processor(images=image, text=prompt,
                               return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8)
            ans = processor.decode(out[0], skip_special_tokens=True).strip().upper()
            val = 1 if "YES" in ans else 0
        except Exception:
            val = 0
        vec.append(val)
    return "[" + ",".join(map(str, vec)) + "]"


def infer_one(model, processor, image: Image.Image, prompt, model_type: str,
              explanation: str = "") -> str:
    """Run single-image inference and return raw generated text."""
    try:
        if model_type in ("blip2", "blip"):
            return infer_blip2_per_symptom(model, processor, image, explanation)

        elif model_type == "instructblip":
            return infer_instructblip_per_symptom(model, processor, image, explanation)

        elif model_type == "llava":
            inputs = processor(text=prompt, images=image, return_tensors="pt",
                               truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            full = processor.decode(out[0], skip_special_tokens=True)
            if "ASSISTANT:" in full:
                full = full.split("ASSISTANT:")[-1].strip()
            return full

        elif model_type == "llava_next":
            # Do NOT truncate: LLaVA-NeXT uses dynamic image tiling and truncation
            # causes an image-token-count mismatch between text and input_ids.
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            full = processor.decode(out[0], skip_special_tokens=True)
            if "ASSISTANT:" in full:
                full = full.split("ASSISTANT:")[-1].strip()
            return full

        elif model_type == "idefics":
            # IDEFICS processor expects list with interleaved text/image
            inputs = processor(
                text=["User: ", image, f"\n{prompt}\nAssistant:"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            full = processor.decode(out[0], skip_special_tokens=True)
            if "Assistant:" in full:
                full = full.split("Assistant:")[-1].strip()
            return full

    except Exception as e:
        raise RuntimeError(f"infer_one failed: {e}") from e

    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_label_vector(s):
    try:
        vals = [int(x) for x in str(s).strip().strip("[]").split(",")]
        return vals if len(vals) == NUM_LABELS else None
    except Exception:
        return None


def compute_metrics(gold_col, pred_col):
    pairs = [(parse_label_vector(g), parse_label_vector(p))
             for g, p in zip(gold_col, pred_col)]
    pairs = [(g, p) for g, p in pairs if g is not None and p is not None]
    if not pairs:
        return None
    G = np.array([x[0] for x in pairs])
    P = np.array([x[1] for x in pairs])
    result = {
        "n_samples":        len(G),
        "macro_f1":         round(float(f1_score(G, P, average="macro",    zero_division=0)), 4),
        "weighted_f1":      round(float(f1_score(G, P, average="weighted", zero_division=0)), 4),
        "macro_precision":  round(float(precision_score(G, P, average="macro", zero_division=0)), 4),
        "macro_recall":     round(float(recall_score(G, P, average="macro",    zero_division=0)), 4),
        "per_label": {},
    }
    for i, sym in enumerate(SYMPTOM_NAMES):
        result["per_label"][sym] = {
            "f1":        round(float(f1_score(G[:, i],        P[:, i], zero_division=0)), 4),
            "precision": round(float(precision_score(G[:, i], P[:, i], zero_division=0)), 4),
            "recall":    round(float(recall_score(G[:, i],    P[:, i], zero_division=0)), 4),
        }
    return result


def check_degenerate(pred_col, model_name):
    vecs = [parse_label_vector(p) for p in pred_col if parse_label_vector(p) is not None]
    if not vecs:
        return
    arr  = np.array(vecs)
    means = arr.mean(axis=0)
    if np.all(means == 0) or np.all(means == 1):
        log_error(model_name, "degenerate predictions — all-zeros or all-ones across dataset")
    elif np.any(means == 0) or np.any(means == 1):
        bad = [SYMPTOM_NAMES[i] for i in range(NUM_LABELS) if means[i] in (0, 1)]
        log_error(model_name, f"degenerate per-label predictions for: {bad}")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------
def run_model(model_id, folder, display_name, model_type, rows_df):
    out_dir  = OUT_ROOT / folder
    out_dir.mkdir(exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    met_json = out_dir / "metrics.json"

    # Resume: detect already-processed image_ids
    done_ids = set()
    if pred_csv.exists():
        try:
            existing = pd.read_csv(pred_csv)
            done_ids = set(existing["image_id"].tolist())
            print(f"  Resume: {len(done_ids)} rows already done.")
        except Exception:
            done_ids = set()

    todo = rows_df[~rows_df["image_id"].isin(done_ids)].reset_index(drop=True)
    if todo.empty:
        print(f"  All rows done — skipping.")
        return

    # --- Sanity check (3 rows) ---
    print(f"  Loading model ...")
    try:
        model, processor = LOADERS[model_type](model_id)
    except Exception as e:
        log_error(display_name, f"Model load failed: {e}\n{traceback.format_exc()}")
        return

    print(f"  Sanity check on 3 rows ...")
    sanity_ok = True
    for i in range(min(3, len(todo))):
        row = todo.iloc[i]
        img_path = find_image(row["image_id"], row["category"])
        if img_path is None:
            print(f"  [sanity] image not found: {row['image_id']}")
            continue
        try:
            img    = Image.open(img_path).convert("RGB")
            prompt = build_prompt(model_type, row["explanation"])
            text   = infer_one(model, processor, img, prompt, model_type,
                               explanation=str(row["explanation"]))
            if text.startswith("[") and text.endswith("]") \
                    and parse_label_vector(text) is not None:
                vec = parse_label_vector(text)
            else:
                vec = parse_yesno_response(text)
            print(f"  [sanity {i}] parsed={vec}  raw={text[:80].replace(chr(10),' ')!r}")
        except Exception as e:
            log_error(display_name, f"Sanity row {i} failed: {e}")
            sanity_ok = False

    if not sanity_ok:
        log_error(display_name, "Sanity check had failures — aborting model.")
        del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return

    # --- Full run ---
    print(f"  Full run on {len(todo)} rows ...")
    t0 = time.time()
    n  = len(todo)
    write_header = not pred_csv.exists()

    for i, row in todo.iterrows():
        img_path = find_image(row["image_id"], row["category"])
        if img_path is None:
            pred_vec = "[IMAGE_NOT_FOUND]"
            raw_text = ""
        else:
            try:
                img    = Image.open(img_path).convert("RGB")
                prompt = build_prompt(model_type, row["explanation"])
                raw_text = infer_one(model, processor, img, prompt, model_type,
                                    explanation=str(row["explanation"]))
                # BLIP2 and InstructBLIP return direct vector strings; others return text
                if raw_text.startswith("[") and raw_text.endswith("]") \
                        and parse_label_vector(raw_text) is not None:
                    pred_vec = raw_text
                else:
                    parsed   = parse_yesno_response(raw_text)
                    pred_vec = format_vector(parsed) if parsed else "[PARSE_FAIL]"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    log_error(display_name, f"OOM at row {i}: {e}")
                    pred_vec = "[OOM]"
                    raw_text = ""
                else:
                    log_error(display_name, f"Inference error at row {i}: {e}")
                    pred_vec = "[INFERENCE_ERROR]"
                    raw_text = ""
            except Exception as e:
                log_error(display_name, f"Error at row {i}: {e}")
                pred_vec = "[INFERENCE_ERROR]"
                raw_text = ""

        rec = pd.DataFrame([{
            "image_id":         row["image_id"],
            "category":         row["category"],
            "gold_labels":      row["gold_labels"],
            "predicted_labels": pred_vec,
            "raw_response":     raw_text[:500],
        }])
        rec.to_csv(pred_csv, mode="a", header=write_header, index=False)
        write_header = False

        done_count = i + 1
        elapsed    = time.time() - t0
        rate       = done_count / elapsed if elapsed > 0 else 0
        eta        = (n - done_count) / rate if rate > 0 else 0
        print(f"\r  {done_count}/{n}  ETA {eta:.0f}s  ", end="", flush=True)

    print()

    # --- Metrics ---
    full_df = pd.read_csv(pred_csv)
    check_degenerate(full_df["predicted_labels"].tolist(), display_name)
    m = compute_metrics(full_df["gold_labels"], full_df["predicted_labels"])
    if m:
        m["model"]  = display_name
        m["input"]  = "explanation+image"
        m["folder"] = folder
        with open(met_json, "w") as f:
            json.dump(m, f, indent=2)
        print(f"  macro_f1={m['macro_f1']}  weighted_f1={m['weighted_f1']}  "
              f"P={m['macro_precision']}  R={m['macro_recall']}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate_metrics():
    summary_rows, per_label_rows = [], []
    for _, folder, display_name, _ in MODEL_CONFIGS:
        met_json = OUT_ROOT / folder / "metrics.json"
        if not met_json.exists():
            continue
        with open(met_json) as f:
            m = json.load(f)
        summary_rows.append({
            "model":           display_name,
            "input":           "explanation+image",
            "n_samples":       m.get("n_samples", 0),
            "macro_f1":        m.get("macro_f1", ""),
            "weighted_f1":     m.get("weighted_f1", ""),
            "macro_precision": m.get("macro_precision", ""),
            "macro_recall":    m.get("macro_recall", ""),
        })
        for sym, vals in m.get("per_label", {}).items():
            per_label_rows.append({
                "model": display_name, "input": "explanation+image",
                "symptom": sym, **vals,
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "metrics_summary.csv", index=False)
        pd.DataFrame(per_label_rows).to_csv(OUT_ROOT / "metrics_per_label.csv", index=False)

    return pd.DataFrame(summary_rows) if summary_rows else None


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(summary_df):
    if summary_df is None or summary_df.empty:
        return
    print("\n" + "=" * 90)
    print("FINAL COMPARISON TABLE (multimodal: explanation+image)")
    print("=" * 90)

    baseline = {}
    if BASELINE_CSV.exists():
        try:
            b = pd.read_csv(BASELINE_CSV)
            for _, row in b.iterrows():
                baseline[str(row["model"])] = row
        except Exception:
            pass

    header = f"{'model':<22} {'input':<17} {'n':>5} {'macro_f1':>9} {'wt_f1':>8} {'prec':>7} {'rec':>7}"
    print(header)
    print("-" * len(header))
    for _, row in summary_df.iterrows():
        print(f"{str(row['model']):<22} {'explanation+image':<17} {int(row['n_samples']):>5} "
              f"{float(row['macro_f1']):>9.4f} {float(row['weighted_f1']):>8.4f} "
              f"{float(row['macro_precision']):>7.4f} {float(row['macro_recall']):>7.4f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Post-process: re-parse raw_response for already-saved CSVs
# ---------------------------------------------------------------------------
def reparse_predictions(folder: str, display_name: str):
    """Re-apply parse_yesno_response to raw_response column of existing CSV."""
    pred_csv = OUT_ROOT / folder / "predictions.csv"
    if not pred_csv.exists():
        return
    df = pd.read_csv(pred_csv)
    if "raw_response" not in df.columns:
        return

    fixed = 0
    for i, row in df.iterrows():
        raw  = str(row.get("raw_response", ""))
        old  = str(row.get("predicted_labels", ""))
        # If raw_response is already a valid direct vector (BLIP2/InstructBLIP),
        # ensure predicted_labels reflects it rather than overwriting with PARSE_FAIL
        if raw.startswith("[") and raw.endswith("]") and parse_label_vector(raw) is not None:
            if old != raw:
                df.at[i, "predicted_labels"] = raw
                fixed += 1
            continue
        parsed = parse_yesno_response(raw)
        new = format_vector(parsed) if parsed else "[PARSE_FAIL]"
        if new != old:
            df.at[i, "predicted_labels"] = new
            fixed += 1

    if fixed:
        df.to_csv(pred_csv, index=False)
        print(f"  Re-parsed {fixed} rows in {folder}/predictions.csv")

        # Re-compute metrics
        met_json = OUT_ROOT / folder / "metrics.json"
        m = compute_metrics(df["gold_labels"], df["predicted_labels"])
        if m:
            m["model"]  = display_name
            m["input"]  = "explanation+image"
            m["folder"] = folder
            with open(met_json, "w") as f:
                json.dump(m, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    apply_path_config(args)

    print("=" * 70)
    print("MULTIMODAL EXPLANATION BENCHMARKING")
    print(f"  Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  Input   : {EXPL_CSV}")
    print(f"  Images  : {IMAGES_DIR}")
    print(f"  Output  : {OUT_ROOT}")
    print("=" * 70 + "\n")

    df    = pd.read_csv(EXPL_CSV)
    valid = df["explanation"].notna() & (df["explanation"].str.strip() != "")
    df    = df[valid].reset_index(drop=True)
    print(f"Loaded {len(df)} rows (skipped {(~valid).sum()} empty/safety-blocked)\n")

    configs = MODEL_CONFIGS
    if args.only:
        configs = [c for c in MODEL_CONFIGS if c[1] == args.only]
        if not configs:
            print(f"Unknown --only {args.only!r}. Valid folder names: {', '.join(c[1] for c in MODEL_CONFIGS)}")
            return

    for model_id, folder, display_name, model_type in configs:
        print(f"\n{'='*60}")
        print(f"[MODEL] {display_name}  ({model_id})")
        try:
            run_model(model_id, folder, display_name, model_type, df)
        except Exception as e:
            log_error(display_name, f"Unhandled exception: {e}\n{traceback.format_exc()}")
            print(f"  FAILED — continuing.\n", file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n\n[Re-parsing predictions with fixed parser ...]")
    for _, folder, display_name, _ in configs:
        reparse_predictions(folder, display_name)

    print("\n[Aggregating metrics ...]")
    summary_df = aggregate_metrics()
    print_comparison(summary_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
