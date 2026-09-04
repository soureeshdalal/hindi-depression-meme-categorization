#!/usr/bin/env python3
"""
Multimodal benchmarking pipeline for Hinglish (Hindi-English) mental health memes.
Reads from a flat directory of 44 Hinglish meme images.
Saves one CSV per model to hinglish_multimodal/{ModelName}/ — never overwrites.
Run one model at a time with --only_model to manage GPU memory:
    python multimodal/run_hinglish_multimodal.py --only_model LLaVA
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)
# These submodules were removed / renamed in some transformers versions; import lazily
try:
    from transformers.models.idefics import IdeficsForVisionText2Text
except ImportError:
    IdeficsForVisionText2Text = None
try:
    from transformers.models.llava_next import LlavaNextForConditionalGeneration, LlavaNextProcessor
except ImportError:
    try:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    except ImportError:
        LlavaNextForConditionalGeneration = None
        LlavaNextProcessor = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HINGLISH_IMAGE_DIR = PROJECT_ROOT / "hinglish_memes"
HINGLISH_PROGRESS_JSON = PROJECT_ROOT / "hinglish_progress.json"
HINGLISH_UNIMODAL_DIR = PROJECT_ROOT / "hinglish_unimodal"
OUTPUT_BASE_DIR = PROJECT_ROOT / "hinglish_multimodal"

NUM_LABELS = 7
SYMPTOM_NAMES = [
    "Feeling Down (उदास महसूस करना)",
    "Lack of Interest (रुचि की कमी)",
    "Self-Harm (आत्म-हानि)",
    "Eating Disorder (खाने का विकार)",
    "Low Self-Esteem (कम आत्म-सम्मान)",
    "Concentration Problem (एकाग्रता की समस्या)",
    "Sleeping Disorder (नींद का विकार)",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
BLIP2_PROMPT = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

MBLIP_PROMPT = """This meme is about mental health. Which of these apply? 1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep. Reply with only the numbers separated by commas (e.g. 1, 5) or the word None. Nothing else."""

LLaVA_PROMPT = """USER: <image>

Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None". Do not list all numbers.
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep
ASSISTANT:"""

INSTRUCTBLIP_PROMPT = """<image>
Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

MINICPM_V_QUESTION = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

LLAVA_NEXT_PROMPT = """USER: <image>
Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep
ASSISTANT:"""

IDEFICS_QUESTION = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep
Assistant:"""

CHITRARTH_QUESTION = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

# (model_id, short_name, display_name, use_hindi, model_type)
MODEL_CONFIGS = [
    ("llava-hf/llava-1.5-7b-hf",               "LLaVA",        "LLaVA-1.5-7B",           False, "llava"),
    ("llava-hf/llava-v1.6-vicuna-7b-hf",        "LLaVA_NeXT",   "LLaVA-NeXT-7B",           False, "llava_next"),
    ("openbmb/MiniCPM-V",                        "MiniCPM_V",    "MiniCPM-V-3B",            False, "minicpm_v"),
    ("HuggingFaceM4/idefics-9b-instruct",        "IDEFICS",      "IDEFICS-9B-Instruct",     False, "idefics"),
    ("Salesforce/instructblip-vicuna-7b",        "InstructBLIP", "InstructBLIP-Vicuna-7B",  False, "instructblip"),
    ("Salesforce/blip2-flan-t5-xl",              "BLIP2",        "BLIP-2-Flan-T5-XL",       False, "blip2"),
    ("Gregor/mblip-mt0-xl",                      "mBLIP",        "mBLIP-mT0-XL",            False, "blip2"),
    ("Salesforce/blip-image-captioning-base",    "BLIP",         "BLIP-Base",               False, "blip"),
    ("MBZUAI/PALO-7B",                           "PALO",         "PALO-7B",                 False, "palo"),
    ("krutrim-ai-labs/Chitrarth",                "Chitrarth",    "Chitrarth-7.5B",          False, "chitrarth"),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_hinglish_images():
    """
    Returns list of dicts: {image_path, category, filename}
    Category comes from hinglish_progress.json.
    Only includes images that exist in HINGLISH_IMAGE_DIR.
    """
    # Build id → category map from progress JSON
    cat_map = {}
    if HINGLISH_PROGRESS_JSON.exists():
        with open(HINGLISH_PROGRESS_JSON) as f:
            prog = json.load(f)
        for entry in prog.get("sample_ids", []):
            img_id, _, category = entry[0], entry[1], entry[2]
            cat_map[img_id] = category

    images = []
    for img_file in sorted(HINGLISH_IMAGE_DIR.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img_id = img_file.stem  # e.g. "TE-104"
        category = cat_map.get(img_id, "HINGLISH")
        images.append({
            "image_path": str(img_file),
            "category": category,
            "filename": img_file.name,
        })
    return images


def load_human_labels():
    """
    Returns dict: filename → human_label string, from any hinglish_unimodal CSV.
    Falls back to empty string if not found.
    """
    labels = {}
    # Use MentalBERT as source of truth for human labels
    label_csv = HINGLISH_UNIMODAL_DIR / "MentalBERT_predictions.csv"
    if label_csv.exists():
        df = pd.read_csv(label_csv)
        if "human_label" in df.columns and "image_id" in df.columns:
            for _, row in df.iterrows():
                labels[row["image_id"]] = str(row["human_label"]) if pd.notna(row["human_label"]) else ""
    return labels


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def clear_model(model):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model loading (same as run_multimodal.py)
# ---------------------------------------------------------------------------

def load_llava_model(model_id):
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except Exception:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def load_blip2_model(model_id):
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda:0",
    )
    if getattr(model.config, "image_token_id", None) is None:
        model.config.image_token_id = -1
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor


def load_blip_model(model_id):
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        use_safetensors=True,
    )
    processor = BlipProcessor.from_pretrained(model_id)
    return model, processor


def load_palo_model(model_id):
    palo_repo = Path(__file__).resolve().parent / "palo_repo"
    if palo_repo.exists() and str(palo_repo) not in sys.path:
        sys.path.insert(0, str(palo_repo))
    from palo.model import PaloForCausalLM
    from transformers import CLIPImageProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = PaloForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    vt = model.get_vision_tower()
    if vt is not None and getattr(vt, "is_loaded", False) is False:
        vt.load_model()
        vt.to(device=next(model.parameters()).device, dtype=model.dtype)
    vision_tower_name = getattr(model.config, "mm_vision_tower", "openai/clip-vit-large-patch14-336")
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    processor = {"tokenizer": tokenizer, "image_processor": image_processor}
    return model, processor


def load_chitrarth_model(model_id):
    import transformers.models.llama.modeling_llama as _llama_mod
    if not hasattr(_llama_mod, "LlamaDynamicNTKScalingRotaryEmbedding"):
        from chitrarth_rope_compat import (
            LlamaDynamicNTKScalingRotaryEmbedding,
            LlamaLinearScalingRotaryEmbedding,
            LlamaRotaryEmbedding as LlamaRotaryEmbeddingCompat,
        )
        _llama_mod.LlamaDynamicNTKScalingRotaryEmbedding = LlamaDynamicNTKScalingRotaryEmbedding
        _llama_mod.LlamaLinearScalingRotaryEmbedding = LlamaLinearScalingRotaryEmbedding
        _llama_mod.LlamaRotaryEmbedding = LlamaRotaryEmbeddingCompat
    try:
        from chitrarth.model.language_model.chitrarth_mpt import ChitrarthForCausalLM
        from chitrarth.model.language_model.mpt.modeling_mpt import MPTForCausalLM
        from chitrarth.model.builder import load_pretrained_model
    except ImportError as e:
        raise ImportError(
            "Chitrarth requires: pip install git+https://github.com/ola-krutrim/Chitrarth.git"
        ) from e
    _orig_chitrarth_init = ChitrarthForCausalLM.__init__
    def _chitrarth_init_with_tied_keys(self, config):
        _orig_chitrarth_init(self, config)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
    ChitrarthForCausalLM.__init__ = _chitrarth_init_with_tied_keys
    _orig_mpt_tie_weights = MPTForCausalLM.tie_weights
    def _mpt_tie_weights_compat(self, missing_keys=None, recompute_mapping=True, **kwargs):
        _orig_mpt_tie_weights(self)
    MPTForCausalLM.tie_weights = _mpt_tie_weights_compat
    from transformers.generation import GenerationMixin
    if GenerationMixin not in ChitrarthForCausalLM.__mro__:
        ChitrarthForCausalLM.__bases__ = (GenerationMixin,) + ChitrarthForCausalLM.__bases__
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id, model_base=None, model_name="chitrarth"
    )
    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = getattr(model.config, "n_layers", 32)
    _orig_forward = model.forward
    def _forward_with_cache_compat(self, input_ids=None, past_key_values=None, attention_mask=None,
            prefix_mask=None, sequence_id=None, labels=None, return_dict=None,
            output_attentions=None, output_hidden_states=None, use_cache=None,
            inputs_embeds=None, images=None, **kwargs):
        if past_key_values is not None and hasattr(past_key_values, "layers"):
            n_layers = getattr(self.config, "n_layers", len(past_key_values.layers))
            past_key_values = [
                (past_key_values.layers[i].keys, past_key_values.layers[i].values)
                for i in range(min(n_layers, len(past_key_values.layers)))
            ]
        kwargs = {k: v for k, v in kwargs.items() if k != "inputs_embeds"}
        return _orig_forward(input_ids=input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id,
            labels=labels, return_dict=return_dict, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, use_cache=use_cache, images=images, **kwargs)
    model.forward = types.MethodType(_forward_with_cache_compat, model)
    _orig_prepare = model.prepare_inputs_for_generation
    def _prepare_with_attention_mask(input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if kwargs.get("attention_mask") is None and input_ids is not None:
            kwargs["attention_mask"] = torch.ones(
                input_ids.shape[:2], dtype=torch.bool, device=input_ids.device
            )
        return _orig_prepare(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
    model.prepare_inputs_for_generation = _prepare_with_attention_mask
    processor = {"tokenizer": tokenizer, "image_processor": image_processor, "context_len": context_len}
    return model, processor


def load_instructblip_model(model_id):
    processor = InstructBlipProcessor.from_pretrained(model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor


def load_llava_next_model(model_id):
    if LlavaNextForConditionalGeneration is None:
        raise ImportError("LlavaNextForConditionalGeneration not available in this transformers version")
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except Exception:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )
    processor = LlavaNextProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def load_minicpm_v_model(model_id):
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def load_idefics_model(model_id):
    if IdeficsForVisionText2Text is None:
        raise ImportError("IdeficsForVisionText2Text not available in this transformers version")
    processor = AutoProcessor.from_pretrained(model_id)
    model = IdeficsForVisionText2Text.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor


def load_model(model_id, model_type):
    if model_type == "llava":
        return load_llava_model(model_id)
    elif model_type == "llava_next":
        return load_llava_next_model(model_id)
    elif model_type == "minicpm_v":
        return load_minicpm_v_model(model_id)
    elif model_type == "idefics":
        return load_idefics_model(model_id)
    elif model_type == "instructblip":
        return load_instructblip_model(model_id)
    elif model_type == "blip2":
        return load_blip2_model(model_id)
    elif model_type == "blip":
        return load_blip_model(model_id)
    elif model_type == "palo":
        return load_palo_model(model_id)
    elif model_type == "chitrarth":
        return load_chitrarth_model(model_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_natural_language_response(text):
    text = (text or "").replace("\u2581", " ")
    text_lower = text.lower()
    binary = [0] * 7
    confidence = [0.0] * 7

    symptom_patterns = [
        {
            'keywords': ['sad', 'depressed', 'hopeless', 'down', 'depression', 'उदास', 'निराश'],
            'strong': ['depressed', 'hopeless', 'suicidal thoughts', 'want to die', 'निराश'],
            'weak': ['sad', 'down', 'उदास']
        },
        {
            'keywords': ['interest', 'pleasure', 'enjoy', 'motivation', 'रुचि', 'मज़ा'],
            'strong': ['no interest', 'lack of interest', 'no pleasure', 'रुचि की कमी'],
            'weak': ['interest', 'enjoy']
        },
        {
            'keywords': ['self-harm', 'suicide', 'kill myself', 'hurt myself', 'आत्म-हानि', 'आत्महत्या'],
            'strong': ['suicide', 'self-harm', 'kill myself', 'hurt myself', 'आत्महत्या'],
            'weak': ['self-harm']
        },
        {
            'keywords': ['eating', 'appetite', 'food', 'hunger', 'overeating', 'not eating', 'खाना', 'भूख'],
            'strong': ['eating disorder', 'no appetite', 'overeating', 'not eating', 'खाने का विकार'],
            'weak': ['eating', 'food', 'appetite']
        },
        {
            'keywords': ['self-esteem', 'failure', 'worthless', 'useless', 'bad about myself', 'आत्म-सम्मान'],
            'strong': ['low self-esteem', 'worthless', 'failure', 'useless', 'कम आत्म-सम्मान'],
            'weak': ['self-esteem', 'failure']
        },
        {
            'keywords': ['concentration', 'focus', 'attention', 'concentrate', 'distracted', 'एकाग्रता'],
            'strong': ["concentration problem", "can't focus", 'trouble concentrating', 'एकाग्रता की समस्या'],
            'weak': ['concentration', 'focus']
        },
        {
            'keywords': ['sleep', 'insomnia', 'sleeping', 'tired', 'exhausted', 'नींद', 'थकान'],
            'strong': ['sleep problem', 'insomnia', "can't sleep", 'sleeping too much', 'नींद की समस्या'],
            'weak': ['sleep', 'tired']
        },
    ]

    for i, patterns in enumerate(symptom_patterns):
        found = False
        conf = 0.0
        for strong_term in patterns['strong']:
            if strong_term in text_lower:
                found = True
                conf = max(conf, 0.8)
                break
        if not found:
            for weak_term in patterns['weak']:
                if weak_term in text_lower:
                    found = True
                    conf = max(conf, 0.5)
        if f"symptom {i+1}" in text_lower:
            found = True
            conf = max(conf, 0.7)
        if f" {i+1}." in text or f"#{i+1}" in text or f"number {i+1}" in text_lower:
            found = True
            conf = max(conf, 0.7)
        if found:
            binary[i] = 1
            confidence[i] = conf

    number_pattern = re.search(r'symptom[s]?[:\s]+([0-7,\s]+)', text_lower)
    if number_pattern:
        numbers = re.findall(r'\d+', number_pattern.group(1))
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 7:
                binary[num - 1] = 1
                confidence[num - 1] = max(confidence[num - 1], 0.9)

    first_part = text_lower[:80].strip()
    numbers_in_text = re.findall(r'\b([1-7])\b', first_part)
    if numbers_in_text:
        cleaned = re.sub(r'[\s,]', '', first_part)
        if cleaned and all(c in '1234567' for c in cleaned):
            for num_str in numbers_in_text:
                num = int(num_str)
                binary[num - 1] = 1
                confidence[num - 1] = max(confidence[num - 1], 0.85)

    if sum(binary) > 0:
        total_symptoms = sum(binary)
        for i in range(7):
            if binary[i] == 1:
                confidence[i] = min(confidence[i] + 0.1 * (total_symptoms - 1), 1.0)

    return binary, confidence


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_symptoms(model, processor, image_path, prompt, max_new_tokens=256, model_type=None):
    try:
        image = Image.open(image_path).convert("RGB")

        if model_type == "minicpm_v":
            msgs = [{"role": "user", "content": prompt}]
            with torch.no_grad():
                res, _, _ = model.chat(image=image, msgs=msgs, context=None,
                                       tokenizer=processor, sampling=False)
            response = (res or "").strip()
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        if model_type == "idefics":
            prompts = ["User:", image, "\n" + prompt.strip()]
            inputs = processor(text=[prompts], return_tensors="pt")
            def _to_device(x, device):
                if hasattr(x, "to"):
                    return x.to(device)
                if isinstance(x, dict):
                    return {k: _to_device(v, device) for k, v in x.items()}
                return x
            inputs = {k: _to_device(v, model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        if model_type == "blip":
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
            generated_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
            response = processor.decode(generated_ids, skip_special_tokens=True).strip()
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        if model_type == "chitrarth":
            try:
                from chitrarth.inference import eval_model
            except ImportError:
                return [0] * 7, [0.0] * 7, "[ERROR: chitrarth package not installed]"
            tokenizer = processor["tokenizer"]
            image_processor = processor["image_processor"]
            context_len = processor["context_len"]
            chitrarth_max = min(max_new_tokens, 64)
            response = eval_model(
                tokenizer, model, image_processor, context_len,
                prompt.strip(), image_file=image_path,
                max_new_tokens=chitrarth_max, temperature=0.0,
            )
            response = (response or "").strip()
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        if model_type == "palo":
            palo_repo = Path(__file__).resolve().parent / "palo_repo"
            if palo_repo.exists() and str(palo_repo) not in sys.path:
                sys.path.insert(0, str(palo_repo))
            from palo.mm_utils import tokenizer_image_token, process_images
            from palo.constants import IMAGE_TOKEN_INDEX
            tokenizer = processor["tokenizer"]
            image_processor = processor["image_processor"]
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            image_tensor = process_images([image], image_processor, model.config).to(device=device, dtype=dtype)
            input_length = input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(input_ids, images=image_tensor,
                                         max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
            generated_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
            if input_length is not None and len(generated_ids) > input_length:
                response = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
            else:
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            response = response.strip()
            for prefix in ("ASSISTANT:", "assistant:"):
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    break
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        # LLaVA, LLaVA-NeXT, InstructBLIP, BLIP-2
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs.get("input_ids", None)
        input_length = input_ids.shape[1] if input_ids is not None and len(input_ids.shape) > 1 else None

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=0.0)

        generated_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
        full_decoded = processor.decode(generated_ids, skip_special_tokens=True)

        if input_length is not None and len(generated_ids) > input_length:
            generated_only = generated_ids[input_length:]
            response = processor.decode(generated_only, skip_special_tokens=True)
            if not response.strip() and full_decoded.strip():
                response = full_decoded
        else:
            response = full_decoded

        prompt_start_50 = prompt[:50].strip().lower()
        response_start_50 = response[:50].strip().lower()
        if response_start_50 == prompt_start_50:
            prompt_len = len(prompt)
            if len(response) > prompt_len:
                for i in range(prompt_len, min(len(response), prompt_len + 200)):
                    if response[i] in ['\n', '.', '!', '?']:
                        response = response[i+1:].strip()
                        break
                else:
                    response = response[prompt_len:].strip()
            else:
                response = ""

        if not response or len(response.strip()) < 10:
            if input_length is not None and len(generated_ids) > input_length:
                response = processor.decode(generated_ids[input_length:], skip_special_tokens=True)

        response = response.strip()

        if model_type in ("llava", "llava_next", "palo") and response:
            for prefix in ("ASSISTANT:", "assistant:"):
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    break

        binary, confidence_list = parse_natural_language_response(response)
        return binary, confidence_list, response

    except Exception as e:
        print(f"  [Error] {image_path}: {str(e)}", file=sys.stderr)
        return [0] * 7, [0.0] * 7, f"[ERROR: {str(e)}]"


# ---------------------------------------------------------------------------
# Per-model CSV save
# ---------------------------------------------------------------------------

def save_model_csv(short_name, images_data, pred_list, raw_list, human_labels, timestamp):
    """Save predictions for one model to hinglish_multimodal/{short_name}/{short_name}_hinglish_predictions_{timestamp}.csv"""
    out_dir = OUTPUT_BASE_DIR / short_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{short_name}_hinglish_predictions_{timestamp}.csv"

    rows = []
    for img_data, pred, raw in zip(images_data, pred_list, raw_list):
        rows.append({
            "image_id": img_data["filename"],
            "category": img_data["category"],
            f"{short_name}_pred": pred,
            f"{short_name}_raw": raw,
            "human_label": human_labels.get(img_data["filename"], ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hinglish multimodal benchmarking pipeline")
    parser.add_argument(
        "--only_model", type=str, default=None,
        help="Short model name to run (e.g. LLaVA, PALO, BLIP). If omitted, all 10 models run sequentially.",
    )
    parser.add_argument(
        "--error_log", type=str, default=None,
        help="Error log path. Defaults to hinglish_multimodal/errors_{timestamp}.log",
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Limit number of images (for testing).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    error_log_path = args.error_log or str(OUTPUT_BASE_DIR / f"errors_{timestamp}.log")
    error_log = open(error_log_path, "w")

    print("=== HINGLISH MULTIMODAL BENCHMARKING PIPELINE ===")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_vram = sum(
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(gpu_count)
        )
        print(f"Hardware: {gpu_count}× NVIDIA GPUs ({total_vram:.1f} GB total VRAM)")
    else:
        print("Warning: CUDA not available, running on CPU")

    images = load_hinglish_images()
    if args.max_images is not None:
        images = images[:args.max_images]
        print(f"Limiting to first {len(images)} images (--max_images={args.max_images})")
    print(f"Found {len(images)} Hinglish test images\n")

    if not images:
        print("No images found. Exiting.")
        error_log.close()
        return

    human_labels = load_human_labels()
    print(f"Loaded human labels for {len(human_labels)} images\n")

    configs_to_run = MODEL_CONFIGS
    if args.only_model:
        configs_to_run = [c for c in MODEL_CONFIGS if c[1] == args.only_model]
        if not configs_to_run:
            print(
                f"Unknown model '{args.only_model}'. Available: {[c[1] for c in MODEL_CONFIGS]}",
                file=sys.stderr,
            )
            error_log.close()
            return
        print(f"Running only: {configs_to_run[0][2]}")

    total_start = time.time()

    for idx, (model_id, short_name, display_name, use_hindi, model_type) in enumerate(configs_to_run, 1):
        print(f"\n[MODEL {idx}/{len(configs_to_run)}] {display_name} ({model_id})")
        print("Loading model...")

        pred_list = []
        raw_list = []
        model = None

        # --- Model loading (if fails, mark all as INFERENCE_ERROR) ---
        try:
            model, processor = load_model(model_id, model_type)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_used = sum(
                    torch.cuda.memory_allocated(i) / (1024 ** 3)
                    for i in range(torch.cuda.device_count())
                )
                print(f"Loaded ({vram_used:.1f} GB VRAM used)")
            else:
                print("Loaded")
        except Exception as e:
            print(f"  [LOAD FAILED] {e}", file=sys.stderr)
            error_log.write(f"{display_name},MODEL_LOAD_ERROR,{str(e)}\n")
            save_model_csv(short_name, images,
                           ["[INFERENCE_ERROR]"] * len(images),
                           ["[INFERENCE_ERROR]"] * len(images),
                           human_labels, timestamp)
            continue

        # --- Inference (runs even if some images fail; partial results preserved) ---
        try:
            # Select prompt
            if model_type == "blip2" and short_name == "mBLIP":
                prompt = MBLIP_PROMPT
            elif model_type in ("blip2", "blip"):
                prompt = BLIP2_PROMPT
            elif model_type == "llava":
                prompt = LLaVA_PROMPT
            elif model_type == "llava_next":
                prompt = LLAVA_NEXT_PROMPT
            elif model_type == "minicpm_v":
                prompt = MINICPM_V_QUESTION
            elif model_type == "idefics":
                prompt = IDEFICS_QUESTION
            elif model_type == "instructblip":
                prompt = INSTRUCTBLIP_PROMPT
            elif model_type == "palo":
                prompt = LLaVA_PROMPT
            elif model_type == "chitrarth":
                prompt = CHITRARTH_QUESTION
            else:
                prompt = BLIP2_PROMPT

            print(f"Running inference on {len(images)} images...")
            start_time = time.time()

            for img_data in tqdm(images, desc=f"  {short_name}"):
                binary, scores, response = predict_symptoms(
                    model, processor, img_data["image_path"], prompt, model_type=model_type
                )
                if response.startswith("[ERROR:"):
                    error_log.write(f"{display_name},{img_data['filename']},{response}\n")
                pred_str = "[" + ",".join(map(str, binary)) + "]"
                raw_str = response[:500] if len(response) <= 500 else response[:500] + "...[truncated]"
                pred_list.append(pred_str)
                raw_list.append(raw_str)

            elapsed = time.time() - start_time
            avg = elapsed / len(images) if images else 0
            print(f"  Done in {elapsed:.0f}s ({avg:.1f}s/image)")

        except Exception as e:
            print(f"  [INFERENCE ERROR after {len(pred_list)} images] {e}", file=sys.stderr)
            error_log.write(f"{display_name},INFERENCE_ERROR,{str(e)}\n")
            # Pad remaining images with error markers so CSV has correct length
            remaining = len(images) - len(pred_list)
            pred_list.extend(["[INFERENCE_ERROR]"] * remaining)
            raw_list.extend([f"[ERROR: {str(e)[:200]}]"] * remaining)

        finally:
            # Free GPU memory; errors here must NOT overwrite pred_list
            try:
                if model is not None:
                    clear_model(model)
            except Exception as clear_err:
                print(f"  [clear_model warning] {clear_err}", file=sys.stderr)

        # Always save CSV (preserves partial results even after errors)
        save_model_csv(short_name, images, pred_list, raw_list, human_labels, timestamp)

    total_elapsed = time.time() - total_start
    hours, minutes = int(total_elapsed // 3600), int((total_elapsed % 3600) // 60)
    print(f"\n=== DONE ===")
    print(f"Total time: {hours}h {minutes}m")
    print(f"Output dir: {OUTPUT_BASE_DIR}")
    error_log.close()


if __name__ == "__main__":
    main()
