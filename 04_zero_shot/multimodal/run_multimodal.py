#!/usr/bin/env python3
"""
Multimodal benchmarking pipeline for Hindi mental health memes.
Processes meme images directly through vision-language models.
Runs 5 multimodal models on the same test set.
"""

import argparse
import gc
import json
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
from transformers.models.idefics import IdeficsForVisionText2Text
from transformers.models.llava_next import LlavaNextForConditionalGeneration, LlavaNextProcessor

# ---------------------------------------------------------------------------
# Hardware: AWS g5.12xlarge, 4× A10G GPUs (96 GB VRAM)
# ---------------------------------------------------------------------------

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

# Natural language prompts optimized for each model type
# BLIP-2 models work best with simple, direct questions
BLIP2_PROMPT = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

# mBLIP often answers "None" or "Yes, it does"; force format: numbers or None only
MBLIP_PROMPT = """This meme is about mental health. Which of these apply? 1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep. Reply with only the numbers separated by commas (e.g. 1, 5) or the word None. Nothing else."""

# LLaVA 1.5 expects "USER: <image>\n<question> ASSISTANT:" so it generates after ASSISTANT.
# The processor expands <image> to the correct number of image tokens.
LLaVA_PROMPT = """USER: <image>

Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None". Do not list all numbers.
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep
ASSISTANT:"""

LLaVA_PROMPT_HINDI = """USER: <image>

इस मीम में कौन से depression symptoms दिख रहे हैं? (1-7 में से, संक्षिप्त व्याख्या दें)
ASSISTANT:"""

# InstructBLIP uses <image> in prompt (processor expands to image tokens)
INSTRUCTBLIP_PROMPT = """<image>
Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

# MiniCPM-V: model.chat(image=..., msgs=[{role, content: question}]) — no <image> in text
MINICPM_V_QUESTION = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

# LLaVA-NeXT (1.6): same format as LLaVA 1.5 with <image>
LLAVA_NEXT_PROMPT = """USER: <image>
Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep
ASSISTANT:"""

# IDEFICS: processor(text=[ "User:", image, "Question\nAssistant:" ])
IDEFICS_QUESTION = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep
Assistant:"""

# PALO uses LLaVA-style prompt; Chitrarth eval_model takes query text only.
CHITRARTH_QUESTION = """Look at this meme. Which depression symptoms (1-7) does it show? Reply with ONLY the numbers that apply, e.g. "1, 5" or "None".
1=Feeling down 2=Lack of interest 3=Self-harm 4=Eating 5=Low self-esteem 6=Concentration 7=Sleep"""

# Model configurations: (model_id, short_name, display_name, use_hindi, model_type)
# Use --only_model to run one VLM at a time on 24GB GPU (avoids OOM).
# InstructBLIP: may hit device-side assert; Chitrarth/PALO: optional, may need fixes.
MODEL_CONFIGS = [
    ("llava-hf/llava-1.5-7b-hf", "LLaVA", "LLaVA-1.5-7B", False, "llava"),
    ("llava-hf/llava-v1.6-vicuna-7b-hf", "LLaVA_NeXT", "LLaVA-NeXT-7B", False, "llava_next"),
    ("openbmb/MiniCPM-V", "MiniCPM_V", "MiniCPM-V-3B", False, "minicpm_v"),
    ("HuggingFaceM4/idefics-9b-instruct", "IDEFICS", "IDEFICS-9B-Instruct", False, "idefics"),
    ("Salesforce/instructblip-vicuna-7b", "InstructBLIP", "InstructBLIP-Vicuna-7B", False, "instructblip"),
    ("Salesforce/blip2-flan-t5-xl", "BLIP2", "BLIP-2-Flan-T5-XL", False, "blip2"),
    ("Gregor/mblip-mt0-xl", "mBLIP", "mBLIP-mT0-XL", False, "blip2"),
    ("Salesforce/blip-image-captioning-base", "BLIP", "BLIP-Base", False, "blip"),
    ("MBZUAI/PALO-7B", "PALO", "PALO-7B", False, "palo"),
    ("krutrim-ai-labs/Chitrarth", "Chitrarth", "Chitrarth-7.5B", False, "chitrarth"),
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


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
                        images.append(
                            {
                                "image_path": str(img_file),
                                "category": category_folder.name,
                                "filename": img_file.name,
                            }
                        )
    return images


def clear_model(model):
    """Free GPU memory between models"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model Loading Functions
# ---------------------------------------------------------------------------


def load_llava_model(model_id):
    """Load LLaVA-style models (PALO, Chitrarth, LLaVA)"""
    try:
        # Try with flash attention first
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        # Fallback without flash attention
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def load_blip2_model(model_id):
    """Load BLIP-2 models (e.g. mBLIP BLOOMZ-7B). Single GPU."""
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    # mBLIP and some checkpoints have image_token_id=None; (input_ids == None) yields a
    # Python bool and breaks .unsqueeze() in get_placeholder_mask. Use -1 so the
    # comparison is a tensor (no positions match). Do not set image_token_index or the
    # model would slice input_ids by num_query_tokens and break our short prompts.
    if getattr(model.config, "image_token_id", None) is None:
        model.config.image_token_id = -1
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor


def load_blip_model(model_id):
    """Load BLIP-style models (legacy BLIP, not BLIP-2). Uses use_safetensors=True to avoid PyTorch version issues."""
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        use_safetensors=True,  # Required to avoid PyTorch 2.6+ requirement
    )
    processor = BlipProcessor.from_pretrained(model_id)
    return model, processor


def load_palo_model(model_id):
    """Load PALO (polyglot LLaVA-style). Use PALO tokenizer + vision image processor so input_ids use IMAGE_TOKEN_INDEX (-200)."""
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
    # Ensure vision tower is loaded (PALO uses delay_load by default)
    vt = model.get_vision_tower()
    if vt is not None and getattr(vt, "is_loaded", False) is False:
        vt.load_model()
        vt.to(device=next(model.parameters()).device, dtype=model.dtype)
    # Use same image processor as PALO's vision tower (from config)
    vision_tower_name = getattr(model.config, "mm_vision_tower", "openai/clip-vit-large-patch14-336")
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    processor = {"tokenizer": tokenizer, "image_processor": image_processor}
    return model, processor


def load_chitrarth_model(model_id):
    """Load Chitrarth (Indic VLM). Requires chitrarth package. Compatible with transformers 5.x via RoPE compat shim."""
    # Patch transformers 5.x: Chitrarth's MPT expects LlamaDynamicNTKScalingRotaryEmbedding / LlamaLinearScalingRotaryEmbedding (removed in HF 5)
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
            "Chitrarth requires the chitrarth package. Install with: pip install git+https://github.com/ola-krutrim/Chitrarth.git"
        ) from e
    # Transformers 5.x expects all_tied_weights_keys; set it on every new instance (before load_pretrained_model)
    _orig_chitrarth_init = ChitrarthForCausalLM.__init__
    def _chitrarth_init_with_tied_keys(self, config):
        _orig_chitrarth_init(self, config)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
    ChitrarthForCausalLM.__init__ = _chitrarth_init_with_tied_keys
    # Transformers 5.x calls tie_weights(missing_keys=..., recompute_mapping=False); MPT's tie_weights() takes no args
    _orig_mpt_tie_weights = MPTForCausalLM.tie_weights
    def _mpt_tie_weights_compat(self, missing_keys=None, recompute_mapping=True, **kwargs):
        _orig_mpt_tie_weights(self)
    MPTForCausalLM.tie_weights = _mpt_tie_weights_compat
    # Transformers 5.x: PreTrainedModel no longer inherits GenerationMixin; Chitrarth needs generate()
    from transformers.generation import GenerationMixin
    if GenerationMixin not in ChitrarthForCausalLM.__mro__:
        ChitrarthForCausalLM.__bases__ = (GenerationMixin,) + ChitrarthForCausalLM.__bases__
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id, model_base=None, model_name="chitrarth"
    )
    # GenerationMixin / transformers 5 expect config.num_hidden_layers; Chitrarth uses n_layers
    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = getattr(model.config, "n_layers", 32)
    # MPT expects past_key_values as list of (k,v) per layer; transformers 5 passes DynamicCache
    _orig_forward = model.forward
    def _forward_with_cache_compat(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        prefix_mask=None,
        sequence_id=None,
        labels=None,
        return_dict=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
        inputs_embeds=None,
        images=None,
        **kwargs,
    ):
        if past_key_values is not None and hasattr(past_key_values, "layers"):
            n_layers = getattr(self.config, "n_layers", len(past_key_values.layers))
            past_key_values = [
                (past_key_values.layers[i].keys, past_key_values.layers[i].values)
                for i in range(min(n_layers, len(past_key_values.layers)))
            ]
        # Chitrarth.forward does not accept inputs_embeds (computes internally)
        kwargs = {k: v for k, v in kwargs.items() if k != "inputs_embeds"}
        return _orig_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            labels=labels,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            images=images,
            **kwargs,
        )
    model.forward = types.MethodType(_forward_with_cache_compat, model)
    # Chitrarth eval_model does not pass attention_mask to generate(); prepare_inputs_for_generation expects it
    _orig_prepare = model.prepare_inputs_for_generation
    def _prepare_with_attention_mask(input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if kwargs.get("attention_mask") is None and input_ids is not None:
            kwargs["attention_mask"] = torch.ones(
                input_ids.shape[:2], dtype=torch.bool, device=input_ids.device
            )
        return _orig_prepare(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
    model.prepare_inputs_for_generation = _prepare_with_attention_mask
    # Return (model, processor) where processor is a dict with tokenizer and image_processor for predict_symptoms
    processor = {"tokenizer": tokenizer, "image_processor": image_processor, "context_len": context_len}
    return model, processor


def load_instructblip_model(model_id):
    """Load InstructBLIP (instruction-tuned BLIP-2). Uses float16 to save VRAM when run after LLaVA."""
    processor = InstructBlipProcessor.from_pretrained(model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor


def load_llava_next_model(model_id):
    """Load LLaVA-NeXT (1.6) — same pattern as LLaVA 1.5."""
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    processor = LlavaNextProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def load_minicpm_v_model(model_id):
    """Load MiniCPM-V (chat API: model.chat(image, msgs, tokenizer)). Returns (model, tokenizer)."""
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def load_idefics_model(model_id):
    """Load IDEFICS (image + text list → generate)."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = IdeficsForVisionText2Text.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor


def load_model(model_id, model_type):
    """Load model based on type"""
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
# Inference Function
# ---------------------------------------------------------------------------


def parse_natural_language_response(text):
    """
    Parse natural language response to extract depression symptoms.
    Returns: (binary_list, confidence_list)
    """
    # Normalize SentencePiece/Unicode space so "▁ 1 , 5" parses as "1, 5"
    text = (text or "").replace("\u2581", " ")  # SentencePiece space -> space
    text_lower = text.lower()
    binary = [0] * 7
    confidence = [0.0] * 7
    
    # Symptom keywords and patterns for each symptom
    symptom_patterns = [
        # Symptom 1: Feeling Down
        {
            'keywords': ['sad', 'depressed', 'hopeless', 'down', 'depression', 'उदास', 'निराश', 'निराशाजनक'],
            'strong': ['depressed', 'hopeless', 'suicidal thoughts', 'want to die', 'निराश'],
            'weak': ['sad', 'down', 'उदास']
        },
        # Symptom 2: Lack of Interest
        {
            'keywords': ['interest', 'pleasure', 'enjoy', 'motivation', 'रुचि', 'मज़ा', 'उत्साह'],
            'strong': ['no interest', 'lack of interest', 'no pleasure', 'रुचि की कमी'],
            'weak': ['interest', 'enjoy']
        },
        # Symptom 3: Self-Harm
        {
            'keywords': ['self-harm', 'suicide', 'kill myself', 'hurt myself', 'आत्म-हानि', 'आत्महत्या'],
            'strong': ['suicide', 'self-harm', 'kill myself', 'hurt myself', 'आत्महत्या'],
            'weak': ['self-harm']
        },
        # Symptom 4: Eating Disorder
        {
            'keywords': ['eating', 'appetite', 'food', 'hunger', 'overeating', 'not eating', 'खाना', 'भूख'],
            'strong': ['eating disorder', 'no appetite', 'overeating', 'not eating', 'खाने का विकार'],
            'weak': ['eating', 'food', 'appetite']
        },
        # Symptom 5: Low Self-Esteem
        {
            'keywords': ['self-esteem', 'failure', 'worthless', 'useless', 'bad about myself', 'आत्म-सम्मान', 'असफल'],
            'strong': ['low self-esteem', 'worthless', 'failure', 'useless', 'कम आत्म-सम्मान'],
            'weak': ['self-esteem', 'failure']
        },
        # Symptom 6: Concentration Problem
        {
            'keywords': ['concentration', 'focus', 'attention', 'concentrate', 'distracted', 'एकाग्रता', 'ध्यान'],
            'strong': ['concentration problem', 'can\'t focus', 'trouble concentrating', 'एकाग्रता की समस्या'],
            'weak': ['concentration', 'focus']
        },
        # Symptom 7: Sleeping Disorder
        {
            'keywords': ['sleep', 'insomnia', 'sleeping', 'tired', 'exhausted', 'नींद', 'थकान'],
            'strong': ['sleep problem', 'insomnia', 'can\'t sleep', 'sleeping too much', 'नींद की समस्या'],
            'weak': ['sleep', 'tired']
        },
    ]
    
    # Check for each symptom
    for i, patterns in enumerate(symptom_patterns):
        found = False
        conf = 0.0
        
        # Check for strong indicators (high confidence)
        for strong_term in patterns['strong']:
            if strong_term in text_lower:
                found = True
                conf = max(conf, 0.8)
                break
        
        # Check for weak indicators (medium confidence)
        if not found:
            for weak_term in patterns['weak']:
                if weak_term in text_lower:
                    found = True
                    conf = max(conf, 0.5)
        
        # Check for symptom numbers (1-7)
        if f"symptom {i+1}" in text_lower or f"symptom {i+1}" in text_lower:
            found = True
            conf = max(conf, 0.7)
        
        # Check for explicit mentions like "symptom 1", "number 1", etc.
        if f" {i+1}." in text or f"#{i+1}" in text or f"number {i+1}" in text_lower:
            found = True
            conf = max(conf, 0.7)
        
        if found:
            binary[i] = 1
            confidence[i] = conf
    
    # If we found symptoms via numbers, boost confidence
    # Look for patterns like "symptoms: 1, 3, 5" or "symptoms 1 and 3"
    number_pattern = re.search(r'symptom[s]?[:\s]+([0-7,\s]+)', text_lower)
    if number_pattern:
        numbers = re.findall(r'\d+', number_pattern.group(1))
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 7:
                binary[num - 1] = 1
                confidence[num - 1] = max(confidence[num - 1], 0.9)

    # LLaVA often replies with "1, 2, 3, 4, 5, 6, 7" or "1, 3, 5" - parse comma-separated 1-7
    first_part = text_lower[:80].strip()
    numbers_in_text = re.findall(r'\b([1-7])\b', first_part)
    if numbers_in_text:
        # Only trust if the start looks like a symptom list (only digits 1-7, commas, spaces)
        cleaned = re.sub(r'[\s,]', '', first_part)
        if cleaned and all(c in '1234567' for c in cleaned):
            for num_str in numbers_in_text:
                num = int(num_str)
                binary[num - 1] = 1
                confidence[num - 1] = max(confidence[num - 1], 0.85)
    
    # Normalize confidence scores (if any found, ensure they're reasonable)
    if sum(binary) > 0:
        # Boost confidence slightly if multiple symptoms found
        total_symptoms = sum(binary)
        for i in range(7):
            if binary[i] == 1:
                confidence[i] = min(confidence[i] + 0.1 * (total_symptoms - 1), 1.0)
    
    return binary, confidence


def predict_symptoms(model, processor, image_path, prompt, max_new_tokens=256, model_type=None):
    """
    Returns: (binary_predictions, raw_scores, raw_response)
    - binary_predictions: list of 7 binary values [0,1,0,1,0,0,0]
    - raw_scores: list of 7 confidence scores [0.82, 0.15, 0.03, ...]
    - raw_response: the full text response from model
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # ---- MiniCPM-V: custom chat API (image + msgs, returns text) ----
        if model_type == "minicpm_v":
            msgs = [{"role": "user", "content": prompt}]
            with torch.no_grad():
                res, _, _ = model.chat(
                    image=image,
                    msgs=msgs,
                    context=None,
                    tokenizer=processor,
                    sampling=False,
                )
            response = (res or "").strip()
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        # ---- IDEFICS: processor(text=["User:", image, "Question\nAssistant:"]) then generate ----
        if model_type == "idefics":
            prompts = ["User:", image, "\n" + prompt.strip()]
            inputs = processor(text=[prompts], return_tensors="pt")
            # Move all tensors to device (pixel_values may be dict of tensors)
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
            # Strip possible "User: ... Assistant:" prefix to get model reply only
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        # ---- BLIP (original): Pure captioning model - no text prompt, just generate caption ----
        if model_type == "blip":
            # BLIP is a captioning model, not instruction-following. Generate pure caption.
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
            generated_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
            response = processor.decode(generated_ids, skip_special_tokens=True).strip()
            # Parse caption for symptom-related keywords
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response
        
        # ---- Chitrarth: custom API (tokenizer + image_processor dict, use their eval) ----
        if model_type == "chitrarth":
            try:
                from chitrarth.inference import eval_model
            except ImportError:
                return [0] * 7, [0.0] * 7, "[ERROR: chitrarth package not installed]"
            tokenizer = processor["tokenizer"]
            image_processor = processor["image_processor"]
            context_len = processor["context_len"]
            # Cap tokens for speed: we only need "1, 5" or "None"; 64 is plenty (256 was ~4x slower)
            chitrarth_max = min(max_new_tokens, 64)
            response = eval_model(
                tokenizer, model, image_processor, context_len,
                prompt.strip(), image_file=image_path,
                max_new_tokens=chitrarth_max, temperature=0.0,
            )
            response = (response or "").strip()
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        # ---- PALO: use PALO tokenizer + vision image processor so input_ids use IMAGE_TOKEN_INDEX (-200) ----
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
            # Prompt already contains "<image>" (LLaVA-style). Build input_ids with -200 at image position.
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            image_tensor = process_images([image], image_processor, model.config).to(device=device, dtype=dtype)
            input_length = input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            generated_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
            # Decode only generated tokens (input_ids contain IMAGE_TOKEN_INDEX=-200 which tokenizer cannot decode)
            if input_length is not None and len(generated_ids) > input_length:
                generated_only = generated_ids[input_length:]
                response = tokenizer.decode(generated_only, skip_special_tokens=True)
            else:
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            response = response.strip()
            for prefix in ("ASSISTANT:", "assistant:"):
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    break
            binary, confidence_list = parse_natural_language_response(response)
            return binary, confidence_list, response

        # ---- LLaVA, LLaVA-NeXT, InstructBLIP, BLIP-2: processor(text=prompt, images=image) then generate ----
        if model_type in ("llava", "llava_next", "instructblip", "blip2"):
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        else:
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Store input length to slice generated tokens
        input_ids = inputs.get("input_ids", None)
        input_length = input_ids.shape[1] if input_ids is not None and len(input_ids.shape) > 1 else None

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for reproducibility
                temperature=0.0,
            )

        # Decode response - handle prompt echoing in BLIP-2 models
        # outputs is a tensor: [batch_size, seq_len] or [seq_len]
        generated_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs
        
        # Debug: Check what we actually got
        full_decoded = processor.decode(generated_ids, skip_special_tokens=True)
        
        # Try to extract only the generated tokens (skip input prompt)
        if input_length is not None and len(generated_ids) > input_length:
            # Slice to get only generated tokens
            generated_only = generated_ids[input_length:]
            response = processor.decode(generated_only, skip_special_tokens=True)
            
            # If sliced response is empty but full isn't, use full (model might not echo)
            if not response.strip() and full_decoded.strip():
                response = full_decoded
        else:
            # Fallback: decode full sequence
            response = full_decoded
        
        # Clean up response - be conservative, only remove if clearly echoed
        # For natural language responses, we want to keep the actual content
        # Only strip if the response STARTS with the prompt verbatim
        
        # Check if response starts with prompt (clear echo case)
        prompt_start_50 = prompt[:50].strip().lower()
        response_start_50 = response[:50].strip().lower()
        
        if response_start_50 == prompt_start_50:
            # Response starts with prompt - try to find where actual answer begins
            # Look for content that's clearly NOT part of the prompt
            prompt_len = len(prompt)
            
            # If response is longer than prompt, take everything after prompt
            if len(response) > prompt_len:
                # Try to find a natural break point
                # Look for newlines or sentence boundaries after prompt length
                for i in range(prompt_len, min(len(response), prompt_len + 200)):
                    if response[i] in ['\n', '.', '!', '?']:
                        # Found a sentence boundary, take from there
                        response = response[i+1:].strip()
                        break
                else:
                    # No clear boundary, just take everything after prompt
                    response = response[prompt_len:].strip()
            else:
                # Response is same length or shorter than prompt - likely all echo
                # Try to extract any meaningful content
                response = ""  # Will be caught by empty check below
        
        # If response is empty or too short, try token-based extraction
        if not response or len(response.strip()) < 10:
            if input_length is not None and len(generated_ids) > input_length:
                # Re-decode just the generated tokens
                generated_only = generated_ids[input_length:]
                response = processor.decode(generated_only, skip_special_tokens=True)
        
        # Final cleanup: remove leading/trailing whitespace
        response = response.strip()

        # For LLaVA / LLaVA-NeXT / PALO (USER/ASSISTANT format), strip any "ASSISTANT:" prefix
        if model_type in ("llava", "llava_next", "palo") and response:
            for prefix in ("ASSISTANT:", "assistant:"):
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    break
        # InstructBLIP may echo prompt; slice already handled above via input_length

        # Parse natural language response to extract symptoms
        binary, confidence_list = parse_natural_language_response(response)
        
        return binary, confidence_list, response

    except Exception as e:
        print(f"  [Error] {image_path}: {str(e)}", file=sys.stderr)
        return [0] * 7, [0.0] * 7, f"[ERROR: {str(e)}]"


# ---------------------------------------------------------------------------
# Benchmarking Function
# ---------------------------------------------------------------------------


def benchmark_model(model_name, model, processor, images_data, prompt, model_type=None):
    """
    images_data: list of dicts with {image_path, category, filename}
    Returns: dict mapping filename to predictions
    """
    results = {}
    start_time = time.time()

    for img_data in tqdm(images_data, desc=f"Running {model_name}"):
        binary_pred, raw_scores, raw_response = predict_symptoms(
            model, processor, img_data["image_path"], prompt, model_type=model_type
        )

        results[img_data["filename"]] = {
            "binary": binary_pred,
            "scores": raw_scores,
            "response": raw_response,
        }

    elapsed = time.time() - start_time
    avg_time = elapsed / len(images_data) if images_data else 0
    return results, elapsed, avg_time


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Multimodal benchmarking pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to translated_categorized_memes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV base name (default: multimodal_predictions). A timestamp is always appended so every run creates a new file and nothing is overwritten.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="If set, only process this many images (for testing)",
    )
    parser.add_argument(
        "--error_log",
        type=str,
        default="multimodal_errors.log",
        help="Error log file path",
    )
    parser.add_argument(
        "--debug_responses",
        action="store_true",
        help="Save raw model responses to debug_responses.log (for first 10 images)",
    )
    parser.add_argument(
        "--only_model",
        type=str,
        default=None,
        help="If set, run only this model (short name, e.g. LLaVA or InstructBLIP). Use to avoid OOM when running multiple 7B VLMs on one GPU.",
    )
    args = parser.parse_args()
    # Always create a new CSV per run (timestamp appended) so existing files are never overwritten
    base = (args.output or "multimodal_predictions").strip()
    if base.endswith(".csv"):
        base = base[:-4]
    args.output = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Setup error logging
    error_log = open(args.error_log, "w")

    
    # Setup debug response logging if requested
    debug_log = None
    if args.debug_responses:
        debug_log = open("debug_responses.log", "w")

    print("=== MULTIMODAL BENCHMARKING PIPELINE ===")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_vram = sum(
            torch.cuda.get_device_properties(i).total_memory / (1024**3)
            for i in range(gpu_count)
        )
        print(f"Hardware: {gpu_count}× NVIDIA GPUs ({total_vram:.1f} GB total VRAM)")
    else:
        print("Warning: CUDA not available, running on CPU")

    # Collect images
    images = collect_test_images(args.data_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
        print(f"Processing first {len(images)} images (--max_images={args.max_images})")

    print(f"Found {len(images)} test images across categories\n")

    if not images:
        print("No test images found. Exiting.")
        error_log.close()
        return

    # Initialize results dictionary
    results_dict = {
        "image_id": [x["filename"] for x in images],
        "category": [x["category"] for x in images],
    }

    total_start_time = time.time()
    inference_errors = 0

    # Optionally filter to one model (avoids OOM when running multiple 7B VLMs on one GPU)
    configs_to_run = MODEL_CONFIGS
    if args.only_model:
        configs_to_run = [c for c in MODEL_CONFIGS if c[1] == args.only_model]
        if not configs_to_run:
            print(f"No model with short name '{args.only_model}'. Available: {[c[1] for c in MODEL_CONFIGS]}", file=sys.stderr)
            return
        print(f"Running only: {configs_to_run[0][2]}")

    # Run each model sequentially
    for idx, (model_id, short_name, display_name, use_hindi, model_type) in enumerate(
        configs_to_run, 1
    ):
        print(f"[MODEL {idx}/{len(configs_to_run)}] {display_name} ({model_id})")
        print("Loading model...")

        try:
            model, processor = load_model(model_id, model_type)

            # Get VRAM usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_used = sum(
                    torch.cuda.memory_allocated(i) / (1024**3)
                    for i in range(torch.cuda.device_count())
                )
                print(f"Done ({vram_used:.1f} GB VRAM)")
            else:
                print("Done")

            # Prepare model-specific prompt
            if model_type == "blip2" and short_name == "mBLIP":
                prompt = MBLIP_PROMPT
            elif model_type == "blip2":
                prompt = BLIP2_PROMPT_HINDI if use_hindi else BLIP2_PROMPT
            elif model_type == "blip":
                prompt = BLIP2_PROMPT_HINDI if use_hindi else BLIP2_PROMPT  # BLIP uses same prompt format as BLIP-2
            elif model_type == "llava":
                prompt = LLaVA_PROMPT_HINDI if use_hindi else LLaVA_PROMPT
            elif model_type == "llava_next":
                prompt = LLAVA_NEXT_PROMPT
            elif model_type == "minicpm_v":
                prompt = MINICPM_V_QUESTION
            elif model_type == "idefics":
                prompt = IDEFICS_QUESTION
            elif model_type == "instructblip":
                prompt = INSTRUCTBLIP_PROMPT
            elif model_type == "palo":
                prompt = LLaVA_PROMPT_HINDI if use_hindi else LLaVA_PROMPT
            elif model_type == "chitrarth":
                prompt = CHITRARTH_QUESTION
            else:
                prompt = BLIP2_PROMPT_HINDI if use_hindi else BLIP2_PROMPT

            # Run inference
            print(f"Running inference on {len(images)} images...")
            model_results, elapsed, avg_time = benchmark_model(
                display_name, model, processor, images, prompt, model_type=model_type
            )

            # Format results
            pred_list = []
            raw_list = []
            debug_count = 0
            for img_data in images:
                filename = img_data["filename"]
                if filename in model_results:
                    binary = model_results[filename]["binary"]
                    scores = model_results[filename]["scores"]
                    response = model_results[filename]["response"]

                    # Debug: log first few responses to see what models actually output
                    if args.debug_responses and debug_count < 10:
                        debug_log.write(f"\n{'='*80}\n")
                        debug_log.write(f"{display_name} - {filename}\n")
                        debug_log.write(f"{'='*80}\n")
                        debug_log.write(f"Response (first 500 chars):\n{response[:500]}\n")
                        debug_log.write(f"\nParsed binary: {binary}\n")
                        debug_log.write(f"Parsed scores: {scores}\n")
                        debug_log.flush()
                        debug_count += 1

                    # Check for errors
                    if response.startswith("[ERROR:"):
                        inference_errors += 1
                        error_log.write(
                            f"{display_name},{filename},{response}\n"
                        )

                    pred_str = "[" + ",".join(map(str, binary)) + "]"
                    # Store actual text response in raw column (not confidence scores)
                    # Truncate very long responses for CSV readability
                    raw_str = response[:500] if len(response) <= 500 else response[:500] + "...[truncated]"
                else:
                    pred_str = "[INFERENCE_ERROR]"
                    raw_str = "[INFERENCE_ERROR]"
                    inference_errors += 1

                pred_list.append(pred_str)
                raw_list.append(raw_str)

            results_dict[f"{short_name}_pred"] = pred_list
            results_dict[f"{short_name}_raw"] = raw_list

            print(f"Average time per image: {avg_time:.1f}s")
            print()

            # Clear model from memory
            clear_model(model)

        except Exception as e:
            print(f"  Failed to load/run model: {e}", file=sys.stderr)
            error_log.write(f"{display_name},MODEL_LOAD_ERROR,{str(e)}\n")
            results_dict[f"{short_name}_pred"] = ["[INFERENCE_ERROR]"] * len(images)
            results_dict[f"{short_name}_raw"] = ["[INFERENCE_ERROR]"] * len(images)
            inference_errors += len(images)
            print()
            continue

    # Add human_label column (empty)
    results_dict["human_label"] = [""] * len(images)

    # Save results to CSV
    print(f"Saving results to {args.output}...")
    df = pd.DataFrame(results_dict)

    # Reorder columns (only for models that were run)
    col_order = ["image_id", "category"]
    for _, short_name, _, _, _ in configs_to_run:
        col_order.extend([f"{short_name}_pred", f"{short_name}_raw"])
    col_order.append("human_label")

    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(args.output, index=False)
    print(f"Done. {len(df)} rows saved.\n")

    # Print summary
    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)

    print("=== SUMMARY ===")
    print(f"Total images: {len(images)}")
    print(f"Total time: {hours}h {minutes}m")
    print(f"Inference errors: {inference_errors} (see {args.error_log})")
    print(f"Output: {args.output}")

    error_log.close()


if __name__ == "__main__":
    main()
