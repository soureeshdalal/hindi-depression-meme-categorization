"""
Pooled hidden-state features for MAMA-style multilabel heads on each VLM in run_multimodal.MODEL_CONFIGS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _pick_hidden_from_outputs(out):
    """Best-effort hidden-state extraction across HF VLM output schemas."""
    # Common direct attributes
    for attr in ("last_hidden_state", "encoder_last_hidden_state", "decoder_last_hidden_state"):
        h = getattr(out, attr, None)
        if h is not None and torch.is_tensor(h):
            return h

    # Common hidden state lists
    for attr in ("hidden_states", "decoder_hidden_states", "encoder_hidden_states"):
        hs = getattr(out, attr, None)
        if hs and isinstance(hs, (list, tuple)) and torch.is_tensor(hs[-1]):
            return hs[-1]

    # Nested outputs used by BLIP-2 / InstructBLIP
    for sub in ("language_model_outputs", "qformer_outputs", "vision_outputs"):
        sub_out = getattr(out, sub, None)
        if sub_out is None:
            continue
        h = _pick_hidden_from_outputs(sub_out)
        if h is not None:
            return h
    return None


def _text_embed_fallback(model, input_ids, attention_mask):
    """Last-resort pooled feature so training can continue when model outputs vary."""
    emb = model.get_input_embeddings()(input_ids).float()
    return _mean_pool_hidden(emb, attention_mask)


def _to_device_structure(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device_structure(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device_structure(v, device) for v in x)
    return x


def _mean_pool_hidden(h: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    h = h.float()
    if attention_mask is None:
        return h.mean(dim=1)
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def pooled_llava_family(model, processor, pil_image: Image.Image, prompt: str, device) -> torch.Tensor:
    inner = getattr(model, "model", model)
    batch = processor(text=prompt, images=pil_image, return_tensors="pt")
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    attn = batch["attention_mask"]
    fwd_kw = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "pixel_values", "position_ids")}
    out = inner(**fwd_kw, return_dict=True, use_cache=False)
    return _mean_pool_hidden(out.last_hidden_state, attn)


def pooled_blip2_like(model, processor, pil_image: Image.Image, prompt: str, device) -> torch.Tensor | None:
    batch = processor(text=prompt, images=pil_image, return_tensors="pt")
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    attn = batch.get("attention_mask")
    try:
        out = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
    except Exception:
        # Some BLIP-family models require explicit subset of kwargs
        try:
            out = model(
                pixel_values=batch.get("pixel_values"),
                input_ids=batch.get("input_ids"),
                attention_mask=attn,
                return_dict=True,
                output_hidden_states=True,
            )
        except Exception:
            if batch.get("input_ids") is not None and attn is not None:
                return _text_embed_fallback(model, batch["input_ids"], attn)
            return None
    h = _pick_hidden_from_outputs(out)
    if h is None:
        if batch.get("input_ids") is not None and attn is not None:
            return _text_embed_fallback(model, batch["input_ids"], attn)
        return None
    return _mean_pool_hidden(h, attn)


def pooled_blip_original(model, processor, pil_image: Image.Image, prompt: str, device) -> torch.Tensor | None:
    try:
        batch = processor(text=prompt, images=pil_image, return_tensors="pt")
    except Exception:
        return None
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    attn = batch.get("attention_mask")
    try:
        out = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
    except Exception:
        if batch.get("input_ids") is not None and attn is not None:
            return _text_embed_fallback(model, batch["input_ids"], attn)
        return None
    h = _pick_hidden_from_outputs(out)
    if h is not None:
        return _mean_pool_hidden(h, attn)
    if batch.get("input_ids") is not None and attn is not None:
        return _text_embed_fallback(model, batch["input_ids"], attn)
    return None


def pooled_palo(model, processor, pil_image: Image.Image, prompt: str, device, dtype) -> torch.Tensor | None:
    try:
        palo_repo = Path(__file__).resolve().parent / "palo_repo"
        if str(palo_repo) not in sys.path:
            sys.path.insert(0, str(palo_repo))
        from palo.mm_utils import tokenizer_image_token, process_images
        from palo.constants import IMAGE_TOKEN_INDEX

        tokenizer = processor["tokenizer"]
        image_processor = processor["image_processor"]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        pad_id = tokenizer.pad_token_id
        attn = (input_ids != pad_id).long() if pad_id is not None else torch.ones_like(input_ids)
        image_tensor = process_images([pil_image], image_processor, model.config).to(device=device, dtype=dtype)
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            images=image_tensor,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        h = _pick_hidden_from_outputs(out)
        if h is None:
            return None
        return _mean_pool_hidden(h, attn)
    except Exception:
        return None


def pooled_idefics(model, processor, pil_image: Image.Image, prompt: str, device) -> torch.Tensor | None:
    try:
        prompts = ["User:", pil_image, "\n" + prompt.strip()]
        inputs = processor(text=[prompts], return_tensors="pt", padding=True)
        # Prefer caller device (matches other VLMs); model.device can be wrong with device_map
        inputs = _to_device_structure(inputs, device)
        out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        h = _pick_hidden_from_outputs(out)
        if h is None:
            return None
        attn = inputs.get("attention_mask")
        if attn is None and h is not None:
            attn = torch.ones(h.shape[:2], device=h.device, dtype=torch.long)
        return _mean_pool_hidden(h, attn)
    except Exception:
        return None


def pooled_minicpm(model, tokenizer, pil_image: Image.Image, prompt: str, device) -> torch.Tensor | None:
    """
    MiniCPM-V: try chat() extra returns; else vision tower (vpm) + mean pool if available.
    """
    msgs = [{"role": "user", "content": prompt}]
    try:
        out = model.chat(
            image=pil_image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=False,
            output_hidden_states=True,
        )
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            hs = out[2]
            if torch.is_tensor(hs):
                if hs.dim() == 3:
                    return hs.mean(1)
                return hs
    except TypeError:
        pass
    except Exception:
        pass

    vpm = getattr(model, "vpm", None) or getattr(model, "vision_model", None)
    if vpm is None:
        return None
    try:
        ip = getattr(tokenizer, "image_processor", None)
        if ip is not None:
            pv = ip(images=pil_image, return_tensors="pt")["pixel_values"].to(device)
        else:
            import torchvision.transforms as T

            t = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize([0.5], [0.5])])
            pv = t(pil_image).unsqueeze(0).to(device)
        vo = vpm(pv)
        if torch.is_tensor(vo):
            if vo.dim() == 4:
                return vo.flatten(2).mean(-1)
            if vo.dim() == 3:
                return vo.mean(1)
            return vo.flatten(1)
    except Exception:
        return None
    return None


def pooled_chitrarth(model, processor: dict[str, Any], pil_image: Image.Image, prompt: str, device, dtype) -> torch.Tensor | None:
    try:
        tokenizer = processor["tokenizer"]
        ip = processor["image_processor"]
        if hasattr(ip, "preprocess"):
            pv = ip.preprocess(pil_image, return_tensors="pt")
        else:
            pv = ip(pil_image, return_tensors="pt")
        pixel_values = pv["pixel_values"].to(device=device, dtype=dtype)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        for kw in ("images", "image", "pixel_values"):
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    **{kw: pixel_values},
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                break
            except TypeError:
                out = None
        else:
            return None
        h = _pick_hidden_from_outputs(out)
        if h is not None:
            return _mean_pool_hidden(h, attn)
        return None
    except Exception:
        return None


def extract_pooled(
    model,
    processor: Any,
    model_type: str,
    pil_image: Image.Image,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if model_type in ("llava", "llava_next"):
        return pooled_llava_family(model, processor, pil_image, prompt, device)
    if model_type in ("blip2", "instructblip"):
        return pooled_blip2_like(model, processor, pil_image, prompt, device)
    if model_type == "blip":
        return pooled_blip_original(model, processor, pil_image, prompt, device)
    if model_type == "palo":
        return pooled_palo(model, processor, pil_image, prompt, device, dtype)
    if model_type == "idefics":
        return pooled_idefics(model, processor, pil_image, prompt, device)
    if model_type == "minicpm_v":
        return pooled_minicpm(model, processor, pil_image, prompt, device)
    if model_type == "chitrarth":
        return pooled_chitrarth(model, processor, pil_image, prompt, device, dtype)
    return None
