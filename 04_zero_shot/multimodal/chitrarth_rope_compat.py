"""
Compatibility shim for Chitrarth (MPT) with transformers 5.x.
Transformers 5 removed LlamaDynamicNTKScalingRotaryEmbedding and LlamaLinearScalingRotaryEmbedding.
These classes match the old HF 4.x API: (dim, max_position_embeddings, base, device, [scaling_factor]).
"""
import math
import torch
from torch import nn


def _default_inv_freq(dim: int, base: float, device) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))


class LlamaRotaryEmbedding(nn.Module):
    """Old API: (dim, max_position_embeddings, base, device)."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = _default_inv_freq(dim, base, device or "cpu")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


class LlamaLinearScalingRotaryEmbedding(nn.Module):
    """Old API: (dim, max_position_embeddings, base, scaling_factor, device)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        inv_freq = _default_inv_freq(dim, base, device or "cpu") / scaling_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(nn.Module):
    """Old API: (dim, max_position_embeddings, base, scaling_factor, device). Dynamic NTK: recompute inv_freq when seq_len > max_position_embeddings."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = max_position_embeddings
        inv_freq = _default_inv_freq(dim, base, device or "cpu")
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    def _update_inv_freq(self, seq_len: int, device: torch.device):
        if seq_len <= self.max_position_embeddings:
            return
        # Dynamic NTK: base is scaled by (factor * seq_len / max_pos - (factor - 1))^(dim/(dim-2))
        scale = (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        scale = max(scale, 1.0) ** (self.dim / (self.dim - 2))
        new_base = self.base * scale
        inv_freq = 1.0 / (
            new_base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        seq_len = int(position_ids.max().item()) + 1
        if seq_len > self.max_seq_len_cached:
            self._update_inv_freq(seq_len, x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin
