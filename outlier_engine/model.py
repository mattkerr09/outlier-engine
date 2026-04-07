"""
Outlier Runtime v0.1 — inference engine for Outlier ternary MoE models.

Loads Outlier checkpoints from HuggingFace (or local directory) and runs
autoregressive generation. Supports both dense ternary models and ternary
MoE models (detected via outlier_num_experts in config.json).

Weight layouts expected in the safetensors checkpoint:

  Dense (non-MoE):
    embed_tokens.weight                              [V, D]
    layers.{i}.attn_norm.weight                      [D]
    layers.{i}.attn.{q,k,v,o}_proj.weight            [D, D]
    layers.{i}.ffn_norm.weight                       [D]
    layers.{i}.ffn.{gate,up,down}_proj.latent_weight [out, in]  FP32
    norm.weight                                      [D]
    lm_head.weight                                   [V, D]

  MoE:
    Same attention/norm keys as above, plus:
    layers.{i}.ffn.router_weight                                [E, D]  FP16
    layers.{i}.ffn.shared.{gate,up,down}_proj.weight            [I, D]  FP16
    layers.{i}.ffn.experts.{j}.{gate,up,down}_proj.weight       [out,in] int8
    layers.{i}.ffn.experts.{j}.{gate,up,down}_proj.scale        scalar   FP16
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute variance in FP32 to avoid overflow when FP16 values exceed ~256
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 4096) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [L, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [L, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)
        self._cached_seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, L, head_dim]"""
        seq_len = x.shape[2]
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len]
        sin = self.sin_cached[:, :, :seq_len]
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    m = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, rope_theta: float, max_seq_len: int) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = _RotaryEmbedding(self.head_dim, base=rope_theta, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        H, d = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# FFN variants
# ---------------------------------------------------------------------------

class _TernaryLinear(nn.Module):
    """Ternary linear for dense models. Stores FP32 latent_weight, quantizes on forward."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.latent_weight = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )

    @staticmethod
    def _quantize(w: torch.Tensor) -> torch.Tensor:
        alpha = w.abs().mean()
        threshold = 0.5 * alpha
        return torch.where(w > threshold, alpha,
                           torch.where(w < -threshold, -alpha, torch.zeros_like(w)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self._quantize(self.latent_weight.float())
        return F.linear(x.float(), qw).to(x.dtype if x.is_floating_point() else torch.float32)


class _TernarySwiGLU(nn.Module):
    """Dense ternary SwiGLU FFN (non-MoE path)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.gate_proj = _TernaryLinear(hidden_dim, intermediate_dim)
        self.up_proj   = _TernaryLinear(hidden_dim, intermediate_dim)
        self.down_proj = _TernaryLinear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _SwiGLU(nn.Module):
    """Standard FP16 SwiGLU — used for the shared expert in MoE."""

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TernaryExpert(nn.Module):
    """
    Ternary MoE expert stored as int8 weights + FP16 scale per projection.

    Forward: output = (W_int8.float() * scale) @ x  (no custom kernels, v0.1)

    Buffer naming: {gate,up,down}_proj_w (int8) + {gate,up,down}_proj_s (float16).
    Loaded via _load_weights from keys:
        layers.{i}.ffn.experts.{j}.{gate,up,down}_proj.weight  → int8
        layers.{i}.ffn.experts.{j}.{gate,up,down}_proj.scale   → float16
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        I, D = intermediate_dim, hidden_dim
        self.register_buffer("gate_proj_w", torch.zeros(I, D, dtype=torch.int8))
        self.register_buffer("gate_proj_s", torch.tensor(1.0, dtype=torch.float16))
        self.register_buffer("up_proj_w",   torch.zeros(I, D, dtype=torch.int8))
        self.register_buffer("up_proj_s",   torch.tensor(1.0, dtype=torch.float16))
        self.register_buffer("down_proj_w", torch.zeros(D, I, dtype=torch.int8))
        self.register_buffer("down_proj_s", torch.tensor(1.0, dtype=torch.float16))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        # Scale buffers may have shape [] (scalar), [1], or [out, 1] (per-channel).
        # PyTorch load_state_dict raises on shape mismatch, so manually set them
        # before delegating the rest to the parent.
        for buf in ("gate_proj_s", "up_proj_s", "down_proj_s"):
            key = prefix + buf
            if key in state_dict:
                self._buffers[buf] = state_dict.pop(key).clone()
                if key in missing_keys:
                    missing_keys.remove(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        # scale may be scalar [], [1], or per-channel [out, 1] — all broadcast
        # correctly against weight shape [out, in] without any special handling.
        gw = self.gate_proj_w.float() * self.gate_proj_s.float()
        uw = self.up_proj_w.float()   * self.up_proj_s.float()
        dw = self.down_proj_w.float() * self.down_proj_s.float()
        return F.linear(F.silu(F.linear(xf, gw)) * F.linear(xf, uw), dw)


class _MoEFFN(nn.Module):
    """
    Ternary Mixture-of-Experts FFN block.

    Architecture:
        output = shared_expert(x) + weighted_sum(top_k ternary experts)

    Router: FP16 linear [hidden_dim → n_experts] → softmax → top-k.
    Shared expert: FP16 SwiGLU.
    Ternary experts: int8 weights + scale, no custom kernels (v0.1).
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 n_experts: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router_weight = nn.Parameter(
            torch.empty(n_experts, hidden_dim), requires_grad=False
        )
        self.shared  = _SwiGLU(hidden_dim, intermediate_dim)
        self.experts = nn.ModuleList(
            [_TernaryExpert(hidden_dim, intermediate_dim) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)    # [N, D]
        N = x_flat.shape[0]

        # Router: softmax → top-k
        logits = F.linear(x_flat.float(), self.router_weight.float())  # [N, E]
        probs  = F.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(probs, k=self.top_k, dim=-1)       # [N, K]
        top_w = top_w / top_w.sum(-1, keepdim=True)                     # normalize

        # Shared expert (FP16)
        shared_out = self.shared(x_flat).float()                        # [N, D]

        # Top-k ternary experts
        expert_out = torch.zeros(N, D, device=x.device, dtype=torch.float32)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    out = self.experts[e](x_flat[mask])                 # [M, D] float32
                    expert_out[mask] += top_w[mask, k:k+1] * out

        combined = (shared_out + expert_out).to(x.dtype)
        return combined.view(B, L, D)

    def get_router_decisions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return top-k expert indices per token without running the full forward.
        Shape: [B*L, top_k].  Used for router analysis / testing.
        """
        x_flat = x.view(-1, x.shape[-1])
        logits = F.linear(x_flat.float(), self.router_weight.float())
        probs  = F.softmax(logits, dim=-1)
        _, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        return top_idx


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, n_heads: int,
                 rope_theta: float, max_seq_len: int,
                 n_experts: int = 0, top_k: int = 2) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(hidden_dim)
        self.attn      = _Attention(hidden_dim, n_heads, rope_theta, max_seq_len)
        self.ffn_norm  = _RMSNorm(hidden_dim)
        if n_experts > 0:
            self.ffn: nn.Module = _MoEFFN(hidden_dim, intermediate_dim, n_experts, top_k)
        else:
            self.ffn = _TernarySwiGLU(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OutlierForCausalLM(nn.Module):
    """
    Outlier causal language model — inference runtime.

    Supports both dense ternary and ternary MoE architectures, selected
    automatically from the outlier_num_experts field in config.json.
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.n_layers        = config_dict["n_layers"]
        self.hidden_dim      = config_dict["hidden_dim"]
        self.n_heads         = config_dict["n_heads"]
        self.intermediate_dim = config_dict["intermediate_dim"]
        self.vocab_size      = config_dict["vocab_size"]
        self.max_seq_len     = config_dict.get("max_seq_len", 4096)
        self.rope_theta      = config_dict.get("rope_theta", 10000.0)
        self.n_experts       = config_dict.get("outlier_num_experts", 0)
        self.top_k           = config_dict.get("outlier_num_experts_per_tok", 2)

        D, I, V = self.hidden_dim, self.intermediate_dim, self.vocab_size

        self.embed_tokens = nn.Embedding(V, D)
        self.layers = nn.ModuleList([
            _TransformerBlock(
                D, I, self.n_heads, self.rope_theta, self.max_seq_len,
                self.n_experts, self.top_k
            )
            for _ in range(self.n_layers)
        ])
        self.norm    = _RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load_from_pretrained(cls, repo_id: str, token: Optional[str] = None) -> "OutlierForCausalLM":
        """
        Load a model from a HuggingFace repo ID or a local directory.

        If repo_id is an existing local path, it is used directly.
        Otherwise, the checkpoint is downloaded from HuggingFace Hub.
        """
        from safetensors.torch import load_file

        if os.path.isdir(repo_id):
            model_dir = Path(repo_id)
        else:
            from huggingface_hub import snapshot_download
            model_dir = Path(snapshot_download(repo_id, token=token))

        # Read config
        with open(model_dir / "config.json") as f:
            config_dict = json.load(f)

        model = cls(config_dict)

        # Load all safetensors shards
        shard_files = sorted(model_dir.glob("*.safetensors"))
        if not shard_files:
            shard_files = sorted(model_dir.glob("model*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        state: Dict[str, torch.Tensor] = {}
        for shard in shard_files:
            state.update(load_file(str(shard), device="cpu"))

        model._load_weights(state)
        model.eval()
        return model

    def _load_weights(self, state: Dict[str, torch.Tensor]) -> None:
        """
        Load a state dict into the model, handling:
          - Ternary FFN latent_weight → stored in _TernaryLinear.latent_weight
          - MoE expert int8 keys: experts.{j}.{proj}.weight / .scale
            remapped to experts.{j}.{proj}_w / {proj}_s buffers
        """
        # Remap MoE expert weight keys before calling load_state_dict
        if self.n_experts > 0:
            remapped: Dict[str, torch.Tensor] = {}
            expert_pat = re.compile(
                r"^(.*\.experts\.\d+\.)([a-z]+)_proj\.(weight|scale)$"
            )
            for k, v in state.items():
                m = expert_pat.match(k)
                if m:
                    prefix, proj, wtype = m.groups()
                    suffix = "w" if wtype == "weight" else "s"
                    remapped[f"{prefix}{proj}_proj_{suffix}"] = v
                else:
                    remapped[k] = v
            state = remapped

        # Strip common "model." prefix (some HF checkpoints add it)
        has_model_prefix = any(k.startswith("model.") for k in state)
        if has_model_prefix:
            state = {
                (k[len("model."):] if k.startswith("model.") else k): v
                for k, v in state.items()
            }

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            warnings.warn(f"Missing weight keys (first 5): {missing[:5]}")
        if unexpected:
            warnings.warn(f"Unexpected weight keys (first 5): {unexpected[:5]}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] integer token ids
        Returns:
            logits: [B, L, vocab_size]
        """
        x = self.embed_tokens(input_ids)   # [B, L, D]

        L = input_ids.shape[-1]
        mask = None
        if L > 1:
            mask = _causal_mask(L, device=input_ids.device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        return self.lm_head(x)             # [B, L, V]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """
        Autoregressive token generation.

        Args:
            input_ids:      [B, prompt_len] integer token ids
            max_new_tokens: number of new tokens to generate
            temperature:    sampling temperature; 0 = greedy argmax

        Returns:
            token ids: [B, prompt_len + max_new_tokens]
        """
        self.eval()
        tokens = input_ids

        for _ in range(max_new_tokens):
            logits = self(tokens)             # [B, L, V]
            next_logits = logits[:, -1, :]   # [B, V]

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if tokens.shape[1] >= self.max_seq_len:
                break

        return tokens
