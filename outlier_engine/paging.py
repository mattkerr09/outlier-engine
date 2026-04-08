"""
OutlierPagedModel v0.3 — expert paging with ternary memory optimisations.
OUTLIER-RUNTIME-002 (paging) + OUTLIER-RUNTIME-003 (memory reduction)

Memory optimisations in v0.3:
  1. Ternary experts stored as packed uint8 in CPU warm cache / local repack cache.
  2. On device: experts stored as int8; matmul via ternary_matmul_direct (never
     materialises float copies of the full weight matrix).
  3. Shared expert quantised to INT8 at load time (saves ~5 GB vs FP16).
  4. Both caches are bounded (LRU):
       device cache : max_experts_in_memory  (default 4)
       CPU warm cache: max_warm_cache         (default 16)

Supports two checkpoint formats, auto-detected at load time:

  "toy" format (test fixtures):
    Prefix: none
    Attention: layers.{i}.attn.{q,k,v,o}_proj.weight   (square, no bias)
    Norms:     layers.{i}.attn_norm.weight / ffn_norm.weight
    MoE:       layers.{i}.ffn.router_weight
               layers.{i}.ffn.shared.{gate,up,down}_proj.weight
               layers.{i}.ffn.experts.{j}.{gate,up,down}_proj.{weight,scale}

  "real" format (actual Outlier checkpoints, e.g. outlier-7b-v0):
    Prefix: base.model.  (lm_head: base.lm_head.weight)
    Attention: base.model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
               GQA: k/v proj may have fewer heads than q
    Norms:     base.model.layers.{i}.{input,post_attention}_layernorm.weight
    MoE:       base.model.layers.{i}.mlp.router.weight
               base.model.layers.{i}.mlp.shared_expert.{gate,up,down}_W
               base.model.layers.{i}.mlp.experts.{j}.{gate,up,down}_{ternary,scale}

Paging policy:
  Device (hot)    ← max_experts_in_memory  (LRU eviction to CPU)
  CPU RAM (warm)  ← max_warm_cache          (LRU eviction to disk/cold)
  Disk (cold)     ← safetensors shards, read on first access only

Target memory budget for outlier-7b-v0 on A100:
  Attention layers (FP16)           ~3.5 GB  always resident
  Shared experts (INT8)             ~5.3 GB  always resident
  Active ternary experts ×4 (int8)  ~776 MB  device cache (int8)
  Warm CPU cache ×16 (2-bit)        ~768 MB  CPU cache
  KV / router / misc overhead       ~1.0 GB
  ─────────────────────────────────────────
  TOTAL                             ~11.3 GB
"""

from __future__ import annotations

import gc
import json
import re
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    _RMSNorm,
    _Attention,
    _SwiGLU,
    _TernarySwiGLU,
    _causal_mask,
    _NoInitEmbedding,
    _NoInitLinear,
)
try:
    from .model import _RotaryEmbedding
except ImportError:
    _RotaryEmbedding = None  # will be caught if GQA path is needed


# ---------------------------------------------------------------------------
# GQA Attention (used for real checkpoint format)
# ---------------------------------------------------------------------------

class _GQAAttention(nn.Module):
    """
    Grouped-query attention with optional projection biases.

    n_kv_heads may differ from n_heads (GQA).  Each KV head is shared by
    n_heads // n_kv_heads query heads via repeat_interleave.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # repetitions per KV head
        self.head_dim   = hidden_dim // n_heads
        self.scale      = self.head_dim ** -0.5

        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = _NoInitLinear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = _NoInitLinear(hidden_dim, kv_dim,     bias=True)
        self.v_proj = _NoInitLinear(hidden_dim, kv_dim,     bias=True)
        self.o_proj = _NoInitLinear(hidden_dim, hidden_dim, bias=False)

        from .model import _RotaryEmbedding
        self.rope = _RotaryEmbedding(
            self.head_dim, base=rope_theta, max_seq_len=max_seq_len
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, H_kv, d = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H,    d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H_kv, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H_kv, d).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# INT8 SwiGLU — shared expert replacement (saves ~5 GB vs FP16)
# ---------------------------------------------------------------------------

class _Int8SwiGLU(nn.Module):
    """
    SwiGLU with INT8-quantized weights + per-tensor scale.

    Replaces the FP16 _SwiGLU shared expert after model load to cut
    shared-expert memory from ~10.5 GB to ~5.3 GB for outlier-7b-v0.

    Forward: chunked dequant matmul (never stores full FP16 weight copy).
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        I, D = intermediate_dim, hidden_dim
        self.register_buffer("gate_w", torch.zeros(I, D, dtype=torch.int8))
        self.register_buffer("gate_s", torch.ones(I, 1, dtype=torch.float16))
        self.register_buffer("up_w",   torch.zeros(I, D, dtype=torch.int8))
        self.register_buffer("up_s",   torch.ones(I, 1, dtype=torch.float16))
        self.register_buffer("down_w", torch.zeros(D, I, dtype=torch.int8))
        self.register_buffer("down_s", torch.ones(D, 1, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .quantize_utils import dequant_int8_matmul
        gate = F.silu(dequant_int8_matmul(x, self.gate_w, self.gate_s))
        up   = dequant_int8_matmul(x, self.up_w, self.up_s)
        out  = dequant_int8_matmul(gate * up, self.down_w, self.down_s)
        return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Expert weight container
# ---------------------------------------------------------------------------

class _ExpertWeights:
    """
    Lightweight container for ternary expert tensors.

    packed=False: weights are int8  {-1, 0, +1}  (native disk format)
                  ~194 MB per expert for outlier-7b-v0
    packed=True:  weights are uint8 in a compact cache format
                  either 2-bit or TQ1_0 base-3 packed.

    Device cache stores int8 (fast matmul via ternary_matmul_direct).
    CPU warm cache stores packed uint8 (memory-efficient).
    """

    __slots__ = (
        "gate_w", "gate_s",
        "up_w",   "up_s",
        "down_w", "down_s",
        "packed",
        "packed_format",
        "shapes",
    )

    def __init__(
        self,
        gate_w: torch.Tensor, gate_s: torch.Tensor,
        up_w:   torch.Tensor, up_s:   torch.Tensor,
        down_w: torch.Tensor, down_s: torch.Tensor,
        packed: bool = False,
        packed_format: str = "raw",
        shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> None:
        self.gate_w = gate_w;  self.gate_s = gate_s
        self.up_w   = up_w;    self.up_s   = up_s
        self.down_w = down_w;  self.down_s = down_s
        self.packed = packed
        self.packed_format = packed_format
        self.shapes = shapes or {}

    def to(self, device: torch.device) -> "_ExpertWeights":
        return _ExpertWeights(
            self.gate_w.to(device), self.gate_s.to(device),
            self.up_w.to(device),   self.up_s.to(device),
            self.down_w.to(device), self.down_s.to(device),
            packed=self.packed,
            packed_format=self.packed_format,
            shapes=dict(self.shapes),
        )

    def cpu(self) -> "_ExpertWeights":
        return self.to(torch.device("cpu"))

    # ------------------------------------------------------------------
    # Packing / unpacking
    # ------------------------------------------------------------------

    def pack_2bit(self) -> "_ExpertWeights":
        """Pack int8 weights → uint8 2-bit (4x RAM reduction)."""
        if self.packed:
            return self
        from .ternary_ops import pack_ternary_2bit
        return _ExpertWeights(
            pack_ternary_2bit(self.gate_w), self.gate_s,
            pack_ternary_2bit(self.up_w),   self.up_s,
            pack_ternary_2bit(self.down_w), self.down_s,
            packed=True,
            packed_format="2bit",
            shapes={
                "gate": tuple(self.gate_w.shape),
                "up": tuple(self.up_w.shape),
                "down": tuple(self.down_w.shape),
            },
        )

    def pack_tq10(self) -> "_ExpertWeights":
        """Pack int8 ternary weights → TQ1_0 uint8 (5 values per byte)."""
        if self.packed and self.packed_format == "tq10":
            return self
        return _ExpertWeights(
            pack_ternary_tq10(self.gate_w), self.gate_s,
            pack_ternary_tq10(self.up_w),   self.up_s,
            pack_ternary_tq10(self.down_w), self.down_s,
            packed=True,
            packed_format="tq10",
            shapes={
                "gate": tuple(self.gate_w.shape),
                "up": tuple(self.up_w.shape),
                "down": tuple(self.down_w.shape),
            },
        )

    def unpack_to_int8(self) -> "_ExpertWeights":
        """Unpack cached uint8 storage back to int8 ternary weights."""
        if not self.packed:
            return self
        if self.packed_format == "2bit":
            from .ternary_ops import unpack_ternary_2bit
            gate_in = self.gate_w.shape[1] * 4
            down_in = self.down_w.shape[1] * 4
            return _ExpertWeights(
                unpack_ternary_2bit(self.gate_w, gate_in), self.gate_s,
                unpack_ternary_2bit(self.up_w,   gate_in), self.up_s,
                unpack_ternary_2bit(self.down_w, down_in), self.down_s,
                packed=False,
            )
        if self.packed_format == "tq10":
            gate_shape = self.shapes.get("gate")
            up_shape = self.shapes.get("up")
            down_shape = self.shapes.get("down")
            if not gate_shape or not up_shape or not down_shape:
                raise RuntimeError("TQ1_0 packed expert missing original shapes.")
            return _ExpertWeights(
                unpack_ternary_tq10(self.gate_w, gate_shape), self.gate_s,
                unpack_ternary_tq10(self.up_w,   up_shape), self.up_s,
                unpack_ternary_tq10(self.down_w, down_shape), self.down_s,
                packed=False,
            )
        raise RuntimeError(f"Unsupported packed format: {self.packed_format}")

    # ------------------------------------------------------------------
    # Forward compute
    # ------------------------------------------------------------------

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ternary SwiGLU: silu(gate(x)) * up(x) → down.

        Dispatches to ternary_matmul_packed (if 2-bit stored) or
        ternary_matmul_direct (if int8 stored).  Both paths are chunked
        and never materialise a full float copy of the weight matrix.

        x: (batch, hidden_dim)   any float dtype
        Returns: (batch, hidden_dim)   float32
        """
        if self.packed:
            unpacked = self.unpack_to_int8()
            return unpacked.run(x)
        from .ternary_ops import ternary_matmul_direct
        gate = F.silu(ternary_matmul_direct(x, self.gate_w, self.gate_s))
        up   = ternary_matmul_direct(x, self.up_w, self.up_s)
        return ternary_matmul_direct(gate * up, self.down_w, self.down_s)


def pack_ternary_tq10(tensor: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1,0,+1} values into base-3 TQ1_0 bytes (5 values / byte)."""
    flat = tensor.reshape(-1).to(torch.int16)
    mapped = flat + 1
    pad_len = (-mapped.numel()) % 5
    if pad_len:
        mapped = torch.cat([mapped, torch.ones(pad_len, dtype=torch.int16)])
    groups = mapped.view(-1, 5)
    multipliers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=groups.device)
    packed = (groups * multipliers).sum(dim=1).to(torch.uint8)
    return packed


def unpack_ternary_tq10(packed: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Unpack TQ1_0 uint8 bytes back into an int8 ternary tensor."""
    total = 1
    for dim in shape:
        total *= dim
    values = packed.reshape(-1).to(torch.int16)
    digits = [((values // divisor) % 3) for divisor in (1, 3, 9, 27, 81)]
    unpacked = torch.stack(digits, dim=1).reshape(-1)[:total] - 1
    return unpacked.reshape(shape).to(torch.int8)


def _default_packed_dir() -> Path:
    return Path.home() / "outlier-engine" / "packed_experts"


def repack_ternary_experts(
    model_ref: str,
    *,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, float]:
    """Repack legacy MoE expert tensors into local TQ1_0 binary files."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = Path(
        snapshot_download(
            repo_id=model_ref,
            token=token,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    packed_dir = Path(output_dir).expanduser() if output_dir else _default_packed_dir()
    packed_dir.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Dict[str, object]] = {}
    total_original = 0
    total_packed = 0
    ternary_tensors = 0
    scale_tensors = 0

    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "experts." not in key or "shared_expert" in key:
                    continue
                tensor = f.get_tensor(key)
                filename = key.replace(".", "_") + ".bin"
                path = packed_dir / filename
                if key.endswith("_ternary"):
                    packed = pack_ternary_tq10(tensor)
                    np.asarray(packed.cpu().numpy(), dtype=np.uint8).tofile(path)
                    total_original += tensor.numel() * tensor.element_size()
                    total_packed += packed.numel()
                    ternary_tensors += 1
                    index[key] = {
                        "file": filename,
                        "shape": list(tensor.shape),
                        "dtype": "uint8",
                        "format": "tq10",
                        "original_bytes": tensor.numel() * tensor.element_size(),
                        "packed_bytes": packed.numel(),
                    }
                elif key.endswith("_scale"):
                    arr = np.asarray(tensor.cpu().numpy(), dtype=np.float16)
                    arr.tofile(path)
                    scale_tensors += 1
                    index[key] = {
                        "file": filename,
                        "shape": list(tensor.shape),
                        "dtype": "float16",
                        "format": "raw",
                        "packed_bytes": arr.nbytes,
                    }

    meta = {
        "model_ref": model_ref,
        "model_dir": str(model_dir),
        "format": "tq10",
        "ternary_tensors": ternary_tensors,
        "scale_tensors": scale_tensors,
        "original_mb": total_original / 1024**2,
        "packed_mb": total_packed / 1024**2,
        "compression_ratio": (total_original / total_packed) if total_packed else 0.0,
    }
    (packed_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (packed_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _run_expert(x: torch.Tensor, w: _ExpertWeights) -> torch.Tensor:
    """Run ternary expert via chunked matmul. Returns float32."""
    return w.run(x)


# ---------------------------------------------------------------------------
# Device-resident layer containers
# ---------------------------------------------------------------------------

class _PagedMoEFFN(nn.Module):
    """Holds router weight and shared expert on device; ternary experts paged."""

    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 n_experts: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.router_weight = nn.Parameter(
            torch.empty(n_experts, hidden_dim), requires_grad=False
        )
        self.shared = _Int8SwiGLU(hidden_dim, intermediate_dim)


class _PagedLayer(nn.Module):
    """Transformer block with device-resident attention and paged MoE FFN."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        max_seq_len: int,
        n_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.attn_norm = _RMSNorm(hidden_dim)
        self.ffn_norm  = _RMSNorm(hidden_dim)

        if n_kv_heads != n_heads:
            self.attn: nn.Module = _GQAAttention(
                hidden_dim, n_heads, n_kv_heads, rope_theta, max_seq_len
            )
        else:
            self.attn = _Attention(
                hidden_dim, n_heads, rope_theta, max_seq_len
            )

        if n_experts > 0:
            self.ffn: nn.Module = _PagedMoEFFN(
                hidden_dim, intermediate_dim, n_experts, top_k
            )
        else:
            self.ffn = _TernarySwiGLU(hidden_dim, intermediate_dim)


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------

def _normalize_config(raw: dict) -> dict:
    cfg = dict(raw)
    _map = {
        "hidden_size":             "hidden_dim",
        "intermediate_size":       "intermediate_dim",
        "num_hidden_layers":       "n_layers",
        "num_attention_heads":     "n_heads",
        "num_key_value_heads":     "n_kv_heads",
        "max_position_embeddings": "max_seq_len",
    }
    for src, dst in _map.items():
        if src in cfg and dst not in cfg:
            cfg[dst] = cfg[src]
    if "rope_theta" not in cfg and "rope_parameters" in cfg:
        cfg["rope_theta"] = cfg["rope_parameters"].get("rope_theta", 10000.0)
    if "outlier_num_experts" in cfg and "n_experts" not in cfg:
        cfg["n_experts"] = cfg["outlier_num_experts"]
    if "outlier_num_experts_per_tok" in cfg and "top_k" not in cfg:
        cfg["top_k"] = cfg["outlier_num_experts_per_tok"]
    cfg.setdefault("n_kv_heads", cfg.get("n_heads", 0))
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint format detection and key remapping
# ---------------------------------------------------------------------------

def _detect_format(sample_keys) -> str:
    """Return 'real' if checkpoint uses base.model.* keys, else 'toy'."""
    for k in sample_keys:
        if k.startswith("base.model.") or k.startswith("base.lm_head"):
            return "real"
    return "toy"


def _remap_real_key(key: str) -> str:
    """Translate one real-format key to the internal model naming."""
    if key.startswith("base.model."):
        key = key[len("base.model."):]
    elif key.startswith("base."):
        key = key[len("base."):]

    key = key.replace(".self_attn.", ".attn.")
    key = key.replace(".input_layernorm.", ".attn_norm.")
    key = key.replace(".post_attention_layernorm.", ".ffn_norm.")
    key = key.replace(".mlp.router.weight", ".ffn.router_weight")
    key = key.replace(".mlp.shared_expert.gate_W", ".ffn.shared.gate_proj.weight")
    key = key.replace(".mlp.shared_expert.up_W", ".ffn.shared.up_proj.weight")
    key = key.replace(".mlp.shared_expert.down_W", ".ffn.shared.down_proj.weight")
    return key


def _remap_real_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Translate real Outlier checkpoint keys to internal model key format.

    Real → internal mapping:
      base.model.X             → X
      base.X                   → X           (lm_head)
      self_attn                → attn
      input_layernorm          → attn_norm
      post_attention_layernorm → ffn_norm
      mlp.router.weight        → ffn.router_weight
      mlp.shared_expert.gate_W → ffn.shared.gate_proj.weight
      mlp.shared_expert.up_W   → ffn.shared.up_proj.weight
      mlp.shared_expert.down_W → ffn.shared.down_proj.weight
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[_remap_real_key(k)] = v
    return out


# ---------------------------------------------------------------------------
# Main paged model
# ---------------------------------------------------------------------------

class OutlierPagedModel(nn.Module):
    """
    Outlier MoE inference with on-demand expert paging and memory optimisations.

    v0.3 changes vs v0.2:
      • Shared experts quantised to INT8 at load time (~5 GB savings)
      • CPU warm cache bounded at max_warm_cache with LRU eviction
      • Experts stored as 2-bit packed uint8 in CPU cache (4x vs int8)
      • Expert device weights kept as int8 — never converted to float16/32
      • ternary_matmul_direct avoids full float materialisation during forward

    Args:
        model_path:            Local directory with config.json + *.safetensors.
        device:                Torch device string (default "mps").
        max_experts_in_memory: LRU device cache capacity (default 4).
        max_warm_cache:        LRU CPU warm cache capacity (default 16).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        max_experts_in_memory: int = 4,
        max_warm_cache: int = 16,
        packed_experts_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        model_dir = Path(model_path)
        with open(model_dir / "config.json") as f:
            cfg = _normalize_config(json.load(f))

        self.device                = torch.device(device)
        self.max_experts_in_memory = max_experts_in_memory
        self.max_warm_cache        = max_warm_cache
        self.n_layers              = cfg["n_layers"]
        self.hidden_dim            = cfg["hidden_dim"]
        self.n_heads               = cfg["n_heads"]
        self.n_kv_heads            = cfg.get("n_kv_heads", cfg["n_heads"])
        self.intermediate_dim      = cfg["intermediate_dim"]
        self.vocab_size            = cfg["vocab_size"]
        self.max_seq_len           = cfg.get("max_seq_len", 4096)
        self.rope_theta            = cfg.get("rope_theta", 10000.0)
        self.n_experts             = cfg.get("n_experts", 0)
        self.top_k                 = cfg.get("top_k", 2)

        D, I, V = self.hidden_dim, self.intermediate_dim, self.vocab_size

        self.embed_tokens = _NoInitEmbedding(V, D)
        self.layers = nn.ModuleList([
            _PagedLayer(
                D, I, self.n_heads, self.n_kv_heads,
                self.rope_theta, self.max_seq_len,
                self.n_experts, self.top_k,
            )
            for _ in range(self.n_layers)
        ])
        self.norm    = _RMSNorm(D)
        self.lm_head = _NoInitLinear(D, V, bias=False)
        # Keep always-resident floating weights in FP16 instead of silently
        # inflating to FP32 during checkpoint-backed initialization.
        self.to(dtype=torch.float16)

        # Two-tier expert cache (both bounded with LRU eviction)
        self._lru_cache: OrderedDict[Tuple[int, int], _ExpertWeights] = OrderedDict()
        self._cpu_cache: OrderedDict[Tuple[int, int], _ExpertWeights] = OrderedDict()

        # Per-generate() cache statistics
        self._cache_hits      = 0
        self._cache_misses    = 0
        self._cache_evictions = 0   # total device→CPU evictions
        self._disk_loads      = 0
        self._disk_load_s     = 0.0

        # Tensor-level shard index: raw_tensor_key → shard_path
        self._tensor_shard_index: Dict[str, str] = {}
        self._packed_index: Dict[str, Dict[str, object]] = {}
        self._packed_experts_dir = Path(packed_experts_dir).expanduser() if packed_experts_dir else None

        # Format detected during shard scan
        self._fmt: str = "toy"

        self._build_shard_index(model_dir)
        self._load_packed_index()
        warnings.warn(
            "Quantizing shared expert weights to INT8 during paged model load.",
            stacklevel=2,
        )
        self._load_non_expert_weights(model_dir)
        self.to(self.device)
        self.eval()

        # Async prefetcher (None = disabled; set via enable_async_prefetch())
        self._async_prefetcher = None

    # ------------------------------------------------------------------
    # Async prefetch integration
    # ------------------------------------------------------------------

    def enable_async_prefetch(self, max_prefetch_ahead: int = 2) -> None:
        """
        Attach an AsyncExpertPrefetcher so the forward() loop overlaps
        attention compute with expert loading.

        OUTLIER-RUNTIME-004: router-first execution — router runs on
        pre-attention hidden state to start prefetching while attention
        computes.  Prefetcher is used in place of direct load_expert()
        calls when enabled.
        """
        from .async_engine import AsyncExpertPrefetcher
        self._async_prefetcher = AsyncExpertPrefetcher(self, max_prefetch_ahead)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_shard_index(self, model_dir: Path) -> None:
        """
        Scan shard headers (no tensor data loaded) to:
          1. Detect checkpoint format.
          2. Map every expert tensor's raw key → shard path.
        """
        from safetensors import safe_open

        toy_pat  = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")
        real_pat = re.compile(r"^base\.model\.layers\.\d+\.mlp\.experts\.\d+\.")

        first_shard_keys: list = []

        for shard in sorted(model_dir.glob("*.safetensors")):
            shard_str = str(shard)
            with safe_open(shard_str, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if not first_shard_keys:
                    first_shard_keys = keys
                for key in keys:
                    if toy_pat.match(key) or real_pat.match(key):
                        self._tensor_shard_index[key] = shard_str

        self._fmt = _detect_format(first_shard_keys)

    def _load_packed_index(self) -> None:
        if self._packed_experts_dir is None:
            return
        index_path = self._packed_experts_dir / "index.json"
        if not index_path.exists():
            return
        try:
            self._packed_index = json.loads(index_path.read_text(encoding="utf-8"))
            warnings.warn(
                f"Using TQ1_0 packed experts from {self._packed_experts_dir}.",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(f"Failed to load packed expert index: {exc}")
            self._packed_index = {}

    # ------------------------------------------------------------------
    # Non-expert weight loading
    # ------------------------------------------------------------------

    def _load_non_expert_weights(self, model_dir: Path) -> None:
        """Stream all non-expert weights directly into resident modules."""
        from safetensors import safe_open
        from .quantize_utils import quantize_to_int8

        toy_expert_pat  = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")
        real_expert_pat = re.compile(
            r"^base\.model\.layers\.\d+\.mlp\.experts\.\d+\."
        )
        assigned = 0
        unexpected: list[str] = []

        for shard in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for raw_key in f.keys():
                    if toy_expert_pat.match(raw_key) or real_expert_pat.match(raw_key):
                        continue
                    key = raw_key
                    if self._fmt == "real":
                        key = _remap_real_key(raw_key)
                    elif key.startswith("model."):
                        key = key[6:]
                    tensor = f.get_tensor(raw_key)

                    if key == "embed_tokens.weight":
                        self.embed_tokens.weight.data.copy_(tensor.to(self.embed_tokens.weight.dtype))
                        assigned += 1
                        continue
                    if key == "norm.weight":
                        self.norm.weight.data.copy_(tensor.to(self.norm.weight.dtype))
                        assigned += 1
                        continue
                    if key == "lm_head.weight":
                        self.lm_head.weight.data.copy_(tensor.to(self.lm_head.weight.dtype))
                        assigned += 1
                        continue

                    layer_match = re.match(r"^layers\.(\d+)\.(.+)$", key)
                    if not layer_match:
                        unexpected.append(key)
                        continue
                    layer_idx = int(layer_match.group(1))
                    suffix = layer_match.group(2)
                    layer = self.layers[layer_idx]

                    if suffix == "attn_norm.weight":
                        layer.attn_norm.weight.data.copy_(tensor.to(layer.attn_norm.weight.dtype))
                    elif suffix == "ffn_norm.weight":
                        layer.ffn_norm.weight.data.copy_(tensor.to(layer.ffn_norm.weight.dtype))
                    elif suffix.startswith("attn."):
                        proj_name, param_name = suffix[len("attn."):].rsplit(".", 1)
                        module = getattr(layer.attn, proj_name, None)
                        target = getattr(module, param_name, None) if module is not None else None
                        if target is None:
                            unexpected.append(key)
                            continue
                        target.data.copy_(tensor.to(target.dtype))
                    elif suffix == "ffn.router_weight":
                        layer.ffn.router_weight.data.copy_(tensor.to(layer.ffn.router_weight.dtype))
                    elif suffix.startswith("ffn.shared."):
                        shared = layer.ffn.shared
                        if not isinstance(shared, _Int8SwiGLU):
                            unexpected.append(key)
                            continue
                        proj = suffix[len("ffn.shared."):]
                        if proj == "gate_proj.weight":
                            q, s = quantize_to_int8(tensor)
                            shared.gate_w.copy_(q)
                            shared.gate_s.copy_(s.to(shared.gate_s.dtype))
                        elif proj == "up_proj.weight":
                            q, s = quantize_to_int8(tensor)
                            shared.up_w.copy_(q)
                            shared.up_s.copy_(s.to(shared.up_s.dtype))
                        elif proj == "down_proj.weight":
                            q, s = quantize_to_int8(tensor)
                            shared.down_w.copy_(q)
                            shared.down_s.copy_(s.to(shared.down_s.dtype))
                        else:
                            unexpected.append(key)
                            continue
                    else:
                        unexpected.append(key)
                        continue
                    assigned += 1

        if assigned == 0:
            raise RuntimeError("No non-expert weights were assigned during paged load.")
        if unexpected:
            warnings.warn(f"Unexpected keys (first 5): {unexpected[:5]}")

    # ------------------------------------------------------------------
    # Expert paging (two-tier LRU)
    # ------------------------------------------------------------------

    def _add_to_warm_cache(
        self, key: Tuple[int, int], weights: _ExpertWeights
    ) -> None:
        """
        Add weights to CPU warm cache.

        If the cache is full, silently evict the LRU entry (it will be
        reloaded from disk on next access — cold miss, not a correctness issue).
        """
        if key in self._cpu_cache:
            self._cpu_cache.move_to_end(key)
            self._cpu_cache[key] = weights
            return
        if len(self._cpu_cache) >= self.max_warm_cache:
            self._cpu_cache.popitem(last=False)   # evict LRU from warm cache
        self._cpu_cache[key] = weights

    def load_expert(self, layer_idx: int, expert_idx: int) -> _ExpertWeights:
        """
        Return expert weights on device, paging through the two-tier cache.

        Device cache hit  → move to MRU, return.
        Device cache miss → evict LRU to CPU (pack to TQ1_0 on non-CPU);
                            load from CPU cache or disk; unpack to int8;
                            move to device.
        """
        key = (layer_idx, expert_idx)
        on_cpu = self.device.type == "cpu"

        # ── Device cache hit ──────────────────────────────────────────
        if key in self._lru_cache:
            self._lru_cache.move_to_end(key)
            self._cache_hits += 1
            return self._lru_cache[key]

        self._cache_misses += 1

        # ── Evict LRU from device cache → CPU warm cache (2-bit packed) ──
        if len(self._lru_cache) >= self.max_experts_in_memory:
            evicted_key, evicted_w = self._lru_cache.popitem(last=False)
            self._cache_evictions += 1
            # On CPU there is no device/host boundary, so avoid wasteful
            # pack/unpack churn and keep the expert in native int8 form.
            cached_w = evicted_w.cpu() if on_cpu else evicted_w.cpu().pack_2bit()
            self._add_to_warm_cache(evicted_key, cached_w)

        # ── Load from warm cache or cold disk ─────────────────────────
        if key in self._cpu_cache:
            self._cpu_cache.move_to_end(key)
            weights_cached = self._cpu_cache[key]
        else:
            # Cold: load int8 from disk, optionally pack for non-CPU warm cache
            start = time.perf_counter()
            weights_int8   = self._load_expert_from_disk(layer_idx, expert_idx)
            self._disk_loads += 1
            self._disk_load_s += time.perf_counter() - start
            weights_cached = weights_int8 if on_cpu else weights_int8.pack_tq10()
            self._add_to_warm_cache(key, weights_cached)

        # ── Move to device: unpack only when the warm cache stores packed bytes ────
        if on_cpu:
            weights_dev = weights_cached
        else:
            weights_dev = weights_cached.unpack_to_int8().to(self.device)
        self._lru_cache[key] = weights_dev
        return weights_dev

    def _load_expert_from_disk(
        self, layer_idx: int, expert_idx: int
    ) -> _ExpertWeights:
        """Load one expert's tensors from shard(s), handling cross-shard splits."""
        packed = self._load_expert_from_packed(layer_idx, expert_idx)
        if packed is not None:
            return packed
        from safetensors import safe_open

        if self._fmt == "real":
            prefix = (
                f"base.model.layers.{layer_idx}"
                f".mlp.experts.{expert_idx}"
            )
            names = {
                "gate_w": f"{prefix}.gate_ternary",
                "gate_s": f"{prefix}.gate_scale",
                "up_w":   f"{prefix}.up_ternary",
                "up_s":   f"{prefix}.up_scale",
                "down_w": f"{prefix}.down_ternary",
                "down_s": f"{prefix}.down_scale",
            }
        else:
            prefix = (
                f"layers.{layer_idx}"
                f".ffn.experts.{expert_idx}"
            )
            names = {
                "gate_w": f"{prefix}.gate_proj.weight",
                "gate_s": f"{prefix}.gate_proj.scale",
                "up_w":   f"{prefix}.up_proj.weight",
                "up_s":   f"{prefix}.up_proj.scale",
                "down_w": f"{prefix}.down_proj.weight",
                "down_s": f"{prefix}.down_proj.scale",
            }

        missing = [v for v in names.values() if v not in self._tensor_shard_index]
        if missing:
            raise KeyError(
                f"Expert ({layer_idx}, {expert_idx}) tensors not found in "
                f"shard index: {missing[:3]}.  "
                f"n_experts={self.n_experts}, fmt={self._fmt!r}"
            )

        shard_to_keys: Dict[str, Dict[str, str]] = {}
        for slot, raw_key in names.items():
            shard = self._tensor_shard_index[raw_key]
            shard_to_keys.setdefault(shard, {})[slot] = raw_key

        tensors: Dict[str, torch.Tensor] = {}
        for shard, slot_map in shard_to_keys.items():
            with safe_open(shard, framework="pt", device="cpu") as f:
                for slot, raw_key in slot_map.items():
                    tensors[slot] = f.get_tensor(raw_key)

        return _ExpertWeights(
            tensors["gate_w"], tensors["gate_s"],
            tensors["up_w"],   tensors["up_s"],
            tensors["down_w"], tensors["down_s"],
            packed=False,
        )

    def _load_expert_from_packed(
        self, layer_idx: int, expert_idx: int
    ) -> Optional[_ExpertWeights]:
        if not self._packed_index:
            return None
        prefix = f"base.model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        projs = {}
        for proj in ("gate", "up", "down"):
            ternary_key = f"{prefix}.{proj}_ternary"
            scale_key = f"{prefix}.{proj}_scale"
            ternary_info = self._packed_index.get(ternary_key)
            scale_info = self._packed_index.get(scale_key)
            if ternary_info is None or scale_info is None:
                return None
            ternary_path = self._packed_experts_dir / str(ternary_info["file"])
            scale_path = self._packed_experts_dir / str(scale_info["file"])
            packed = torch.from_numpy(np.fromfile(ternary_path, dtype=np.uint8).copy())
            scale = torch.from_numpy(np.fromfile(scale_path, dtype=np.float16).copy()).reshape(scale_info["shape"])
            projs[proj] = (unpack_ternary_tq10(packed, tuple(ternary_info["shape"])), scale)

        return _ExpertWeights(
            projs["gate"][0], projs["gate"][1],
            projs["up"][0],   projs["up"][1],
            projs["down"][0], projs["down"][1],
            packed=False,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] integer token ids (on self.device)
        Returns:
            logits: [B, L, vocab_size]
        """
        x = self.embed_tokens(input_ids)   # [B, L, D]
        B, L, D = x.shape

        causal_mask: Optional[torch.Tensor] = None
        if L > 1:
            causal_mask = _causal_mask(L, device=input_ids.device, dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            prefetcher = self._async_prefetcher

            # ── Router-first: async prefetch (OUTLIER-RUNTIME-004) ────────
            # Run router on pre-attention x to start expert loading while
            # attention computes.  Only when prefetcher is enabled.
            if prefetcher is not None and layer.n_experts > 0:
                ffn_pre: _PagedMoEFFN = layer.ffn   # type: ignore[assignment]
                h_pre      = layer.ffn_norm(x)
                h_pre_flat = h_pre.view(-1, D)
                logits_pre = F.linear(
                    h_pre_flat.float(), ffn_pre.router_weight.float()
                )
                _, top_idx_pre = torch.topk(
                    F.softmax(logits_pre, dim=-1), k=self.top_k, dim=-1
                )
                # Prefetch current layer's likely experts
                prefetcher.prefetch_experts_async(i, top_idx_pre)
                # Predict and prefetch next layer's experts
                if i + 1 < self.n_layers:
                    next_pred = prefetcher.predict_next_experts(
                        i, logits_pre, topk=self.top_k
                    )
                    prefetcher.prefetch_experts_async(i + 1, next_pred)

            # ── Attention (always device-resident) ────────────────────────
            # On CUDA/MPS this overlaps with the prefetch kicked off above.
            x = x + layer.attn(layer.attn_norm(x), mask=causal_mask)

            # ── FFN ───────────────────────────────────────────────────────
            h = layer.ffn_norm(x)

            if layer.n_experts > 0:
                ffn: _PagedMoEFFN = layer.ffn   # type: ignore[assignment]
                h_flat = h.view(-1, D)           # [N, D]
                N = h_flat.shape[0]

                # Router: softmax → top-k (accurate, on post-attention x)
                logits = F.linear(h_flat.float(), ffn.router_weight.float())
                probs  = F.softmax(logits, dim=-1)
                top_w, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
                top_w = top_w / top_w.sum(-1, keepdim=True)

                # Sync prefetch before accessing expert weights
                if prefetcher is not None:
                    prefetcher.sync_prefetch()

                # Shared expert (always device-resident, INT8 quantised)
                shared_out = ffn.shared(h_flat).float()   # [N, D]

                # Ternary experts (paged — int8 on device, 2-bit in CPU cache)
                expert_out = torch.zeros(
                    N, D, device=self.device, dtype=torch.float32
                )
                for k in range(self.top_k):
                    for e in range(self.n_experts):
                        mask_e = (top_idx[:, k] == e)
                        if mask_e.any():
                            if prefetcher is not None:
                                w = prefetcher.get_expert(i, e)
                            else:
                                w = self.load_expert(i, e)
                            out = _run_expert(h_flat[mask_e], w)   # float32
                            expert_out[mask_e] += top_w[mask_e, k:k+1] * out

                if prefetcher is not None:
                    prefetcher.clear_layer(i)

                x = x + (shared_out + expert_out).to(x.dtype).view(B, L, D)
            else:
                # Dense FFN — no paging needed
                x = x + layer.ffn(h)

        x = self.norm(x)
        return self.lm_head(x)   # [B, L, V]

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
        Autoregressive generation.  Prints expert cache stats on completion.

        Args:
            input_ids:      [B, prompt_len] on self.device
            max_new_tokens: tokens to generate
            temperature:    sampling temperature; 0 = greedy

        Returns:
            [B, prompt_len + new_tokens]
        """
        self.eval()
        self._cache_hits      = 0
        self._cache_misses    = 0
        self._cache_evictions = 0
        self._disk_loads      = 0
        self._disk_load_s     = 0.0

        tokens = input_ids

        for _ in range(max_new_tokens):
            logits      = self.forward(tokens)
            next_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs      = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if tokens.shape[1] >= self.max_seq_len:
                break

        total    = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        print(
            f"Expert cache: {self._cache_hits} hits / {total} lookups "
            f"({hit_rate:.1%} hit rate), "
            f"{self._cache_evictions} device evictions"
        )
        return tokens

    def cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "lookups": total,
            "hit_rate": hit_rate,
            "evictions": self._cache_evictions,
            "disk_loads": self._disk_loads,
            "disk_load_s": self._disk_load_s,
            "avg_disk_load_ms": (self._disk_load_s * 1000.0 / self._disk_loads) if self._disk_loads else 0.0,
        }
