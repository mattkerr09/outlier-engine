"""
OutlierPagedModel v0.3 — expert paging with ternary memory optimisations.
OUTLIER-RUNTIME-002 (paging) + OUTLIER-RUNTIME-003 (memory reduction)

Memory optimisations in v0.3:
  1. Ternary experts stored as packed uint8 in CPU warm cache / local repack cache.
  2. On device: experts stored as int8; matmul via ternary_matmul_direct (never
     materialises float copies of the full weight matrix).
  3. Shared expert quantised to INT8 at load time (saves ~5 GB vs FP16).
  4. Both caches are bounded (LRU):
       hot cache : max_experts_in_memory  (default 64)
       CPU warm cache: max_warm_cache         (default 256)

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
  Device-ready hot cache ← max_experts_in_memory  (LRU eviction only)
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
import os
import re
import threading
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .prefetch import ExpertPrefetcher
from .routing_predictor import RoutingPredictor
from .et_routing import ETRouter
from .cache_prior_routing import CachePriorRouter
from .model import (
    _RMSNorm,
    _Attention,
    KVCache,
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


def _profile_enabled() -> bool:
    return bool(os.environ.get("OUTLIER_PROFILE"))


def _batched_enabled() -> bool:
    """OUTLIER_BATCHED=1 (default ON) — use batched GEMM in MoE forward."""
    return os.environ.get("OUTLIER_BATCHED", "1").strip() != "0"


def _cache_prior_enabled() -> bool:
    """OUTLIER_CACHE_PRIOR=1 (default ON) — bias routing toward cached experts."""
    return os.environ.get("OUTLIER_CACHE_PRIOR", "1").strip() != "0"


def _cache_prior_lambda() -> float:
    raw = os.environ.get("OUTLIER_CACHE_PRIOR_LAMBDA", "0.5").strip()
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"Invalid OUTLIER_CACHE_PRIOR_LAMBDA={raw!r}; falling back to 0.5", stacklevel=2)
        return 0.5


def _cache_prior_top_j() -> int:
    raw = os.environ.get("OUTLIER_CACHE_PRIOR_TOP_J", "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        warnings.warn(f"Invalid OUTLIER_CACHE_PRIOR_TOP_J={raw!r}; falling back to 1", stacklevel=2)
        return 1


def _cache_prior_alpha() -> float:
    raw = os.environ.get("OUTLIER_CACHE_PRIOR_ALPHA", "0.99").strip()
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"Invalid OUTLIER_CACHE_PRIOR_ALPHA={raw!r}; falling back to 0.99", stacklevel=2)
        return 0.99


def _sync_device(device: torch.device | str) -> None:
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_log(message: str) -> None:
    if _profile_enabled():
        print(f"[OUTLIER_PROFILE] {message}", flush=True)


def _resolve_monolith_path(
    model_dir: Path,
    packed_experts_dir: Optional[Path],
    monolith_path: Optional[str | "Path"] = None,
) -> Optional[Path]:
    """Return the first existing monolith (experts.bin) candidate, or None."""
    candidates: list[Path] = []
    if monolith_path:
        candidates.append(Path(monolith_path).expanduser())
    env_path = os.environ.get("OUTLIER_MONOLITH_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if packed_experts_dir is not None:
        candidates.append(packed_experts_dir / "experts.bin")
    if model_dir is not None:
        candidates.append(model_dir / "packed_experts" / "experts.bin")
    candidates.append(Path.home() / "outlier-engine" / "packed_experts" / "experts.bin")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


class MonolithExpertLoader:
    """
    Persistent O_RDONLY random-access reader for the packed experts.bin monolith.

    Keeps a single file descriptor open for the lifetime of the page manager so
    every expert lookup is a single pread() syscall instead of open+seek+read+close.
    Requires a populated ``packed_index`` (from index.json) for tensor shape metadata.
    """

    def __init__(
        self,
        store_path: str | Path,
        packed_index: Dict[str, Dict[str, object]],
    ) -> None:
        from .expert_store import ExpertStore, SUB_FILE_ORDER

        self.store_path = Path(store_path).expanduser().resolve()
        self._packed_index = packed_index
        self._fd = os.open(str(self.store_path), os.O_RDONLY)
        with open(self.store_path, "rb") as handle:
            header = ExpertStore._read_header(handle)
            self._index = ExpertStore._read_index(handle, header["num_entries"])
        self._sub_sizes: Dict[str, int] = {
            name: int(size)
            for name, size in zip(SUB_FILE_ORDER, header["sub_sizes"])
        }
        self._sub_file_order: list[str] = list(SUB_FILE_ORDER)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meta(self, layer_idx: int, expert_idx: int, proj: str, suffix: str) -> Dict[str, object]:
        key = f"base.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}_{suffix}"
        info = self._packed_index.get(key)
        if info is None:
            raise KeyError(f"Packed metadata missing for {key!r}")
        return info

    # ------------------------------------------------------------------
    # Main load
    # ------------------------------------------------------------------

    def load_expert(self, layer_idx: int, expert_idx: int) -> "_ExpertWeights":
        """Load one expert from the monolith via a single pread() syscall."""
        key = (layer_idx, expert_idx)
        if key not in self._index:
            raise KeyError(f"Expert ({layer_idx}, {expert_idx}) not present in monolith")

        offset, size = self._index[key]
        raw = os.pread(self._fd, size, offset)
        if len(raw) != size:
            raise IOError(
                f"Short read for expert ({layer_idx}, {expert_idx}): "
                f"expected {size} B, got {len(raw)} B"
            )

        view = memoryview(raw)
        cursor = 0
        chunks: dict[str, memoryview] = {}
        for name in self._sub_file_order:
            chunk_size = self._sub_sizes[name]
            chunks[name] = view[cursor : cursor + chunk_size]
            cursor += chunk_size

        tensors: dict[str, tuple] = {}
        shapes: dict[str, tuple] = {}
        for proj in ("gate", "up", "down"):
            t_info = self._meta(layer_idx, expert_idx, proj, "ternary")
            s_info = self._meta(layer_idx, expert_idx, proj, "scale")
            ternary = torch.from_numpy(
                np.frombuffer(chunks[f"{proj}_ternary"], dtype=np.uint8).copy()
            )
            scale_shape = tuple(int(d) for d in s_info["shape"])
            scale = torch.from_numpy(
                np.frombuffer(chunks[f"{proj}_scale"], dtype=np.float16).copy()
            ).reshape(scale_shape)
            tensors[proj] = (ternary, scale)
            shapes[proj] = tuple(int(d) for d in t_info["shape"])

        return _ExpertWeights(
            tensors["gate"][0], tensors["gate"][1],
            tensors["up"][0],   tensors["up"][1],
            tensors["down"][0], tensors["down"][1],
            packed=True,
            dequantized=False,
            packed_format="tq10",
            shapes=shapes,
        )


def _parse_layer_idx(value: str) -> int | None:
    for pattern in (
        r"^(?:layers?|layer)[._](\d+)$",
        r"(?:^|[._])(?:layers?|layer)[._](\d+)(?:[._]|$)",
    ):
        match = re.search(pattern, value)
        if match:
            return int(match.group(1))
    return None


def _parse_expert_idx(value: str) -> int | None:
    for pattern in (
        r"^(?:experts?|expert)[._](\d+)$",
        r"(?:^|[._])(?:experts?|expert)[._](\d+)(?:[._]|$)",
    ):
        match = re.search(pattern, value)
        if match:
            return int(match.group(1))
    return None


def _scalar_alpha(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    return None


def _collect_alpha_entries(
    payload: Any,
    out: dict[int, dict[int, float]],
    *,
    layer_idx: int | None = None,
    expert_idx: int | None = None,
) -> None:
    if isinstance(payload, torch.Tensor):
        if payload.ndim == 0:
            payload = payload.item()
        else:
            payload = payload.detach().cpu().tolist()
    elif isinstance(payload, np.ndarray):
        if payload.ndim == 0:
            payload = payload.item()
        else:
            payload = payload.tolist()

    scalar = _scalar_alpha(payload)
    if scalar is not None:
        if layer_idx is not None and expert_idx is not None:
            out.setdefault(layer_idx, {})[expert_idx] = scalar
        return

    if isinstance(payload, (list, tuple)):
        if layer_idx is None:
            for next_layer_idx, value in enumerate(payload):
                _collect_alpha_entries(value, out, layer_idx=next_layer_idx)
            return
        if expert_idx is None:
            for next_expert_idx, value in enumerate(payload):
                _collect_alpha_entries(value, out, layer_idx=layer_idx, expert_idx=next_expert_idx)
        return

    if not isinstance(payload, dict):
        return

    for raw_key, value in payload.items():
        key = str(raw_key)
        key_lower = key.lower()
        next_layer_idx = layer_idx
        next_expert_idx = expert_idx

        parsed_layer_idx = _parse_layer_idx(key_lower)
        if parsed_layer_idx is not None:
            next_layer_idx = parsed_layer_idx
        parsed_expert_idx = _parse_expert_idx(key_lower)
        if parsed_expert_idx is not None:
            next_expert_idx = parsed_expert_idx
        if key.isdigit():
            if layer_idx is None:
                next_layer_idx = int(key)
            elif expert_idx is None:
                next_expert_idx = int(key)

        if key_lower in {
            "alpha",
            "alphas",
            "expert_alpha",
            "expert_alphas",
            "experts",
            "layer_alpha",
            "layer_alphas",
            "layers",
            "per_expert",
            "per_layer",
            "values",
        }:
            _collect_alpha_entries(value, out, layer_idx=layer_idx, expert_idx=expert_idx)
            continue

        _collect_alpha_entries(value, out, layer_idx=next_layer_idx, expert_idx=next_expert_idx)


def _load_hybrid_expert_alphas(
    model_dir: Path,
    *,
    n_layers: int,
    n_experts: int,
) -> dict[int, dict[int, float]]:
    alphas: dict[int, dict[int, float]] = {}

    for alpha_path in sorted(model_dir.rglob("alpha.json")):
        try:
            payload = json.loads(alpha_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.warn(f"Failed to read alpha sidecar {alpha_path}: {exc}", stacklevel=2)
            continue
        _collect_alpha_entries(payload, alphas)

    try:
        from safetensors import safe_open
    except ImportError:
        safe_open = None

    if safe_open is not None:
        for shard in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                alpha_keys = [key for key in f.keys() if "alpha" in key.lower()]
                for key in alpha_keys:
                    _collect_alpha_entries({key: f.get_tensor(key)}, alphas)

    filtered: dict[int, dict[int, float]] = {}
    for layer_idx, layer_alphas in alphas.items():
        if layer_idx < 0 or layer_idx >= n_layers:
            continue
        filtered_layer = {
            expert_idx: float(alpha)
            for expert_idx, alpha in layer_alphas.items()
            if 0 <= expert_idx < n_experts
        }
        if filtered_layer:
            filtered[layer_idx] = filtered_layer
    return filtered


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
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        H, H_kv, d = self.n_heads, self.n_kv_heads, self.head_dim
        past_k, past_v = past_key_value if past_key_value is not None else (None, None)
        past_len = 0 if past_k is None else past_k.shape[2]

        q = self.q_proj(x).view(B, L, H,    d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H_kv, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H_kv, d).transpose(1, 2)

        q = self.rope(q, position_offset=past_len)
        k = self.rope(k, position_offset=past_len)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        if use_cache:
            cached_k = k if self.n_rep == 1 else k[:, :: self.n_rep, :, :].contiguous()
            cached_v = v if self.n_rep == 1 else v[:, :: self.n_rep, :, :].contiguous()
            return out, (cached_k, cached_v)
        return out


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
        self.register_buffer("gate_s", torch.ones(I, 1, dtype=torch.bfloat16))
        self.register_buffer("up_w",   torch.zeros(I, D, dtype=torch.int8))
        self.register_buffer("up_s",   torch.ones(I, 1, dtype=torch.bfloat16))
        self.register_buffer("down_w", torch.zeros(D, I, dtype=torch.int8))
        self.register_buffer("down_s", torch.ones(D, 1, dtype=torch.bfloat16))

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

    Hot cache stores dequantized weights ready for matmul.
    CPU warm cache stores packed uint8 (memory-efficient).
    """

    __slots__ = (
        "gate_w", "gate_s",
        "up_w",   "up_s",
        "down_w", "down_s",
        "packed",
        "dequantized",
        "packed_format",
        "shapes",
    )

    def __init__(
        self,
        gate_w: torch.Tensor, gate_s: torch.Tensor,
        up_w:   torch.Tensor, up_s:   torch.Tensor,
        down_w: torch.Tensor, down_s: torch.Tensor,
        packed: bool = False,
        dequantized: bool = False,
        packed_format: str = "raw",
        shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> None:
        self.gate_w = gate_w;  self.gate_s = gate_s
        self.up_w   = up_w;    self.up_s   = up_s
        self.down_w = down_w;  self.down_s = down_s
        self.packed = packed
        self.dequantized = dequantized
        self.packed_format = packed_format
        self.shapes = shapes or {}

    def to(self, device: torch.device) -> "_ExpertWeights":
        return _ExpertWeights(
            self.gate_w.to(device), self.gate_s.to(device),
            self.up_w.to(device),   self.up_s.to(device),
            self.down_w.to(device), self.down_s.to(device),
            packed=self.packed,
            dequantized=self.dequantized,
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
            dequantized=False,
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
            dequantized=False,
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
                dequantized=False,
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
                dequantized=False,
            )
        raise RuntimeError(f"Unsupported packed format: {self.packed_format}")

    def hot_ready(self, device: torch.device) -> "_ExpertWeights":
        """Return dequantized weights ready for matmul on the target device."""
        weights = self.unpack_to_int8() if self.packed else self
        compute_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        gate_s = weights.gate_s.to(device=device, dtype=compute_dtype)
        up_s = weights.up_s.to(device=device, dtype=compute_dtype)
        down_s = weights.down_s.to(device=device, dtype=compute_dtype)
        gate_w = weights.gate_w.to(device=device, dtype=compute_dtype) * gate_s
        up_w = weights.up_w.to(device=device, dtype=compute_dtype) * up_s
        down_w = weights.down_w.to(device=device, dtype=compute_dtype) * down_s
        return _ExpertWeights(
            gate_w.contiguous(),
            gate_s,
            up_w.contiguous(),
            up_s,
            down_w.contiguous(),
            down_s,
            packed=False,
            dequantized=True,
        )

    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.gate_w, self.gate_s, self.up_w, self.up_s, self.down_w, self.down_s)
        )

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
        Returns: (batch, hidden_dim) in the active compute dtype
        """
        if self.packed:
            unpacked = self.unpack_to_int8()
            return unpacked.run(x)

        if self.dequantized:
            compute_dtype = torch.float32 if x.device.type == "cpu" else torch.bfloat16
            x_dev = x if (x.dtype == compute_dtype and x.device == self.gate_w.device) else x.to(
                device=self.gate_w.device, dtype=compute_dtype
            )
            if _profile_enabled():
                _sync_device(x_dev.device)
                stage_t0 = time.perf_counter()
                gate_raw = F.linear(x_dev, self.gate_w)
                _sync_device(x_dev.device)
                gate_ms = (time.perf_counter() - stage_t0) * 1000.0
                stage_t0 = time.perf_counter()
                up = F.linear(x_dev, self.up_w)
                _sync_device(x_dev.device)
                up_ms = (time.perf_counter() - stage_t0) * 1000.0
                stage_t0 = time.perf_counter()
                gate = F.silu(gate_raw)
                _sync_device(x_dev.device)
                activation_ms = (time.perf_counter() - stage_t0) * 1000.0
                stage_t0 = time.perf_counter()
                out = F.linear(gate * up, self.down_w)
                _sync_device(x_dev.device)
                down_ms = (time.perf_counter() - stage_t0) * 1000.0
                _profile_log(
                    "expert_run "
                    f"mode=dequantized x_device={x_dev.device} w_device={self.gate_w.device} "
                    f"gate_ms={gate_ms:.2f} up_ms={up_ms:.2f} activation_ms={activation_ms:.2f} "
                    f"down_ms={down_ms:.2f}"
                )
                return out

            gate = F.silu(F.linear(x_dev, self.gate_w))
            up = F.linear(x_dev, self.up_w)
            return F.linear(gate * up, self.down_w)

        if x.device.type != "cpu":
            compute_dtype = torch.bfloat16
            x_dev = x.to(compute_dtype)
            gate_w = self.gate_w.to(compute_dtype) * self.gate_s.to(compute_dtype)
            up_w = self.up_w.to(compute_dtype) * self.up_s.to(compute_dtype)
            down_w = self.down_w.to(compute_dtype) * self.down_s.to(compute_dtype)
            gate = F.silu(F.linear(x_dev, gate_w))
            up = F.linear(x_dev, up_w)
            return F.linear(gate * up, down_w)

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


def load_hybrid_paged_qwen(
    model_path: str | Path,
    *,
    device: str = "mps",
    max_experts_in_memory: int = 64,
    max_warm_cache: int = 256,
    packed_experts_dir: Optional[str] = None,
    monolith_path: Optional[str | Path] = None,
):
    """Build a native Qwen2 model and swap only the MLP path for paged MoE."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model_dir = Path(model_path)
    with open(model_dir / "config.json") as f:
        cfg = _normalize_config(json.load(f))
    alpha_sidecar_present = any(model_dir.rglob("alpha.json"))
    layer_alphas = _load_hybrid_expert_alphas(
        model_dir,
        n_layers=cfg["n_layers"],
        n_experts=cfg.get("n_experts", 0),
    )

    hf_cfg = Qwen2Config(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_dim"],
        intermediate_size=cfg["intermediate_dim"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg.get("n_kv_heads", cfg["n_heads"]),
        hidden_act="silu",
        max_position_embeddings=cfg.get("max_seq_len", 32768),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        use_cache=True,
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        rope_parameters=cfg.get("rope_parameters", {"rope_theta": cfg.get("rope_theta", 1000000.0), "rope_type": "default"}),
        use_sliding_window=cfg.get("use_sliding_window", False),
        sliding_window=cfg.get("sliding_window"),
        max_window_layers=cfg.get("max_window_layers", cfg["n_layers"]),
        layer_types=cfg.get("layer_types"),
        attention_dropout=cfg.get("attention_dropout", 0.0),
        bos_token_id=cfg.get("bos_token_id"),
        eos_token_id=cfg.get("eos_token_id"),
        pad_token_id=cfg.get("pad_token_id"),
    )

    # V3.2 publishes `moe_layer_indices` (and an alias `v3_2_moe_layers`)
    # enumerating which layers are sparse MoE.  When the config declares
    # either list, only those layers get `_HybridPagedMLP`; all other
    # layers keep the stock Qwen2MLP that `Qwen2ForCausalLM(hf_cfg)`
    # already constructed.  For legacy V1/V2 "fully sparse" checkpoints
    # where every layer is MoE, neither key is present and the original
    # "replace every layer" behaviour is preserved.
    moe_indices_cfg = cfg.get("moe_layer_indices") or cfg.get("v3_2_moe_layers")
    if moe_indices_cfg is not None:
        moe_layer_set = {int(i) for i in moe_indices_cfg}
    else:
        moe_layer_set = None  # None = all layers are MoE (legacy semantics)

    with torch.device("meta"):
        model = Qwen2ForCausalLM(hf_cfg)
        for layer_idx, layer in enumerate(model.model.layers):
            is_moe_layer = (moe_layer_set is None) or (layer_idx in moe_layer_set)
            if is_moe_layer:
                layer.mlp = _HybridPagedMLP(
                    cfg["hidden_dim"],
                    cfg["intermediate_dim"],
                    cfg.get("n_experts", 0),
                    cfg.get("top_k", 2),
                    layer_idx=layer_idx,
                    page_manager=None,
                    alphas=layer_alphas.get(layer_idx),
                    alpha_default=0.0 if alpha_sidecar_present else 1.0,
                )
            # else: dense layer — leave the stock Qwen2MLP in place.  It is
            # populated below from the standard `mlp.gate_proj/up_proj/
            # down_proj` safetensors keys.

    model = model.to_empty(device=torch.device(device))
    model = model.to(dtype=torch.bfloat16)

    page_manager = ExpertPageManager(
        model_dir,
        device=device,
        n_experts=cfg.get("n_experts", 0),
        n_layers=cfg["n_layers"],
        top_k=cfg.get("top_k", 2),
        max_experts_in_memory=max_experts_in_memory,
        max_warm_cache=max_warm_cache,
        packed_experts_dir=packed_experts_dir,
        monolith_path=monolith_path,
    )

    for layer in model.model.layers:
        if isinstance(layer.mlp, _HybridPagedMLP):
            layer.mlp.page_manager = page_manager

    from safetensors import safe_open
    from .quantize_utils import quantize_to_int8

    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for raw_key in f.keys():
                # Routed experts are loaded on demand by the page manager.
                # Match both "base.model.layers.N.mlp.experts.M.*" (legacy
                # V1/V2 "real" format) and "model.layers.N.mlp.experts.M.*"
                # (V3.2 custom-modeling format, no `base.` prefix).
                if (
                    raw_key.startswith("base.model.layers.")
                    or raw_key.startswith("model.layers.")
                ) and ".mlp.experts." in raw_key:
                    continue
                # Per-expert alpha tensor in V3.2 is loaded via the alpha.json
                # sidecar → layer_alphas; skip the safetensors copy here.
                if raw_key.endswith(".mlp.alpha_values"):
                    continue
                tensor = f.get_tensor(raw_key)
                if raw_key.startswith("base.model."):
                    key = raw_key[len("base."):]
                elif raw_key.startswith("base."):
                    key = raw_key[len("base."):]
                else:
                    key = raw_key

                if key == "model.embed_tokens.weight":
                    model.model.embed_tokens.weight.data.copy_(tensor.to(model.model.embed_tokens.weight.dtype))
                    continue
                if key == "model.norm.weight":
                    model.model.norm.weight.data.copy_(tensor.to(model.model.norm.weight.dtype))
                    continue
                if key == "lm_head.weight":
                    model.lm_head.weight.data.copy_(tensor.to(model.lm_head.weight.dtype))
                    continue

                layer_match = re.match(r"^model\.layers\.(\d+)\.(.+)$", key)
                if not layer_match:
                    continue
                layer_idx = int(layer_match.group(1))
                suffix = layer_match.group(2)
                layer = model.model.layers[layer_idx]

                if suffix == "input_layernorm.weight":
                    layer.input_layernorm.weight.data.copy_(tensor.to(layer.input_layernorm.weight.dtype))
                elif suffix == "post_attention_layernorm.weight":
                    layer.post_attention_layernorm.weight.data.copy_(tensor.to(layer.post_attention_layernorm.weight.dtype))
                elif suffix.startswith("self_attn."):
                    proj_name, param_name = suffix[len("self_attn."):].rsplit(".", 1)
                    module = getattr(layer.self_attn, proj_name, None)
                    target = getattr(module, param_name, None) if module is not None else None
                    if target is not None:
                        target.data.copy_(tensor.to(target.dtype))
                elif suffix == "mlp.router.weight" and isinstance(layer.mlp, _HybridPagedMLP):
                    layer.mlp.router_weight.data.copy_(tensor.to(layer.mlp.router_weight.dtype))
                elif suffix.startswith("mlp.shared_expert.") and isinstance(layer.mlp, _HybridPagedMLP):
                    # Legacy V1/V2 "real" format: mlp.shared_expert.gate_W/up_W/down_W
                    shared = layer.mlp.shared
                    proj = suffix[len("mlp.shared_expert."):]
                    if proj == "gate_W":
                        q, s = quantize_to_int8(tensor)
                        shared.gate_w.copy_(q)
                        shared.gate_s.copy_(s.to(shared.gate_s.dtype))
                    elif proj == "up_W":
                        q, s = quantize_to_int8(tensor)
                        shared.up_w.copy_(q)
                        shared.up_s.copy_(s.to(shared.up_s.dtype))
                    elif proj == "down_W":
                        q, s = quantize_to_int8(tensor)
                        shared.down_w.copy_(q)
                        shared.down_s.copy_(s.to(shared.down_s.dtype))
                elif suffix in (
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ):
                    # V3.2 custom-modeling format: every MoE layer has a
                    # shared dense path stored under `mlp.{gate,up,down}_proj`
                    # and every dense layer has only this.
                    proj_name = suffix[len("mlp."):-len(".weight")]  # "gate_proj" etc.
                    if isinstance(layer.mlp, _HybridPagedMLP):
                        # MoE layer — dense keys populate the shared (_Int8SwiGLU) expert.
                        shared = layer.mlp.shared
                        q, s = quantize_to_int8(tensor)
                        if proj_name == "gate_proj":
                            shared.gate_w.copy_(q)
                            shared.gate_s.copy_(s.to(shared.gate_s.dtype))
                        elif proj_name == "up_proj":
                            shared.up_w.copy_(q)
                            shared.up_s.copy_(s.to(shared.up_s.dtype))
                        elif proj_name == "down_proj":
                            shared.down_w.copy_(q)
                            shared.down_s.copy_(s.to(shared.down_s.dtype))
                    else:
                        # Dense layer — stock Qwen2MLP, copy weight directly
                        # into gate_proj/up_proj/down_proj.weight.
                        module = getattr(layer.mlp, proj_name, None)
                        if module is not None and hasattr(module, "weight"):
                            module.weight.data.copy_(tensor.to(module.weight.dtype))

    model.eval()
    model.outlier_device = torch.device(device)
    model.outlier_page_manager = page_manager
    model.cache_stats = MethodType(lambda self: self.outlier_page_manager.cache_stats(), model)
    model.prefetch_stats = MethodType(lambda self: self.outlier_page_manager.prefetch_stats(), model)

    def _enable_expert_prefetch(self) -> None:
        self.outlier_page_manager.enable_expert_prefetch()
        if getattr(self, "_outlier_prefetch_hooks", None):
            return

        handles = []
        for layer_idx, layer in enumerate(self.model.layers):
            def _wait_hook(_module, _inputs, current_layer=layer_idx):
                self.outlier_page_manager.wait_for_layer(current_layer)

            handles.append(layer.register_forward_pre_hook(_wait_hook))
        self._outlier_prefetch_hooks = handles

    model.enable_expert_prefetch = MethodType(_enable_expert_prefetch, model)

    def _enable_et_routing(self) -> None:
        self.outlier_page_manager.enable_et_routing()

    model.enable_et_routing = MethodType(_enable_et_routing, model)
    model.routing_stats = MethodType(lambda self: self.outlier_page_manager.et_routing_stats(), model)

    def _enable_roe(self, roe_top_k: int) -> None:
        self.outlier_page_manager.enable_roe(roe_top_k)

    model.enable_roe = MethodType(_enable_roe, model)
    return model


def _run_expert(x: torch.Tensor, w: _ExpertWeights) -> torch.Tensor:
    """Run ternary expert via chunked matmul. Returns float32."""
    return w.run(x)


def _run_single_token_experts_batched(x: torch.Tensor, experts: list[_ExpertWeights]) -> torch.Tensor:
    """
    Run a small set of dequantized device experts as a single batched GEMM.

    OUTLIER-ENGINE-BATCHED-GEMM-001: 4 kernel launches for all experts combined
    vs. 5 × n_experts in the sequential path.

    Fallback to sequential when OUTLIER_BATCHED=0 or experts not all on device.
    """
    if not experts:
        raise ValueError("Expected at least one expert for batched execution.")
    if len(experts) == 1:
        # _run_expert returns [batch, H]; wrap to [1, H] == [E, H] for consistency
        out = _run_expert(x, experts[0])
        return out if out.dim() == 2 else out.unsqueeze(0)

    first = experts[0]
    if (
        not _batched_enabled()
        or first.gate_w.device.type == "cpu"
        or not all(w.dequantized and w.gate_w.device == first.gate_w.device for w in experts)
    ):
        return torch.cat([_run_expert(x, expert) for expert in experts], dim=0)

    compute_dtype = first.gate_w.dtype
    x_dev = x if (x.dtype == compute_dtype and x.device == first.gate_w.device) else x.to(
        device=first.gate_w.device, dtype=compute_dtype
    )
    # x_vec: [H]  →  x_col: [H, 1] for batched matmul [E, I, H] × [H, 1] = [E, I, 1]
    x_col = x_dev.squeeze(0).unsqueeze(-1)

    # GATHER: Python loop + torch.stack (no GPU kernel)
    gate_w = torch.stack([e.gate_w for e in experts], dim=0)  # [E, I, H]
    up_w   = torch.stack([e.up_w   for e in experts], dim=0)  # [E, I, H]
    down_w = torch.stack([e.down_w for e in experts], dim=0)  # [E, H, I]

    # BATCHED GEMM: 4 kernel launches total
    gate_out = torch.matmul(gate_w, x_col).squeeze(-1)        # [E, I]
    up_out   = torch.matmul(up_w,   x_col).squeeze(-1)        # [E, I]
    act_out  = F.silu(gate_out) * up_out                      # [E, I]
    down_out = torch.matmul(down_w, act_out.unsqueeze(-1)).squeeze(-1)  # [E, H]
    return down_out  # [E, H]


def _summarize_et_stats(routers: list[ETRouter] | None) -> Dict[str, object]:
    if not routers:
        return {
            "et_routing_enabled": False,
            "avg_experts_per_token": 0.0,
            "total_tokens_routed": 0,
            "expert_count_histogram": {},
        }

    total_tokens = 0
    total_selected = 0.0
    histogram: Counter[int] = Counter()
    threshold_values: list[list[float]] = []
    for router in routers:
        stats = router.stats
        tokens = int(stats["total_tokens_routed"])
        total_tokens += tokens
        total_selected += float(stats["avg_experts_per_token"]) * tokens
        histogram.update(stats.get("expert_count_histogram", {}))
        threshold_values.append(list(stats["threshold_values"]))
    avg = total_selected / total_tokens if total_tokens else 0.0
    return {
        "et_routing_enabled": True,
        "avg_experts_per_token": avg,
        "total_tokens_routed": total_tokens,
        "expert_count_histogram": dict(sorted(histogram.items())),
        "threshold_values": threshold_values,
    }


def _summarize_cache_prior_stats(routers: list[CachePriorRouter] | None) -> Dict[str, object]:
    if not routers:
        return {
            "cache_prior_enabled": False,
            "cache_prior_overrides": 0,
            "cache_prior_tokens": 0,
            "cache_prior_override_rate": 0.0,
        }

    overrides = 0
    tokens = 0
    lam = None
    top_j = None
    for router in routers:
        stats = router.stats
        overrides += int(stats["cache_prior_overrides"])
        tokens += int(stats["cache_prior_tokens"])
        lam = stats["cache_prior_lambda"]
        top_j = stats["cache_prior_top_j"]
    return {
        "cache_prior_enabled": True,
        "cache_prior_overrides": overrides,
        "cache_prior_tokens": tokens,
        "cache_prior_override_rate": (overrides / tokens) if tokens else 0.0,
        "cache_prior_lambda": lam,
        "cache_prior_top_j": top_j,
    }


def _route_with_cache_prior(
    router: CachePriorRouter,
    logits: torch.Tensor,
    *,
    layer_id: int,
    initial_cache_state: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 1:
        return router.route(logits, layer_id=layer_id, cache_state=initial_cache_state)

    cache_state = set(initial_cache_state)
    selected_idx_rows = []
    selected_w_rows = []
    for row in range(logits.shape[0]):
        row_idx, row_w = router.route(
            logits[row : row + 1],
            layer_id=layer_id,
            cache_state=cache_state,
        )
        selected_idx_rows.append(row_idx)
        selected_w_rows.append(row_w)
        cache_state.update(int(expert_idx) for expert_idx in row_idx[0].tolist() if expert_idx >= 0)
    return torch.cat(selected_idx_rows, dim=0), torch.cat(selected_w_rows, dim=0)


# ---------------------------------------------------------------------------
# RoE (Routing on Ensemble) helpers
# ---------------------------------------------------------------------------

def _roe_augment(
    probs: torch.Tensor,
    selected_idx: torch.Tensor,
    *,
    roe_top_k: int,
    cached_ids: "set[int]",
    single_token: bool,
) -> "tuple[list[int], torch.Tensor | None]":
    """
    Augment standard top-k expert selection with additional experts from cache.

    OUTLIER-ENGINE-ROE-001: sample up to ``roe_top_k`` experts, drawing extras
    ONLY from the hot+warm cache so no SSD reads are triggered.  Weights are
    re-normalised over the selected set (standard soft-MoE weighting).

    Args:
        probs:        softmax probabilities, shape [1, n_experts] (single-token only).
        selected_idx: top-k expert indices, shape [1, k].
        roe_top_k:    desired ensemble size (must be > k to have any effect).
        cached_ids:   set of expert IDs in the hot or warm cache for this layer.
        single_token: True when batch × seq_len == 1 (decode step).

    Returns:
        (extra_ids, roe_weights) where:
            extra_ids  – list of extra expert IDs (may be empty)
            roe_weights – 1-D tensor of renormalised weights over ALL selected
                          experts (required + extras), or None if no extras.
    """
    if not single_token or roe_top_k <= 0:
        return [], None

    required_ids: list[int] = selected_idx[0].tolist()
    n_required = len(required_ids)
    if roe_top_k <= n_required:
        return [], None

    # Full sorted expert list by probability (descending)
    all_probs_1d = probs[0]  # [n_experts]
    n_to_consider = min(roe_top_k, int(all_probs_1d.shape[0]))
    _, cand_idx = torch.topk(all_probs_1d, k=n_to_consider)

    required_set = set(required_ids)
    extras: list[int] = []
    for eid_t in cand_idx.tolist():
        eid = int(eid_t)
        if eid in required_set:
            continue
        if eid in cached_ids:
            extras.append(eid)
        if n_required + len(extras) >= roe_top_k:
            break

    if not extras:
        return [], None

    all_ids = required_ids + extras
    idx_t = torch.tensor(all_ids, dtype=torch.long, device=all_probs_1d.device)
    raw_w = all_probs_1d[idx_t]
    roe_w = raw_w / raw_w.sum()  # renormalise over selected set
    return extras, roe_w


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


class _HybridPagedMLP(nn.Module):
    """Drop-in replacement for HF Qwen2 MLP that pages ternary experts."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        n_experts: int,
        top_k: int,
        layer_idx: int,
        page_manager: "ExpertPageManager | None" = None,
        alphas: Optional[Dict[int, float]] = None,
        alpha_default: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.page_manager = page_manager
        self.alphas = {int(expert_idx): float(alpha) for expert_idx, alpha in (alphas or {}).items()}
        self.alpha_default = float(alpha_default)
        # Register alphas as nn.Parameters so they're visible to named_parameters()
        # and optimizable by TTT experiments.
        for expert_idx, alpha_val in self.alphas.items():
            self.register_parameter(
                f"alpha_e{expert_idx}",
                nn.Parameter(torch.tensor(alpha_val), requires_grad=True),
            )
        self.router_weight = nn.Parameter(
            torch.empty(n_experts, hidden_dim), requires_grad=False
        )
        self.shared = _Int8SwiGLU(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.page_manager is None:
            raise RuntimeError("Hybrid paged MLP has no page manager attached.")

        batch, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        compute_dtype = x_flat.dtype if x_flat.device.type != "cpu" else torch.float32
        shared_out = self.shared(x_flat.to(compute_dtype))
        if os.environ.get("OUTLIER_SHARED_ONLY") == "1":
            return shared_out.to(x.dtype).view(batch, seq_len, hidden_dim)
        logits = F.linear(x_flat.to(compute_dtype), self.router_weight.to(compute_dtype))
        et_router = self.page_manager.get_et_router(self.layer_idx)
        _probs_for_roe: torch.Tensor | None = None
        if et_router is not None:
            selected_idx, selected_w = et_router.route(logits)
        else:
            cache_prior_router = self.page_manager.get_cache_prior_router(self.layer_idx)
            if cache_prior_router is not None:
                selected_idx, selected_w = _route_with_cache_prior(
                    cache_prior_router,
                    logits,
                    layer_id=self.layer_idx,
                    initial_cache_state=self.page_manager.cached_expert_ids(self.layer_idx),
                )
            else:
                probs = F.softmax(logits, dim=-1)
                effective_top_k = min(self.top_k, probs.shape[-1])
                selected_w, selected_idx = torch.topk(probs, k=effective_top_k, dim=-1)
                selected_w = selected_w / selected_w.sum(-1, keepdim=True)
                _probs_for_roe = probs  # captured for RoE augmentation below
        used_expert_ids = [int(expert_id) for expert_id in torch.unique(selected_idx[selected_idx >= 0]).tolist()]
        self.page_manager.record_layer_routing(self.layer_idx, logits.detach(), used_expert_ids)

        # OUTLIER-ENGINE-ROE-001: augment decode with cached extras (no SSD reads)
        _roe_w: torch.Tensor | None = None
        roe_top_k = getattr(self.page_manager, "roe_top_k", 0)
        if _probs_for_roe is not None and roe_top_k > 0 and x_flat.shape[0] == 1:
            _roe_extras, _roe_w = _roe_augment(
                probs=_probs_for_roe,
                selected_idx=selected_idx,
                roe_top_k=roe_top_k,
                cached_ids=self.page_manager.cached_expert_ids(self.layer_idx),
                single_token=True,
            )
            if _roe_extras:
                used_expert_ids = used_expert_ids + _roe_extras

        expert_out = torch.zeros_like(shared_out)

        # OUTLIER-ENGINE-BATCHED-GEMM-001: batched path for single-token decode
        # Gate: OUTLIER_BATCHED=1 (default ON)
        if _batched_enabled() and x_flat.shape[0] == 1 and len(used_expert_ids) >= 1:
            experts = [self.page_manager.get_expert(self.layer_idx, int(eid)) for eid in used_expert_ids]
            # batched_out: [E, H] — all experts computed in 4 kernel launches
            batched_out = _run_single_token_experts_batched(x_flat, experts).to(shared_out.dtype)

            # Use RoE weights (renormalised over required+extras) when available
            if _roe_w is not None:
                w_vec = _roe_w.to(dtype=batched_out.dtype, device=batched_out.device)  # [E]
            else:
                # Vectorized weighted combine — no Python loop over experts
                w_vec = torch.stack([
                    (selected_w[0] * (selected_idx[0] == eid).to(selected_w.dtype)).sum()
                    for eid in used_expert_ids
                ]).to(dtype=batched_out.dtype, device=batched_out.device)  # [E]
            alpha_vec = torch.stack([
                getattr(self, f"alpha_e{int(eid)}", None) or torch.tensor(self.alpha_default)
                for eid in used_expert_ids
            ]).to(dtype=batched_out.dtype, device=batched_out.device)
            expert_out[0] = (batched_out * (w_vec * alpha_vec).unsqueeze(-1)).sum(dim=0)  # [H]
            return (shared_out + expert_out).to(x.dtype).view(batch, seq_len, hidden_dim)

        # Multi-token path (prefill) or OUTLIER_BATCHED=0: sequential per-expert loop
        for expert_idx in used_expert_ids:
            assignment = selected_idx == expert_idx
            token_mask = assignment.any(dim=-1)
            if not token_mask.any():
                continue
            weights = (selected_w * assignment.to(selected_w.dtype)).sum(dim=-1, keepdim=True)
            expert = self.page_manager.get_expert(self.layer_idx, int(expert_idx))
            out = _run_expert(x_flat[token_mask], expert).to(shared_out.dtype)
            alpha_param = getattr(self, f"alpha_e{int(expert_idx)}", None)
            alpha_val = alpha_param if alpha_param is not None else self.alpha_default
            out = out * alpha_val
            expert_out[token_mask] += weights[token_mask] * out

        return (shared_out + expert_out).to(x.dtype).view(batch, seq_len, hidden_dim)


class ExpertUsageTracker:
    """Tracks per-expert selection frequency to support hot expert pinning.

    After ``pin_after_tokens`` forward passes, the ``pin_top_k`` most-used
    experts are frozen into ``pinned`` — a frozenset of (layer, expert) keys
    that the page manager will never evict from the hot cache.
    """

    def __init__(self, pin_after_tokens: int = 100, pin_top_k: int = 50) -> None:
        self.pin_after_tokens = pin_after_tokens
        self.pin_top_k = pin_top_k
        self._counts: Counter = Counter()
        self._tokens_seen: int = 0
        self._pinned: frozenset = frozenset()
        self._pin_hits: int = 0
        self._pin_lookups: int = 0

    def record(self, layer_idx: int, expert_idx: int) -> None:
        """Record an expert selection; updates pin-hit counters."""
        key = (layer_idx, expert_idx)
        self._counts[key] += 1
        self._pin_lookups += 1
        if key in self._pinned:
            self._pin_hits += 1

    def on_token(self) -> None:
        """Call once per generated token. Locks in the pinned set at threshold."""
        self._tokens_seen += 1
        if self._tokens_seen == self.pin_after_tokens:
            top = self._counts.most_common(self.pin_top_k)
            self._pinned = frozenset(k for k, _ in top)

    @property
    def pinned(self) -> frozenset:
        return self._pinned

    @property
    def pin_hit_rate(self) -> float:
        return self._pin_hits / self._pin_lookups if self._pin_lookups > 0 else 0.0


class ExpertPageManager:
    """Reusable expert pager for hybrid HF and legacy paged runtimes."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "mps",
        *,
        n_experts: int,
        n_layers: int = 0,
        top_k: int = 2,
        max_experts_in_memory: int = 64,
        max_warm_cache: int = 256,
        packed_experts_dir: Optional[str] = None,
        monolith_path: Optional[str | Path] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.top_k = top_k
        self.roe_top_k: int = 0
        self.max_experts_in_memory = max_experts_in_memory
        self.max_warm_cache = max_warm_cache
        self._debug = bool(os.environ.get("OUTLIER_CACHE_DEBUG"))
        self._forward_passes = 0
        self._lock = threading.RLock()

        self._hot_cache: OrderedDict[Tuple[int, int], _ExpertWeights] = OrderedDict()
        self._cpu_cache: OrderedDict[Tuple[int, int], _ExpertWeights] = OrderedDict()
        self._hot_hits = 0
        self._warm_hits = 0
        self._cold_misses = 0
        self._hot_evictions = 0
        self._disk_loads = 0
        self._disk_load_s = 0.0
        self._usage_tracker: ExpertUsageTracker | None = None
        self._pinned: frozenset = frozenset()
        self._tensor_shard_index: Dict[str, str] = {}
        self._packed_index: Dict[str, Dict[str, object]] = {}
        if packed_experts_dir:
            self._packed_experts_dir: Path | None = Path(packed_experts_dir).expanduser()
        else:
            _default = _default_packed_dir()
            self._packed_experts_dir = _default if (_default / "index.json").exists() else None
        self._monolith_loader: MonolithExpertLoader | None = None
        self._monolith_path: Path | None = None
        self._fmt = "toy"
        self._prefetcher: ExpertPrefetcher | None = None
        self._routing_predictor: RoutingPredictor | None = None
        self._last_routed_layer: int | None = None
        self._last_routed_experts: list[int] = []
        self._et_routers: list[ETRouter] | None = None
        self._cache_prior_routers: list[CachePriorRouter] | None = None
        if _cache_prior_enabled() and self.n_experts > 0 and self.n_layers > 0:
            self._cache_prior_routers = [
                CachePriorRouter(
                    self.n_experts,
                    top_k=self.top_k,
                    top_j=min(_cache_prior_top_j(), self.top_k, self.n_experts),
                    lam=_cache_prior_lambda(),
                    alpha=_cache_prior_alpha(),
                )
                for _ in range(self.n_layers)
            ]

        self._build_shard_index()
        self._load_packed_index()
        self._init_monolith_loader(monolith_path)

    def _get_lock(self) -> threading.RLock:
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lock = lock
        return lock

    def _debug_log(self, message: str) -> None:
        if self._debug:
            print(f"[OUTLIER_CACHE_DEBUG] {message}", flush=True)

    def debug_forward_start(self, label: str | None = None) -> None:
        self._forward_passes += 1
        tag = label or f"forward_{self._forward_passes}"
        self._debug_log(
            f"{tag}: hot_cache={len(self._hot_cache)}/{self.max_experts_in_memory} "
            f"warm_cache={len(self._cpu_cache)}/{self.max_warm_cache}"
        )
        tracker = getattr(self, "_usage_tracker", None)
        if tracker is not None:
            tracker.on_token()
            self._pinned = tracker.pinned

    def _build_shard_index(self) -> None:
        from safetensors import safe_open

        toy_pat = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")
        real_pat = re.compile(r"^base\.model\.layers\.\d+\.mlp\.experts\.\d+\.")
        first_shard_keys: list[str] = []

        for shard in sorted(self.model_dir.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if not first_shard_keys:
                    first_shard_keys = keys
                for key in keys:
                    if toy_pat.match(key) or real_pat.match(key):
                        self._tensor_shard_index[key] = str(shard)

        self._fmt = _detect_format(first_shard_keys)

    def _load_packed_index(self) -> None:
        if self._packed_experts_dir is None:
            return
        index_path = self._packed_experts_dir / "index.json"
        if not index_path.exists():
            return
        self._packed_index = json.loads(index_path.read_text(encoding="utf-8"))

    def _init_monolith_loader(self, monolith_path: Optional[str | Path]) -> None:
        """Auto-detect and open the monolith experts.bin if one is available.

        Requires ``self._packed_index`` to already be populated (called after
        ``_load_packed_index``).  Falls back gracefully if no monolith is found
        or if the packed index is empty (shape metadata unavailable).
        """
        if not self._packed_index:
            # Can't parse monolith tensors without shape metadata from index.json.
            return
        resolved = _resolve_monolith_path(
            self.model_dir,
            self._packed_experts_dir,
            monolith_path,
        )
        if resolved is None:
            return
        try:
            self._monolith_loader = MonolithExpertLoader(resolved, self._packed_index)
            self._monolith_path = resolved
            _profile_log(f"monolith loader active: {resolved}")
        except Exception as exc:
            warnings.warn(
                f"Failed to initialise monolith loader from {resolved}: {exc}; "
                "falling back to per-expert disk loads.",
                stacklevel=3,
            )

    def _add_to_warm_cache(self, key: Tuple[int, int], weights: _ExpertWeights) -> None:
        if key in self._cpu_cache:
            self._cpu_cache.move_to_end(key)
            self._cpu_cache[key] = weights
            return
        if len(self._cpu_cache) >= self.max_warm_cache:
            self._cpu_cache.popitem(last=False)
        self._cpu_cache[key] = weights

    def enable_expert_prefetch(self) -> None:
        if getattr(self, "_prefetcher", None) is None:
            self._prefetcher = ExpertPrefetcher(self)
            self._routing_predictor = RoutingPredictor()
            self._last_routed_layer = None
            self._last_routed_experts = []

    def enable_et_routing(self) -> None:
        if self._et_routers is None:
            self._et_routers = [
                ETRouter(
                    n_experts=self.n_experts,
                    top_k_fallback=self.top_k,
                    min_experts=1,
                    max_experts=min(4, self.n_experts),
                )
                for _ in range(self.n_layers)
            ]

    def enable_roe(self, roe_top_k: int) -> None:
        """Enable RoE (Routing on Ensemble) for single-token decode steps.

        When active, up to ``roe_top_k`` experts are used per forward pass.
        Experts beyond the standard top-k are drawn ONLY from the hot+warm
        cache — no SSD reads are triggered for the extra slots.
        """
        self.roe_top_k = max(0, int(roe_top_k))

    def get_et_router(self, layer_idx: int) -> ETRouter | None:
        routers = getattr(self, "_et_routers", None)
        if routers is None or layer_idx >= len(routers):
            return None
        return routers[layer_idx]

    def et_routing_stats(self) -> Dict[str, object]:
        return _summarize_et_stats(getattr(self, "_et_routers", None))

    def cache_prior_stats(self) -> Dict[str, object]:
        return _summarize_cache_prior_stats(getattr(self, "_cache_prior_routers", None))

    def get_cache_prior_router(self, layer_idx: int) -> CachePriorRouter | None:
        routers = getattr(self, "_cache_prior_routers", None)
        if routers is None or layer_idx >= len(routers):
            return None
        return routers[layer_idx]

    def warmup(self, n_experts_per_layer: int = 2, callback=None) -> Dict[str, float]:
        """Pre-load the most common experts into hot cache to eliminate cold-start latency.

        Loads ``n_experts_per_layer`` experts per layer (default: top-2 = typical routing).
        For layers 0-6 and 21-27, loads experts 0 and 1 (empirically most common).
        For layers 7-20, loads the first ``n_experts_per_layer`` experts.

        Args:
            n_experts_per_layer: How many experts to pre-load per layer.
            callback: Optional callable(layer_idx, expert_idx, total_loaded, total) for progress.

        Returns:
            Dict with warmup stats: duration_s, experts_loaded, hot_cache_mb.
        """
        import time as _time
        t0 = _time.perf_counter()
        total = self.n_layers * n_experts_per_layer
        loaded = 0
        for layer_idx in range(self.n_layers):
            experts_to_load = list(range(min(n_experts_per_layer, self.n_experts)))
            for expert_idx in experts_to_load:
                self.get_expert(layer_idx, expert_idx)
                loaded += 1
                if callback is not None:
                    callback(layer_idx, expert_idx, loaded, total)
        duration = _time.perf_counter() - t0
        stats = self.cache_stats()
        return {
            "warmup_duration_s": duration,
            "experts_loaded": loaded,
            "hot_cache_mb": stats.get("hot_cache_mb", 0),
            "hot_cache_entries": stats.get("hot_cache_entries", 0),
        }

    def enable_expert_pinning(self, pin_top_k: int = 50, pin_after_tokens: int = 100) -> None:
        """Enable hot expert pinning. After ``pin_after_tokens`` tokens the top-K
        most-used experts are permanently pinned and never evicted from the hot cache."""
        self._usage_tracker = ExpertUsageTracker(
            pin_after_tokens=pin_after_tokens,
            pin_top_k=pin_top_k,
        )
        self._pinned = frozenset()

    def cached_expert_ids(self, layer_idx: int) -> set[int]:
        with self._get_lock():
            cached = {
                expert_idx
                for cache_layer_idx, expert_idx in list(self._hot_cache.keys()) + list(self._cpu_cache.keys())
                if cache_layer_idx == layer_idx
            }
        return cached

    def wait_for_layer(self, layer_idx: int) -> None:
        prefetcher = getattr(self, "_prefetcher", None)
        if prefetcher is not None:
            prefetcher.wait(layer_idx)

    def prefetch_expert(self, layer_idx: int, expert_idx: int) -> bool:
        key = (layer_idx, expert_idx)
        on_cpu = self.device.type == "cpu"

        with self._get_lock():
            if key in self._hot_cache or key in self._cpu_cache:
                return False

        start = time.perf_counter()
        weights_int8 = self._load_expert_from_disk(layer_idx, expert_idx)
        load_s = time.perf_counter() - start
        weights_cached = weights_int8 if on_cpu else weights_int8.pack_tq10()

        with self._get_lock():
            if key in self._hot_cache or key in self._cpu_cache:
                return False
            self._disk_loads += 1
            self._disk_load_s += load_s
            self._add_to_warm_cache(key, weights_cached)
        return True

    def record_layer_routing(self, layer_idx: int, routing_logits: torch.Tensor, expert_ids: list[int]) -> None:
        prefetcher = getattr(self, "_prefetcher", None)
        predictor = getattr(self, "_routing_predictor", None)
        if prefetcher is None or predictor is None:
            return

        prefetcher.record_usage(layer_idx, expert_ids)
        if self._last_routed_layer is not None and self._last_routed_layer + 1 == layer_idx:
            predictor.update(self._last_routed_layer, self._last_routed_experts, expert_ids)

        self._last_routed_layer = layer_idx
        self._last_routed_experts = list(expert_ids)

        if layer_idx + 1 >= self.n_layers:
            return

        predicted = predictor.predict(layer_idx, expert_ids, top_k=self.top_k)
        prefetcher.prefetch(
            layer_idx + 1,
            routing_logits=routing_logits,
            top_k=self.top_k,
            predicted_expert_ids=predicted,
        )

    def prefetch_stats(self) -> Dict[str, float]:
        prefetcher = getattr(self, "_prefetcher", None)
        if prefetcher is None:
            return {
                "prefetches_issued": 0,
                "prefetch_hits": 0,
                "prefetch_wastes": 0,
                "prefetch_accuracy": 0.0,
            }
        return prefetcher.prefetch_stats

    def get_expert(self, layer_idx: int, expert_idx: int) -> _ExpertWeights:
        key = (layer_idx, expert_idx)
        on_cpu = self.device.type == "cpu"
        self._debug_log(f"lookup key={key}")
        lookup_t0 = time.perf_counter() if _profile_enabled() else 0.0

        tracker = getattr(self, "_usage_tracker", None)
        if tracker is not None:
            tracker.record(layer_idx, expert_idx)

        with self._get_lock():
            if key in self._hot_cache:
                self._hot_cache.move_to_end(key)
                self._hot_hits += 1
                self._debug_log(f"hit level=hot key={key}")
                if _profile_enabled():
                    _profile_log(
                        f"expert_lookup key={key} level=hot elapsed_ms={(time.perf_counter() - lookup_t0) * 1000.0:.2f} "
                        f"device={self._hot_cache[key].gate_w.device}"
                    )
                return self._hot_cache[key]

            if key in self._cpu_cache:
                self._cpu_cache.move_to_end(key)
                weights_cached = self._cpu_cache[key]
                self._warm_hits += 1
                self._debug_log(f"hit level=warm key={key}")
                load_ms = 0.0
            else:
                weights_cached = None
                load_ms = 0.0

        if weights_cached is None:
            self._debug_log(f"miss key={key}")
            start = time.perf_counter()
            weights_int8 = self._load_expert_from_disk(layer_idx, expert_idx)
            load_s = time.perf_counter() - start
            weights_cached = weights_int8 if on_cpu else weights_int8.pack_tq10()
            load_ms = load_s * 1000.0
            with self._get_lock():
                if key in self._hot_cache:
                    self._hot_cache.move_to_end(key)
                    self._hot_hits += 1
                    return self._hot_cache[key]
                if key in self._cpu_cache:
                    self._cpu_cache.move_to_end(key)
                    weights_cached = self._cpu_cache[key]
                    self._warm_hits += 1
                    load_ms = 0.0
                else:
                    self._cold_misses += 1
                    self._disk_loads += 1
                    self._disk_load_s += load_s
                    self._add_to_warm_cache(key, weights_cached)

        prep_t0 = time.perf_counter() if _profile_enabled() else 0.0
        weights_hot = weights_cached.hot_ready(self.device)
        with self._get_lock():
            if key in self._hot_cache:
                self._hot_cache.move_to_end(key)
                self._hot_hits += 1
                return self._hot_cache[key]
            if len(self._hot_cache) >= self.max_experts_in_memory:
                pinned = getattr(self, "_pinned", frozenset())
                evicted_key = None
                for candidate in self._hot_cache:
                    if candidate not in pinned:
                        evicted_key = candidate
                        break
                if evicted_key is None:
                    evicted_key, _ = self._hot_cache.popitem(last=False)
                else:
                    del self._hot_cache[evicted_key]
                self._hot_evictions += 1
                self._debug_log(f"evict hot key={evicted_key}")
            self._hot_cache[key] = weights_hot
        if _profile_enabled():
            _profile_log(
                f"expert_lookup key={key} level={'warm' if key in self._cpu_cache and load_ms == 0.0 else 'cold'} "
                f"disk_ms={load_ms:.2f} hot_ready_ms={(time.perf_counter() - prep_t0) * 1000.0:.2f} "
                f"elapsed_ms={(time.perf_counter() - lookup_t0) * 1000.0:.2f} device={weights_hot.gate_w.device}"
            )
        return weights_hot

    def _load_expert_from_packed(self, layer_idx: int, expert_idx: int) -> Optional[_ExpertWeights]:
        if not self._packed_index or self._packed_experts_dir is None:
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
            projs["up"][0], projs["up"][1],
            projs["down"][0], projs["down"][1],
            packed=False,
        )

    def _load_expert_from_disk(self, layer_idx: int, expert_idx: int) -> _ExpertWeights:
        # 1. Monolith (fastest: single pread syscall, 4KB-aligned layout)
        loader = getattr(self, "_monolith_loader", None)
        if loader is not None:
            try:
                return loader.load_expert(layer_idx, expert_idx)
            except (KeyError, IOError):
                pass  # expert not in monolith — fall through

        # 2. Per-expert packed binary files (packed_experts_dir)
        packed = self._load_expert_from_packed(layer_idx, expert_idx)
        if packed is not None:
            return packed

        # 3. Original safetensors shards (cold path)
        from safetensors import safe_open

        if self._fmt == "real":
            prefix = f"base.model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            names = {
                "gate_w": f"{prefix}.gate_ternary",
                "gate_s": f"{prefix}.gate_scale",
                "up_w": f"{prefix}.up_ternary",
                "up_s": f"{prefix}.up_scale",
                "down_w": f"{prefix}.down_ternary",
                "down_s": f"{prefix}.down_scale",
            }
        else:
            prefix = f"layers.{layer_idx}.ffn.experts.{expert_idx}"
            names = {
                "gate_w": f"{prefix}.gate_proj.weight",
                "gate_s": f"{prefix}.gate_proj.scale",
                "up_w": f"{prefix}.up_proj.weight",
                "up_s": f"{prefix}.up_proj.scale",
                "down_w": f"{prefix}.down_proj.weight",
                "down_s": f"{prefix}.down_proj.scale",
            }

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
            tensors["up_w"], tensors["up_s"],
            tensors["down_w"], tensors["down_s"],
            packed=False,
        )

    def cache_stats(self) -> Dict[str, float]:
        total_hits = self._hot_hits + self._warm_hits
        total = total_hits + self._cold_misses
        hit_rate = total_hits / total if total > 0 else 0.0
        hot_cache_bytes = sum(weights.nbytes() for weights in self._hot_cache.values())
        warm_cache_bytes = sum(weights.nbytes() for weights in self._cpu_cache.values())
        stats = {
            "hits": total_hits,
            "hot_hits": self._hot_hits,
            "warm_hits": self._warm_hits,
            "cold_misses": self._cold_misses,
            "misses": self._cold_misses,
            "lookups": total,
            "hit_rate": hit_rate,
            "hot_evictions": self._hot_evictions,
            "evictions": self._hot_evictions,
            "disk_loads": self._disk_loads,
            "disk_load_s": self._disk_load_s,
            "avg_disk_load_ms": (self._disk_load_s * 1000.0 / self._disk_loads) if self._disk_loads else 0.0,
            "hot_cache_entries": len(self._hot_cache),
            "hot_cache_mb": hot_cache_bytes / 1024**2,
            "warm_cache_entries": len(self._cpu_cache),
            "warm_cache_mb": warm_cache_bytes / 1024**2,
        }
        tracker = getattr(self, "_usage_tracker", None)
        pinned = getattr(self, "_pinned", frozenset())
        if tracker is not None:
            pinned_in_hot = sum(1 for k in self._hot_cache if k in pinned)
            stats["pinning_enabled"] = True
            stats["pinned_expert_count"] = len(pinned)
            stats["pinned_in_hot_cache"] = pinned_in_hot
            stats["pin_hit_rate"] = tracker.pin_hit_rate
        else:
            stats["pinning_enabled"] = False
            stats["pinned_expert_count"] = 0
            stats["pinned_in_hot_cache"] = 0
            stats["pin_hit_rate"] = 0.0
        stats.update(self.prefetch_stats())
        stats.update(self.et_routing_stats())
        stats.update(self.cache_prior_stats())
        return stats


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
        max_warm_cache:        LRU CPU warm cache capacity (default 256).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        max_experts_in_memory: int = 4,
        max_warm_cache: int = 256,
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
        self.to(dtype=torch.bfloat16)

        # Two-tier expert cache (both bounded with LRU eviction)
        self._lru_cache: OrderedDict[Tuple[int, int], _ExpertWeights] = OrderedDict()
        self._cpu_cache: OrderedDict[Tuple[int, int], _ExpertWeights] = OrderedDict()

        # Per-generate() cache statistics
        self._cache_hits      = 0
        self._cache_misses    = 0
        self._cache_evictions = 0   # total device→CPU evictions
        self._disk_loads      = 0
        self._disk_load_s     = 0.0
        self._lock            = threading.RLock()

        # Tensor-level shard index: raw_tensor_key → shard_path
        self._tensor_shard_index: Dict[str, str] = {}
        self._packed_index: Dict[str, Dict[str, object]] = {}
        if packed_experts_dir:
            self._packed_experts_dir: Path | None = Path(packed_experts_dir).expanduser()
        else:
            _default = _default_packed_dir()
            self._packed_experts_dir = _default if (_default / "index.json").exists() else None

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

        self._expert_prefetcher: ExpertPrefetcher | None = None
        self._routing_predictor: RoutingPredictor | None = None
        self._last_routed_layer: int | None = None
        self._last_routed_experts: list[int] = []
        self._et_routers: list[ETRouter] | None = None
        self._cache_prior_routers: list[CachePriorRouter] | None = None
        if _cache_prior_enabled() and self.n_experts > 0 and self.n_layers > 0:
            self._cache_prior_routers = [
                CachePriorRouter(
                    self.n_experts,
                    top_k=self.top_k,
                    top_j=min(_cache_prior_top_j(), self.top_k, self.n_experts),
                    lam=_cache_prior_lambda(),
                    alpha=_cache_prior_alpha(),
                )
                for _ in range(self.n_layers)
            ]

    # ------------------------------------------------------------------
    # Async prefetch integration
    # ------------------------------------------------------------------

    def enable_expert_prefetch(self) -> None:
        if self._expert_prefetcher is None:
            self._expert_prefetcher = ExpertPrefetcher(self)
            self._routing_predictor = RoutingPredictor()
            self._last_routed_layer = None
            self._last_routed_experts = []

    def enable_et_routing(self) -> None:
        if self._et_routers is None:
            self._et_routers = [
                ETRouter(
                    n_experts=self.n_experts,
                    top_k_fallback=self.top_k,
                    min_experts=1,
                    max_experts=min(4, self.n_experts),
                )
                for _ in range(self.n_layers)
            ]

    def routing_stats(self) -> Dict[str, object]:
        return _summarize_et_stats(getattr(self, "_et_routers", None))

    def cache_prior_stats(self) -> Dict[str, object]:
        return _summarize_cache_prior_stats(getattr(self, "_cache_prior_routers", None))

    def enable_async_prefetch(self, max_prefetch_ahead: int = 2) -> None:
        del max_prefetch_ahead
        self.enable_expert_prefetch()

    def _cached_expert_ids(self, layer_idx: int) -> set[int]:
        with self._get_lock():
            return {
                expert_idx
                for cache_layer_idx, expert_idx in list(self._lru_cache.keys()) + list(self._cpu_cache.keys())
                if cache_layer_idx == layer_idx
            }

    def _get_lock(self) -> threading.RLock:
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lock = lock
        return lock

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

    def prefetch_expert(self, layer_idx: int, expert_idx: int) -> bool:
        key = (layer_idx, expert_idx)
        on_cpu = self.device.type == "cpu"

        with self._get_lock():
            if key in self._lru_cache or key in self._cpu_cache:
                return False

        start = time.perf_counter()
        weights_int8 = self._load_expert_from_disk(layer_idx, expert_idx)
        load_s = time.perf_counter() - start
        weights_cached = weights_int8 if on_cpu else weights_int8.pack_tq10()

        with self._get_lock():
            if key in self._lru_cache or key in self._cpu_cache:
                return False
            self._disk_loads += 1
            self._disk_load_s += load_s
            self._add_to_warm_cache(key, weights_cached)
        return True

    def _record_layer_routing(self, layer_idx: int, routing_logits: torch.Tensor, expert_ids: list[int]) -> None:
        prefetcher = self._expert_prefetcher
        predictor = self._routing_predictor
        if prefetcher is None or predictor is None:
            return

        prefetcher.record_usage(layer_idx, expert_ids)
        if self._last_routed_layer is not None and self._last_routed_layer + 1 == layer_idx:
            predictor.update(self._last_routed_layer, self._last_routed_experts, expert_ids)

        self._last_routed_layer = layer_idx
        self._last_routed_experts = list(expert_ids)

        if layer_idx + 1 >= self.n_layers:
            return

        predicted = predictor.predict(layer_idx, expert_ids, top_k=self.top_k)
        prefetcher.prefetch(
            layer_idx + 1,
            routing_logits=routing_logits,
            top_k=self.top_k,
            predicted_expert_ids=predicted,
        )

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

        with self._get_lock():
            if key in self._lru_cache:
                self._lru_cache.move_to_end(key)
                self._cache_hits += 1
                return self._lru_cache[key]

            self._cache_misses += 1

            if len(self._lru_cache) >= self.max_experts_in_memory:
                evicted_key, evicted_w = self._lru_cache.popitem(last=False)
                self._cache_evictions += 1
                cached_w = evicted_w.cpu() if on_cpu else evicted_w.cpu().pack_tq10()
                self._add_to_warm_cache(evicted_key, cached_w)

            if key in self._cpu_cache:
                self._cpu_cache.move_to_end(key)
                weights_cached = self._cpu_cache[key]
            else:
                weights_cached = None

        if weights_cached is None:
            start = time.perf_counter()
            weights_int8 = self._load_expert_from_disk(layer_idx, expert_idx)
            load_s = time.perf_counter() - start
            weights_cached = weights_int8 if on_cpu else weights_int8.pack_tq10()
            with self._get_lock():
                if key in self._lru_cache:
                    self._lru_cache.move_to_end(key)
                    self._cache_hits += 1
                    return self._lru_cache[key]
                if key in self._cpu_cache:
                    self._cpu_cache.move_to_end(key)
                    weights_cached = self._cpu_cache[key]
                else:
                    self._disk_loads += 1
                    self._disk_load_s += load_s
                    self._add_to_warm_cache(key, weights_cached)

        if on_cpu:
            weights_dev = weights_cached
        else:
            weights_dev = weights_cached.unpack_to_int8().to(self.device)

        with self._get_lock():
            if key in self._lru_cache:
                self._lru_cache.move_to_end(key)
                self._cache_hits += 1
                return self._lru_cache[key]
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

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] integer token ids (on self.device)
        Returns:
            logits: [B, L, vocab_size]
        """
        x = self.embed_tokens(input_ids)   # [B, L, D]
        B, L, D = x.shape

        causal_mask: Optional[torch.Tensor] = None
        cache_empty = kv_cache is None or not kv_cache.cache
        if cache_empty and L > 1:
            causal_mask = _causal_mask(L, device=input_ids.device, dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            prefetcher = self._expert_prefetcher
            if prefetcher is not None:
                prefetcher.wait(i)

            # ── Attention (always device-resident) ────────────────────────
            # On CUDA/MPS this overlaps with the prefetch kicked off above.
            attn_input = layer.attn_norm(x)
            past = kv_cache.get(i) if kv_cache is not None else (None, None)
            attn_out = layer.attn(
                attn_input,
                mask=causal_mask,
                past_key_value=past if past[0] is not None else None,
                use_cache=use_cache,
            )
            if use_cache:
                attn_delta, present = attn_out
                if kv_cache is not None:
                    kv_cache.set(i, present[0], present[1])
            else:
                attn_delta = attn_out
            x = x + attn_delta

            # ── FFN ───────────────────────────────────────────────────────
            h = layer.ffn_norm(x)

            if layer.n_experts > 0:
                ffn: _PagedMoEFFN = layer.ffn   # type: ignore[assignment]
                h_flat = h.view(-1, D)           # [N, D]
                N = h_flat.shape[0]

                # Router: softmax → top-k (accurate, on post-attention x)
                logits = F.linear(h_flat.float(), ffn.router_weight.float())
                et_router = None if self._et_routers is None else self._et_routers[i]
                if et_router is not None:
                    selected_idx, selected_w = et_router.route(logits)
                else:
                    cache_prior_router = None if self._cache_prior_routers is None else self._cache_prior_routers[i]
                    if cache_prior_router is not None:
                        selected_idx, selected_w = _route_with_cache_prior(
                            cache_prior_router,
                            logits,
                            layer_id=i,
                            initial_cache_state=self._cached_expert_ids(i),
                        )
                    else:
                        probs  = F.softmax(logits, dim=-1)
                        effective_top_k = min(self.top_k, probs.shape[-1])
                        selected_w, selected_idx = torch.topk(probs, k=effective_top_k, dim=-1)
                        selected_w = selected_w / selected_w.sum(-1, keepdim=True)
                used_expert_ids = [int(expert_id) for expert_id in torch.unique(selected_idx[selected_idx >= 0]).tolist()]
                self._record_layer_routing(i, logits.detach(), used_expert_ids)

                # Shared expert (always device-resident, INT8 quantised)
                shared_out = ffn.shared(h_flat).float()   # [N, D]

                # Ternary experts: load each unique expert once for all tokens
                expert_out = torch.zeros(
                    N, D, device=self.device, dtype=torch.float32
                )

                # OUTLIER-ENGINE-BATCHED-GEMM-001: single-token batched path
                # Gate: OUTLIER_BATCHED=1 (default ON)
                if _batched_enabled() and N == 1 and len(used_expert_ids) >= 1:
                    _experts = [self.load_expert(i, int(e)) for e in used_expert_ids]
                    batched_out = _run_single_token_experts_batched(h_flat, _experts)  # [E, D]
                    for row_idx, e in enumerate(used_expert_ids):
                        weight = (selected_w[0] * (selected_idx[0] == e).to(selected_w.dtype)).sum()
                        expert_out[0] += float(weight) * batched_out[row_idx].to(torch.float32)
                else:
                    for e in used_expert_ids:
                        assignment = (selected_idx == e)
                        token_mask = assignment.any(dim=-1)
                        if not token_mask.any():
                            continue
                        weights = (selected_w * assignment.to(selected_w.dtype)).sum(dim=-1, keepdim=True)
                        w = self.load_expert(i, int(e))
                        out = _run_expert(h_flat[token_mask], w)
                        expert_out[token_mask] += weights[token_mask] * out

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
        self._last_routed_layer = None
        self._last_routed_experts = []

        tokens = input_ids
        kv_cache = KVCache()

        logits = self.forward(tokens, kv_cache=kv_cache, use_cache=True)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs      = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)
            logits = self.forward(next_token, kv_cache=kv_cache, use_cache=True)

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
        stats = {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "lookups": total,
            "hit_rate": hit_rate,
            "evictions": self._cache_evictions,
            "disk_loads": self._disk_loads,
            "disk_load_s": self._disk_load_s,
            "avg_disk_load_ms": (self._disk_load_s * 1000.0 / self._disk_loads) if self._disk_loads else 0.0,
        }
        if self._expert_prefetcher is not None:
            stats.update(self._expert_prefetcher.prefetch_stats)
        stats.update(self.routing_stats())
        stats.update(self.cache_prior_stats())
        return stats
