"""
OUTLIER-ENGINE-BATCHED-GEMM-001

BatchedExpertMLP: replaces per-expert kernel launches with 4 batched GEMM calls.

Current slow path (per layer, single-token decode):
  gate = F.linear(x, expert.gate_w)   ← 1 kernel per expert
  up   = F.linear(x, expert.up_w)     ← 1 kernel per expert
  act  = F.silu(gate) * up             ← 2 kernels per expert
  down = F.linear(act, expert.down_w) ← 1 kernel per expert
  → 5 kernels × n_experts per layer

Fast path (this module):
  Stack weights → 4 bmm/matmul ops for ALL experts at once
  → ~5 kernels total per layer, regardless of n_experts

Usage:
  runner = BatchedExpertMLP()
  out = runner.forward(hidden, expert_list, routing_weights)
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .paging import _ExpertWeights

_BATCHED_ENABLED = os.environ.get("OUTLIER_BATCHED", "1").strip() != "0"


def batched_enabled() -> bool:
    """Check OUTLIER_BATCHED env var (default: ON)."""
    return os.environ.get("OUTLIER_BATCHED", "1").strip() != "0"


class BatchedExpertMLP:
    """
    Runs a list of dequantized experts as a single batched GEMM.

    All experts must be dequantized (float16) and on the same device.
    Falls back to sequential if any expert violates these constraints.

    Kernel launch count: 4 per call (gate bmm, up bmm, silu+mul, down bmm)
    vs. 5 × n_experts for the sequential path.
    """

    def forward(
        self,
        hidden_state: torch.Tensor,
        expert_weights_list: List["_ExpertWeights"],
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state:       [1, hidden_dim] or [hidden_dim] — single token
            expert_weights_list: list of _ExpertWeights, all dequantized + same device
            routing_weights:    [n_experts] scalar weights (sum to ~1)

        Returns:
            [1, hidden_dim] weighted combination of all expert outputs
        """
        n = len(expert_weights_list)
        if n == 0:
            raise ValueError("expert_weights_list must be non-empty")

        # Ensure hidden is 2-D: [1, hidden_dim]
        x = hidden_state if hidden_state.dim() == 2 else hidden_state.unsqueeze(0)
        first = expert_weights_list[0]

        # Fast path: all experts dequantized on the same non-CPU device
        if (
            first.gate_w.device.type != "cpu"
            and all(
                w.dequantized and w.gate_w.device == first.gate_w.device
                for w in expert_weights_list
            )
        ):
            return self._batched_forward(x, expert_weights_list, routing_weights)

        # Slow fallback: run sequentially, then combine
        return self._sequential_forward(x, expert_weights_list, routing_weights)

    def _batched_forward(
        self,
        x: torch.Tensor,
        experts: List["_ExpertWeights"],
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """4 kernel launches for all experts."""
        first = experts[0]
        compute_dtype = first.gate_w.dtype
        x_dev = x if (x.dtype == compute_dtype and x.device == first.gate_w.device) else x.to(
            device=first.gate_w.device, dtype=compute_dtype
        )
        # x_dev: [1, H] → x_vec: [H]
        x_vec = x_dev.squeeze(0)

        # GATHER: stack weight tensors — Python loop but no GPU kernels
        # Shapes: [n_experts, intermediate_dim, hidden_dim]
        gate_w = torch.stack([e.gate_w for e in experts], dim=0)  # [E, I, H]
        up_w   = torch.stack([e.up_w   for e in experts], dim=0)  # [E, I, H]
        down_w = torch.stack([e.down_w for e in experts], dim=0)  # [E, H, I]

        # BATCHED GEMM — 4 kernel launches total
        # x_vec: [H, 1]
        x_col = x_vec.unsqueeze(-1)                                # [H, 1]
        gate_out = torch.matmul(gate_w, x_col).squeeze(-1)        # [E, I]
        up_out   = torch.matmul(up_w,   x_col).squeeze(-1)        # [E, I]
        act_out  = F.silu(gate_out) * up_out                      # [E, I]
        down_out = torch.matmul(down_w, act_out.unsqueeze(-1)).squeeze(-1)  # [E, H]

        # WEIGHTED COMBINE — vectorized, no Python loop
        # routing_weights: [E] → [E, 1]
        w = routing_weights.to(device=down_out.device, dtype=down_out.dtype).unsqueeze(-1)
        combined = (down_out * w).sum(dim=0, keepdim=True)         # [1, H]
        return combined

    def _sequential_forward(
        self,
        x: torch.Tensor,
        experts: List["_ExpertWeights"],
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: run each expert separately and accumulate."""
        out = None
        for i, expert in enumerate(experts):
            w = float(routing_weights[i])
            expert_out = expert.run(x)  # [1, H]
            if out is None:
                out = w * expert_out.to(x.dtype)
            else:
                out = out + w * expert_out.to(x.dtype)
        if out is None:
            return torch.zeros_like(x)
        return out
