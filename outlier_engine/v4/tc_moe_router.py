"""TC-MoE: Ternary-Coded Mixture-of-Experts skip routing.

Replaces standard top-k softmax routing with ternary gates in {-1, 0, +1}:
  - 0: skip the expert entirely (no compute, no memory access)
  - +1: full activation (standard positive contribution)
  - -1: subtractive activation (negative-weighted contribution)

The key advantage: experts gated to 0 are never loaded from disk, reducing
both compute and I/O.  The -1 gate enables "anti-experts" that actively
suppress certain knowledge directions, which can improve routing specificity.

The gate function uses a learned threshold per expert to decide the ternary
assignment from the router logits:
  gate(logit) = +1 if logit > +threshold
                -1 if logit < -threshold
                 0 otherwise

Reference: Inspired by ternary quantization routing concepts from TC-MoE
and skip-routing mechanisms in Switch Transformer / Expert Choice routing.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryGateFunction(torch.autograd.Function):
    """Ternary gate with straight-through estimator for gradient flow."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        gates = torch.zeros_like(logits)
        gates[logits > threshold] = 1.0
        gates[logits < -threshold] = -1.0
        ctx.save_for_backward(logits, threshold)
        return gates

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, threshold = ctx.saved_tensors
        # Straight-through: pass gradient through as if identity
        return grad_output, None


ternary_gate = TernaryGateFunction.apply


class TcMoeRouter(nn.Module):
    """Ternary-coded MoE router with configurable skip target.

    Args:
        hidden_size: input dimension
        n_experts: number of experts
        top_k: maximum experts to activate (soft cap)
        target_skip_rate: target fraction of expert slots to skip (0=no skip, 1=skip all)
        initial_threshold: starting threshold for ternary gate
    """

    def __init__(
        self,
        hidden_size: int,
        n_experts: int,
        top_k: int = 2,
        target_skip_rate: float = 0.5,
        initial_threshold: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.target_skip_rate = target_skip_rate

        self.router_weight = nn.Parameter(
            torch.empty(n_experts, hidden_size), requires_grad=False
        )
        self.threshold = nn.Parameter(
            torch.full((n_experts,), initial_threshold), requires_grad=False
        )

        # Tracking
        self._total_slots = 0
        self._skipped_slots = 0
        self._negative_slots = 0

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route input tokens to experts.

        Args:
            x: input tensor, shape (batch*seq, hidden_size)

        Returns:
            gates: ternary gate values in {-1, 0, +1}, shape (batch*seq, n_experts)
            selected_idx: indices of active (non-zero) experts per token, shape (batch*seq, top_k)
            selected_weights: weights for active experts, shape (batch*seq, top_k)
        """
        logits = F.linear(x.float(), self.router_weight.float())

        # Apply ternary gating
        gates = ternary_gate(logits, self.threshold.abs())

        # Track stats
        n_tokens = gates.shape[0]
        total = n_tokens * self.n_experts
        self._total_slots += total
        self._skipped_slots += (gates == 0).sum().item()
        self._negative_slots += (gates == -1).sum().item()

        # Select top-k non-zero experts per token
        # Use absolute gate value as priority (both +1 and -1 are important)
        abs_gates = gates.abs()

        # If all gates are zero for a token, fall back to top-k by logit magnitude
        all_zero_mask = abs_gates.sum(dim=-1) == 0
        if all_zero_mask.any():
            # Fallback: use top-k from raw logits for tokens with all-zero gates
            fallback_probs = F.softmax(logits[all_zero_mask], dim=-1)
            _, fallback_idx = torch.topk(fallback_probs, k=min(self.top_k, self.n_experts), dim=-1)
            for i, token_pos in enumerate(all_zero_mask.nonzero(as_tuple=True)[0]):
                for j in range(fallback_idx.shape[1]):
                    gates[token_pos, fallback_idx[i, j]] = 1.0

        # Get active expert indices and weights
        effective_k = min(self.top_k, (gates != 0).sum(dim=-1).max().item())
        effective_k = max(effective_k, 1)

        # Weight by gate sign × softmax probability
        probs = F.softmax(logits, dim=-1)
        weighted = gates * probs
        abs_weighted = weighted.abs()

        selected_w, selected_idx = torch.topk(abs_weighted, k=int(effective_k), dim=-1)
        # Restore sign from gates
        gate_signs = torch.gather(gates, 1, selected_idx)
        selected_w = selected_w * gate_signs.sign()
        # Renormalize weights
        weight_sum = selected_w.abs().sum(dim=-1, keepdim=True).clamp_min(1e-9)
        selected_w = selected_w / weight_sum

        return gates, selected_idx, selected_w

    @property
    def skip_rate(self) -> float:
        if self._total_slots == 0:
            return 0.0
        return self._skipped_slots / self._total_slots

    @property
    def negative_rate(self) -> float:
        if self._total_slots == 0:
            return 0.0
        return self._negative_slots / self._total_slots

    @property
    def activation_ratio(self) -> float:
        """Fraction of expert slots that are activated (non-zero)."""
        return 1.0 - self.skip_rate

    def reset_stats(self):
        self._total_slots = 0
        self._skipped_slots = 0
        self._negative_slots = 0

    @classmethod
    def from_existing_router(
        cls,
        router_weight: torch.Tensor,
        n_experts: int,
        top_k: int = 2,
        target_skip_rate: float = 0.5,
    ) -> "TcMoeRouter":
        """Create a TC-MoE router from an existing router's weight matrix."""
        hidden_size = router_weight.shape[1]
        router = cls(
            hidden_size=hidden_size,
            n_experts=n_experts,
            top_k=top_k,
            target_skip_rate=target_skip_rate,
        )
        router.router_weight.copy_(router_weight)

        # Calibrate threshold from the weight distribution
        # Set threshold so that ~target_skip_rate of activations are skipped
        with torch.no_grad():
            # Use random inputs to estimate logit distribution
            x_sample = torch.randn(256, hidden_size)
            logits = F.linear(x_sample, router_weight.float())
            abs_logits = logits.abs()
            # Set threshold at the percentile that achieves target skip rate
            threshold_val = torch.quantile(abs_logits.flatten(), target_skip_rate)
            router.threshold.fill_(threshold_val.item())

        return router
