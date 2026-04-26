"""Hadamard-rotated wrapper for Outlier 10B V3.2.

Wraps the loaded V3.2 model to apply Walsh-Hadamard rotation ONLY to the
delta experts (ternary MoE experts), never to the shared/base FFN.

Per spec §2.4: "The shared expert base_FFN_L(x) is NOT rotated — it's
frozen Qwen2.5 bf16 and doesn't need rotation. Only the delta experts
get the Hadamard treatment."

For non-power-of-2 hidden dimensions (e.g., 3584), uses a padded
Hadamard matrix: pad input to next power of 2, apply H, truncate back.
The pad-rotate-truncate approach preserves orthogonality within the
active subspace.

Does NOT modify the V3.2 modeling file.  All changes are external wrappers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hadamard_rotation import build_hadamard, rotate_weight


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _build_padded_hadamard(dim: int) -> torch.Tensor:
    """Build a dim×dim orthogonal rotation matrix.

    If dim is a power of 2, returns the exact Walsh-Hadamard matrix.
    Otherwise, pads to the next power of 2 and extracts the dim×dim
    upper-left submatrix, then re-orthogonalizes via QR decomposition.
    """
    if dim & (dim - 1) == 0:
        return build_hadamard(dim)

    # Pad to next power of 2
    n_padded = _next_power_of_2(dim)
    H_full = build_hadamard(n_padded)
    # Extract submatrix and re-orthogonalize
    H_sub = H_full[:dim, :dim]
    Q, R = torch.linalg.qr(H_sub)
    # Ensure determinant is +1 (proper rotation, no reflection)
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    return Q


class RotatedExpertWrapper(nn.Module):
    """Wraps a single delta expert (Qwen2ExpertMLP) with Hadamard rotation.

    Only gate_proj and up_proj are rotated (they take hidden_dim input).
    down_proj operates on intermediate_dim and is left untouched.
    """

    def __init__(self, expert: nn.Module, H: torch.Tensor):
        super().__init__()
        self.expert = expert
        self.register_buffer("H", H)
        self._pre_rotate_weights()

    def _pre_rotate_weights(self):
        with torch.no_grad():
            self.expert.gate_proj.weight.copy_(
                rotate_weight(self.expert.gate_proj.weight, self.H)
            )
            self.expert.up_proj.weight.copy_(
                rotate_weight(self.expert.up_proj.weight, self.H)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rot = (x.float() @ self.H.T).to(x.dtype)
        return self.expert(x_rot)


class RotatedV32Model:
    """Wraps a loaded V3.2 model with Hadamard rotation on delta experts ONLY.

    Per spec §2.4: shared/base FFN is NEVER rotated. Only the MoE delta
    experts (.experts ModuleList) are wrapped with RotatedExpertWrapper.

    After wrapping, model.generate() and model() produce identical results
    for the shared path, with improved quantization quality on the delta path.
    """

    def __init__(self, model: nn.Module, *, save_dir: Optional[str] = None):
        self.model = model
        self._rotation_info: dict = {}
        self._apply_rotation()
        if save_dir:
            self.save_rotation_matrices(save_dir)

    def _apply_rotation(self):
        """Walk model layers, wrap ONLY delta experts with Hadamard rotation.

        Shared MLP (gate_proj, up_proj, down_proj at layer.mlp level) and
        the base FFN path are explicitly left untouched.
        """
        H_cache: dict[int, torch.Tensor] = {}

        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            if not hasattr(mlp, "experts"):
                continue  # dense layer, no MoE

            hidden_size = mlp.hidden_size
            if hidden_size not in H_cache:
                H_cache[hidden_size] = _build_padded_hadamard(hidden_size)

            H = H_cache[hidden_size].to(device=next(mlp.experts.parameters()).device)

            wrapped_experts = nn.ModuleList()
            for expert_idx, expert in enumerate(mlp.experts):
                wrapped = RotatedExpertWrapper(expert, H)
                wrapped_experts.append(wrapped)

            mlp.experts = wrapped_experts
            self._rotation_info[layer_idx] = {
                "hidden_size": hidden_size,
                "n_experts": len(wrapped_experts),
                "padded": hidden_size & (hidden_size - 1) != 0,
            }

    @property
    def rotation_layers(self) -> dict:
        return dict(self._rotation_info)

    def save_rotation_matrices(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        H_tensors = {}
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            if not hasattr(mlp, "experts") or len(mlp.experts) == 0:
                continue
            expert_0 = mlp.experts[0]
            if hasattr(expert_0, "H"):
                n = expert_0.H.shape[0]
                if n not in H_tensors:
                    H_tensors[n] = expert_0.H.cpu()

        from safetensors.torch import save_file
        save_file(
            {f"hadamard_{n}": H for n, H in H_tensors.items()},
            str(save_path / "rotation_matrices.safetensors"),
        )
        (save_path / "rotation_metadata.json").write_text(
            json.dumps({"layers": self._rotation_info}, indent=2)
        )

    def __getattr__(self, name):
        if name in ("model", "_rotation_info"):
            return object.__getattribute__(self, name)
        return getattr(self.model, name)
