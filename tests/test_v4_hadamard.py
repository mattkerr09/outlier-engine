"""Tests for Walsh-Hadamard pre-rotation (V4 Phase 1)."""

import torch
import pytest
from outlier_engine.v4.hadamard_rotation import (
    build_hadamard,
    fast_hadamard_transform,
    rotate_weight,
    rotate_input,
    quantize_ternary,
    quantization_error,
)


class TestBuildHadamard:
    def test_identity_property(self):
        """H @ H^T must equal identity (to 1e-5) for normalized Hadamard."""
        for n in [2, 4, 8, 16, 32, 64, 128, 256]:
            H = build_hadamard(n)
            product = H @ H.T
            identity = torch.eye(n)
            assert torch.allclose(product, identity, atol=1e-5), (
                f"H @ H^T != I for n={n}, max error={torch.abs(product - identity).max():.2e}"
            )

    def test_non_power_of_2_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            build_hadamard(3)
        with pytest.raises(ValueError, match="power of 2"):
            build_hadamard(6)

    def test_unnormalized(self):
        H = build_hadamard(4, normalize=False)
        # Unnormalized: H @ H^T = n * I
        product = H @ H.T
        assert torch.allclose(product, 4.0 * torch.eye(4), atol=1e-5)

    def test_values_are_plus_minus(self):
        """Unnormalized Hadamard entries must be exactly +1 or -1."""
        H = build_hadamard(8, normalize=False)
        assert torch.all((H == 1.0) | (H == -1.0))


class TestRotationRoundTrip:
    def test_rotate_then_unrotate_roundtrips(self):
        """Rotating and unrotating a tensor must preserve it exactly."""
        torch.manual_seed(42)
        for n in [16, 64, 256]:
            H = build_hadamard(n)
            W = torch.randn(n, n)
            W_rot = rotate_weight(W, H)
            W_back = rotate_weight(W_rot, H.T)  # unrotate: (W @ H^T) @ H = W
            assert torch.allclose(W, W_back, atol=1e-4), (
                f"Round-trip failed for n={n}, max error={torch.abs(W - W_back).max():.2e}"
            )

    def test_input_rotation_cancels(self):
        """y = W @ x must equal W_rot @ x_rot for rotated weight+input."""
        torch.manual_seed(42)
        n = 64
        H = build_hadamard(n)
        W = torch.randn(n, n)
        x = torch.randn(1, n)

        y_original = x @ W.T  # standard linear
        W_rot = rotate_weight(W, H)
        x_rot = rotate_input(x, H)
        y_rotated = x_rot @ W_rot.T

        assert torch.allclose(y_original, y_rotated, atol=1e-4), (
            f"Rotation cancellation failed, max error={torch.abs(y_original - y_rotated).max():.2e}"
        )

    def test_batch_input_rotation(self):
        """Rotation must work on batched inputs (..., n)."""
        torch.manual_seed(42)
        n = 32
        H = build_hadamard(n)
        x = torch.randn(4, 8, n)
        x_rot = rotate_input(x, H)
        x_back = rotate_input(x_rot, H.T)
        assert torch.allclose(x, x_back, atol=1e-4)


class TestQuantizationImprovement:
    def test_rotated_quantization_reduces_error(self):
        """Rotated-then-quantized must have lower L2 error than unrotated
        for a deliberately outlier-heavy weight tensor."""
        torch.manual_seed(42)
        n = 256
        # Create a tensor with heavy outliers (one column has 10x magnitude)
        W = torch.randn(n, n) * 0.1
        W[:, 0] = torch.randn(n) * 10.0  # extreme outlier column
        W[:, 1] = torch.randn(n) * 5.0   # moderate outlier

        # Quantize without rotation
        q_plain, s_plain = quantize_ternary(W)
        err_plain = quantization_error(W, q_plain, s_plain)

        # Quantize with rotation
        H = build_hadamard(n)
        W_rot = rotate_weight(W, H)
        q_rot, s_rot = quantize_ternary(W_rot)
        err_rot = quantization_error(W_rot, q_rot, s_rot)

        assert err_rot < err_plain, (
            f"Rotation did not reduce quantization error: "
            f"rotated={err_rot:.4f} >= plain={err_plain:.4f}"
        )

    def test_uniform_weights_rotation_neutral(self):
        """For already-uniform weights, rotation should not significantly
        increase quantization error (within 10%)."""
        torch.manual_seed(42)
        n = 128
        W = torch.randn(n, n)  # already approximately uniform

        q_plain, s_plain = quantize_ternary(W)
        err_plain = quantization_error(W, q_plain, s_plain)

        H = build_hadamard(n)
        W_rot = rotate_weight(W, H)
        q_rot, s_rot = quantize_ternary(W_rot)
        err_rot = quantization_error(W_rot, q_rot, s_rot)

        # Allow 10% tolerance — rotation shouldn't make uniform weights worse
        assert err_rot < err_plain * 1.1, (
            f"Rotation hurt uniform weights too much: "
            f"rotated={err_rot:.4f} vs plain={err_plain:.4f}"
        )


class TestFastHadamardTransform:
    def test_matches_matrix_multiply(self):
        """Fast transform must match explicit H @ x."""
        torch.manual_seed(42)
        for n in [4, 16, 64]:
            H = build_hadamard(n)
            x = torch.randn(n)
            y_matrix = H @ x
            y_fast = fast_hadamard_transform(x.unsqueeze(0)).squeeze(0)
            assert torch.allclose(y_matrix, y_fast, atol=1e-4), (
                f"Fast transform != matrix for n={n}, "
                f"max error={torch.abs(y_matrix - y_fast).max():.2e}"
            )
