from __future__ import annotations


def metal_available() -> bool:
    try:
        import mlx.core as mx  # noqa: F401
    except Exception:
        return False
    return True
