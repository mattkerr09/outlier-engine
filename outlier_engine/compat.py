"""
Outlier runtime compatibility shims.

Currently hosts the transformers>=4.45 tokenizer compat patch that is required
to load Outlier-*-V3.2 repos, whose `tokenizer_config.json` ships
`extra_special_tokens` as a LIST (legacy Qwen convention). transformers 4.45+
expects a DICT and calls `.keys()` on it during
`_set_model_specific_special_tokens`, raising:

    AttributeError: 'list' object has no attribute 'keys'

Ported verbatim (idempotent) from ~/Outlier/outlier/runtime/alpha_loader.py as
part of OUTLIER-DAY17-MEGA-INTEGRATION-001 Phase 2 so the outlier-engine repo
no longer has to take an import dependency on the ~/Outlier repo to load the
tokenizer for any V3.2 model.

Call `install_tokenizer_shim()` before the first `AutoTokenizer.from_pretrained`
call. Repeated calls are safe — the patch sets a sentinel attribute.
"""
from __future__ import annotations


def install_tokenizer_shim() -> None:
    """Monkey-patch PreTrainedTokenizerBase._set_model_specific_special_tokens.

    Safe to call multiple times. No-op if transformers is unavailable.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    _orig = getattr(
        PreTrainedTokenizerBase, "_set_model_specific_special_tokens", None
    )
    if _orig is None or getattr(_orig, "_outlier_shimmed", False):
        return

    def _patched(self, special_tokens=None, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(special_tokens, list):
            current = list(getattr(self, "additional_special_tokens", []) or [])
            merged = current + [t for t in special_tokens if t not in current]
            try:
                self.additional_special_tokens = merged
            except Exception:
                pass
            self.extra_special_tokens = {}
            special_tokens = {}
        try:
            return _orig(self, special_tokens=special_tokens, **kwargs)
        except TypeError:
            return _orig(self, special_tokens)

    _patched._outlier_shimmed = True  # type: ignore[attr-defined]
    PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched  # type: ignore[assignment]


__all__ = ["install_tokenizer_shim"]
