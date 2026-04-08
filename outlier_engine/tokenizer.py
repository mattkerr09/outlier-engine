"""
Outlier Runtime v0.1 — tokenizer wrapper.

Thin wrapper around a HuggingFace AutoTokenizer that provides the
encode / decode interface used by the runtime.
"""

from __future__ import annotations

from typing import List, Union


class OutlierTokenizer:
    """Wraps a HuggingFace tokenizer with a minimal encode/decode API."""

    def __init__(self, hf_tokenizer) -> None:
        self._tok = hf_tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token ids (no special tokens added)."""
        return self._tok.encode(text, add_special_tokens=False)

    def prepare_prompt(self, text: str) -> str:
        """
        Format chat/instruct prompts when the tokenizer exposes a chat template.

        Raw special-token prompts are passed through unchanged so power users can
        still provide an explicit template by hand.
        """
        if "<|im_start|>" in text or "<|im_end|>" in text:
            return text
        if hasattr(self._tok, "apply_chat_template") and getattr(self._tok, "chat_template", None):
            messages = [{"role": "user", "content": text}]
            return self._tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return text

    def decode(self, token_ids: Union[List[int], "torch.Tensor"]) -> str:  # type: ignore[name-defined]
        """Decode token ids back to text, skipping special tokens."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            # Batch: decode first item
            token_ids = token_ids[0]
        return self._tok.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eos_token_id(self):
        return self._tok.eos_token_id

    @property
    def bos_token_id(self):
        return self._tok.bos_token_id

    # Allow direct access to the underlying HF tokenizer
    def __getattr__(self, name: str):
        return getattr(self._tok, name)


def load_tokenizer(repo_id: str, token: str = None) -> OutlierTokenizer:
    """
    Load the tokenizer for an Outlier model.

    Args:
        repo_id: HuggingFace repo ID (e.g. "Outlier-AI/Outlier-7B-v0")
                 or path to a local directory containing tokenizer files.
        token:   Optional HuggingFace auth token.

    Returns:
        OutlierTokenizer wrapping the loaded HF tokenizer.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        repo_id,
        token=token,
        trust_remote_code=False,
    )
    return OutlierTokenizer(tok)
