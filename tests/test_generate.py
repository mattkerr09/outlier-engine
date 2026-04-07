from __future__ import annotations

from dataclasses import dataclass

import torch

from outlier_engine.generate import generate_text


class DummyTokenizer:
    eos_token_id = 3

    def encode(self, text: str):
        return [1, 2]

    def decode(self, token_ids):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return "".join(chr(64 + token) for token in token_ids)


class DummyModel:
    def __init__(self):
        self.device = "cpu"
        self.calls = 0

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        vocab_size = 8
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], vocab_size)
        next_id = [1, 2, 3][min(self.calls, 2)]
        logits[:, -1, next_id] = 100.0
        self.calls += 1
        return logits


@dataclass
class DummyLoaded:
    tokenizer: DummyTokenizer
    model: DummyModel
    config: dict
    device: str = "cpu"


def test_generate_text_streams_until_eos():
    loaded = DummyLoaded(
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        config={"max_seq_len": 32},
    )

    text = generate_text(loaded, "hello", max_tokens=8, temperature=0.0)
    assert text == "ABC"
