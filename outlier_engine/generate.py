from __future__ import annotations

import sys
import time
from typing import Generator, Iterable, Optional

import torch
import torch.nn.functional as F

from .loader import LoadedOutlier


def _sample_next_token(
    next_logits: torch.Tensor,
    *,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> torch.Tensor:
    if temperature == 0.0:
        return next_logits.argmax(dim=-1, keepdim=True)

    scaled = next_logits / max(temperature, 1e-5)
    probs = F.softmax(scaled, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_indices, -1, sampled)

    return torch.multinomial(probs, num_samples=1)


def stream_generate(
    loaded: LoadedOutlier,
    prompt: str,
    *,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    file=None,
) -> Generator[str, None, torch.Tensor]:
    tokenizer = loaded.tokenizer
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt encoded to an empty token list.")

    output_file = file
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.model.device if hasattr(loaded.model, "device") else loaded.device)
    tokens = input_ids
    generated_ids: list[int] = []
    previous_text = ""

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = loaded.model(tokens)
            next_logits = logits[:, -1, :]
            next_token = _sample_next_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
            )
            tokens = torch.cat([tokens, next_token], dim=1)
            generated_ids.append(int(next_token[0, 0].item()))

            current_text = tokenizer.decode(generated_ids)
            delta = current_text[len(previous_text):] if current_text.startswith(previous_text) else current_text
            previous_text = current_text

            if delta and output_file is not None:
                output_file.write(delta)
                output_file.flush()
            if delta:
                yield delta

            if getattr(tokenizer, "eos_token_id", None) is not None and generated_ids[-1] == tokenizer.eos_token_id:
                break
            if tokens.shape[1] >= loaded.config.get("max_seq_len", 4096):
                break

    return tokens


def generate_text(
    loaded: LoadedOutlier,
    prompt: str,
    *,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> str:
    chunks = list(
        stream_generate(
            loaded,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            file=None,
        )
    )
    return "".join(chunks)


def benchmark_generation(
    loaded: LoadedOutlier,
    prompt: str,
    *,
    max_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    t0 = time.perf_counter()
    text = generate_text(
        loaded,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.perf_counter() - t0
    return {
        "prompt": prompt,
        "generated_text": text,
        "elapsed_s": elapsed,
        "tokens_per_s": max_tokens / max(elapsed, 1e-6),
    }
