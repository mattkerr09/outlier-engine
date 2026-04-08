from __future__ import annotations

import sys
import threading
import time
from typing import Generator, Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import TextIteratorStreamer

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


class _TokenTextIteratorStreamer(TextIteratorStreamer):
    """Text streamer that also carries the raw generated token ids."""

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextIteratorStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        token_ids = value.tolist()
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(token_ids)
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text((printable_text, token_ids))

    def end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text((printable_text, []), stream_end=True)


def stream_generate(
    loaded: LoadedOutlier,
    prompt: str,
    *,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    file=None,
    verbose: bool = False,
    verbose_file=None,
) -> Generator[str, None, torch.Tensor]:
    tokenizer = loaded.tokenizer
    prompt_text = tokenizer.prepare_prompt(prompt) if hasattr(tokenizer, "prepare_prompt") else prompt
    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        raise ValueError("Prompt encoded to an empty token list.")

    output_file = file
    debug_file = verbose_file if verbose_file is not None else sys.stderr
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.model.device if hasattr(loaded.model, "device") else loaded.device)

    if getattr(loaded, "backend", "custom") == "hf":
        hf_tokenizer = getattr(tokenizer, "_tok", tokenizer)
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        streamer = _TokenTextIteratorStreamer(
            hf_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "streamer": streamer,
            "pad_token_id": hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
        }
        if temperature == 0.0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        error: dict[str, BaseException] = {}

        def _run_generate() -> None:
            try:
                loaded.model.generate(**generation_kwargs)
            except BaseException as exc:  # pragma: no cover - streamed exception path
                error["exception"] = exc
                streamer.end()

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        generated_tokens = 0
        for item in streamer:
            delta, token_ids = item if isinstance(item, tuple) else (item, [])
            generated_tokens += len(token_ids)
            if verbose and token_ids:
                print(
                    f"[token_ids={token_ids} delta={delta!r}]",
                    file=debug_file,
                    flush=True,
                )
            if delta and output_file is not None:
                output_file.write(delta)
                output_file.flush()
            if delta:
                yield delta

        thread.join()
        if "exception" in error:
            raise RuntimeError("HF generation failed") from error["exception"]
        return {"tokens": generated_tokens}

    tokens = input_ids
    generated_ids: list[int] = []
    previous_text = ""

    with torch.no_grad():
        for _ in range(max_tokens):
            model_out = loaded.model(tokens)
            logits = model_out.logits if hasattr(model_out, "logits") else model_out
            next_logits = logits[:, -1, :]
            next_token = _sample_next_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
            )
            tokens = torch.cat([tokens, next_token], dim=1)
            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)

            current_text = tokenizer.decode(generated_ids)
            delta = current_text[len(previous_text):] if current_text.startswith(previous_text) else current_text
            previous_text = current_text

            if verbose:
                print(
                    f"[token_id={token_id} delta={delta!r} text={current_text!r}]",
                    file=debug_file,
                    flush=True,
                )

            if delta and output_file is not None:
                output_file.write(delta)
                output_file.flush()
            if delta:
                yield delta

            if getattr(tokenizer, "eos_token_id", None) is not None and generated_ids[-1] == tokenizer.eos_token_id:
                break
            if tokens.shape[1] >= loaded.config.get("max_seq_len", 4096):
                break

    return {"tokens": len(generated_ids), "token_ids": generated_ids}


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
