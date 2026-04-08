#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outlier_engine.loader import load_model


def _default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> None:
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class TokenBenchResult:
    latencies: list[float]
    text: list[str]
    cache_stats: dict[str, Any]


class ExpertMathModule(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        return F.linear(F.silu(gate) * up, down_w)


class FusedExpertMLP(nn.Module):
    def __init__(self, expert) -> None:
        super().__init__()
        self.register_buffer("gate_w", expert.gate_w.contiguous())
        self.register_buffer("up_w", expert.up_w.contiguous())
        self.register_buffer("down_w", expert.down_w.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_w)
        up = F.linear(x, self.up_w)
        return F.linear(F.silu(gate) * up, self.down_w)


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(prompt)


def _warm_model(loaded, prompt: str = "Hi", warm_tokens: int = 3):
    tokenizer = loaded.tokenizer
    prompt_ids = _encode_prompt(tokenizer, prompt)
    model = loaded.model
    device = getattr(model, "outlier_device", loaded.device)
    device_name = device.type if isinstance(device, torch.device) else str(device)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device_name)
    attention_mask = torch.ones_like(generated, device=generated.device)
    past_key_values = None
    target_layer = model.model.layers[-1]
    captured_inputs: list[torch.Tensor] = []

    def _capture(_module, args):
        captured_inputs.append(args[0].detach())

    handle = target_layer.mlp.register_forward_pre_hook(_capture)
    for _ in range(warm_tokens):
        with torch.no_grad():
            step_input = generated if past_key_values is None else generated[:, -1:]
            outputs = model(
                input_ids=step_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            _sync(device_name)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            next_token = int(logits[0, -1].argmax().item())
            generated = torch.cat(
                [generated, torch.tensor([[next_token]], dtype=torch.long, device=generated.device)],
                dim=1,
            )
            attention_mask = torch.ones_like(generated, device=generated.device)
            past_key_values = getattr(outputs, "past_key_values", None)
    handle.remove()
    if not captured_inputs:
        raise RuntimeError("Failed to capture MLP input during warmup.")
    return generated, past_key_values, attention_mask, device_name, captured_inputs[-1]


def _find_hot_expert(loaded):
    page_manager = loaded.model.outlier_page_manager
    if not page_manager._hot_cache:
        raise RuntimeError("Hot cache is empty after warmup.")
    (layer_idx, expert_idx), expert = next(reversed(page_manager._hot_cache.items()))
    return layer_idx, expert_idx, expert


def _bench_module(module: nn.Module, x: torch.Tensor, *, label: str, device: str, steps: int = 100) -> float:
    with torch.no_grad():
        for _ in range(10):
            _ = module(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = module(x)
        _sync(device)
    ms = (time.perf_counter() - t0) * 1000.0 / steps
    print(f"{label}_ms={ms:.3f}")
    return ms


def _bench_function(fn, *, label: str, device: str, steps: int = 100) -> float:
    with torch.no_grad():
        for _ in range(10):
            _ = fn()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = fn()
        _sync(device)
    ms = (time.perf_counter() - t0) * 1000.0 / steps
    print(f"{label}_ms={ms:.3f}")
    return ms


def _run_token_bench(loaded, *, prompt: str = "Hi", tokens: int = 5) -> TokenBenchResult:
    tokenizer = loaded.tokenizer
    prompt_ids = _encode_prompt(tokenizer, prompt)
    model = loaded.model
    device = getattr(model, "outlier_device", loaded.device)
    device_name = device.type if isinstance(device, torch.device) else str(device)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device_name)
    attention_mask = torch.ones_like(generated, device=generated.device)
    past_key_values = None
    latencies: list[float] = []
    text: list[str] = []
    cache_stats: dict[str, Any] = {}
    with torch.no_grad():
        for _ in range(tokens):
            step_input = generated if past_key_values is None else generated[:, -1:]
            t0 = time.perf_counter()
            outputs = model(
                input_ids=step_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            _sync(device_name)
            latencies.append(time.perf_counter() - t0)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            next_token = int(logits[0, -1].argmax().item())
            text.append(tokenizer.decode([next_token]))
            generated = torch.cat(
                [generated, torch.tensor([[next_token]], dtype=torch.long, device=generated.device)],
                dim=1,
            )
            attention_mask = torch.ones_like(generated, device=generated.device)
            past_key_values = getattr(outputs, "past_key_values", None)
            if hasattr(model, "cache_stats"):
                cache_stats = model.cache_stats()
    return TokenBenchResult(latencies=latencies, text=text, cache_stats=cache_stats)


def _bench_mlp_module(module: nn.Module, x: torch.Tensor, *, label: str, device: str, steps: int = 10) -> float:
    with torch.no_grad():
        for _ in range(3):
            _ = module(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = module(x)
        _sync(device)
    ms = (time.perf_counter() - t0) * 1000.0 / steps
    print(f"{label}_ms={ms:.3f}", flush=True)
    return ms


def main() -> int:
    device = _default_device()
    print(f"device={device}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"has_compile={hasattr(torch, 'compile')}", flush=True)

    load_t0 = time.perf_counter()
    loaded = load_model(
        "Outlier-Ai/Outlier-10B",
        paged=True,
        device=device,
        max_experts_in_memory=20,
        max_warm_cache=64,
    )
    _sync(device)
    print(f"load_time_s={time.perf_counter() - load_t0:.2f}", flush=True)

    generated, past_key_values, attention_mask, device_name, mlp_input = _warm_model(loaded, warm_tokens=3)
    print(f"warmup_device={device_name}", flush=True)
    print(f"prompt_plus_warm_tokens={generated.shape[1]}", flush=True)
    print(f"cache_after_warmup={loaded.model.cache_stats() if hasattr(loaded.model, 'cache_stats') else {}}", flush=True)
    print(f"captured_mlp_input_shape={tuple(mlp_input.shape)}", flush=True)
    print(f"captured_mlp_input_dtype={mlp_input.dtype}", flush=True)
    print(f"captured_mlp_input_device={mlp_input.device}", flush=True)

    layer_idx, expert_idx, expert = _find_hot_expert(loaded)
    x = torch.randn(1, expert.gate_w.shape[1], device=expert.gate_w.device, dtype=expert.gate_w.dtype)
    print(f"hot_expert_layer={layer_idx}", flush=True)
    print(f"hot_expert_idx={expert_idx}", flush=True)
    print(f"expert_weight_device={expert.gate_w.device}", flush=True)
    print(f"expert_weight_dtype={expert.gate_w.dtype}", flush=True)
    print(f"hidden_dtype={x.dtype}", flush=True)
    print(f"hidden_device={x.device}", flush=True)

    math_mod = ExpertMathModule().to(device=x.device).eval()
    fused_mod = FusedExpertMLP(expert).to(device=x.device).eval()

    _bench_function(
        lambda: math_mod(x, expert.gate_w, expert.up_w, expert.down_w),
        label="expert_math_eager",
        device=device_name,
    )
    _bench_module(fused_mod, x, label="fused_module_eager", device=device_name)

    if hasattr(torch, "compile"):
        try:
            compiled_math = torch.compile(math_mod, mode="reduce-overhead")
            _bench_function(
                lambda: compiled_math(x, expert.gate_w, expert.up_w, expert.down_w),
                label="expert_math_compile",
                device=device_name,
            )
        except Exception as exc:
            print(f"expert_math_compile_error={type(exc).__name__}: {exc}")

        try:
            compiled_fused = torch.compile(fused_mod, mode="reduce-overhead")
            _bench_module(compiled_fused, x, label="fused_module_compile", device=device_name)
        except Exception as exc:
            print(f"fused_module_compile_error={type(exc).__name__}: {exc}")

    try:
        traced_math = torch.jit.trace(fused_mod, x)
        _bench_module(traced_math, x, label="fused_module_jit_trace", device=device_name)
    except Exception as exc:
        print(f"fused_module_jit_trace_error={type(exc).__name__}: {exc}")

    model = loaded.model
    target_layer = model.model.layers[-1]
    _bench_mlp_module(target_layer.mlp, mlp_input, label="paged_mlp_eager", device=device_name)

    full_compile_error = None
    compiled_token_bench = None
    if hasattr(torch, "compile"):
        try:
            compiled_mlp = torch.compile(target_layer.mlp, mode="reduce-overhead")
            _bench_mlp_module(compiled_mlp, mlp_input, label="paged_mlp_compile", device=device_name)
        except Exception as exc:
            full_compile_error = exc
            print(f"paged_mlp_compile_error={type(exc).__name__}: {exc}")

    if compiled_token_bench is not None:
        token3to5 = compiled_token_bench.latencies[2:]
        token3to5_avg = sum(token3to5) / len(token3to5) if token3to5 else 0.0
        print(f"paged_mlp_compile_token_latencies_s={[round(v, 3) for v in compiled_token_bench.latencies]}")
        print(f"paged_mlp_compile_token_3_to_5_avg_s={token3to5_avg:.3f}")
        print(f"paged_mlp_compile_text={''.join(compiled_token_bench.text)!r}")
        print(f"paged_mlp_compile_cache_stats={compiled_token_bench.cache_stats}")
    elif full_compile_error is not None:
        print("paged_mlp_compile_result=failed")
    else:
        print("paged_mlp_compile_result=fixed_input_only")

    print("baseline_reference_token_3_to_5_avg_s=0.80")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
