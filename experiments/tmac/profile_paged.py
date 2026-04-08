#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outlier_engine.loader import load_model


def sync(device: str) -> None:
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def main() -> int:
    device = default_device()
    loaded = load_model("Outlier-Ai/Outlier-10B", paged=True, device=device)
    tokenizer = loaded.tokenizer
    prompt_text = tokenizer.prepare_prompt("Hello") if hasattr(tokenizer, "prepare_prompt") else "Hello"
    prompt_ids = tokenizer.encode(prompt_text)
    model = loaded.model
    model_device = getattr(model, "outlier_device", loaded.device)
    device_name = model_device.type if isinstance(model_device, torch.device) else str(model_device)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device_name)
    attention_mask = torch.ones_like(input_ids)
    past_key_values = None
    generated = input_ids

    target_layer_idx = len(model.model.layers) - 1
    target_layer = model.model.layers[target_layer_idx]
    captures: list[torch.Tensor] = []

    def capture_hook(_module, args):
        x = args[0]
        captures.append(x.detach())

    handle = target_layer.mlp.register_forward_pre_hook(capture_hook)

    warm_timings: list[dict[str, float]] = []
    warm_meta: dict[str, str] = {}

    try:
        with torch.no_grad():
            for step in range(5):
                captures.clear()
                step_input = generated if past_key_values is None else generated[:, -1:]
                outputs = model(
                    input_ids=step_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                sync(device_name)
                logits = outputs.logits
                next_token = int(logits[0, -1].argmax().item())
                generated = torch.cat(
                    [generated, torch.tensor([[next_token]], dtype=torch.long, device=generated.device)],
                    dim=1,
                )
                attention_mask = torch.ones_like(generated)
                past_key_values = getattr(outputs, "past_key_values", None)

                if step < 2 or not captures:
                    continue

                x = captures[-1].view(-1, captures[-1].shape[-1])[-1:].contiguous()
                mlp = target_layer.mlp
                compute_dtype = x.dtype if x.device.type != "cpu" else torch.float32
                logits_router = F.linear(x.to(compute_dtype), mlp.router_weight.to(compute_dtype))
                probs = F.softmax(logits_router, dim=-1)
                top_w, top_idx = torch.topk(probs, k=mlp.top_k, dim=-1)
                expert_idx = int(top_idx[0, 0].item())
                route_weight = float(top_w[0, 0].item())
                page_manager = model.outlier_page_manager

                expert, lookup_s = timed(lambda: page_manager.get_expert(target_layer_idx, expert_idx))
                sync(device_name)

                x_dev = x if (x.dtype == expert.gate_w.dtype and x.device == expert.gate_w.device) else x.to(
                    device=expert.gate_w.device, dtype=expert.gate_w.dtype
                )
                warm_meta = {
                    "device": str(device_name),
                    "hidden_dtype": str(x_dev.dtype),
                    "hidden_device": str(x_dev.device),
                    "weight_dtype": str(expert.gate_w.dtype),
                    "weight_device": str(expert.gate_w.device),
                    "layer_idx": str(target_layer_idx),
                    "expert_idx": str(expert_idx),
                }

                gate_raw, gate_s = timed(lambda: F.linear(x_dev, expert.gate_w))
                sync(device_name)
                up, up_s = timed(lambda: F.linear(x_dev, expert.up_w))
                sync(device_name)
                gate, act_s = timed(lambda: F.silu(gate_raw))
                sync(device_name)
                expert_out, down_s = timed(lambda: F.linear(gate * up, expert.down_w))
                sync(device_name)
                _, combine_s = timed(lambda: expert_out * route_weight)
                sync(device_name)

                warm_timings.append(
                    {
                        "lookup_ms": lookup_s * 1000.0,
                        "gate_ms": gate_s * 1000.0,
                        "up_ms": up_s * 1000.0,
                        "activation_ms": act_s * 1000.0,
                        "down_ms": down_s * 1000.0,
                        "combine_ms": combine_s * 1000.0,
                    }
                )
    finally:
        handle.remove()

    if not warm_timings:
        raise SystemExit("No warm-token timings collected.")

    print("profiled_tokens=3")
    for key, value in warm_meta.items():
        print(f"{key}={value}")
    for name in ("lookup_ms", "gate_ms", "up_ms", "activation_ms", "down_ms", "combine_ms"):
        values = [row[name] for row in warm_timings]
        print(f"{name}_avg={mean(values):.3f}")
        print(f"{name}_samples={[round(v, 3) for v in values]}")
    total_ms = sum(mean([row[name] for row in warm_timings]) for name in ("lookup_ms", "gate_ms", "up_ms", "activation_ms", "down_ms", "combine_ms"))
    print(f"warm_expert_total_ms_avg={total_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
