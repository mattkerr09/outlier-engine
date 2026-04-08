#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outlier_engine.loader import load_model


LOG_PATH = ROOT / "diagnose_generation.log"


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8")

    def emit(self, message: str = "") -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a_f = a.detach().float().reshape(1, -1)
    b_f = b.detach().float().reshape(1, -1)
    cosine = F.cosine_similarity(a_f, b_f, dim=-1).item()
    max_abs = (a_f - b_f).abs().max().item()
    return cosine, max_abs


def _log_compare(logger: TeeLogger, label: str, a: torch.Tensor | None, b: torch.Tensor | None) -> None:
    if a is None or b is None:
        logger.emit(f"{label}: unavailable")
        return
    cosine, max_abs = _tensor_stats(a, b)
    logger.emit(
        f"{label}: shape={tuple(a.shape)} cosine={cosine:.6f} max_abs_diff={max_abs:.6e}"
    )


def _decode_topk(tokenizer: Any, logits: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    rows = []
    for value, index in zip(values[0].tolist(), indices[0].tolist()):
        rows.append(
            {
                "token_id": int(index),
                "text": tokenizer.decode([int(index)]),
                "logit": float(value),
            }
        )
    return rows


def _get_layers(loaded) -> Any:
    model = loaded.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Unable to locate transformer layers for loaded model.")


def _get_first_moe_module(loaded):
    for layer in _get_layers(loaded):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router_weight"):
            return layer.mlp
        if hasattr(layer, "ffn") and hasattr(layer.ffn, "router_weight"):
            return layer.ffn
    return None


def _run_model(loaded, input_ids: torch.Tensor, *, use_cache: bool = False, past_key_values=None):
    if getattr(loaded, "backend", "custom") == "hf":
        return loaded.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
    return loaded.model(input_ids)


def _capture_first_layer(loaded, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    layers = _get_layers(loaded)
    first_layer = layers[0]
    first_moe = _get_first_moe_module(loaded)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _pre(name: str):
        def inner(_module, args):
            if args and torch.is_tensor(args[0]):
                captured[name] = args[0].detach().cpu()
        return inner

    def _post(name: str):
        def inner(_module, _args, output):
            if torch.is_tensor(output):
                captured[name] = output.detach().cpu()
            elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                captured[name] = output[0].detach().cpu()
        return inner

    handles.append(first_layer.register_forward_pre_hook(_pre("layer_input")))
    handles.append(first_layer.register_forward_hook(_post("layer_output")))
    if first_moe is not None:
        handles.append(first_moe.register_forward_pre_hook(_pre("moe_input")))
        handles.append(first_moe.register_forward_hook(_post("moe_output")))

    with torch.no_grad():
        outputs = _run_model(loaded, input_ids)

    for handle in handles:
        handle.remove()

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    captured["final_logits"] = logits[:, -1, :].detach().cpu()
    return captured


def _analyze_moe(module, x: torch.Tensor) -> dict[str, torch.Tensor] | None:
    if module is None or not hasattr(module, "shared") or not hasattr(module, "router_weight"):
        return None

    x = x.detach()
    batch, seq_len, hidden_dim = x.shape
    x_flat = x.view(-1, hidden_dim)

    if hasattr(module, "page_manager"):
        compute_dtype = x_flat.dtype if x_flat.device.type != "cpu" else torch.float32
        logits = F.linear(x_flat.to(compute_dtype), module.router_weight.to(compute_dtype))
        probs = F.softmax(logits, dim=-1)
        effective_top_k = min(module.top_k, probs.shape[-1])
        top_w, top_idx = torch.topk(probs, k=effective_top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        shared_out = module.shared(x_flat.to(compute_dtype)).float()
        expert_out = torch.zeros_like(shared_out)
        used_expert_ids = [int(eid) for eid in torch.unique(top_idx).tolist()]
        for expert_idx in used_expert_ids:
            assignment = top_idx == expert_idx
            token_mask = assignment.any(dim=-1)
            if not token_mask.any():
                continue
            weights = (top_w * assignment.to(top_w.dtype)).sum(dim=-1, keepdim=True)
            expert = module.page_manager.get_expert(module.layer_idx, int(expert_idx))
            out = expert.run(x_flat[token_mask]).float()
            expert_out[token_mask] += weights[token_mask] * out
        combined = (shared_out + expert_out).view(batch, seq_len, hidden_dim)
        return {
            "moe_input": x.cpu(),
            "shared_out": shared_out.view(batch, seq_len, hidden_dim).cpu(),
            "expert_out": expert_out.view(batch, seq_len, hidden_dim).cpu(),
            "combined_out": combined.cpu(),
            "moe_output": combined.cpu(),
        }

    if hasattr(module, "experts"):
        logits = F.linear(x_flat.float(), module.router_weight.float())
        probs = F.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(probs, k=module.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        shared_out = module.shared(x_flat).float()
        expert_out = torch.zeros_like(shared_out)
        for k in range(module.top_k):
            for expert_idx in range(module.n_experts):
                mask = top_idx[:, k] == expert_idx
                if not mask.any():
                    continue
                out = module.experts[expert_idx](x_flat[mask]).float()
                expert_out[mask] += top_w[mask, k : k + 1] * out
        combined = (shared_out + expert_out).view(batch, seq_len, hidden_dim)
        return {
            "moe_input": x.cpu(),
            "shared_out": shared_out.view(batch, seq_len, hidden_dim).cpu(),
            "expert_out": expert_out.view(batch, seq_len, hidden_dim).cpu(),
            "combined_out": combined.cpu(),
            "moe_output": combined.cpu(),
        }

    return None


@contextlib.contextmanager
def _shared_only_mode(loaded):
    modules = []
    for layer in _get_layers(loaded):
        module = getattr(layer, "mlp", None)
        if module is None:
            module = getattr(layer, "ffn", None)
        if module is not None and hasattr(module, "shared") and hasattr(module, "router_weight"):
            modules.append((module, module.forward))

    if not modules:
        yield False
        return

    def _shared_only_forward(module, x):
        batch, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        compute_dtype = x_flat.dtype if x_flat.device.type != "cpu" else torch.float32
        shared_out = module.shared(x_flat.to(compute_dtype))
        return shared_out.to(x.dtype).view(batch, seq_len, hidden_dim)

    try:
        for module, _original in modules:
            module.forward = _shared_only_forward.__get__(module, module.__class__)
        yield True
    finally:
        for module, original in modules:
            module.forward = original


def _teacher_forcing_probe(loaded, prompt_ids: list[int], teacher_token: int) -> dict[str, Any]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        first = _run_model(loaded, input_ids, use_cache=True)
        first_logits = first.logits[:, -1, :].detach().cpu() if hasattr(first, "logits") else first[:, -1, :].detach().cpu()
        if hasattr(first, "past_key_values") and first.past_key_values is not None:
            second = _run_model(
                loaded,
                torch.tensor([[teacher_token]], dtype=torch.long),
                use_cache=True,
                past_key_values=first.past_key_values,
            )
        else:
            second_ids = torch.tensor([prompt_ids + [teacher_token]], dtype=torch.long)
            second = _run_model(loaded, second_ids, use_cache=False)
        second_logits = second.logits[:, -1, :].detach().cpu() if hasattr(second, "logits") else second[:, -1, :].detach().cpu()
    return {
        "position_0_top5": _decode_topk(loaded.tokenizer, first_logits),
        "position_1_top5": _decode_topk(loaded.tokenizer, second_logits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose paged vs non-paged generation divergence.")
    parser.add_argument("--model", default="Outlier-Ai/Outlier-10B")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logger = TeeLogger(LOG_PATH)
    try:
        logger.emit("Diagnose paged generation")
        logger.emit(f"model={args.model}")
        logger.emit(f"device={args.device}")
        logger.emit(f"prompt={args.prompt!r}")
        logger.emit("")

        non_paged = load_model(args.model, paged=False, device=args.device)
        paged = load_model(args.model, paged=True, device=args.device)

        logger.emit("Resolved models")
        for label, loaded in (("nonpaged", non_paged), ("paged", paged)):
            logger.emit(
                f"{label}: model_ref={loaded.model_ref} model_dir={loaded.model_dir} "
                f"backend={loaded.backend} paged={loaded.paged} "
                f"model_type={loaded.config.get('model_type')} n_experts={loaded.config.get('n_experts', 0)}"
            )
        logger.emit(
            f"same_model_dir={Path(non_paged.model_dir) == Path(paged.model_dir)} "
            f"same_model_ref={non_paged.model_ref == paged.model_ref}"
        )
        logger.emit("")

        tokenizer = non_paged.tokenizer
        prompt_text = tokenizer.prepare_prompt(args.prompt) if hasattr(tokenizer, "prepare_prompt") else args.prompt
        prompt_ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        logger.emit("Task 1: one forward pass")
        non_paged_capture = _capture_first_layer(non_paged, input_ids)
        paged_capture = _capture_first_layer(paged, input_ids)
        _log_compare(logger, "hidden_before_first_layer", non_paged_capture.get("layer_input"), paged_capture.get("layer_input"))
        _log_compare(logger, "hidden_after_first_layer", non_paged_capture.get("layer_output"), paged_capture.get("layer_output"))
        _log_compare(logger, "moe_input", non_paged_capture.get("moe_input"), paged_capture.get("moe_input"))

        non_paged_moe = _analyze_moe(_get_first_moe_module(non_paged), non_paged_capture.get("moe_input")) if non_paged_capture.get("moe_input") is not None else None
        paged_moe = _analyze_moe(_get_first_moe_module(paged), paged_capture.get("moe_input")) if paged_capture.get("moe_input") is not None else None
        if non_paged_moe is None or paged_moe is None:
            logger.emit("shared_out/expert_out/combined_out: unavailable (one or both resolved models do not expose a comparable MoE block)")
        else:
            _log_compare(logger, "shared_out", non_paged_moe.get("shared_out"), paged_moe.get("shared_out"))
            _log_compare(logger, "expert_out", non_paged_moe.get("expert_out"), paged_moe.get("expert_out"))
            _log_compare(logger, "combined_out", non_paged_moe.get("combined_out"), paged_moe.get("combined_out"))
            _log_compare(logger, "hidden_after_moe", non_paged_moe.get("moe_output"), paged_moe.get("moe_output"))

        _log_compare(logger, "final_logits", non_paged_capture.get("final_logits"), paged_capture.get("final_logits"))
        logger.emit(f"nonpaged_top5={json.dumps(_decode_topk(tokenizer, non_paged_capture['final_logits']), ensure_ascii=False)}")
        logger.emit(f"paged_top5={json.dumps(_decode_topk(tokenizer, paged_capture['final_logits']), ensure_ascii=False)}")
        logger.emit("")

        logger.emit("Task 2: shared-only paged probe")
        with _shared_only_mode(paged) as enabled:
            if not enabled:
                logger.emit("shared_only_probe: skipped (resolved paged model has no MoE experts to disable)")
            else:
                with torch.no_grad():
                    shared_only = _run_model(paged, input_ids)
                    shared_only_logits = shared_only.logits[:, -1, :].detach().cpu() if hasattr(shared_only, "logits") else shared_only[:, -1, :].detach().cpu()
                logger.emit(f"shared_only_top5={json.dumps(_decode_topk(tokenizer, shared_only_logits), ensure_ascii=False)}")
        logger.emit("")

        logger.emit("Task 3: teacher forcing probe")
        teacher_token = int(non_paged_capture["final_logits"].argmax(dim=-1).item())
        logger.emit(f"teacher_token_id={teacher_token} teacher_token_text={tokenizer.decode([teacher_token])!r}")
        logger.emit(
            f"nonpaged_teacher_forcing={json.dumps(_teacher_forcing_probe(non_paged, prompt_ids, teacher_token), ensure_ascii=False)}"
        )
        logger.emit(
            f"paged_teacher_forcing={json.dumps(_teacher_forcing_probe(paged, prompt_ids, teacher_token), ensure_ascii=False)}"
        )
        logger.emit("")

        logger.emit("Diagnosis summary")
        if Path(non_paged.model_dir) != Path(paged.model_dir):
            logger.emit(
                "Root cause candidate: paged and non-paged resolved to different checkpoint directories, so the previous comparison was not apples-to-apples."
            )
        elif _get_first_moe_module(paged) is None:
            logger.emit(
                "Paged request fell back to the canonical dense model, so there is no expert-combination bug on the active public path."
            )
        else:
            logger.emit(
                "Paged and non-paged resolved to the same checkpoint. Inspect the tensor comparisons above for the first divergence point."
            )
        logger.emit(f"log_path={LOG_PATH}")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
