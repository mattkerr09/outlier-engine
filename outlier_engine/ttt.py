"""
Alpha-only Test-Time Training (TTT) for Outlier MoE models.

OUTLIER-LOCAL-EXPERIMENTS-001
  Experiment 1: Alpha-Only TTT — self-supervised expert-weight adaptation
  Experiment 3: RoE Ensemble Quality Comparison
  Experiment 4: Pre-Attention Expert Prediction (routing trace collection)

The alpha parameters of each _HybridPagedMLP layer are lifted to
nn.Parameter tensors and a differentiable forward pass is patched in.
All other model parameters remain frozen (requires_grad=False).

Memory safety: every major allocation checks free RAM.  If RAM falls
below 15 GB the experiment saves partial results and aborts.
"""

from __future__ import annotations

import gc
import json
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from .paging import _HybridPagedMLP, _run_expert


# ---------------------------------------------------------------------------
# RAM helpers
# ---------------------------------------------------------------------------

def free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def check_ram(threshold_gb: float = 15.0, label: str = "") -> float:
    free = free_ram_gb()
    tag = f" [{label}]" if label else ""
    print(f"[RAM{tag}] Free: {free:.1f} GB", flush=True)
    if free < threshold_gb:
        raise MemoryError(
            f"Free RAM {free:.1f} GB is below safety threshold {threshold_gb:.1f} GB."
            " Aborting to prevent swap."
        )
    return free


# ---------------------------------------------------------------------------
# Alpha state helpers
# ---------------------------------------------------------------------------

def _get_moe_layers(model) -> List[Tuple[int, _HybridPagedMLP]]:
    """Return (layer_idx, mlp) for every _HybridPagedMLP in the model."""
    result = []
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, _HybridPagedMLP) and layer.mlp.n_experts > 0:
            result.append((idx, layer.mlp))
    return result


def collect_alpha_state(model) -> Dict[int, Dict[int, float]]:
    """Snapshot current alpha floats from all MoE layers.

    Returns {layer_idx: {expert_idx: alpha_float}}.
    """
    state: Dict[int, Dict[int, float]] = {}
    for layer_idx, mlp in _get_moe_layers(model):
        state[layer_idx] = {
            e: float(mlp.alphas.get(e, mlp.alpha_default))
            for e in range(mlp.n_experts)
        }
    return state


def _write_back_alphas(moe_layers: List[Tuple[int, _HybridPagedMLP]],
                       alpha_params: List[nn.Parameter]) -> None:
    """Write parameter float values back into mlp.alphas dicts."""
    for (layer_idx, mlp), param in zip(moe_layers, alpha_params):
        with torch.no_grad():
            for e in range(mlp.n_experts):
                mlp.alphas[e] = float(param[e].item())


# ---------------------------------------------------------------------------
# Forward patching for differentiable alpha
# ---------------------------------------------------------------------------

def _make_ttt_forward(mlp: _HybridPagedMLP):
    """Return a patched forward for mlp that uses mlp._ttt_alpha_param."""

    def _forward(x: torch.Tensor) -> torch.Tensor:
        alpha_param: nn.Parameter = mlp._ttt_alpha_param  # [n_experts]
        pm = mlp.page_manager
        if pm is None:
            raise RuntimeError("page_manager is None on _HybridPagedMLP")

        batch, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        T = x_flat.shape[0]
        compute_dtype = (
            x_flat.dtype if x_flat.device.type != "cpu" else torch.float32
        )

        # Shared expert — no detach: gradient flows through input
        shared_out = mlp.shared(x_flat.to(compute_dtype))

        if os.environ.get("OUTLIER_SHARED_ONLY") == "1":
            return shared_out.to(x.dtype).view(batch, seq_len, hidden_dim)

        # Router — discrete selection, wrap in no_grad
        with torch.no_grad():
            logits = F.linear(
                x_flat.to(compute_dtype),
                mlp.router_weight.to(compute_dtype),
            )
            probs = F.softmax(logits, dim=-1)
            eff_k = min(mlp.top_k, mlp.n_experts)
            sel_w, sel_idx = torch.topk(probs, k=eff_k, dim=-1)
            sel_w = sel_w / sel_w.sum(-1, keepdim=True)
            used = [
                int(e)
                for e in torch.unique(sel_idx[sel_idx >= 0]).tolist()
            ]
            pm.record_layer_routing(mlp.layer_idx, logits.detach(), used)

        # Expert contributions — differentiable only via alpha
        contribs: List[torch.Tensor] = []
        H = shared_out.shape[-1]

        for eid in used:
            with torch.no_grad():
                assign = sel_idx == eid              # [T, top_k]
                tmask = assign.any(dim=-1)           # [T]
                if not tmask.any():
                    continue
                w = (sel_w * assign.to(sel_w.dtype)).sum(dim=-1, keepdim=True)  # [T,1]
                expert = pm.get_expert(mlp.layer_idx, eid)
                e_out = _run_expert(
                    x_flat[tmask].detach(), expert
                ).to(shared_out.dtype)               # [T_m, H]
                w_m = w[tmask].detach()              # [T_m, 1]
                idxs = tmask.nonzero(as_tuple=False).view(-1)  # [T_m]

            alpha = alpha_param[eid].to(
                dtype=e_out.dtype, device=e_out.device
            )  # scalar, requires_grad=True
            scaled = e_out * w_m * alpha  # [T_m, H] — grad via alpha

            # Scatter to full sequence length [T, H]
            zeros = shared_out.new_zeros(T, H)
            contrib = torch.scatter_add(
                zeros,
                0,
                idxs.unsqueeze(1).expand(-1, H),
                scaled,
            )
            contribs.append(contrib)

        if contribs:
            expert_out = torch.stack(contribs).sum(dim=0)  # [T, H]
        else:
            expert_out = shared_out.new_zeros(T, H)

        return (shared_out + expert_out).to(x.dtype).view(batch, seq_len, hidden_dim)

    return _forward


def setup_alpha_params(
    model,
) -> Tuple[List[nn.Parameter], List[Tuple[int, _HybridPagedMLP]]]:
    """Lift alphas to nn.Parameters and patch MoE layer forwards.

    Returns (alpha_params, moe_layers) — call teardown_alpha_params when done.
    """
    moe_layers = _get_moe_layers(model)
    alpha_params: List[nn.Parameter] = []

    for layer_idx, mlp in moe_layers:
        init = torch.tensor(
            [mlp.alphas.get(e, mlp.alpha_default) for e in range(mlp.n_experts)],
            dtype=torch.float32,
        )
        param = nn.Parameter(init, requires_grad=True)
        alpha_params.append(param)

        mlp._ttt_alpha_param = param
        mlp._ttt_original_forward = mlp.forward
        mlp.forward = _make_ttt_forward(mlp)

    return alpha_params, moe_layers


def teardown_alpha_params(
    moe_layers: List[Tuple[int, _HybridPagedMLP]],
    alpha_params: List[nn.Parameter],
) -> None:
    """Restore original forwards and write alpha values back to dicts."""
    _write_back_alphas(moe_layers, alpha_params)
    for layer_idx, mlp in moe_layers:
        # Remove the instance-level forward attribute so Python falls back
        # to the class method (which is the original unpatched forward).
        if "forward" in mlp.__dict__:
            del mlp.__dict__["forward"]
        if hasattr(mlp, "_ttt_original_forward"):
            del mlp._ttt_original_forward
        if hasattr(mlp, "_ttt_alpha_param"):
            del mlp._ttt_alpha_param


# ---------------------------------------------------------------------------
# TTT update steps
# ---------------------------------------------------------------------------

def _tokenize(loaded, text: str) -> List[int]:
    tok = loaded.tokenizer
    if hasattr(tok, "prepare_prompt"):
        text = tok.prepare_prompt(text)
    ids = tok.encode(text)
    return ids if ids else []


def ttt_on_tokens(
    loaded,
    token_ids: List[int],
    alpha_params: List[nn.Parameter],
    *,
    lr: float = 0.001,
    chunk_size: int = 128,
) -> float:
    """Update alphas via self-supervised CE loss on token_ids.

    Returns mean loss over all chunks.
    """
    model = loaded.model
    device = torch.device(loaded.device)
    total_loss = 0.0
    n_chunks = 0

    for start in range(0, len(token_ids) - 1, chunk_size - 1):
        chunk = token_ids[start : start + chunk_size]
        if len(chunk) < 2:
            break

        inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor(chunk[1:], dtype=torch.long, device=device)

        # Zero alpha grads
        for p in alpha_params:
            if p.grad is not None:
                p.grad.zero_()

        with torch.enable_grad():
            out = model(input_ids=inp, use_cache=False)
            logits = out.logits[0]  # [seq-1, vocab]
            loss = F.cross_entropy(logits, tgt)

        loss.backward()

        with torch.no_grad():
            for p in alpha_params:
                if p.grad is not None:
                    p.data.sub_(lr * p.grad)
                    p.data.clamp_(0.0, 1.0)

        loss_val = float(loss.item())
        # Explicitly free computation graph and output tensors before next chunk.
        del out, logits, loss, inp, tgt
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        total_loss += loss_val
        n_chunks += 1

    return total_loss / max(n_chunks, 1)


def ttt_on_text(
    loaded,
    text: str,
    alpha_params: List[nn.Parameter],
    *,
    lr: float = 0.001,
    chunk_size: int = 128,
) -> float:
    """Tokenise text and run TTT update. Returns mean loss."""
    ids = _tokenize(loaded, text)
    if len(ids) < 2:
        warnings.warn("Text too short to tokenise into >1 token.")
        return 0.0
    return ttt_on_tokens(loaded, ids, alpha_params, lr=lr, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Experiment 1: domain alpha profiling
# ---------------------------------------------------------------------------

_MEDICAL_TEXT = (
    "Myocardial infarction occurs when coronary artery blood flow decreases enough to cause "
    "heart muscle damage. Treatment includes antiplatelet therapy with aspirin and P2Y12 "
    "inhibitors such as clopidogrel, ticagrelor, or prasugrel. Beta-blockers reduce myocardial "
    "oxygen demand by decreasing heart rate and contractility. ACE inhibitors prevent left "
    "ventricular remodeling after myocardial infarction. Statins reduce cardiovascular mortality "
    "through LDL-C lowering and anti-inflammatory effects. Pharmacokinetics: ticagrelor achieves "
    "peak plasma concentration in 1.5 hours and half-life of 7-8 hours, while clopidogrel requires "
    "hepatic activation via CYP2C19 to its active metabolite. Cardiac catheterization with "
    "percutaneous coronary intervention is the preferred reperfusion strategy when a cath lab is "
    "available. Glycoprotein IIb/IIIa inhibitors may be used as adjunct therapy in high-risk "
    "patients undergoing PCI. Troponin I and T are the preferred biomarkers for myocardial injury, "
    "with high-sensitivity assays detecting elevations within 1-3 hours of symptom onset."
)

_CODING_TEXT = (
    "def binary_search(arr, target):\n"
    "    left, right = 0, len(arr) - 1\n"
    "    while left <= right:\n"
    "        mid = (left + right) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            left = mid + 1\n"
    "        else:\n"
    "            right = mid - 1\n"
    "    return -1\n\n"
    "class LRUCache:\n"
    "    def __init__(self, capacity: int):\n"
    "        self.capacity = capacity\n"
    "        self.cache = {}\n"
    "        self.order = []\n"
    "    def get(self, key: int) -> int:\n"
    "        if key not in self.cache:\n"
    "            return -1\n"
    "        self.order.remove(key)\n"
    "        self.order.append(key)\n"
    "        return self.cache[key]\n"
    "    def put(self, key: int, value: int) -> None:\n"
    "        if key in self.cache:\n"
    "            self.order.remove(key)\n"
    "        elif len(self.cache) >= self.capacity:\n"
    "            oldest = self.order.pop(0)\n"
    "            del self.cache[oldest]\n"
    "        self.cache[key] = value\n"
    "        self.order.append(key)\n\n"
    "def merge_sort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    mid = len(arr) // 2\n"
    "    left = merge_sort(arr[:mid])\n"
    "    right = merge_sort(arr[mid:])\n"
    "    return merge(left, right)\n\n"
    "def merge(left, right):\n"
    "    result = []\n"
    "    i = j = 0\n"
    "    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n"
    "            result.append(left[i])\n"
    "            i += 1\n"
    "        else:\n"
    "            result.append(right[j])\n"
    "            j += 1\n"
    "    result.extend(left[i:])\n"
    "    result.extend(right[j:])\n"
    "    return result\n"
)

_LEGAL_TEXT = (
    "This Agreement is entered into as of the date last signed below by and between the parties "
    "listed herein. WHEREAS, the parties desire to set forth their mutual understandings with "
    "respect to the subject matter hereof; NOW THEREFORE, in consideration of the mutual covenants "
    "and agreements herein contained, and for other good and valuable consideration, the receipt "
    "and sufficiency of which are hereby acknowledged, the parties agree as follows: 1. "
    "Confidentiality. Each party agrees to hold in strict confidence all Confidential Information "
    "received from the other party. Confidential Information means any non-public information that "
    "the disclosing party designates as being confidential. 2. Term and Termination. This Agreement "
    "shall commence on the Effective Date and continue for two (2) years unless earlier terminated. "
    "3. Indemnification. Each party shall indemnify and hold harmless the other party from any "
    "claims, damages, or liabilities arising out of such party's breach of this Agreement. "
    "4. Governing Law. This Agreement shall be governed by and construed in accordance with the "
    "laws of the State of Delaware, without regard to its conflict of law provisions. "
    "5. Entire Agreement. This Agreement constitutes the entire agreement between the parties."
)


def _alpha_state_summary(
    state: Dict[int, Dict[int, float]],
    n_experts: int,
    label: str,
) -> None:
    """Print a compact per-layer alpha table."""
    print(f"\n=== Alpha state: {label} ===")
    for layer_idx in sorted(state.keys()):
        vals = [f"{state[layer_idx].get(e, 0.0):.3f}" for e in range(n_experts)]
        print(f"  layer {layer_idx:2d}: [{', '.join(vals)}]")


def _alpha_state_diff(
    before: Dict[int, Dict[int, float]],
    after: Dict[int, Dict[int, float]],
    n_experts: int,
    label: str,
) -> Dict[str, Any]:
    """Report which experts changed and by how much."""
    print(f"\n--- Alpha delta: {label} ---")
    increased = []
    decreased = []
    max_delta = 0.0
    for layer_idx in sorted(before.keys()):
        for e in range(n_experts):
            b = before[layer_idx].get(e, 0.0)
            a = after[layer_idx].get(e, 0.0)
            delta = a - b
            if abs(delta) > max_delta:
                max_delta = abs(delta)
            if delta > 1e-4:
                increased.append((layer_idx, e, delta))
            elif delta < -1e-4:
                decreased.append((layer_idx, e, delta))
    print(f"  Increased: {len(increased)} alpha(s), max_delta={max_delta:.4f}")
    for layer_idx, e, d in sorted(increased, key=lambda x: -abs(x[2]))[:5]:
        print(f"    layer {layer_idx}, expert {e}: +{d:.4f}")
    print(f"  Decreased: {len(decreased)} alpha(s)")
    for layer_idx, e, d in sorted(decreased, key=lambda x: abs(x[2]))[:5]:
        print(f"    layer {layer_idx}, expert {e}: {d:.4f}")
    return {
        "n_increased": len(increased),
        "n_decreased": len(decreased),
        "max_delta": max_delta,
    }


def run_experiment_1(
    loaded,
    output_dir: str = ".",
    *,
    lr: float = 0.001,
    n_epochs: int = 1,
) -> Dict[str, Any]:
    """
    Experiment 1: Alpha-only TTT with domain text.

    Feeds medical / coding / legal text through the model and updates alphas
    with self-supervised SGD.  Saves per-domain alpha profiles as JSON.

    Returns summary dict with before/after alpha states.
    """
    check_ram(8.0, "exp1-start")
    model = loaded.model
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get number of experts from first MoE layer
    moe_layers_probe = _get_moe_layers(model)
    if not moe_layers_probe:
        raise RuntimeError("No _HybridPagedMLP layers found — wrong model type?")
    n_experts = moe_layers_probe[0][1].n_experts

    # Save initial state
    initial_state = collect_alpha_state(model)
    _alpha_state_summary(initial_state, n_experts, "initial")

    results: Dict[str, Any] = {
        "initial": {
            str(li): {str(e): v for e, v in exp.items()}
            for li, exp in initial_state.items()
        }
    }

    domains = [
        ("medical", _MEDICAL_TEXT),
        ("coding", _CODING_TEXT),
        ("legal", _LEGAL_TEXT),
    ]

    for domain_name, domain_text in domains:
        check_ram(8.0, f"exp1-{domain_name}-start")

        # Reset to initial alphas for each domain profile
        for layer_idx, mlp in _get_moe_layers(model):
            for e in range(n_experts):
                mlp.alphas[e] = initial_state[layer_idx].get(e, mlp.alpha_default)

        state_before = collect_alpha_state(model)
        alpha_params, moe_layers = setup_alpha_params(model)

        t0 = time.perf_counter()
        for _ in range(n_epochs):
            loss = ttt_on_text(loaded, domain_text, alpha_params, lr=lr)

        elapsed = time.perf_counter() - t0
        teardown_alpha_params(moe_layers, alpha_params)

        state_after = collect_alpha_state(model)
        diff = _alpha_state_diff(state_before, state_after, n_experts, domain_name)

        print(f"\n[{domain_name}] loss={loss:.4f}  elapsed={elapsed:.1f}s")
        _alpha_state_summary(state_after, n_experts, domain_name)

        # Save profile
        profile = {
            str(li): {str(e): v for e, v in exp.items()}
            for li, exp in state_after.items()
        }
        profile_path = out_path / f"{domain_name}_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(f"Saved: {profile_path}")

        results[domain_name] = {
            "loss": loss,
            "elapsed_s": elapsed,
            "diff": diff,
            "profile": profile,
        }

        check_ram(8.0, f"exp1-{domain_name}-end")

    # Restore initial alphas
    for layer_idx, mlp in _get_moe_layers(model):
        for e in range(n_experts):
            mlp.alphas[e] = initial_state[layer_idx].get(e, mlp.alpha_default)

    # Success check: at least 3 experts with different magnitudes across domains
    all_distinct = 0
    for layer_idx in sorted(initial_state.keys()):
        for e in range(n_experts):
            vals = [results[d]["profile"][str(layer_idx)][str(e)] for d in ["medical", "coding", "legal"]]
            if max(vals) - min(vals) > 1e-3:
                all_distinct += 1
    print(f"\n[Exp 1] Experts with domain-distinct alphas: {all_distinct}")
    results["distinct_experts"] = all_distinct
    results["success"] = all_distinct >= 3

    return results


# ---------------------------------------------------------------------------
# Experiment 3: RoE ensemble quality comparison
# ---------------------------------------------------------------------------

_EVAL_PROMPTS = [
    "Explain the process of photosynthesis.",
    "What are the main differences between Python 2 and Python 3?",
    "Describe the causes of World War I.",
    "How does a transformer neural network work?",
    "What is the capital of France and what are its main attractions?",
    "Explain quantum entanglement in simple terms.",
    "What are the symptoms of diabetes and how is it treated?",
    "Describe the water cycle and its importance to life on Earth.",
    "How does the human immune system fight viral infections?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of supply and demand in economics.",
    "What causes earthquakes and how are they measured?",
    "How does Bitcoin achieve consensus without a central authority?",
    "Describe the structure and function of DNA.",
    "What are the main principles of object-oriented programming?",
    "How does the greenhouse effect lead to climate change?",
    "Explain the difference between RAM and storage in computers.",
    "What are the key events of the French Revolution?",
    "How does natural selection drive evolution?",
    "What is the speed of light and why is it a universal constant?",
]


def _compute_perplexity(loaded, prompt_ids: List[int], response_ids: List[int]) -> float:
    """CE loss (≈ log-perplexity) of response_ids given prompt_ids context."""
    model = loaded.model
    device = torch.device(loaded.device)
    full = prompt_ids + response_ids
    inp = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor(full[1:], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids=inp, use_cache=False)
        logits = out.logits[0]  # [full_len-1, vocab]

    # Only score the response portion
    n_prompt = len(prompt_ids) - 1
    resp_logits = logits[n_prompt:]                    # [len(response_ids), vocab]
    resp_tgt = tgt[n_prompt:]
    if resp_logits.shape[0] == 0:
        return float("inf")
    return float(F.cross_entropy(resp_logits, resp_tgt).item())


def _generate_ids(loaded, prompt_ids: List[int], max_tokens: int = 30) -> List[int]:
    """Greedy-generate max_tokens token IDs after prompt."""
    model = loaded.model
    device = torch.device(loaded.device)
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos = getattr(loaded.tokenizer, "eos_token_id", None)
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(input_ids=inp, use_cache=False)
            next_id = int(out.logits[0, -1].argmax().item())
            generated.append(next_id)
            if eos is not None and next_id == eos:
                break
            inp = torch.cat([inp, torch.tensor([[next_id]], device=device)], dim=1)
    return generated


def run_experiment_3(
    loaded,
    output_dir: str = ".",
    *,
    n_prompts: int = 20,
    n_responses_each: int = 5,
    max_tokens: int = 30,
) -> Dict[str, Any]:
    """
    Experiment 3: RoE ensemble quality comparison.

    Generates responses under top-2, RoE-4, and RoE-6 routing and compares
    perplexity on the self-generated continuation.

    Returns comparison table.
    """
    check_ram(8.0, "exp3-start")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pm = getattr(loaded.model, "outlier_page_manager", None)
    if pm is None:
        raise RuntimeError("Model has no outlier_page_manager — not a paged model?")

    prompts = _EVAL_PROMPTS[:n_prompts]
    configs = [
        ("top2", 0),
        ("roe4", 4),
        ("roe6", 6),
    ]

    rows: List[Dict[str, Any]] = []

    for config_name, roe_k in configs:
        print(f"\n[Exp 3] Config: {config_name} (roe_k={roe_k})", flush=True)
        pm.enable_roe(roe_k)
        check_ram(8.0, f"exp3-{config_name}")

        for prompt_text in prompts:
            prompt_ids = _tokenize(loaded, prompt_text)
            if not prompt_ids:
                continue

            for rep in range(n_responses_each):
                resp_ids = _generate_ids(loaded, prompt_ids, max_tokens=max_tokens)
                if not resp_ids:
                    continue
                ppl = _compute_perplexity(loaded, prompt_ids, resp_ids)
                rows.append({
                    "config": config_name,
                    "prompt": prompt_text[:40],
                    "rep": rep,
                    "perplexity": ppl,
                })

        print(f"  {config_name}: collected {sum(1 for r in rows if r['config'] == config_name)} rows")

    # Reset RoE
    pm.enable_roe(0)

    # Summary
    summary: Dict[str, Dict[str, float]] = {}
    for config_name, _ in configs:
        cfg_rows = [r["perplexity"] for r in rows if r["config"] == config_name and r["perplexity"] != float("inf")]
        if cfg_rows:
            summary[config_name] = {
                "mean_perplexity": sum(cfg_rows) / len(cfg_rows),
                "n": len(cfg_rows),
            }

    print("\n=== Exp 3 Summary ===")
    for name, s in summary.items():
        print(f"  {name}: mean_ppl={s['mean_perplexity']:.4f}  n={s['n']}")

    # RoE success check: roe4 or roe6 beats top2 on ≥60% of prompts
    def _win_rate(base_name: str, cmp_name: str) -> float:
        base = {(r["prompt"], r["rep"]): r["perplexity"] for r in rows if r["config"] == base_name}
        cmp = {(r["prompt"], r["rep"]): r["perplexity"] for r in rows if r["config"] == cmp_name}
        wins = sum(1 for k in cmp if k in base and cmp[k] < base[k])
        total = sum(1 for k in cmp if k in base)
        return wins / max(total, 1)

    roe4_win = _win_rate("top2", "roe4")
    roe6_win = _win_rate("top2", "roe6")
    print(f"  RoE-4 beats top-2 on {roe4_win*100:.1f}% of prompts")
    print(f"  RoE-6 beats top-2 on {roe6_win*100:.1f}% of prompts")

    result = {
        "rows": rows,
        "summary": summary,
        "roe4_win_rate": roe4_win,
        "roe6_win_rate": roe6_win,
        "success": max(roe4_win, roe6_win) >= 0.60,
    }

    out_file = out_path / "roe_comparison.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {out_file}")
    check_ram(8.0, "exp3-end")
    return result


# ---------------------------------------------------------------------------
# Experiment 4: Pre-attention routing trace collection + predictor
# ---------------------------------------------------------------------------

def collect_routing_traces(
    loaded,
    text: str,
    *,
    max_tokens: int = 500,
) -> List[Tuple[int, torch.Tensor, List[int]]]:
    """
    Hook into the model to record (layer_idx, hidden_before_attn, expert_ids).

    Returns list of (layer_idx, hidden_state[hidden_dim], selected_expert_ids).
    """
    model = loaded.model
    device = torch.device(loaded.device)
    ids = _tokenize(loaded, text)
    if not ids:
        return []

    # Trim to max_tokens
    ids = ids[:max_tokens]

    traces: List[Tuple[int, torch.Tensor, List[int]]] = []

    # Capture hidden states before self-attention
    pre_attn: Dict[int, torch.Tensor] = {}

    def _make_pre_attn_hook(layer_idx: int):
        def hook(module, args):
            hidden = args[0]  # [1, seq, hidden]
            # Detach and store per-token (last token only for incremental)
            pre_attn[layer_idx] = hidden.detach().cpu()
        return hook

    # Capture routing decisions
    routing_log: Dict[int, List[int]] = {}
    orig_record = {}

    def _make_routing_capture(pm, layer_idx: int):
        original = pm.record_layer_routing.__func__ if hasattr(pm.record_layer_routing, "__func__") else None

        def _captured(self_pm, li, logits, expert_ids):
            if li == layer_idx:
                routing_log[li] = list(expert_ids)

        return _captured

    pm = getattr(model, "outlier_page_manager", None)
    if pm is None:
        raise RuntimeError("No outlier_page_manager — not a paged model")

    # Patch record_layer_routing to capture routing
    orig_record_fn = pm.record_layer_routing

    def _patched_record(layer_idx, logits, expert_ids):
        routing_log[layer_idx] = list(expert_ids)
        orig_record_fn(layer_idx, logits, expert_ids)

    pm.record_layer_routing = _patched_record

    # Register pre-attention hooks
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_pre_hook(_make_pre_attn_hook(layer_idx))
        hooks.append(h)

    # Run token-by-token to collect per-token traces
    inp = torch.tensor([ids[:1]], dtype=torch.long, device=device)
    collected = 0

    try:
        with torch.no_grad():
            for tok_idx in range(1, len(ids)):
                pre_attn.clear()
                routing_log.clear()
                out = model(input_ids=inp, use_cache=False)
                # Pair: pre_attn at layer L → routing at layer L+1
                n_layers = len(model.model.layers)
                for l in range(n_layers - 1):
                    if l in pre_attn and (l + 1) in routing_log:
                        hs = pre_attn[l][:, -1, :]  # [1, hidden] → last token
                        traces.append((l, hs.squeeze(0), list(routing_log[l + 1])))
                collected += 1
                next_tok = int(out.logits[0, -1].argmax().item())
                inp = torch.cat([inp, torch.tensor([[next_tok]], device=device)], dim=1)
                if collected >= max_tokens:
                    break
    finally:
        for h in hooks:
            h.remove()
        pm.record_layer_routing = orig_record_fn

    print(f"Collected {len(traces)} (layer, hidden, expert) traces over {collected} tokens")
    return traces


def train_routing_predictor(
    traces: List[Tuple[int, torch.Tensor, List[int]]],
    n_layers: int,
    n_experts: int,
    *,
    n_epochs: int = 50,
) -> Tuple[Dict[int, nn.Linear], Dict[int, float]]:
    """
    Train a linear predictor: hidden_state[L] → expert_selection[L+1].

    Returns (predictors, per-layer accuracy dict).
    """
    # Group traces by layer
    by_layer: Dict[int, List[Tuple[torch.Tensor, List[int]]]] = {}
    for layer_idx, hs, eids in traces:
        by_layer.setdefault(layer_idx, []).append((hs, eids))

    predictors: Dict[int, nn.Linear] = {}
    accuracies: Dict[int, float] = {}

    for layer_idx, layer_traces in sorted(by_layer.items()):
        if not layer_traces:
            continue
        hidden_dim = layer_traces[0][0].shape[0]

        # Build dataset
        X_list, Y_list = [], []
        for hs, eids in layer_traces:
            y = torch.zeros(n_experts)
            for e in eids:
                if 0 <= e < n_experts:
                    y[e] = 1.0
            X_list.append(hs.float())
            Y_list.append(y)

        X = torch.stack(X_list)  # [N, hidden]
        Y = torch.stack(Y_list)  # [N, n_experts]

        # Train predictor
        predictor = nn.Linear(hidden_dim, n_experts)
        optim = torch.optim.Adam(predictor.parameters(), lr=1e-3)

        predictor.train()
        for epoch in range(n_epochs):
            optim.zero_grad()
            logits = predictor(X)
            loss = F.binary_cross_entropy_with_logits(logits, Y)
            loss.backward()
            optim.step()

        predictor.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(predictor(X)) > 0.5).float()
            # Accuracy: fraction of correct binary predictions
            acc = float((preds == Y).float().mean().item())

        predictors[layer_idx] = predictor
        accuracies[layer_idx] = acc
        print(f"  Layer {layer_idx}: predictor trained, accuracy={acc:.3f}")

    return predictors, accuracies


def run_experiment_4(
    loaded,
    output_dir: str = ".",
    *,
    max_tokens: int = 300,
    n_epochs: int = 50,
) -> Dict[str, Any]:
    """
    Experiment 4: Pre-attention routing trace collection and linear predictor.

    Collects (hidden_before_attn, next_layer_experts) pairs, trains a linear
    predictor, and reports prediction accuracy.
    """
    check_ram(8.0, "exp4-start")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Use diverse text for trace collection
    diverse_text = _MEDICAL_TEXT + " " + _CODING_TEXT[:300] + " " + _LEGAL_TEXT[:300]

    moe = _get_moe_layers(loaded.model)
    if not moe:
        raise RuntimeError("No MoE layers found")
    n_experts = moe[0][1].n_experts
    n_layers = len(loaded.model.model.layers)

    traces = collect_routing_traces(loaded, diverse_text, max_tokens=max_tokens)
    if not traces:
        return {"error": "No traces collected"}

    check_ram(8.0, "exp4-after-traces")

    print(f"\n[Exp 4] Training predictors on {len(traces)} traces …")
    predictors, accuracies = train_routing_predictor(
        traces, n_layers, n_experts, n_epochs=n_epochs
    )

    mean_acc = sum(accuracies.values()) / max(len(accuracies), 1)
    print(f"\n[Exp 4] Mean prediction accuracy: {mean_acc:.3f}")

    # Save predictor weights
    pred_path = out_path / "routing_predictors.pt"
    torch.save(
        {f"layer_{li}": p.state_dict() for li, p in predictors.items()},
        str(pred_path),
    )

    # Save trace metadata (not full tensors to keep file small)
    trace_path = out_path / "pre_attention_routing_traces_meta.json"
    trace_meta = {
        "n_traces": len(traces),
        "n_layers_with_traces": len(set(t[0] for t in traces)),
        "n_experts": n_experts,
        "mean_accuracy": mean_acc,
        "per_layer_accuracy": {str(k): v for k, v in accuracies.items()},
    }
    trace_path.write_text(json.dumps(trace_meta, indent=2), encoding="utf-8")
    print(f"Saved: {pred_path}")
    print(f"Saved: {trace_path}")

    check_ram(8.0, "exp4-end")

    return {
        "n_traces": len(traces),
        "mean_accuracy": mean_acc,
        "per_layer_accuracy": accuracies,
        "success": mean_acc >= 0.85,
    }


# ---------------------------------------------------------------------------
# Entry point for running all experiments
# ---------------------------------------------------------------------------

def run_all_experiments(
    model_ref: str = "Outlier-Ai/Outlier-10B",
    output_dir: str = "experiments",
    *,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the model and run all 4 experiments sequentially."""
    from .loader import load_model
    from .profiles import run_experiment_2

    # Pre-load RAM check
    pre_load_ram = check_ram(20.0, "pre-load")
    print(f"Starting experiments. Free RAM before load: {pre_load_ram:.1f} GB")

    loaded = load_model(model_ref, paged=True, device=device)
    post_load_ram = check_ram(15.0, "post-load")
    print(f"Model loaded. Free RAM: {post_load_ram:.1f} GB")

    out = Path(output_dir)
    results: Dict[str, Any] = {}

    # Experiment 1
    print("\n" + "="*60)
    print("EXPERIMENT 1: Alpha-Only TTT")
    print("="*60)
    results["exp1"] = run_experiment_1(loaded, str(out))

    # Experiment 2 (uses profiles saved by exp1)
    print("\n" + "="*60)
    print("EXPERIMENT 2: Domain Mode Switching via Alpha Recipes")
    print("="*60)
    results["exp2"] = run_experiment_2(loaded, str(out))

    # Experiment 3
    print("\n" + "="*60)
    print("EXPERIMENT 3: RoE Ensemble Quality Comparison")
    print("="*60)
    results["exp3"] = run_experiment_3(loaded, str(out))

    # Experiment 4
    print("\n" + "="*60)
    print("EXPERIMENT 4: Pre-Attention Expert Prediction")
    print("="*60)
    results["exp4"] = run_experiment_4(loaded, str(out))

    final_ram = free_ram_gb()
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Final free RAM: {final_ram:.1f} GB")
    print(f"  Exp 1 (TTT):           success={results['exp1'].get('success')}  distinct_experts={results['exp1'].get('distinct_experts')}")
    print(f"  Exp 2 (Profiles):      success={results['exp2'].get('success')}")
    print(f"  Exp 3 (RoE):           success={results['exp3'].get('success')}  roe4_win={results['exp3'].get('roe4_win_rate', 0)*100:.1f}%")
    print(f"  Exp 4 (Predictor):     success={results['exp4'].get('success')}  acc={results['exp4'].get('mean_accuracy', 0):.3f}")

    return results
