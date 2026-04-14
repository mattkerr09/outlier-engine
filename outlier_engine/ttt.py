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
    """Greedy-generate max_tokens token IDs after prompt.

    Uses use_cache=False with a manual decode loop to avoid KV-cache
    attention shapes that require unique Metal shader compilations at every
    step on MPS.  O(n²) in tokens but safe for small max_tokens.
    """
    model = loaded.model
    device = torch.device(loaded.device)
    eos = getattr(loaded.tokenizer, "eos_token_id", None)

    current_ids = list(prompt_ids)
    new_ids: List[int] = []

    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([current_ids], dtype=torch.long, device=device)
            out = model(input_ids=inp, use_cache=False)
            next_id = int(out.logits[0, -1].argmax().item())
            del out, inp   # ref-count drop; MPS buffers reclaimed lazily
            if eos is not None and next_id == eos:
                break
            new_ids.append(next_id)
            current_ids.append(next_id)

    return new_ids


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

        for p_idx, prompt_text in enumerate(prompts):
            prompt_ids = _tokenize(loaded, prompt_text)
            if not prompt_ids:
                continue
            print(f"  [{config_name}] prompt {p_idx+1}/{len(prompts)}: {prompt_text[:40]!r}",
                  flush=True)

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
                print(f"    rep {rep+1}: ppl={ppl:.4f}", flush=True)

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
            # Stay on device — calling .cpu() here stalls the MPS pipeline at
            # every layer boundary (28× per forward pass).  We batch-transfer
            # all hidden states to CPU once after the full forward pass.
            pre_attn[layer_idx] = hidden[:, 0, :].detach().clone()  # [1, hidden] on device
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

    # Register pre-attention hooks on the decoder layer (not self_attn directly).
    # Newer transformers may call self_attn with keyword-only args, making args[]
    # empty.  Decoder layers are always called with hidden_states as args[0].
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = layer.register_forward_pre_hook(_make_pre_attn_hook(layer_idx))
        hooks.append(h)

    # Run single-token forward passes (seq_len=1 always) to collect traces.
    # This avoids the O(n³) cost and MPS shader-compilation storm of the
    # growing-sequence approach: MPS compiles ONE kernel for [1,1,hidden] and
    # reuses it for every subsequent token.  expert_ids from record_layer_routing
    # is the set of experts selected for that single token — a clean per-token
    # training label.
    n_layers = len(model.model.layers)
    collected = 0

    try:
        with torch.no_grad():
            for tok_id in ids:
                pre_attn.clear()
                routing_log.clear()
                inp = torch.tensor([[tok_id]], dtype=torch.long, device=device)
                out = model(input_ids=inp, use_cache=False)
                del out  # free logits; we only need pre_attn + routing_log
                # Batch-transfer hidden states to CPU in one shot — avoids 28
                # per-layer MPS pipeline stalls that were costing ~6 s/token.
                pre_attn_cpu = {k: v.cpu() for k, v in pre_attn.items()}
                del inp
                # Pair: pre_attn at layer L → routing at layer L+1
                for l in range(n_layers - 1):
                    if l in pre_attn_cpu and (l + 1) in routing_log:
                        hs = pre_attn_cpu[l]  # [1, hidden] on CPU
                        traces.append((l, hs.squeeze(0), list(routing_log[l + 1])))
                collected += 1
                if collected >= max_tokens:
                    break
    finally:
        for h in hooks:
            h.remove()
        pm.record_layer_routing = orig_record_fn
        # Single cache flush after all tokens: for seq_len=1 passes the MPS
        # allocator reuses freed weight temporaries across steps, so per-step
        # empty_cache() is unnecessary and expensive (~5 s/call).
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"Collected {len(traces)} (layer, hidden, expert) traces over {collected} tokens", flush=True)
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

    print(f"\n[Exp 4] Training predictors on {len(traces)} traces …", flush=True)
    predictors, accuracies = train_routing_predictor(
        traces, n_layers, n_experts, n_epochs=n_epochs
    )

    mean_acc = sum(accuracies.values()) / max(len(accuracies), 1)
    print(f"\n[Exp 4] Mean prediction accuracy: {mean_acc:.3f}", flush=True)

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
# MCQ data — Experiments 5, 6, 7
# Format: {"question": str, "choices": [A, B, C, D], "answer": 0-3}
# ---------------------------------------------------------------------------

_MCQ_MEDICAL: List[Dict[str, Any]] = [
    {"question": "Which chamber of the heart has the thickest muscular wall?",
     "choices": ["Right atrium", "Right ventricle", "Left atrium", "Left ventricle"], "answer": 3},
    {"question": "Metformin's primary mechanism of action in type 2 diabetes is:",
     "choices": ["Stimulating pancreatic insulin secretion", "Reducing hepatic glucose production",
                 "Blocking intestinal glucose absorption", "Increasing renal glucose excretion via SGLT2"],
     "answer": 1},
    {"question": "The vagus nerve (CN X) provides parasympathetic innervation to:",
     "choices": ["Only the parotid gland", "The eye sphincter only",
                 "The heart and most abdominal viscera", "Only the thoracic organs"],
     "answer": 2},
    {"question": "Penicillin antibiotics inhibit bacterial cell wall synthesis by targeting:",
     "choices": ["DNA gyrase", "Transpeptidases (penicillin-binding proteins)",
                 "The 30S ribosomal subunit", "Dihydrofolate reductase"],
     "answer": 1},
    {"question": "Which coagulation pathway is initiated by tissue factor (factor III)?",
     "choices": ["Intrinsic pathway", "Common pathway", "Fibrinolytic pathway", "Extrinsic pathway"],
     "answer": 3},
    {"question": "Warfarin prevents clotting by inhibiting:",
     "choices": ["Thrombin directly", "Factor Xa directly",
                 "Synthesis of vitamin K-dependent clotting factors", "Platelet aggregation"],
     "answer": 2},
    {"question": "First-line treatment for generalized status epilepticus in adults is:",
     "choices": ["IV phenytoin", "IV levetiracetam", "IV or rectal benzodiazepine", "IV phenobarbital"],
     "answer": 2},
    {"question": "HbA1c reflects average blood glucose levels over approximately:",
     "choices": ["1 week", "2-4 weeks", "2-3 months", "6 months"],
     "answer": 2},
    {"question": "Atropine is a competitive antagonist at which receptor type?",
     "choices": ["Nicotinic ACh receptors", "Muscarinic ACh receptors",
                 "Beta-adrenergic receptors", "Alpha-1 adrenergic receptors"],
     "answer": 1},
    {"question": "Sensitivity of a diagnostic test is calculated as:",
     "choices": ["TP / (TP + FP)", "TN / (TN + FP)", "TP / (TP + FN)", "TN / (TN + FN)"],
     "answer": 2},
    {"question": "Loop diuretics such as furosemide act on the:",
     "choices": ["Proximal convoluted tubule", "Thin descending loop of Henle",
                 "Thick ascending limb of the loop of Henle", "Collecting duct"],
     "answer": 2},
    {"question": "Phase 0 of the ventricular action potential (rapid upstroke) is caused by:",
     "choices": ["Potassium efflux", "Calcium influx via L-type channels",
                 "Rapid sodium influx via fast Na+ channels", "Chloride influx"],
     "answer": 2},
    {"question": "High-dose dopamine (>10 mcg/kg/min) causes vasoconstriction primarily via:",
     "choices": ["Beta-1 adrenergic receptors", "Beta-2 adrenergic receptors",
                 "Alpha-1 adrenergic receptors", "D1 dopaminergic receptors"],
     "answer": 2},
    {"question": "The anatomical structure separating the thoracic and abdominal cavities is:",
     "choices": ["Pleura", "Peritoneum", "Diaphragm", "Mediastinum"],
     "answer": 2},
    {"question": "Which cranial nerve carries the afferent limb of the gag reflex?",
     "choices": ["CN V (trigeminal)", "CN VII (facial)", "CN IX (glossopharyngeal)", "CN X (vagus)"],
     "answer": 2},
    {"question": "A patient with periorbital edema, frothy urine, and hypoalbuminemia most likely has:",
     "choices": ["Nephritic syndrome", "Nephrotic syndrome", "Acute tubular necrosis", "Renal artery stenosis"],
     "answer": 1},
    {"question": "Which class of antibiotics inhibits bacterial protein synthesis at the 50S subunit?",
     "choices": ["Fluoroquinolones", "Aminoglycosides", "Macrolides", "Beta-lactams"],
     "answer": 2},
    {"question": "Normal glomerular filtration rate (GFR) in a healthy young adult is approximately:",
     "choices": ["25 mL/min", "60 mL/min", "120 mL/min", "200 mL/min"],
     "answer": 2},
    {"question": "The troponin complex inhibits muscle contraction by:",
     "choices": ["Providing the myosin power stroke", "Cross-linking actin filaments",
                 "Blocking actin-myosin interaction in the absence of Ca2+", "Hydrolyzing ATP"],
     "answer": 2},
    {"question": "ACE inhibitors are contraindicated in pregnancy primarily due to risk of:",
     "choices": ["Maternal hypertension", "Fetal renal dysgenesis and oligohydramnios",
                 "Maternal thrombosis", "Neonatal respiratory distress syndrome"],
     "answer": 1},
]

_MCQ_CODING: List[Dict[str, Any]] = [
    {"question": "What is the average-case time complexity of binary search on a sorted array of n elements?",
     "choices": ["O(1)", "O(n)", "O(log n)", "O(n log n)"],
     "answer": 2},
    {"question": "In Python, `[x for x in range(5) if x % 2 == 0]` produces:",
     "choices": ["[1, 3]", "[0, 2, 4]", "[0, 1, 2, 3, 4]", "[2, 4]"],
     "answer": 1},
    {"question": "A stack data structure follows which insertion/removal order?",
     "choices": ["FIFO", "LIFO", "Priority-based", "Random access"],
     "answer": 1},
    {"question": "Which sorting algorithm guarantees O(n log n) worst-case time complexity?",
     "choices": ["Quicksort", "Insertion sort", "Merge sort", "Bubble sort"],
     "answer": 2},
    {"question": "In Python, `dict.get(key, default)` returns what when the key is absent?",
     "choices": ["Raises KeyError", "Returns None always", "Returns the default value", "Returns an empty dict"],
     "answer": 2},
    {"question": "Which data structure provides O(1) average-case lookup by key?",
     "choices": ["Sorted array", "Linked list", "Binary search tree", "Hash table"],
     "answer": 3},
    {"question": "In SQL, which clause filters rows AFTER a GROUP BY aggregation?",
     "choices": ["WHERE", "FROM", "HAVING", "ORDER BY"],
     "answer": 2},
    {"question": "The gradient descent parameter update subtracts:",
     "choices": ["The loss value times the learning rate",
                 "The learning rate times the gradient of the loss",
                 "The gradient squared",
                 "The parameter divided by the loss"],
     "answer": 1},
    {"question": "A Python generator function uses which keyword to yield values?",
     "choices": ["return", "async", "yield", "lambda"],
     "answer": 2},
    {"question": "Big-O notation O(1) describes what type of algorithmic growth?",
     "choices": ["Linear", "Logarithmic", "Constant", "Quadratic"],
     "answer": 2},
    {"question": "In REST API design, a POST request is conventionally used to:",
     "choices": ["Retrieve a resource", "Update an existing resource completely",
                 "Create a new resource", "Delete a resource"],
     "answer": 2},
    {"question": "In machine learning, regularization is primarily used to:",
     "choices": ["Speed up gradient descent", "Reduce overfitting by penalizing large weights",
                 "Increase model capacity", "Normalize input features"],
     "answer": 1},
    {"question": "A binary heap data structure is commonly used to implement:",
     "choices": ["A FIFO queue", "A priority queue", "A hash table", "A sorted linked list"],
     "answer": 1},
    {"question": "The TCP protocol guarantees:",
     "choices": ["Minimal latency", "Broadcast delivery",
                 "Ordered, reliable byte-stream delivery", "Connectionless datagram delivery"],
     "answer": 2},
    {"question": "Which activation function is most susceptible to the vanishing gradient problem?",
     "choices": ["ReLU", "Leaky ReLU", "Sigmoid", "ELU"],
     "answer": 2},
    {"question": "In databases, ACID stands for:",
     "choices": ["Abstraction, Concurrency, Isolation, Durability",
                 "Atomicity, Consistency, Isolation, Durability",
                 "Atomicity, Concurrency, Integrity, Distribution",
                 "Availability, Consistency, Integrity, Durability"],
     "answer": 1},
    {"question": "In Git, `git stash` is used to:",
     "choices": ["Permanently delete uncommitted changes", "Merge two branches",
                 "Temporarily save uncommitted changes without committing", "Revert the last commit"],
     "answer": 2},
    {"question": "Which algorithm uses a greedy approach to find single-source shortest paths?",
     "choices": ["Floyd-Warshall", "Bellman-Ford", "Dijkstra's algorithm", "A* search"],
     "answer": 2},
    {"question": "In two's complement representation, the most significant bit indicates:",
     "choices": ["The magnitude of the number", "Whether the number is positive or negative",
                 "A carry bit for addition", "Whether the number is even or odd"],
     "answer": 1},
    {"question": "In Python, `*args` in a function definition captures:",
     "choices": ["Exactly one required positional argument", "Keyword-only arguments",
                 "An arbitrary number of positional arguments", "Type annotations"],
     "answer": 2},
]

_MCQ_LEGAL: List[Dict[str, Any]] = [
    {"question": "In contract law, 'consideration' refers to:",
     "choices": ["The subjective intent of the offeror",
                 "Something of legal value exchanged by each party",
                 "The writing requirement for certain contracts",
                 "The capacity of the contracting parties"],
     "answer": 1},
    {"question": "The Fourth Amendment to the U.S. Constitution protects citizens against:",
     "choices": ["Self-incrimination", "Cruel and unusual punishment",
                 "Unreasonable searches and seizures", "Double jeopardy"],
     "answer": 2},
    {"question": "Under respondeat superior, an employer is liable for:",
     "choices": ["All intentional torts of any employee regardless of scope",
                 "Torts of independent contractors",
                 "Employee torts committed within the scope of employment",
                 "Only criminal acts of employees"],
     "answer": 2},
    {"question": "A contract entered under duress is best characterized as:",
     "choices": ["Void ab initio and unenforceable by either party",
                 "Voidable at the option of the coerced party",
                 "Fully enforceable because both parties agreed",
                 "Illegal and subject to criminal sanctions"],
     "answer": 1},
    {"question": "The standard of proof in a civil lawsuit is:",
     "choices": ["Beyond a reasonable doubt", "Clear and convincing evidence",
                 "Preponderance of the evidence", "Probable cause"],
     "answer": 2},
    {"question": "The 'reasonable person' standard in negligence law is:",
     "choices": ["Subjective to the individual defendant's capabilities",
                 "That of a licensed professional",
                 "An objective standard of ordinary care under similar circumstances",
                 "That of the specific plaintiff"],
     "answer": 2},
    {"question": "First Amendment protection of free speech does NOT protect:",
     "choices": ["Political dissent against the government", "Religious expression in public spaces",
                 "True threats of imminent violence", "Satirical commentary on public officials"],
     "answer": 2},
    {"question": "A writ of habeas corpus is used to:",
     "choices": ["Order a party to appear as a witness",
                 "Challenge the legality of a person's detention",
                 "Transfer a civil case to federal court",
                 "Enjoin a party from taking an action"],
     "answer": 1},
    {"question": "Under Article I of the U.S. Constitution, Congress has express power to:",
     "choices": ["Regulate interstate commerce", "Establish a national religion",
                 "Override Supreme Court decisions by simple majority", "Abolish state governments"],
     "answer": 0},
    {"question": "A valid 'offer' in contract law must include:",
     "choices": ["Acceptance signed by all parties",
                 "Sufficient definiteness and manifestation of intent to be bound",
                 "A written document with witnesses",
                 "Monetary consideration stated explicitly"],
     "answer": 1},
    {"question": "The tort of battery requires proof of:",
     "choices": ["Specific intent to harm the plaintiff",
                 "Intentional, harmful or offensive contact with the plaintiff's person",
                 "Reasonable apprehension of imminent contact",
                 "Actual physical injury"],
     "answer": 1},
    {"question": "Under promissory estoppel, a promise is enforceable when:",
     "choices": ["It is supported by traditional consideration",
                 "The promisee reasonably relied on it to their detriment",
                 "It is reduced to a signed writing",
                 "The promisor had an attorney draft it"],
     "answer": 1},
    {"question": "The Supremacy Clause of the U.S. Constitution establishes that:",
     "choices": ["The President is supreme over Congress in foreign policy",
                 "Federal law is the supreme law of the land",
                 "The Supreme Court automatically invalidates any conflicting state law",
                 "Congress may override state constitutions by statute"],
     "answer": 1},
    {"question": "In criminal law, mens rea refers to:",
     "choices": ["The prohibited physical act",
                 "The guilty mind or criminal intent required for a crime",
                 "The harm caused to the victim",
                 "The defendant's legal defense"],
     "answer": 1},
    {"question": "An ex post facto law unconstitutionally:",
     "choices": ["Applies only prospectively to future conduct",
                 "Retroactively criminalizes conduct that was lawful when performed",
                 "Is passed without legislative debate",
                 "Creates a new constitutional right"],
     "answer": 1},
    {"question": "The doctrine of stare decisis requires courts to:",
     "choices": ["Follow the opinions of leading legal scholars",
                 "Follow precedent set by prior decisions on similar facts",
                 "Defer to executive agencies on questions of law",
                 "Apply strict textualism to all statutes"],
     "answer": 1},
    {"question": "A 'void contract' differs from a 'voidable contract' in that a void contract:",
     "choices": ["Can be ratified by the parties later",
                 "Is unenforceable from the start and cannot be ratified",
                 "Is enforceable only if reduced to writing",
                 "Requires court approval to cancel"],
     "answer": 1},
    {"question": "Miranda warnings must be given before interrogation when a suspect is:",
     "choices": ["Briefly stopped for a traffic violation", "Voluntarily speaking with police",
                 "In custody and subject to interrogation", "Represented by an attorney"],
     "answer": 2},
    {"question": "The parol evidence rule generally prohibits:",
     "choices": ["Oral modification of a written contract after formation",
                 "Prior oral agreements introduced to contradict a fully integrated written contract",
                 "Any testimony about pre-contract negotiations",
                 "Written evidence in oral contract disputes"],
     "answer": 1},
    {"question": "The Equal Protection Clause of the 14th Amendment was designed primarily to:",
     "choices": ["Guarantee the right to vote to all citizens",
                 "Protect against state discrimination, particularly against freed slaves",
                 "Prohibit the federal government from creating classifications",
                 "Guarantee equal pay for equal work"],
     "answer": 1},
]

# New medical questions for Experiment 6 (distinct from _MCQ_MEDICAL)
_MCQ_MEDICAL_NEW: List[Dict[str, Any]] = [
    {"question": "Digoxin toxicity is potentiated by:",
     "choices": ["Hyperkalemia", "Hypokalemia", "Hypernatremia", "Metabolic alkalosis only"],
     "answer": 1},
    {"question": "Initial fluid resuscitation of choice in septic shock is:",
     "choices": ["5% albumin", "Packed red blood cells",
                 "Balanced crystalloid (0.9% NS or lactated Ringer's)", "Hypertonic saline"],
     "answer": 2},
    {"question": "SSRIs treat depression primarily by:",
     "choices": ["Blocking dopamine receptors", "Inhibiting monoamine oxidase",
                 "Blocking presynaptic serotonin reuptake transporters", "Stimulating GABA-A receptors"],
     "answer": 2},
    {"question": "The most common cause of chronic kidney disease worldwide is:",
     "choices": ["Hypertension", "Diabetes mellitus", "Glomerulonephritis", "Polycystic kidney disease"],
     "answer": 1},
    {"question": "INR (International Normalized Ratio) measures function of the:",
     "choices": ["Intrinsic pathway only", "Extrinsic and common pathways",
                 "Platelet aggregation", "Fibrinolytic pathway"],
     "answer": 1},
    {"question": "Which of the following is a live attenuated vaccine?",
     "choices": ["Hepatitis B vaccine", "Inactivated influenza vaccine",
                 "Measles-mumps-rubella (MMR) vaccine", "Tetanus toxoid"],
     "answer": 2},
    {"question": "A patient on MAO inhibitors who eats tyramine-rich food risks:",
     "choices": ["Severe hypotension", "Hypertensive crisis", "Serotonin syndrome", "Agranulocytosis"],
     "answer": 1},
    {"question": "The normal resting membrane potential of cardiac ventricular myocytes is approximately:",
     "choices": ["-90 mV", "-70 mV", "-55 mV", "0 mV"],
     "answer": 0},
    {"question": "Type 1 diabetes mellitus is primarily characterized by:",
     "choices": ["Peripheral insulin resistance", "Autoimmune destruction of pancreatic beta cells",
                 "Excessive glucagon secretion", "Impaired incretin effect"],
     "answer": 1},
    {"question": "Which lipoprotein is primarily responsible for reverse cholesterol transport?",
     "choices": ["LDL", "VLDL", "HDL", "IDL"],
     "answer": 2},
    {"question": "The Fick principle calculates cardiac output from:",
     "choices": ["Blood pressure divided by heart rate",
                 "O2 consumption divided by the arteriovenous O2 difference",
                 "Stroke volume times ejection fraction",
                 "Mean arterial pressure divided by SVR"],
     "answer": 1},
    {"question": "Heparin-induced thrombocytopenia (HIT) is caused by:",
     "choices": ["Direct bone marrow suppression",
                 "Antibodies against platelet factor 4-heparin complexes",
                 "Excessive fibrinolysis", "Complement activation directly"],
     "answer": 1},
    {"question": "Which neurotransmitter is depleted in Parkinson's disease?",
     "choices": ["Acetylcholine", "GABA", "Dopamine", "Serotonin"],
     "answer": 2},
    {"question": "The zona glomerulosa of the adrenal cortex primarily secretes:",
     "choices": ["Cortisol", "Androgens", "Aldosterone", "Epinephrine"],
     "answer": 2},
    {"question": "Iron-deficiency anemia produces which CBC pattern?",
     "choices": ["Macrocytic, hypochromic", "Microcytic, hypochromic",
                 "Normocytic, normochromic", "Macrocytic, normochromic"],
     "answer": 1},
    {"question": "The Krebs (TCA) cycle occurs in which cellular compartment?",
     "choices": ["Cytoplasm", "Mitochondrial matrix",
                 "Inner mitochondrial membrane", "Smooth endoplasmic reticulum"],
     "answer": 1},
    {"question": "Which beta-blocker is cardioselective (beta-1 selective)?",
     "choices": ["Propranolol", "Carvedilol", "Metoprolol", "Labetalol"],
     "answer": 2},
    {"question": "Omeprazole (a PPI) irreversibly inhibits:",
     "choices": ["H2 receptors on parietal cells", "Carbonic anhydrase",
                 "H+/K+-ATPase on gastric parietal cells", "Gastrin receptors"],
     "answer": 2},
    {"question": "The complement membrane attack complex (MAC) is formed by:",
     "choices": ["C3b opsonization on bacteria",
                 "C5a-mediated neutrophil chemotaxis",
                 "Polymerization of C5b-9",
                 "MHC class II antigen presentation"],
     "answer": 2},
    {"question": "Acute subarachnoid hemorrhage classically presents with:",
     "choices": ["Gradual onset unilateral headache with aura",
                 "Sudden 'thunderclap' worst headache of life",
                 "Frontal headache worsened by leaning forward",
                 "Unilateral throbbing headache with photophobia for days"],
     "answer": 1},
]

# General knowledge questions for Experiment 6 control (non-domain)
_MCQ_GENERAL: List[Dict[str, Any]] = [
    {"question": "The capital city of Australia is:",
     "choices": ["Sydney", "Melbourne", "Canberra", "Brisbane"],
     "answer": 2},
    {"question": "Who wrote the dystopian novel '1984'?",
     "choices": ["Aldous Huxley", "George Orwell", "Ray Bradbury", "Ernest Hemingway"],
     "answer": 1},
    {"question": "The speed of light in a vacuum is approximately:",
     "choices": ["3×10^5 km/s", "3×10^8 m/s", "3×10^10 m/s", "3×10^6 m/s"],
     "answer": 1},
    {"question": "The French Revolution began in the year:",
     "choices": ["1776", "1789", "1804", "1815"],
     "answer": 1},
    {"question": "The Pythagorean theorem states that for a right triangle:",
     "choices": ["a + b = c", "a^2 + b^2 = c^2", "a x b = c^2", "a^2 - b^2 = c"],
     "answer": 1},
    {"question": "The currency of Japan is the:",
     "choices": ["Yuan", "Won", "Rupee", "Yen"],
     "answer": 3},
    {"question": "Which planet is closest to the Sun?",
     "choices": ["Venus", "Mars", "Mercury", "Earth"],
     "answer": 2},
    {"question": "'To be, or not to be' is from which Shakespeare play?",
     "choices": ["Macbeth", "Hamlet", "King Lear", "Othello"],
     "answer": 1},
    {"question": "The chemical symbol Au represents:",
     "choices": ["Silver", "Gold", "Platinum", "Aluminum"],
     "answer": 1},
    {"question": "The United Nations was founded in:",
     "choices": ["1920", "1939", "1945", "1955"],
     "answer": 2},
    {"question": "Who published 'On the Origin of Species' in 1859?",
     "choices": ["Gregor Mendel", "Charles Darwin", "Alfred Wallace", "Thomas Huxley"],
     "answer": 1},
    {"question": "Mount Everest is located on the border between:",
     "choices": ["India and China", "Nepal and Tibet (China)", "India and Nepal", "Bhutan and China"],
     "answer": 1},
    {"question": "The human body contains how many pairs of chromosomes?",
     "choices": ["22 pairs", "23 pairs", "24 pairs", "46 pairs"],
     "answer": 1},
    {"question": "The Amazon River flows through which continent?",
     "choices": ["Africa", "Asia", "North America", "South America"],
     "answer": 3},
    {"question": "Which organelle is called the powerhouse of the cell?",
     "choices": ["Ribosome", "Nucleus", "Mitochondria", "Golgi apparatus"],
     "answer": 2},
    {"question": "Photosynthesis converts CO2 and water into:",
     "choices": ["ATP and water", "Glucose and oxygen", "Amino acids and nitrogen", "Lipids and CO2"],
     "answer": 1},
    {"question": "The Magna Carta was signed in the year:",
     "choices": ["1066", "1215", "1265", "1453"],
     "answer": 1},
    {"question": "Einstein's equation E = mc^2 relates to:",
     "choices": ["Gravitational potential energy", "Quantum uncertainty",
                 "Mass-energy equivalence", "Electromagnetic force"],
     "answer": 2},
    {"question": "The Pacific Ocean is:",
     "choices": ["The deepest but not the largest ocean",
                 "Both the largest and deepest ocean",
                 "The largest but not the deepest ocean",
                 "The saltiest ocean"],
     "answer": 1},
    {"question": "DNA replication occurs during which phase of the cell cycle?",
     "choices": ["G1 phase", "S phase", "G2 phase", "M phase"],
     "answer": 1},
]


# ---------------------------------------------------------------------------
# MCQ scoring helpers — shared by Experiments 5, 6, 7
# ---------------------------------------------------------------------------

# Fixed padding length for MCQ forward passes.  All MCQ questions are
# right-padded to this length so that MPS compiles ONE kernel configuration
# for the entire sprint instead of recompiling at every unique seq_len
# (~18-34 s/compile on Apple Silicon).
_MCQ_PAD_LEN: int = 256


def _score_mcq(
    loaded,
    question: str,
    choices: List[str],
    correct_idx: int,
) -> Tuple[int, float]:
    """
    Score a 4-choice MCQ question using log-likelihood at the answer token.

    Formats as "Question: ...\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nAnswer:"
    and compares log-prob of tokens ' A', ' B', ' C', ' D' at the final
    position.  Returns (predicted_idx, log_prob_of_correct_answer).
    """
    tok = loaded.tokenizer
    model = loaded.model
    device = torch.device(loaded.device)
    letters = ["A", "B", "C", "D"]

    prefix = f"Question: {question}\n"
    for i, ch in enumerate(choices):
        prefix += f"{letters[i]}. {ch}\n"
    prefix += "Answer:"

    prefix_ids = tok.encode(prefix)
    n_real = len(prefix_ids)

    # Right-pad to _MCQ_PAD_LEN so all MCQ forward passes share one MPS kernel
    # configuration (compiled once, not once per unique seq_len — each compile
    # costs ~18-34 s on Apple Silicon).  Causal attention ensures padding tokens
    # after position n_real-1 cannot affect logits at that position.
    if n_real >= _MCQ_PAD_LEN:
        prefix_ids = prefix_ids[-_MCQ_PAD_LEN:]  # keep trailing "Answer:" token
        n_real = _MCQ_PAD_LEN
    else:
        prefix_ids = prefix_ids + [0] * (_MCQ_PAD_LEN - n_real)

    # Resolve token ID for each answer letter (" A", " B", ...)
    letter_tids: List[int] = []
    for letter in letters:
        ids = tok.encode(" " + letter)  # OutlierTokenizer.encode never adds special tokens
        letter_tids.append(ids[-1])  # last token if multi-token (rare)

    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=inp, use_cache=False)
    # Score at the last REAL token (n_real-1), not the last padding position.
    last_logits = out.logits[0, n_real - 1].float()
    log_probs = F.log_softmax(last_logits, dim=-1)
    scores = [log_probs[lid].item() for lid in letter_tids]

    del out, inp, last_logits, log_probs
    # Release MPS memory between questions.  The sequential expert loop at
    # seq_len>1 allocates large bf16 temporaries; not flushing between questions
    # can leave the MPS allocator fragmented and cause silent OOM crashes.
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    predicted = max(range(len(choices)), key=lambda i: scores[i])
    return predicted, scores[correct_idx]


def _eval_questions(loaded, questions: List[Dict[str, Any]]) -> Tuple[float, List[bool]]:
    """Run _score_mcq on a list of questions, return (accuracy, per-q bool list)."""
    hits: List[bool] = []
    for q in questions:
        pred, _ = _score_mcq(loaded, q["question"], q["choices"], q["answer"])
        hits.append(pred == q["answer"])
    return sum(hits) / max(len(hits), 1), hits


# ---------------------------------------------------------------------------
# Experiment 5: Domain-specific quality measurement
# ---------------------------------------------------------------------------

def run_experiment_5(
    loaded,
    output_dir: str = ".",
    *,
    profile_dir: Optional[str] = None,
    n_questions: int = 5,
) -> Dict[str, Any]:
    """
    Experiment 5: Does the alive model actually improve domain quality?

    For each domain (medical/coding/legal), measures MCQ accuracy under:
      - Default profile (all alphas = 1.0)
      - Matched profile (domain-appropriate alpha recipe)
      - Mismatched profile (wrong-domain recipe)

    Uses log-likelihood scoring — one forward pass per question, no generation.
    """
    from .profiles import load_alpha_profile, reset_alpha_profile

    check_ram(8.0, "exp5-start")
    out_path = Path(output_dir)
    profile_base = Path(profile_dir) if profile_dir else out_path

    domains = [
        ("medical", _MCQ_MEDICAL, "coding"),
        ("coding",  _MCQ_CODING,  "legal"),
        ("legal",   _MCQ_LEGAL,   "medical"),
    ]

    results: Dict[str, Any] = {}

    for domain_name, questions, mismatch_name in domains:
        questions = questions[:n_questions]
        matched_path  = str(profile_base / f"{domain_name}_profile.json")
        mismatch_path = str(profile_base / f"{mismatch_name}_profile.json")

        if not Path(matched_path).exists():
            print(f"[Exp 5] WARNING: missing profile {matched_path}, skipping {domain_name}")
            results[domain_name] = {"error": f"missing profile: {matched_path}"}
            continue

        # ── Default ────────────────────────────────────────────────────────
        reset_alpha_profile(loaded.model)
        print(f"\n[Exp 5 / {domain_name}] Scoring {len(questions)} questions — DEFAULT profile …",
              flush=True)
        default_acc, default_hits = _eval_questions(loaded, questions)
        print(f"  Default accuracy: {default_acc*100:.1f}%")

        # ── Matched ────────────────────────────────────────────────────────
        load_alpha_profile(loaded.model, matched_path)
        print(f"[Exp 5 / {domain_name}] Scoring — MATCHED ({domain_name}) profile …", flush=True)
        matched_acc, matched_hits = _eval_questions(loaded, questions)
        print(f"  Matched accuracy: {matched_acc*100:.1f}%  "
              f"delta={matched_acc - default_acc:+.3f}")

        # ── Mismatched ─────────────────────────────────────────────────────
        if Path(mismatch_path).exists():
            load_alpha_profile(loaded.model, mismatch_path)
            print(f"[Exp 5 / {domain_name}] Scoring — MISMATCHED ({mismatch_name}) profile …",
                  flush=True)
            mismatch_acc, mismatch_hits = _eval_questions(loaded, questions)
            print(f"  Mismatch accuracy: {mismatch_acc*100:.1f}%  "
                  f"delta_vs_matched={mismatch_acc - matched_acc:+.3f}")
        else:
            mismatch_acc = None
            mismatch_hits = []
            print(f"[Exp 5 / {domain_name}] MISMATCHED profile not found — skipped")

        reset_alpha_profile(loaded.model)
        check_ram(8.0, f"exp5-{domain_name}-end")

        results[domain_name] = {
            "n_questions":   len(questions),
            "default_acc":   default_acc,
            "matched_acc":   matched_acc,
            "mismatch_acc":  mismatch_acc,
            "matched_delta": matched_acc - default_acc,
        }

    reset_alpha_profile(loaded.model)

    # Success: matched > default on at least 2 of 3 domains
    wins = sum(
        1 for d in ["medical", "coding", "legal"]
        if results.get(d, {}).get("matched_delta", -1) > 0
    )
    success = wins >= 2

    print(f"\n=== Exp 5 Summary ===")
    print(f"{'Domain':<10} {'Default':>8} {'Matched':>8} {'Mismatch':>10} {'Delta':>8}")
    for domain_name in ["medical", "coding", "legal"]:
        r = results.get(domain_name, {})
        if "error" in r:
            print(f"  {domain_name:<8}  ERROR: {r['error']}")
            continue
        mis = f"{r['mismatch_acc']*100:.1f}%" if r["mismatch_acc"] is not None else "  n/a"
        print(f"  {domain_name:<8}  {r['default_acc']*100:6.1f}%  "
              f"{r['matched_acc']*100:6.1f}%  {mis:>8}  "
              f"{r['matched_delta']:+.3f}")
    print(f"Matched beats default in {wins}/3 domains — success={success}")

    results["wins"] = wins
    results["success"] = success
    check_ram(8.0, "exp5-end")
    return results


# ---------------------------------------------------------------------------
# Experiment 6: Profile persistence across new prompts
# ---------------------------------------------------------------------------

def run_experiment_6(
    loaded,
    output_dir: str = ".",
    *,
    profile_dir: Optional[str] = None,
    n_questions: int = 5,
) -> Dict[str, Any]:
    """
    Experiment 6: Does the medical profile generalise to NEW questions?

    Tests whether a profile trained on one paragraph of medical text
    transfers to medical MCQs it has never seen, and whether it avoids
    hurting performance on general knowledge.
    """
    from .profiles import load_alpha_profile, reset_alpha_profile

    check_ram(8.0, "exp6-start")
    out_path = Path(output_dir)
    profile_base = Path(profile_dir) if profile_dir else out_path
    medical_path = str(profile_base / "medical_profile.json")

    if not Path(medical_path).exists():
        return {"error": f"Missing profile: {medical_path}", "success": False}

    med_new_qs  = _MCQ_MEDICAL_NEW[:n_questions]
    general_qs  = _MCQ_GENERAL[:n_questions]

    # ── Default — new medical questions ───────────────────────────────────
    reset_alpha_profile(loaded.model)
    print(f"\n[Exp 6] Scoring {len(med_new_qs)} NEW medical questions — DEFAULT …",
          flush=True)
    med_default_acc, _ = _eval_questions(loaded, med_new_qs)
    print(f"  New medical default: {med_default_acc*100:.1f}%")

    # ── Medical profile — new medical questions ────────────────────────────
    load_alpha_profile(loaded.model, medical_path)
    print(f"[Exp 6] Scoring new medical questions — MEDICAL profile …", flush=True)
    med_profile_acc, _ = _eval_questions(loaded, med_new_qs)
    print(f"  New medical (medical profile): {med_profile_acc*100:.1f}%  "
          f"delta={med_profile_acc - med_default_acc:+.3f}")

    # ── Default — general knowledge ────────────────────────────────────────
    reset_alpha_profile(loaded.model)
    print(f"[Exp 6] Scoring {len(general_qs)} general knowledge questions — DEFAULT …",
          flush=True)
    gen_default_acc, _ = _eval_questions(loaded, general_qs)
    print(f"  General default: {gen_default_acc*100:.1f}%")

    # ── Medical profile — general knowledge ───────────────────────────────
    load_alpha_profile(loaded.model, medical_path)
    print(f"[Exp 6] Scoring general questions — MEDICAL profile …", flush=True)
    gen_profile_acc, _ = _eval_questions(loaded, general_qs)
    print(f"  General (medical profile): {gen_profile_acc*100:.1f}%  "
          f"delta={gen_profile_acc - gen_default_acc:+.3f}")

    reset_alpha_profile(loaded.model)

    med_delta = med_profile_acc - med_default_acc
    gen_delta  = gen_profile_acc - gen_default_acc

    # Success: profile helps on new medical AND doesn't degrade general by >2%
    success = (med_delta > 0) and (gen_delta > -0.02)

    print(f"\n=== Exp 6 Summary ===")
    print(f"  New medical: default={med_default_acc*100:.1f}%  "
          f"medical_profile={med_profile_acc*100:.1f}%  delta={med_delta:+.3f}")
    print(f"  General:     default={gen_default_acc*100:.1f}%  "
          f"medical_profile={gen_profile_acc*100:.1f}%  delta={gen_delta:+.3f}")
    print(f"  success={success}  (needs med_delta>0 and gen_delta>-2%)")

    check_ram(8.0, "exp6-end")
    return {
        "med_default_acc":  med_default_acc,
        "med_profile_acc":  med_profile_acc,
        "med_delta":        med_delta,
        "gen_default_acc":  gen_default_acc,
        "gen_profile_acc":  gen_profile_acc,
        "gen_delta":        gen_delta,
        "success":          success,
    }


# ---------------------------------------------------------------------------
# Experiment 7: Cumulative TTT — does more training produce better profiles?
# ---------------------------------------------------------------------------

def run_experiment_7(
    loaded,
    output_dir: str = ".",
    *,
    n_questions: int = 5,
    token_budgets: Optional[List[int]] = None,
    ttt_chunk_size: int = 128,
) -> Dict[str, Any]:
    """
    Experiment 7: Cumulative TTT quality curve.

    Runs TTT on medical text for increasing token budgets (each starting
    from the default alpha state), saves each profile, then evaluates on
    the Exp 5 medical MCQs.

    token_budgets: list of token counts to train on (default [200, 500, 1000]).
    ttt_chunk_size: chunk_size passed to ttt_on_tokens (default 128).
      Use 2 to force single-token forward passes via the batched GEMM path.

    IMPORTANT: Alphas are reset to default BEFORE each TTT run.
    """
    from .profiles import save_alpha_profile, reset_alpha_profile

    check_ram(8.0, "exp7-start")
    out_path = Path(output_dir)

    model = loaded.model
    moe_layers = _get_moe_layers(model)
    if not moe_layers:
        return {"error": "No MoE layers found", "success": False}
    n_experts = moe_layers[0][1].n_experts

    # Use caller-supplied budgets/chunk_size, falling back to fast single-token
    # defaults.  chunk_size=2 → each TTT chunk is seq_len=1 (batched GEMM path).
    _resolved_budgets: List[int] = token_budgets if token_budgets is not None else [8, 16, 32]
    _resolved_chunk: int = ttt_chunk_size if ttt_chunk_size != 128 else 2

    # Build a long medical corpus large enough for the largest budget
    _max_budget = max(_resolved_budgets) if _resolved_budgets else 32
    repeats = max(6, (_max_budget // max(len(_MEDICAL_TEXT.split()), 1)) + 2)
    long_medical = (_MEDICAL_TEXT + " ") * repeats
    all_medical_ids = _tokenize(loaded, long_medical)
    if len(all_medical_ids) < 2:
        return {"error": f"Too few tokens after tokenisation: {len(all_medical_ids)}", "success": False}

    # Snapshot initial alpha state so we can reset between runs
    initial_alpha_state = collect_alpha_state(model)

    # Clamp to what we actually have
    token_budgets = [b for b in _resolved_budgets if b <= len(all_medical_ids)]
    ttt_chunk_size = _resolved_chunk
    print(f"\n[Exp 7] Token budgets: {token_budgets}  chunk_size={ttt_chunk_size}  "
          f"(corpus length: {len(all_medical_ids)} tokens)",
          flush=True)

    ttt_results: Dict[str, Any] = {}

    for budget in token_budgets:
        label = f"medical_{budget}"
        check_ram(8.0, f"exp7-{budget}-start")

        # Reset to default alphas
        for layer_idx, mlp in moe_layers:
            for e in range(n_experts):
                mlp.alphas[e] = initial_alpha_state[layer_idx].get(e, mlp.alpha_default)

        ids_slice = all_medical_ids[:budget]
        alpha_params, moe_layers_patched = setup_alpha_params(model)
        t0 = time.perf_counter()
        loss = ttt_on_tokens(loaded, ids_slice, alpha_params, lr=0.1, chunk_size=ttt_chunk_size)
        elapsed = time.perf_counter() - t0
        teardown_alpha_params(moe_layers_patched, alpha_params)
        del alpha_params

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        profile_path = str(out_path / f"{label}_profile.json")
        save_alpha_profile(model, profile_path, label=label)
        print(f"[Exp 7 / {budget} tokens] loss={loss:.4f}  elapsed={elapsed:.1f}s  "
              f"saved: {profile_path}", flush=True)

        # Evaluate on medical MCQs
        med_eval_qs = _MCQ_MEDICAL[:n_questions]
        print(f"[Exp 7 / {budget} tokens] Evaluating on {len(med_eval_qs)} medical MCQs …",
              flush=True)
        acc, _ = _eval_questions(loaded, med_eval_qs)
        print(f"  Medical accuracy: {acc*100:.1f}%")

        ttt_results[str(budget)] = {
            "token_budget": budget,
            "ttt_loss":     loss,
            "elapsed_s":    elapsed,
            "medical_acc":  acc,
        }

        check_ram(8.0, f"exp7-{budget}-end")

    # Restore initial alphas
    for layer_idx, mlp in moe_layers:
        for e in range(n_experts):
            mlp.alphas[e] = initial_alpha_state[layer_idx].get(e, mlp.alpha_default)

    accs = [ttt_results[str(b)]["medical_acc"] for b in token_budgets if str(b) in ttt_results]
    max_acc = max(accs) if accs else 0.0

    # Also get default accuracy for comparison
    med_eval_qs = _MCQ_MEDICAL[:n_questions]
    print(f"\n[Exp 7] Scoring medical MCQs under DEFAULT (post-restore) alphas …", flush=True)
    default_acc, _ = _eval_questions(loaded, med_eval_qs)
    ttt_results["default_acc"] = default_acc

    # Success: max accuracy across budgets beats default
    success = max_acc > default_acc

    print(f"\n=== Exp 7 Summary ===")
    print(f"  Default accuracy: {default_acc*100:.1f}%")
    for b in token_budgets:
        r = ttt_results.get(str(b), {})
        print(f"  {b:4d} tokens: acc={r.get('medical_acc', 0)*100:.1f}%  "
              f"loss={r.get('ttt_loss', 0):.4f}  t={r.get('elapsed_s', 0):.1f}s")
    print(f"  success={success}  (any budget > default {default_acc*100:.1f}%)")

    check_ram(8.0, "exp7-end")
    ttt_results["success"] = success
    ttt_results["max_acc"] = max_acc
    return ttt_results


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
