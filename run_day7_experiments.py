"""
OUTLIER-LOCAL-EXPERIMENTS-001 runner.
Loads the 10B model once, runs all 4 experiments, saves results to ./results_day7/
"""

import gc
import sys
import os
import json
import time
import psutil
import warnings

warnings.filterwarnings("ignore")

def ram_gb():
    return psutil.virtual_memory().available / 1e9

print(f"[START] Free RAM: {ram_gb():.1f} GB", flush=True)
assert ram_gb() >= 20.0, f"Not enough RAM: {ram_gb():.1f} GB"

# ── activate venv path ──────────────────────────────────────────────────────
sys.path.insert(0, "/Users/matthewkerr/outlier-engine")
os.chdir("/Users/matthewkerr/outlier-engine")

import torch
from outlier_engine.loader import load_model
from outlier_engine.ttt import (
    check_ram, free_ram_gb,
    collect_alpha_state, _get_moe_layers,
    setup_alpha_params, teardown_alpha_params, ttt_on_text,
    _alpha_state_summary, _alpha_state_diff,
    run_experiment_3, run_experiment_4,
    _MEDICAL_TEXT, _CODING_TEXT, _LEGAL_TEXT,
)
from outlier_engine.profiles import (
    save_alpha_profile, load_alpha_profile, reset_alpha_profile, list_profiles,
    run_experiment_2,
)

OUT = "/Users/matthewkerr/outlier-engine/results_day7"
os.makedirs(OUT, exist_ok=True)

# ── Load model ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Loading Outlier-10B (paged, bf16) ...")
print("="*60, flush=True)

# Use local snapshot path to bypass the HF alias (Outlier-10B → Outlier-10B-V2)
# that would redirect to the dense model.
MODEL_PATH = (
    "/Users/matthewkerr/.cache/huggingface/hub/"
    "models--Outlier-Ai--Outlier-10B/snapshots/"
    "11ff21cad8faa97dafab2362f20bf790da7f3ae3"
)
t_load = time.perf_counter()
loaded = load_model(
    MODEL_PATH,
    paged=True,
    packed_experts_dir="/Users/matthewkerr/outlier-engine/packed_experts",
)
print(f"Load time: {time.perf_counter() - t_load:.1f}s   Free RAM after load: {ram_gb():.1f} GB",
      flush=True)
check_ram(15.0, "post-load")

model = loaded.model
moe_layers_list = _get_moe_layers(model)
n_experts = moe_layers_list[0][1].n_experts
n_moe_layers = len(moe_layers_list)
print(f"MoE layers: {n_moe_layers}, experts/layer: {n_experts}, total alphas: {n_moe_layers*n_experts}",
      flush=True)

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Alpha-only TTT
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 1: Alpha-only TTT (domain profiling)")
print("="*60, flush=True)

initial_state = collect_alpha_state(model)
_alpha_state_summary(initial_state, n_experts, "initial (first 4 layers shown)")

domains = [
    ("medical", _MEDICAL_TEXT),
    ("coding",  _CODING_TEXT),
    ("legal",   _LEGAL_TEXT),
]
exp1_results = {"initial": {str(li): {str(e): v for e, v in exp.items()} for li, exp in initial_state.items()}}
all_profiles = {}

for domain_name, domain_text in domains:
    # After the first domain the page cache is hot (stays in RAM).
    # Subsequent domains don't allocate significant new RAM, so the
    # per-domain threshold is 8 GB instead of the global 15 GB.
    check_ram(8.0, f"exp1-{domain_name}")
    print(f"\n--- Domain: {domain_name} ---", flush=True)

    # Reset to initial state
    for layer_idx, mlp in moe_layers_list:
        for e in range(n_experts):
            mlp.alphas[e] = initial_state[layer_idx].get(e, mlp.alpha_default)

    state_before = collect_alpha_state(model)
    alpha_params, moe_layers = setup_alpha_params(model)

    t0 = time.perf_counter()
    # LR=0.1 produces ~0.1 alpha shift per domain (10-100× more than 0.001),
    # which is needed to get distinguishable per-domain profiles.
    loss = ttt_on_text(loaded, domain_text, alpha_params, lr=0.1, chunk_size=128)
    elapsed = time.perf_counter() - t0

    teardown_alpha_params(moe_layers, alpha_params)
    del alpha_params
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    state_after = collect_alpha_state(model)

    diff = _alpha_state_diff(state_before, state_after, n_experts, domain_name)
    print(f"  loss={loss:.4f}  elapsed={elapsed:.1f}s", flush=True)

    # Print top shifted layers/experts
    shifts = []
    for li in sorted(state_before.keys()):
        for e in range(n_experts):
            b = state_before[li].get(e, 0.0)
            a = state_after[li].get(e, 0.0)
            shifts.append((abs(a - b), li, e, b, a))
    shifts.sort(reverse=True)
    print(f"  Top 8 alpha shifts for {domain_name}:")
    for delta, li, e, b, a in shifts[:8]:
        direction = "+" if a > b else "-"
        print(f"    layer={li:2d} expert={e}: {b:.4f} -> {a:.4f} ({direction}{delta:.4f})")

    profile = {str(li): {str(e): v for e, v in exp.items()} for li, exp in state_after.items()}
    all_profiles[domain_name] = profile
    save_alpha_profile(model, f"{OUT}/{domain_name}_profile.json", label=domain_name)
    exp1_results[domain_name] = {"loss": loss, "elapsed_s": elapsed, "diff": diff, "profile": profile}

# Distinct-expert check
distinct = 0
for li in sorted(initial_state.keys()):
    for e in range(n_experts):
        vals = [float(all_profiles[d][str(li)][str(e)]) for d in ["medical","coding","legal"]]
        if max(vals) - min(vals) > 1e-3:
            distinct += 1
print(f"\n[Exp 1] Experts with domain-distinct alphas: {distinct}/{n_moe_layers*n_experts}")
exp1_results["distinct_experts"] = distinct
exp1_results["success"] = distinct >= 3

# Write compact summary for the three profiles (layers 0-3, all experts)
print("\n=== Alpha comparison: first 4 MoE layers ===")
print(f"{'Layer':>6}  {'Expert':>6}  {'Medical':>9}  {'Coding':>9}  {'Legal':>9}  {'MaxDiff':>9}")
for li in sorted(initial_state.keys())[:4]:
    for e in range(n_experts):
        med = float(all_profiles["medical"][str(li)][str(e)])
        cod = float(all_profiles["coding"][str(li)][str(e)])
        leg = float(all_profiles["legal"][str(li)][str(e)])
        diff_val = max(med, cod, leg) - min(med, cod, leg)
        print(f"  {li:4d}  {e:6d}  {med:9.4f}  {cod:9.4f}  {leg:9.4f}  {diff_val:9.4f}")

with open(f"{OUT}/exp1_summary.json", "w") as f:
    json.dump(exp1_results, f, indent=2)
print(f"\nSaved: {OUT}/exp1_summary.json", flush=True)

# Restore to initial
for layer_idx, mlp in moe_layers_list:
    for e in range(n_experts):
        mlp.alphas[e] = initial_state[layer_idx].get(e, mlp.alpha_default)

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Domain profile switching
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 2: Profile switching speed + domain generation")
print("="*60, flush=True)

exp2_results = run_experiment_2(loaded, OUT, max_tokens=50)
with open(f"{OUT}/exp2_summary.json", "w") as f:
    json.dump(exp2_results, f, indent=2)
print(f"Saved: {OUT}/exp2_summary.json", flush=True)

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: RoE comparison
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 3: RoE ensemble quality comparison")
print("="*60, flush=True)
check_ram(8.0, "exp3-start")

exp3_results = run_experiment_3(loaded, OUT, n_prompts=20, n_responses_each=5, max_tokens=30)
with open(f"{OUT}/exp3_summary.json", "w") as f:
    json.dump(exp3_results, f, indent=2)
print(f"Saved: {OUT}/exp3_summary.json", flush=True)

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Pre-attention routing predictor
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 4: Pre-attention routing trace + predictor")
print("="*60, flush=True)
check_ram(8.0, "exp4-start")

exp4_results = run_experiment_4(loaded, OUT, max_tokens=300, n_epochs=50)
with open(f"{OUT}/exp4_summary.json", "w") as f:
    json.dump(exp4_results, f, indent=2)
print(f"Saved: {OUT}/exp4_summary.json", flush=True)

# ────────────────────────────────────────────────────────────────────────────
# Final summary
# ────────────────────────────────────────────────────────────────────────────
final_ram = free_ram_gb()
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Final free RAM: {final_ram:.1f} GB")
print(f"  Exp 1 (Alpha TTT):     success={exp1_results['success']}  distinct={exp1_results['distinct_experts']}/{n_moe_layers*n_experts}")
print(f"  Exp 2 (Profiles):      success={exp2_results.get('success')}  max_switch={exp2_results.get('max_switch_ms',0):.3f}ms")
print(f"  Exp 3 (RoE):           success={exp3_results.get('success')}  roe4_win={exp3_results.get('roe4_win_rate',0)*100:.1f}%  roe6_win={exp3_results.get('roe6_win_rate',0)*100:.1f}%")
print(f"  Exp 4 (Predictor):     success={exp4_results.get('success')}  mean_acc={exp4_results.get('mean_accuracy',0):.3f}")
print("All results saved to", OUT, flush=True)
