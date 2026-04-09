"""
OUTLIER-LOCAL-EXPERIMENTS-001 continuation runner.
Skips Exp 1 (profiles already in results_day7/) and runs Exp 2/3/4 only.
"""

import sys
import gc
import os
import json
import time
import psutil
import warnings

warnings.filterwarnings("ignore")

def ram_gb():
    return psutil.virtual_memory().available / 1e9

print(f"[START] Free RAM: {ram_gb():.1f} GB", flush=True)
assert ram_gb() >= 8.0, f"Not enough RAM: {ram_gb():.1f} GB"

sys.path.insert(0, "/Users/matthewkerr/outlier-engine")
os.chdir("/Users/matthewkerr/outlier-engine")

import torch
from outlier_engine.loader import load_model
from outlier_engine.ttt import (
    check_ram, free_ram_gb,
    _get_moe_layers,
    run_experiment_3, run_experiment_4,
)
from outlier_engine.profiles import run_experiment_2

OUT = "/Users/matthewkerr/outlier-engine/results_day7"
os.makedirs(OUT, exist_ok=True)

# Check Exp 1 profiles exist
for domain in ["medical", "coding", "legal"]:
    pf = f"{OUT}/{domain}_profile.json"
    assert os.path.exists(pf), f"Missing Exp 1 profile: {pf} — run full runner first"
print("Exp 1 profiles found.", flush=True)

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
check_ram(8.0, "post-load")

# ── Experiment 2 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 2: Profile switching speed + domain generation")
print("="*60, flush=True)

exp2_results = run_experiment_2(loaded, OUT, max_tokens=50)
with open(f"{OUT}/exp2_summary.json", "w") as f:
    json.dump(exp2_results, f, indent=2)
print(f"Saved: {OUT}/exp2_summary.json", flush=True)

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ── Experiment 3 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 3: RoE ensemble quality comparison")
print("="*60, flush=True)

exp3_results = run_experiment_3(loaded, OUT, n_prompts=20, n_responses_each=5, max_tokens=15)
with open(f"{OUT}/exp3_summary.json", "w") as f:
    json.dump(exp3_results, f, indent=2)
print(f"Saved: {OUT}/exp3_summary.json", flush=True)

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ── Experiment 4 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 4: Pre-attention routing trace + predictor")
print("="*60, flush=True)

exp4_results = run_experiment_4(loaded, OUT, max_tokens=300, n_epochs=50)
with open(f"{OUT}/exp4_summary.json", "w") as f:
    json.dump(exp4_results, f, indent=2)
print(f"Saved: {OUT}/exp4_summary.json", flush=True)

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY (Exp 2-4)")
print(f"{'='*60}")
print(f"Final free RAM: {free_ram_gb():.1f} GB")
print(f"  Exp 2 (Profiles):  success={exp2_results.get('success')}  max_switch={exp2_results.get('max_switch_ms',0):.3f}ms")
print(f"  Exp 3 (RoE):       success={exp3_results.get('success')}  roe4_win={exp3_results.get('roe4_win_rate',0)*100:.1f}%  roe6_win={exp3_results.get('roe6_win_rate',0)*100:.1f}%")
print(f"  Exp 4 (Predictor): success={exp4_results.get('success')}  mean_acc={exp4_results.get('mean_accuracy',0):.3f}")
print("All results saved to", OUT, flush=True)
