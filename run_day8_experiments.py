"""
OUTLIER-LOCAL-EXPERIMENTS-002 combined runner.

Runs (in order):
  Exp 3  — RoE ensemble quality comparison       (from sprint 001)
  Exp 4  — Pre-attention routing predictor        (from sprint 001)
  Exp 5  — Domain-specific quality measurement    (new)
  Exp 6  — Profile persistence / generalisation   (new)
  Exp 7  — Cumulative TTT quality curve           (new)

Skips Exp 3/4 if their JSON summaries already exist in OUT.
Loads the model once, runs all experiments, commits results.

Shell bootstrap:
  cd ~/outlier-engine && source .venv/bin/activate
  python run_day8_experiments.py 2>&1 | tee results_day7/run_day8.log
"""

import gc
import sys
import os
import json
import time
import psutil
import warnings

warnings.filterwarnings("ignore")


def ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _banner(title: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(title, flush=True)
    print("="*60, flush=True)


print(f"[START] Free RAM: {ram_gb():.1f} GB", flush=True)
assert ram_gb() >= 8.0, f"Insufficient RAM: {ram_gb():.1f} GB"

sys.path.insert(0, "/Users/matthewkerr/outlier-engine")
os.chdir("/Users/matthewkerr/outlier-engine")

import torch
from outlier_engine.loader import load_model
from outlier_engine.ttt import (
    check_ram, free_ram_gb,
    run_experiment_3, run_experiment_4,
    run_experiment_5, run_experiment_6, run_experiment_7,
)

OUT = "/Users/matthewkerr/outlier-engine/results_day7"
os.makedirs(OUT, exist_ok=True)

MODEL_PATH = (
    "/Users/matthewkerr/outlier-engine/checkpoints/outlier-10b-v3/v3_checkpoints"
)

# ── Load model ────────────────────────────────────────────────────────────────
# v3 checkpoint: use cold path from safetensors (no packed_experts_dir) so the
# page manager loads experts directly from the v3 shards — coherent checkpoint.
# 64 hot slots (12.5 GB) leaves ~12 GB of physical RAM for warm cache (~64 warm
# slots).  With 128 slots the entire 25 GB free budget goes to hot cache,
# leaving the warm cache starved — evictions go straight to disk.
_banner("Loading Outlier-10B-v3 (paged, bf16, max_experts_in_memory=64)")
t_load = time.perf_counter()
loaded = load_model(
    MODEL_PATH,
    paged=True,
    packed_experts_dir=None,
    max_experts_in_memory=64,
)
print(f"Load time: {time.perf_counter() - t_load:.1f}s   Free RAM: {ram_gb():.1f} GB", flush=True)
assert ram_gb() >= 8.0, f"RAM too low after load: {ram_gb():.1f} GB"
check_ram(8.0, "post-load")

# ── Pre-warm: force expert loading + MPS shader compilation ───────────────────
_banner("Pre-warm: single forward pass (loads experts + compiles Metal shaders)")
from outlier_engine.ttt import _tokenize
_warmup_text = "The quick brown fox jumps over the lazy dog. " * 6
_warmup_ids = _tokenize(loaded, _warmup_text)[:64]
_warmup_inp = torch.tensor([_warmup_ids], dtype=torch.long, device=torch.device(loaded.device))
t_warm = time.perf_counter()
with torch.no_grad():
    _ = loaded.model(input_ids=_warmup_inp, use_cache=False)
print(f"Pre-warm done in {time.perf_counter()-t_warm:.1f}s   Free RAM: {ram_gb():.1f} GB",
      flush=True)
del _warmup_inp, _warmup_ids, _warmup_text, _
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
check_ram(8.0, "post-warmup")

# ── MCQ pre-warm at seq_len=256 ────────────────────────────────────────────────
# _score_mcq right-pads all questions to _MCQ_PAD_LEN=256.  Running one dummy
# forward pass here pre-compiles all MPS kernels for that shape so no question
# incurs a 18-34 s JIT penalty during Exps 5/6/7.
from outlier_engine.ttt import _MCQ_PAD_LEN
_banner(f"MCQ pre-warm: forward pass at seq_len={_MCQ_PAD_LEN} (compile MCQ kernels)")
_mcq_warm_inp = torch.zeros(1, _MCQ_PAD_LEN, dtype=torch.long,
                            device=torch.device(loaded.device))
t_mcq_warm = time.perf_counter()
with torch.no_grad():
    _ = loaded.model(input_ids=_mcq_warm_inp, use_cache=False)
print(f"MCQ pre-warm done in {time.perf_counter()-t_mcq_warm:.1f}s   Free RAM: {ram_gb():.1f} GB",
      flush=True)
del _mcq_warm_inp, _
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
check_ram(8.0, "post-mcq-warmup")

# ── Exp 3 ─────────────────────────────────────────────────────────────────────
exp3_path = f"{OUT}/exp3_summary.json"
if os.path.exists(exp3_path):
    print(f"\n[SKIP] Exp 3 already complete: {exp3_path}", flush=True)
    with open(exp3_path) as f:
        exp3_results = json.load(f)
else:
    _banner("EXPERIMENT 3: RoE ensemble quality comparison")
    exp3_results = run_experiment_3(loaded, OUT, n_prompts=3, n_responses_each=1, max_tokens=5)
    with open(exp3_path, "w") as f:
        json.dump(exp3_results, f, indent=2)
    print(f"Saved: {exp3_path}", flush=True)

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ── Exp 4 ─────────────────────────────────────────────────────────────────────
exp4_path = f"{OUT}/exp4_summary.json"
if os.path.exists(exp4_path):
    print(f"\n[SKIP] Exp 4 already complete: {exp4_path}", flush=True)
    with open(exp4_path) as f:
        exp4_results = json.load(f)
else:
    _banner("EXPERIMENT 4: Pre-attention routing trace + predictor")
    # 50 tokens: O(n^3) = 50^3 vs 300^3 → 216× faster; ~1300 traces
    # sufficient for the linear predictor.  mps.empty_cache() each step
    # prevents memory pressure from int8→bf16 weight temporaries.
    exp4_results = run_experiment_4(loaded, OUT, max_tokens=50, n_epochs=50)
    with open(exp4_path, "w") as f:
        json.dump(exp4_results, f, indent=2)
    print(f"Saved: {exp4_path}", flush=True)

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

_exp5_path = f"{OUT}/exp5_summary.json"
_exp6_path = f"{OUT}/exp6_summary.json"

# ── Exp 5 ─────────────────────────────────────────────────────────────────────
if os.path.exists(_exp5_path):
    print(f"\n[SKIP] Exp 5 already complete: {_exp5_path}", flush=True)
    with open(_exp5_path) as _f:
        exp5_results = json.load(_f)
else:
    _banner("EXPERIMENT 5: Domain-specific quality measurement")
    exp5_results = run_experiment_5(loaded, OUT, profile_dir=OUT, n_questions=1)
    with open(_exp5_path, "w") as f:
        json.dump(exp5_results, f, indent=2)
    print(f"Saved: {_exp5_path}", flush=True)

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ── Exp 6 ─────────────────────────────────────────────────────────────────────
if os.path.exists(_exp6_path):
    print(f"\n[SKIP] Exp 6 already complete: {_exp6_path}", flush=True)
    with open(_exp6_path) as _f:
        exp6_results = json.load(_f)
else:
    _banner("EXPERIMENT 6: Profile persistence / generalisation")
    exp6_results = run_experiment_6(loaded, OUT, profile_dir=OUT, n_questions=1)
    with open(_exp6_path, "w") as f:
        json.dump(exp6_results, f, indent=2)
    print(f"Saved: {_exp6_path}", flush=True)

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ── Exp 7 ─────────────────────────────────────────────────────────────────────
_banner("EXPERIMENT 7: Cumulative TTT quality curve")
# ttt_chunk_size=2: each TTT chunk is a single-token forward pass (batched GEMM
# path fires when x_flat.shape[0]==1), keeping peak memory low and each step
# to ~0.5 sec.  token_budgets=[8,16,32] gives 7/15/31 gradient steps.
exp7_results = run_experiment_7(
    loaded, OUT,
    n_questions=1,
    token_budgets=[8, 16, 32],
    ttt_chunk_size=2,
)
with open(f"{OUT}/exp7_summary.json", "w") as f:
    json.dump(exp7_results, f, indent=2)
print(f"Saved: {OUT}/exp7_summary.json", flush=True)

# ── Final master table ────────────────────────────────────────────────────────
final_ram = free_ram_gb()
print(f"\n{'='*60}", flush=True)
print("MASTER RESULTS TABLE — ALL 7 EXPERIMENTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"Final free RAM: {final_ram:.1f} GB")

# Load Exp 1/2 summaries for completeness
def _safe_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

exp1 = _safe_load(f"{OUT}/exp1_summary.json")
exp2 = _safe_load(f"{OUT}/exp2_summary.json")

print(f"\n  Exp 1 (Alpha TTT):       success={exp1.get('success')}  "
      f"distinct={exp1.get('distinct_experts')}")
print(f"  Exp 2 (Profile switch):  success={exp2.get('success')}  "
      f"max_switch={exp2.get('max_switch_ms', 0):.3f}ms")
print(f"  Exp 3 (RoE):             success={exp3_results.get('success')}  "
      f"roe4_win={exp3_results.get('roe4_win_rate', 0)*100:.1f}%  "
      f"roe6_win={exp3_results.get('roe6_win_rate', 0)*100:.1f}%")
print(f"  Exp 4 (Predictor):       success={exp4_results.get('success')}  "
      f"mean_acc={exp4_results.get('mean_accuracy', 0):.3f}")
print(f"  Exp 5 (Quality):         success={exp5_results.get('success')}  "
      f"wins={exp5_results.get('wins')}/3")
for d in ["medical", "coding", "legal"]:
    r = exp5_results.get(d, {})
    if "error" not in r:
        print(f"    {d:<8}: default={r.get('default_acc', 0)*100:.1f}%  "
              f"matched={r.get('matched_acc', 0)*100:.1f}%  "
              f"delta={r.get('matched_delta', 0):+.3f}")
print(f"  Exp 6 (Persistence):     success={exp6_results.get('success')}  "
      f"med_delta={exp6_results.get('med_delta', 0):+.3f}  "
      f"gen_delta={exp6_results.get('gen_delta', 0):+.3f}")
print(f"  Exp 7 (Cumulative TTT):  success={exp7_results.get('success')}  "
      f"max_acc={exp7_results.get('max_acc', 0)*100:.1f}%  "
      f"default={exp7_results.get('default_acc', 0)*100:.1f}%")
for b in [200, 500, 1000]:
    r = exp7_results.get(str(b), {})
    if r:
        print(f"    {b:4d} tokens: {r.get('medical_acc', 0)*100:.1f}%")

print(f"\nAll results saved to {OUT}", flush=True)
