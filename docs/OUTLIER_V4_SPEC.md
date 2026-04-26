# OUTLIER V4 — Architecture Spec for Sprint Implementation
## Outlier-specific context the generic arxiv papers don't cover
**Purpose:** Give the V4 sprint code (Hadamard, TC-MoE, predictor v2, HESTIA)
the Outlier-specific architectural facts it needs to apply correctly to V3.2
models. This is NOT a general V4 doc — it's the scaffolding needed to bridge
arxiv math to Outlier's specific V3.2 architecture.

---

## §1 — V3.2 base architecture (what the V4 layers apply ON TOP of)

### §1.1 Shared expert + routed experts
Every MoE layer in V3.2 has:
- ONE shared expert = the frozen Qwen2.5 base FFN at that layer, bf16, always active
- 8-16 routed ternary experts (delta experts), top-2 routing
- Expert correction equation:

```expert_i_out(x) = base_FFN_L(x) + alpha_i * delta_i(x)

Where:
- `base_FFN_L(x)` = frozen Qwen2.5 FFN at layer L (bf16, always runs)
- `delta_i(x)` = three ternary linear layers (gate/up/down), stored as ternary {-1, 0, +1}
- `alpha_i` = single learnable scalar per expert, initialized to 0.1

**Key point:** the delta experts are NOT standalone experts. They are learned
deviations from the base FFN. At init, delta = zero, so
expert_i_out(x) == base_FFN_L(x). Training learns small deviations.

### §1.2 Zero-delta initialization
- All delta ternary weights start at zero
- Alpha scalars start at 0.1
- Routers are trained per-layer via CAKLD distillation
- Loss: CAKLD + L1 delta regularization
- Converges in ~1500 steps at 10B scale (vs 5000+ for dropout-clone init)

**Implication for V4 layers:** any transformation applied to the delta experts
must preserve the zero-init property at initialization, OR the V4 layer must
re-initialize alpha to a matched value such that the initial behavior matches
V3.2.

### §1.3 Layer pattern (Dense-Sparse-Dense)
- Bottom ~25% of layers: dense (no MoE)
- Middle ~55% of layers: MoE layers (where V4 layers apply)
- Top ~20% of layers: dense (no MoE)
- V4 layers ONLY apply to the middle MoE layers. Dense layers are unchanged.

### §1.4 Ternary storage format
- Weights stored as int8 packed with 4 ternary values per byte
- Scale factor (per-expert or per-channel) stored separately
- Dequant pathway: ternary → bf16 using `GPUResidentExpert` class at runtime
- Source of truth: 150B modeling file at
  `Outlier-Ai/Outlier-150B-V3.2/modeling_outlier_150b_rexmoe.py` line 49

### §1.5 Router architecture
- Standard top-k=2 router per MoE layer
- Router input = hidden state at layer L
- Router output = (expert_indices, expert_weights) normalized to sum to 1
- Router is full-precision (bf16), not ternary

### §1.6 ReXMoE (150B only, ignore for 10B sprint)
150B shares expert weights across 4 adjacent layers with progressive scaling
routing (alpha multipliers [0.7, 0.9, 1.1, 1.3]). **NOT relevant to 10B V4 sprint.**

---

## §2 — Where Hadamard applies (Phase 1)

### §2.1 The goal
Hadamard pre-rotation per expert reduces the dynamic range of the delta weights
before ternary quantization, which lets the STE (or HESTIA) hit better quality
at the same quantization level. This is the PolarQuant insight: rotation alone
gives 98% of the quality improvement from quantization-aware training.

### §2.2 Where to apply
Apply Hadamard rotation to each delta expert's gate/up/down projections BEFORE
ternary quantization, AFTER full-precision training. The rotation is stored as
32 bytes per expert (just the sign pattern of the Hadamard matrix — the matrix
itself is derived).For each delta_i in MoE layer:
for proj in [gate_proj, up_proj, down_proj]:
H = hadamard_matrix(proj.shape[1])  # deterministic from shape
proj_rotated = proj @ H
proj_ternary = ternary_quantize(proj_rotated)
store(proj_ternary, H_sign_pattern_32_bytes)

### §2.3 Inference path
At inference, the Hadamard rotation is absorbed into the input:x_rotated = x @ H.T  # (one-time per layer, amortized)
delta_out = proj_ternary @ x_rotated
Result is mathematically equivalent to unrotated version, but better
quantization quality

### §2.4 Outlier-specific gotcha (CRITICAL)
The shared expert `base_FFN_L(x)` is NOT rotated — it's frozen Qwen2.5 bf16 and
doesn't need rotation. Only the delta experts get the Hadamard treatment. The
alpha gating `alpha_i * delta_i(x)` is applied AFTER the rotated delta produces
its output.

Your existing `RotatedV32Model` wraps ALL experts in a MoE layer — this is the
fix you already flagged. Correct behavior: iterate only the delta experts
(NOT the shared MLP / base FFN) when applying rotation. Sanity check in test:
loading an unmodified Qwen2.5 FFN through your rotation wrapper should produce
bitwise-identical output to the unwrapped FFN, because the shared path is
untouched.

### §2.5 Phase 1 success criterion
After applying Hadamard rotation to all delta experts in a 10B V3.2 model:
- Coherent generation must still work (sanity check with 50 tokens)
- MMLU delta vs unrotated baseline should be in [-0.5%, +3%] range per PolarQuant
- If MMLU drops more than 0.5%, something is wrong — check the rotation absorption

### §2.6 Arxiv sources
- QuIP# (Tseng et al., arXiv:2402.04396) — Hadamard incoherence for LLM quantization
- PolarQuant — rotation-only ablation showing 98% of quality improvement
- Already implemented in `hadamard_rotation.py` per Claude Code status

---

## §3 — Where TC-MoE applies (Phase 3)

### §3.1 The goal
TC-MoE (ternary-coded routing) extends the router output from {0, 1} (off/on)
to {-1, 0, +1} (negative/off/positive). The 0 value means "skip this delta
expert" — the delta contribution is zero regardless of alpha. The ±1 values
apply the expert with positive or negative sign.

### §3.2 Why this works for Outlier specifically
Outlier's alpha gating already learns per-expert scalars that can be positive
or negative. TC-MoE's -1 case is equivalent to "alpha is negative for this
token," which is already learnable in V3.2. But the 0 case (skip) is NEW —
V3.2 always applies all top-k experts even when the weight is tiny.

Measured benefit from TC-MoE paper: 9% fewer expert activations + 1.1%
accuracy improvement. For Outlier, the 9% fewer activations translates to
9% fewer paging stalls, which is a larger absolute speedup than on dense
FP16 MoE.

### §3.3 Where to apply
Modify the router output processing, NOT the router itself:V3.2:
expert_indices, expert_weights = router(x)  # top-k=2
ALL top-k experts applyV4 with TC-MoE:
expert_indices, expert_weights, expert_signs = tc_moe_router(x)  # top-k=2
expert_signs is in {-1, 0, +1}
Only experts with sign != 0 actually execute
for idx, weight, sign in zip(expert_indices, expert_weights, expert_signs):
if sign == 0:
continue  # SKIP — no paging stall, no compute
delta_out += sign * weight * alpha_i * delta_i(x)

### §3.4 Training consideration
TC-MoE requires a small adjustment to the routing loss to encourage ternary
routing outputs. Per the TC-MoE paper: add an auxiliary loss term that pushes
router outputs toward {-1, 0, +1}.

**Decision for 10B V4 sprint:** Apply TC-MoE in inference-only mode for Phase 3
(no retraining). The benefit will be smaller (~3-5% fewer activations instead
of 9%) but the code path is much simpler and doesn't conflict with HESTIA
training. Full TC-MoE with routing loss waits for V4.5.

### §3.5 Outlier-specific gotcha (CRITICAL)
The shared expert `base_FFN_L(x)` is ALWAYS applied. TC-MoE skip ONLY applies
to delta experts. Never skip the base FFN — that would break the model.

Your existing `TcMoeRouter` only handles expert routing (per your status). That
is correct. The concern is at the MoE layer forward level, NOT the router
level: the V3.2 `Qwen2MoEMLP.forward` must always execute the shared_out path
regardless of what TC-MoE says. Verification test: with a TC-MoE output that
says sign=0 for all top-k experts on a token, the layer output for that token
should still equal `base_FFN_L(x)` — non-zero, coherent. If it equals zero,
the shared path got wrongly gated.

### §3.6 Phase 3 success criterion
- Measured activation count reduction on delta experts (target: 3-5% for
  inference-only mode)
- Measured paging stall reduction (target: ~3-5% correlated with activation
  reduction)
- MMLU delta vs Phase 1 baseline should be in [-1%, +1.5%] range
- If MMLU drops more than 1%, the 0-threshold is too aggressive — tune threshold

---

## §4 — Where cross-layer predictor v2 applies (Phase 4)

### §4.1 The goal
Improve the cross-layer expert prediction accuracy from the measured 81.5%
baseline to ≥90%. The predictor takes the hidden state at layer L and predicts
which experts will be selected at layer L+1 (or L+2), so the paging system can
start prefetching from NVMe while layer L is still computing.

### §4.2 Where to apply
The predictor is a separate small model (not part of the main Outlier model)
that runs alongside during inference:During layer L forward:
h_L = compute_layer_L(x)
predicted_experts_L1 = predictor_v2(h_L)  # top-k predictions for layer L+1
async_prefetch(predicted_experts_L1)  # kicks off NVMe read in backgroundDuring layer L+1 forward:
actual_experts_L1 = router_L1(h_L)
If predicted correctly, experts are already in hot cache (no stall)
If predicted wrong, pay the NVMe cost (~5-10ms)

### §4.3 The 81.5% baseline and why v2 can beat it
V1 predictor was trained on a subset of layers with a simple MLP architecture.
V2 targets:
1. Train on ALL MoE layers (not a subset)
2. Use a deeper architecture (2 hidden layers instead of 1)
3. Train on a larger dataset of hidden states (not just the original toy set)
4. Loss = cross-entropy on the actual router selections at layer L+1

### §4.4 Outlier-specific gotcha (CRITICAL)
The predictor must run FAST — the whole point is to hide NVMe latency. If the
predictor itself takes longer than the NVMe read, there's no speedup. Target:
predictor inference < 1ms per token, measured on M1 Ultra.

Additional gotcha: predictor only predicts DELTA EXPERT selections for layer
L+1. The shared expert is always active, so there's nothing to predict for
it. Do not include the shared expert as a "class" in the predictor output.

### §4.5 Phase 4 success criterion
- Held-out predictor accuracy ≥ 90% (target)
- Predictor inference latency < 1ms per token on M1 Ultra
- Paging speedup vs V3.2 baseline measured end-to-end (target: 2-4x paged tok/s)

### §4.6 Data source for training
Capture hidden states from 10B V3.2 on a calibration dataset (e.g., WikiText-2
or a slice of The Pile) by running inference and saving (h_L, experts_L+1)
pairs. ~100K pairs should be enough for training a 2-hidden-layer MLP.

---

## §5 — Where HESTIA applies (Phase 2, waits for teacher cache)

### §5.1 The goal
Replace Straight-Through Estimator (STE) with HESTIA (Hessian-aware Soft
Ternary Intermediate Approximation) for the ternary quantization gradient.
STE treats the quantizer as identity in the backward pass, which causes
gradient mismatch at the decision boundaries. HESTIA uses a soft approximation
that's differentiable.

### §5.2 Where to apply
In the training loop, replace every `ternary_quantize(w)` backward pass:V3.2 (STE):
forward:  w_q = sign(w) * (abs(w) > threshold)  # ternary {-1, 0, +1}
backward: grad_w = grad_w_q  # identity pass-throughV4 (HESTIA):
forward:  w_q = sign(w) * (abs(w) > threshold)  # same ternary output
backward: grad_w = hestia_derivative(w, w_q, threshold) * grad_w_q
hestia_derivative approximates d(w_q)/d(w) with a smooth function

### §5.3 HESTIA derivative formulation
Per the HESTIA paper (Claude Code should research from arxiv when Phase 2
unlocks — do NOT implement from memory). The key idea: use a temperature-
scaled soft step function in the backward pass:def hestia_derivative(w, w_q, threshold, temperature=1.0):
# Soft approximation to the step function d|w|/dw near threshold
# Returns a scalar multiplier in [0, 1]
dist = abs(w) - threshold
return torch.sigmoid(-dist / temperature) * torch.sigmoid(dist / temperature + 1)

The exact formulation needs to match the paper — read the paper before coding
Phase 2. If the arxiv number isn't found via search, the fallback is the
classic QAT soft-quantization family (LSQ, PACT, DSQ) — any of those would
work as a Phase 2 placeholder and the comparison vs STE is the measurable
quantity.

### §5.4 Training data (Phase 2A teacher cache)
HESTIA training uses CAKLD loss from a teacher model's logits. The teacher is
Kimi K2.5 (primary) or MiMo-V2-Flash (fallback), cached offline via Phase 0.5
daemon so training is teacher-free after the cache is built.

Cache target: ~80 GB of (prompt, teacher_logits) pairs on common instruction
datasets (OpenOrca, UltraChat, SlimOrca, etc.) — enough for 1500 training
steps at batch size 4, context length 2048.

### §5.5 Training schedule
- 1500 steps (matches V3.2 convergence)
- Batch size 4-8 (limited by Mac memory at fp32)
- Learning rate 1e-5 to 5e-5 (lower than V3.2 because HESTIA gradients are
  smoother)
- Optimizer: AdamW for 10B (Adafactor only needed for ≥70B per Rule 58)
- Loss: CAKLD against cached teacher logits + L1 delta regularization

### §5.6 Outlier-specific gotcha
The base Qwen2.5 FFN is FROZEN during HESTIA training. Only the delta experts
and alpha scalars update. This is the same as V3.2 training — V4 doesn't
change which parameters are trainable.

Also: HESTIA training runs on the Mac at fp32 (MPS fp16 NaN bug per Rule 38).
This is slow but correct. Expected wall-clock for 1500 steps on M1 Ultra:
~5-10 days. This is the critical path of the sprint.

### §5.7 Phase 2 success criterion
- Training converges (loss monotonically decreasing, no NaN)
- Checkpoint at step 1500 loads cleanly
- Inference on checkpoint produces coherent generation (50 tokens sanity check)
- MMLU delta vs Phase 1 baseline should be in [+1%, +5%] range
- If MMLU drops vs Phase 1, diagnose per Rule 79: HESTIA temperature tuning,
  learning rate, teacher cache integrity, or CAKLD loss scaling

---

## §6 — Phase 5 stacked eval

After Phases 1, 2, 3, 4 all DONE, apply all four layers to a final 10B V4
candidate:
1. Start from 10B V3.2 weights
2. Apply Hadamard rotation (Phase 1 output)
3. HESTIA-retrained checkpoint (Phase 2 output)
4. TC-MoE skip routing (Phase 3 inference-only mode)
5. Cross-layer predictor v2 (Phase 4 output, used for paging but not quality)

Run full eval suite on the stacked V4 candidate:
- MMLU n=500 (same as Phase 0 anchor, for direct delta)
- HellaSwag n=2000 (headline number)
- ARC-Easy n=500
- Paged inference tok/s (with predictor v2 active)
- Non-paged inference tok/s (quality baseline)

All numbers in the Phase 5 DONE file must be Rule 66 compliant: source file,
exact command, n, stderr, date.

---

## §7 — Phase 0.5 teacher caching daemon (needs this info)

### §7.1 Corpus for teacher logit caching
Use public instruction datasets:
- OpenOrca (primary) — ~4M high-quality instruction-response pairs from GPT-4
- SlimOrca (secondary) — 518K deduplicated subset of OpenOrca, cleaner
- UltraChat (tertiary) — multi-turn conversations for diversity

Target: ~50K-80K prompts cached with teacher logits, enough for 1500 HESTIA
steps at batch size 4-8, context 2048.

### §7.2 Cache storage format
- Filename: `~/outlier-engine/data/teacher_cache/{dataset}_{shard_id}.safetensors`
- Each shard: ~1000 (prompt_ids, teacher_logits) pairs
- Logits stored as bf16 (half precision — full fp32 is 2x disk for no signal)
- Shard atomically (write to .tmp then rename) so crashes don't corrupt cache
- Index file: `~/outlier-engine/data/teacher_cache/index.json` — maps
  (dataset, prompt_hash) → shard_path for dedup

### §7.3 DeepSeek V3.2 API integration (pivoted from Moonshot)
- Base URL: https://api.deepseek.com/v1/chat/completions
- Model name: `deepseek-chat` (DeepSeek V3.2)
- Auth: `Authorization: Bearer <key>` — read from
  `~/.outlier/secrets/deepseek_api_key` at daemon startup, never print
- OpenAI-compatible chat/completions endpoint
- Logprobs: `logprobs=true` and `top_logprobs=20` — top-20 token-level
  logprobs per output position, enough for CAKLD distillation
- Pricing: $0.28/M input + $1.10/M output — hard cap at $12 spend
- Corpus: 48,692 prompts from existing DeepSeek-generated data at
  `~/Outlier/data/cloud/*deepseek_ultra.jsonl`, subsetted to
  `~/outlier-engine/data/teacher_cache/corpus.jsonl` (seed=42)
- Note: Moonshot Kimi K2.5 was original plan but account suspended
  (insufficient balance). DeepSeek is the same teacher that generated
  the original training data, so logprob distributions match exactly.

### §7.4 Fallback (deprecated)
Moonshot and Xiaomi fallbacks are no longer active. DeepSeek is primary
and only teacher endpoint for this sprint.

### §7.5 Daemon pacing
- Concurrent requests: start at 2, ramp to 4-8 based on rate-limit headers
- Exponential backoff on 429s (2s, 4s, 8s, 16s, 32s)
- Total target cache build time: 24-48 hours (faster is fine, slower is a
  blocker)
- Heartbeat: append to `~/OUTLIER_V4_MAC_HEARTBEAT` every 60s with
  (completed_prompts, failed_prompts, current_rate)
- Write `~/OUTLIER_V4_MAC_PHASE0_5_DONE` only when total cached prompts ≥ 50K
  AND verify step can sample and load 10 random shards successfully

---

## §8 — Expected stacked deltas (projections, NOT measurements)

Realistic case from ARCHITECTURE_v5 §6.3:
- Phase 1 (Hadamard): +1 to +2% MMLU (within PolarQuant range)
- Phase 2 (HESTIA): +2 to +3% MMLU (lower than paper estimate due to Outlier's
  zero-delta init already being quite good)
- Phase 3 (TC-MoE inference-only): +0 to +0.5% MMLU (mostly a speed play)
- Phase 4 (predictor v2): +0% MMLU (pure paging speedup, doesn't change quality)
- **Stacked realistic: +3 to +5% MMLU → 80-81% at 10B**

Pessimistic case: layers interact badly, gains sub-additive, end at 77-79%.
Optimistic case: layers stack well, +6 to +8% → 82-84%.

See `OUTLIER_ARCHITECTURE_v5.md §6.3` for the full projection table (held
separately in project knowledge).

---

## §9 — Rules applied to this sprint

- **Rule 79** (diagnose before deciding) — every failure triggers 5-attempt
  diagnose-then-retry before BLOCKED
- **Rule 91** (no write+self-verify in same session) — every DONE file must
  contain externally reproducible commands
- **Rule 94** (belief is not evidence) — watched-it-train is not a reason to
  skip eval
- **Rule 95** (optimization on unverified baselines) — V4 is running on 10B
  V3.2 which has a Rule 66-verified baseline (76.19% MMLU). Rule 95 applies
  loosely here because the baseline is the most-verified Outlier number.
- **Rule 96** (suspicion goes UP when confirming) — if Phase 5 lands exactly
  at the "realistic 80-81%" target, run additional external verification
  before writing the master DONE file.
