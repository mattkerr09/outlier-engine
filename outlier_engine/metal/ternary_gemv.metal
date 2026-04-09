#include <metal_stdlib>
using namespace metal;

// Minimal no-op kernel for measuring per-dispatch overhead.
kernel void noop(uint tid [[thread_position_in_grid]]) { (void)tid; }

// =============================================================================
// Production TQ1_0 GEMV + Fused SwiGLU — Outlier Engine
//
// TQ1_0 encoding: 1 byte → 5 ternary values {-1,0,+1} in base-3 positional form
//   w[j] = ((byte / 3^j) % 3) - 1,  j = 0..4
// group_size = 32 weights per fp16 scale factor
//
// Optimizations over naive prototype:
//   1. LUT in threadgroup shared memory — O(1) decode per byte vs O(5) div/mod
//   2. Input vector x cached in threadgroup shared memory — avoid global reads
//   3. Scale hoisted per 32-weight group — 112 instead of 3584 global scale reads
//   4. Gate + Up computed in a single pass — shared x accesses, ILP between streams
//   5. Float32 accumulation — avoids fp16 precision loss in 3584-dim dot products
//
// Target dimensions (Qwen2.5-7B expert):
//   D = 3584  (hidden dim)
//   I = 18944 (intermediate dim)
// =============================================================================

// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
inline float silu(float x) {
    return x * (1.0f / (1.0f + exp(-x)));
}

// Threadgroup memory layout for gate/up kernels:
//   lut[243][8]   — 243 * 8 = 1944 bytes  (padded to 8 for alignment)
//   x_shared[D_MAX] half — 4096 * 2 = 8192 bytes
//   Total: ~10 KB  (well within 32 KB M1 threadgroup limit)

constant uint TG_SIZE = 128;   // threads per threadgroup
constant uint D_MAX   = 4096;  // max hidden dim (covers D=3584 with headroom)

// =============================================================================
// KERNEL 1: tq1_gate_up_swiglu
//
// Fused gate + up projection with SwiGLU activation.
// Each thread i computes:
//   gate[i] = dot(x, gate_W[i, :])
//   up[i]   = dot(x, up_W[i, :])
//   mid[i]  = silu(gate[i]) * up[i]
//
// Dispatch: I threads total, TG_SIZE per threadgroup.
// =============================================================================
[[max_total_threads_per_threadgroup(128)]]
kernel void tq1_gate_up_swiglu(
    device const half*  x            [[buffer(0)]],  // [D] input hidden state
    device const uchar* gate_packed  [[buffer(1)]],  // [I * bytes_per_row]
    device const half*  gate_scales  [[buffer(2)]],  // [I * scales_per_row]
    device const uchar* up_packed    [[buffer(3)]],  // [I * bytes_per_row]
    device const half*  up_scales    [[buffer(4)]],  // [I * scales_per_row]
    device       float* mid          [[buffer(5)]],  // [I] output (float32)
    constant     uint&  I            [[buffer(6)]],
    constant     uint&  D            [[buffer(7)]],
    uint tid    [[thread_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]]
) {
    // ---- Phase 0: build shared state ----------------------------------------
    // LUT: lut[b] encodes the 5 ternary values for byte value b.
    // Padded to stride-8 for coalesced threadgroup loads.
    threadgroup int8_t lut[243][8];
    threadgroup half   x_shared[D_MAX];

    // Distribute LUT init across threads.
    for (uint e = tg_tid; e < 243u; e += TG_SIZE) {
        uint tmp = e;
        for (int j = 0; j < 5; j++) {
            lut[e][j] = (int8_t)((int)(tmp % 3u) - 1);
            tmp /= 3u;
        }
    }

    // Load x into threadgroup shared memory.
    for (uint k = tg_tid; k < D; k += TG_SIZE) {
        x_shared[k] = x[k];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 1: fused gate + up dot products --------------------------------
    if (tid >= I) return;

    const uint bytes_per_row  = (D + 4u) / 5u;
    const uint scales_per_row = (D + 31u) / 32u;

    device const uchar* gate_row = gate_packed + tid * bytes_per_row;
    device const half*  gate_sc  = gate_scales  + tid * scales_per_row;
    device const uchar* up_row   = up_packed    + tid * bytes_per_row;
    device const half*  up_sc    = up_scales    + tid * scales_per_row;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    // Hoist first scale; update every 32 weights.
    float gate_scale = (float)gate_sc[0];
    float up_scale   = (float)up_sc[0];
    uint  next_sw    = 32u;   // next weight index where scale changes
    uint  w          = 0u;

    for (uint b = 0u; w < D; b++) {
        // Load both gate and up bytes for this column group.
        const threadgroup int8_t* ge = lut[(uint)gate_row[b]];
        const threadgroup int8_t* ue = lut[(uint)up_row[b]];

        // Unrolled 5-value inner loop with scale-boundary check.
        for (int j = 0; j < 5 && w < D; j++, w++) {
            if (w == next_sw) {
                uint si  = w >> 5u;  // w / 32
                gate_scale = (float)gate_sc[si];
                up_scale   = (float)up_sc[si];
                next_sw   += 32u;
            }
            float xv     = (float)x_shared[w];
            gate_acc += (float)ge[j] * gate_scale * xv;
            up_acc   += (float)ue[j] * up_scale   * xv;
        }
    }

    mid[tid] = silu(gate_acc) * up_acc;
}


// =============================================================================
// KERNEL 2: tq1_down_proj
//
// Down projection: output[d] = dot(mid[I], down_W[d, :])
// mid is float32 (from gate_up_swiglu), output is fp16.
//
// mid[I=18944] = 75.8 KB — too large for threadgroup; read from global memory.
// The M1 12 MB L2 cache keeps mid hot after the first threadgroup warms it.
//
// Dispatch: D threads total, TG_SIZE per threadgroup.
// =============================================================================
[[max_total_threads_per_threadgroup(128)]]
kernel void tq1_down_proj(
    device const float* mid          [[buffer(0)]],  // [I] float32 from gate_up
    device const uchar* down_packed  [[buffer(1)]],  // [D * bytes_per_row]
    device const half*  down_scales  [[buffer(2)]],  // [D * scales_per_row]
    device       half*  out          [[buffer(3)]],  // [D] output
    constant     uint&  D            [[buffer(4)]],
    constant     uint&  I            [[buffer(5)]],
    uint tid    [[thread_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]]
) {
    // LUT in threadgroup memory for fast base-3 decode.
    threadgroup int8_t lut[243][8];

    for (uint e = tg_tid; e < 243u; e += TG_SIZE) {
        uint tmp = e;
        for (int j = 0; j < 5; j++) {
            lut[e][j] = (int8_t)((int)(tmp % 3u) - 1);
            tmp /= 3u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= D) return;

    const uint bytes_per_row  = (I + 4u) / 5u;
    const uint scales_per_row = (I + 31u) / 32u;

    device const uchar* dn_row = down_packed + tid * bytes_per_row;
    device const half*  dn_sc  = down_scales  + tid * scales_per_row;

    float acc       = 0.0f;
    float cur_scale = (float)dn_sc[0];
    uint  next_sw   = 32u;
    uint  w         = 0u;

    for (uint b = 0u; w < I; b++) {
        const threadgroup int8_t* e = lut[(uint)dn_row[b]];

        for (int j = 0; j < 5 && w < I; j++, w++) {
            if (w == next_sw) {
                cur_scale = (float)dn_sc[w >> 5u];
                next_sw  += 32u;
            }
            acc += (float)e[j] * cur_scale * mid[w];
        }
    }

    out[tid] = (half)acc;
}


// =============================================================================
// KERNEL 3: tq1_gemv
//
// General TQ1_0 GEMV: y[M] = W[M, D] @ x[D]
// One thread per output element. x cached in threadgroup shared memory.
// Used for unit testing and standalone benchmarking of a single projection.
// =============================================================================
[[max_total_threads_per_threadgroup(128)]]
kernel void tq1_gemv(
    device const half*  x        [[buffer(0)]],  // [D]
    device const uchar* W_packed [[buffer(1)]],  // [M * bytes_per_row]
    device const half*  W_scales [[buffer(2)]],  // [M * scales_per_row]
    device       half*  y        [[buffer(3)]],  // [M]
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  D        [[buffer(5)]],
    uint tid    [[thread_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]]
) {
    threadgroup int8_t lut[243][8];
    threadgroup half   x_shared[D_MAX];

    for (uint e = tg_tid; e < 243u; e += TG_SIZE) {
        uint tmp = e;
        for (int j = 0; j < 5; j++) {
            lut[e][j] = (int8_t)((int)(tmp % 3u) - 1);
            tmp /= 3u;
        }
    }
    for (uint k = tg_tid; k < D; k += TG_SIZE) {
        x_shared[k] = x[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= M) return;

    const uint bytes_per_row  = (D + 4u) / 5u;
    const uint scales_per_row = (D + 31u) / 32u;

    device const uchar* row_bytes  = W_packed + tid * bytes_per_row;
    device const half*  row_scales = W_scales  + tid * scales_per_row;

    float acc       = 0.0f;
    float cur_scale = (float)row_scales[0];
    uint  next_sw   = 32u;
    uint  w         = 0u;

    for (uint b = 0u; w < D; b++) {
        const threadgroup int8_t* e = lut[(uint)row_bytes[b]];

        for (int j = 0; j < 5 && w < D; j++, w++) {
            if (w == next_sw) {
                cur_scale = (float)row_scales[w >> 5u];
                next_sw  += 32u;
            }
            acc += (float)e[j] * cur_scale * (float)x_shared[w];
        }
    }

    y[tid] = (half)acc;
}
