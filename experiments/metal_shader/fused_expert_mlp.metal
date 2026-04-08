#include <metal_stdlib>
using namespace metal;

// Fused SwiGLU expert MLP kernel with TQ1_0 dequantization.
//
// Architecture:
//   gate_proj: [I, D] (I rows, D cols) — TQ1_0 packed
//   up_proj:   [I, D] (I rows, D cols) — TQ1_0 packed
//   down_proj: [D, I] (D rows, I cols) — TQ1_0 packed
//
// Forward pass:
//   gate_vec[i] = dot(hidden[D], gate_proj[i, :])
//   up_vec[i]   = dot(hidden[D], up_proj[i, :])
//   mid[i]      = silu(gate_vec[i]) * up_vec[i]
//   output[d]   = sum_i(mid[i] * down_proj[d, i])
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// TQ1_0: 1 byte per 5 weights, group_size=32 for scales.
// bytes_per_row(D cols) = ceil(D / 5)
// scales_per_row(D cols) = ceil(D / 32)

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// Decode a TQ1_0 byte and return the i-th ternary value (0-indexed, i in 0..4)
inline int decode_tq1_val(uint byte_val, uint pos) {
    uint tmp = byte_val;
    for (uint k = 0; k < pos; k++) tmp /= 3u;
    return (int)(tmp % 3u) - 1;
}

// Dot product of hidden[D] with a TQ1_0-packed row.
// packed_row: pointer to ceil(D/5) bytes
// row_scales: pointer to ceil(D/32) scales
// D: number of elements in hidden
inline float tq1_dot(
    device const half*  hidden,
    device const uchar* packed_row,
    device const half*  row_scales,
    uint                D
) {
    float acc = 0.0f;
    uint byte_idx = 0u;
    uint w_idx = 0u;

    while (w_idx < D) {
        uint b = (uint)packed_row[byte_idx];
        uint tmp = b;
        for (int pos = 0; pos < 5 && w_idx < D; pos++, w_idx++) {
            int w = (int)(tmp % 3u) - 1;
            tmp /= 3u;
            uint scale_idx = w_idx / 32u;
            float scale = (float)row_scales[scale_idx];
            acc += (float)w * scale * (float)hidden[w_idx];
        }
        byte_idx++;
    }
    return acc;
}

// Fused expert MLP kernel.
// Thread = one intermediate neuron (i in 0..I-1).
// Phase 1: compute gate[i] and up[i] dot products.
// Phase 2: compute silu(gate[i]) * up[i] -> mid[i], stored in threadgroup.
// Phase 3: each thread (now repurposed as output dim d) computes down projection.
//
// For simplicity and correctness, this kernel computes one output at a time,
// using a single threadgroup of 256 threads covering intermediate dims.
// Output dim loop is done serially per-thread in phase 3.
//
// Layout constants:
//   bytes_per_row_GU = ceil(D / 5)   — gate/up rows
//   scales_per_row_GU = ceil(D / 32) — gate/up scales
//   bytes_per_row_D_proj = ceil(I / 5)   — down rows
//   scales_per_row_D_proj = ceil(I / 32) — down scales

[[max_total_threads_per_threadgroup(256)]]
kernel void fused_expert_mlp(
    device const half*  hidden           [[buffer(0)]],
    device const uchar* gate_bytes       [[buffer(1)]],
    device const half*  gate_scales      [[buffer(2)]],
    device const uchar* up_bytes         [[buffer(3)]],
    device const half*  up_scales        [[buffer(4)]],
    device const uchar* down_bytes       [[buffer(5)]],
    device const half*  down_scales      [[buffer(6)]],
    device       half*  output           [[buffer(7)]],
    constant     uint&  D                [[buffer(8)]],
    constant     uint&  I                [[buffer(9)]],
    uint                tg_tid           [[thread_position_in_threadgroup]],
    uint                tg_size          [[threads_per_threadgroup]]
) {
    // Strides for gate/up projections (I rows of D weights each)
    uint bytes_per_gu_row   = (D + 4u) / 5u;
    uint scales_per_gu_row  = (D + 31u) / 32u;
    uint bytes_per_dn_row   = (I + 4u) / 5u;
    uint scales_per_dn_row  = (I + 31u) / 32u;

    // Threadgroup storage for mid vector [I]
    // We can't allocate variable-length arrays easily; use fixed size.
    // Max intermediate dim we support in threadgroup = 256 (matches tg_size for phase 1).
    // For I > 256, threads handle multiple intermediate neurons.
    threadgroup float mid_tg[256]; // stores mid[0..min(I,256)-1] per threadgroup

    // Phase 1 & 2: each thread computes gate[i] and up[i] for i = tg_tid
    for (uint i = tg_tid; i < I; i += tg_size) {
        // Gate projection dot product
        device const uchar* gate_row   = gate_bytes + i * bytes_per_gu_row;
        device const half*  gscale_row = gate_scales + i * scales_per_gu_row;
        float gate_val = tq1_dot(hidden, gate_row, gscale_row, D);

        // Up projection dot product
        device const uchar* up_row     = up_bytes + i * bytes_per_gu_row;
        device const half*  uscale_row = up_scales + i * scales_per_gu_row;
        float up_val = tq1_dot(hidden, up_row, uscale_row, D);

        // SwiGLU activation: silu(gate) * up
        mid_tg[i % 256u] = silu(gate_val) * up_val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: each thread computes output[d] for d = tg_tid
    for (uint d = tg_tid; d < D; d += tg_size) {
        device const uchar* dn_row     = down_bytes + d * bytes_per_dn_row;
        device const half*  dscale_row = down_scales + d * scales_per_dn_row;

        // Dot product: sum_i( mid[i] * down_proj[d, i] )
        float acc = 0.0f;
        uint byte_idx = 0u;
        uint i = 0u;
        while (i < I) {
            uint b = (uint)dn_row[byte_idx];
            uint tmp = b;
            for (int pos = 0; pos < 5 && i < I; pos++, i++) {
                int w = (int)(tmp % 3u) - 1;
                tmp /= 3u;
                uint scale_idx = i / 32u;
                float scale = (float)dscale_row[scale_idx];
                // mid[i] is stored in mid_tg[i % 256]
                acc += (float)w * scale * mid_tg[i % 256u];
            }
            byte_idx++;
        }
        output[d] = (half)acc;
    }
}
