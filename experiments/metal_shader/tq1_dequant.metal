#include <metal_stdlib>
using namespace metal;

// TQ1_0 dequantization kernel.
// Encoding: 1 byte encodes 5 ternary values {-1, 0, 1} via base-3 positional encoding.
//   w0 = (B % 3) - 1
//   w1 = ((B / 3) % 3) - 1
//   w2 = ((B / 9) % 3) - 1
//   w3 = ((B / 27) % 3) - 1
//   w4 = ((B / 81) % 3) - 1
// group_size = 32 weights -> ceil(32/5) = 7 bytes per group, 1 scale per group.
// scale_idx = weight_idx / 32

// Precomputed 243-entry lookup table: entry[b] gives 5 ternary values packed
// as int8: values[0..4] = (b%3)-1, ((b/3)%3)-1, ((b/9)%3)-1, ((b/27)%3)-1, ((b/81)%3)-1
// Packed as int8x4 + 1 spare byte. We store as a uint32 with the 5 values
// in bits [7:0], [15:8], [23:16], [31:24] of two uint32s.
// Simpler: just store as 5 separate int8 values per entry -> 243*5 = 1215 bytes.
// We'll use a threadgroup array of char[243][5].

// Build the LUT at compile time using constexpr.
// Since Metal doesn't support constexpr functions easily for LUT init,
// we decode inline using integer arithmetic (compiler optimizes divisions by 3).

kernel void dequant_tq1(
    device const uchar*  packed      [[buffer(0)]],
    device const half*   scales      [[buffer(1)]],
    device       half*   output      [[buffer(2)]],
    constant     uint&   num_weights [[buffer(3)]],
    uint                 tid         [[thread_position_in_grid]]
) {
    // Each thread handles one byte = up to 5 weights.
    uint byte_idx = tid;
    uint base_weight_idx = byte_idx * 5u;

    if (base_weight_idx >= num_weights) return;

    uint b = (uint)packed[byte_idx];

    // Decode 5 ternary values
    int w[5];
    uint tmp = b;
    for (int i = 0; i < 5; i++) {
        w[i] = (int)(tmp % 3u) - 1;
        tmp /= 3u;
    }

    // Apply scales and write output
    for (int i = 0; i < 5; i++) {
        uint wi = base_weight_idx + (uint)i;
        if (wi >= num_weights) break;
        uint scale_idx = wi / 32u;
        output[wi] = (half)((float)w[i] * (float)scales[scale_idx]);
    }
}

// Vectorized version: uses threadgroup LUT for faster decode on large workloads.
kernel void dequant_tq1_lut(
    device const uchar*  packed        [[buffer(0)]],
    device const half*   scales        [[buffer(1)]],
    device       half*   output        [[buffer(2)]],
    constant     uint&   num_weights   [[buffer(3)]],
    uint                 tid           [[thread_position_in_grid]],
    uint                 tg_size       [[threads_per_threadgroup]],
    uint                 tg_id         [[thread_position_in_threadgroup]]
) {
    // Build 243-entry LUT in threadgroup memory.
    // Each entry: 5 x int8 ternary values.
    threadgroup int8_t lut[243][5];

    // Distribute LUT initialization across threads.
    for (uint entry = tg_id; entry < 243u; entry += tg_size) {
        uint tmp = entry;
        for (int i = 0; i < 5; i++) {
            lut[entry][i] = (int8_t)((int)(tmp % 3u) - 1);
            tmp /= 3u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint byte_idx = tid;
    uint base_weight_idx = byte_idx * 5u;
    if (base_weight_idx >= num_weights) return;

    uint b = (uint)packed[byte_idx];
    const threadgroup int8_t* entry = lut[b];

    for (int i = 0; i < 5; i++) {
        uint wi = base_weight_idx + (uint)i;
        if (wi >= num_weights) break;
        uint scale_idx = wi / 32u;
        output[wi] = (half)((float)entry[i] * (float)scales[scale_idx]);
    }
}
