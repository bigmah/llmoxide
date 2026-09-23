// Prefill GEMM for Q6_K weights on Apple GPUs, using simdgroup matrices.
//
// The same computation as `gemm_q6k` in gemm.wgsl — Y[t, row] = W[row] . X[t]
// — but the inner product runs on `simdgroup_float8x8` multiply-accumulates,
// which WGSL has no way to express. That is worth several times the plain
// vec4-FMA kernel, which tops out near 2.5 TFLOP/s on an M4 Pro because every
// FMA needs its operands staged through threadgroup memory by hand.
//
// Loaded through wgpu's MSL passthrough, so nothing checks it against the
// pipeline layout. It must agree with `QuantKernels::layout` by hand: wgpu's
// Metal backend numbers buffers in bind-group entry order, giving
//   buffer(0) weights, buffer(1) x, buffer(2) y, buffer(3) params,
// and buffer(4) the token ids, which this kernel ignores.
//
// Everything accumulates in f32; the weights are dequantized to f32 as well,
// so the result matches the CPU reference as closely as the WGSL kernel does.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint w_base;
    uint in_dim;
    uint out_dim;
    uint n_tokens;
};

// Output tile: BM rows x BN tokens per threadgroup, K in steps of BK (one
// 32-element quarter of a Q6_K half-block). Four simdgroups, each owning a
// 32-row x 16-token corner as 4 x 2 accumulator matrices. Keep in sync with
// `GEMM_MSL_BM` / `GEMM_MSL_BN`.
constant constexpr uint BM = 64;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;
constant constexpr uint THREADS = 128;

// Repacked Q6_K block, 56 u32: ql at words 0..32, qh at 32..48, sixteen i8
// scales at 48..52, d (f16) in the low half of word 52. Element
// n*128 + j*32 + l decodes as
//   (nibble (j>>1) of ql[n*64 + (j&1)*32 + l] | bits 2j.. of qh[n*32 + l] << 4) - 32
// times d * scale[n*8 + j*2 + l/16].
static inline float4 q6k_4(device const uint *blk, uint n, uint j, uint l, float s) {
    uint lo = blk[(n * 64 + (j & 1) * 32 + l) / 4] >> ((j >> 1) * 4);
    uint hi = blk[32 + (n * 32 + l) / 4] >> (j * 2);
    uint4 q4 = uint4(lo, lo >> 8, lo >> 16, lo >> 24) & 15u;
    uint4 q2 = uint4(hi, hi >> 8, hi >> 16, hi >> 24) & 3u;
    return (float4(q4 | (q2 << 4)) - 32.0f) * s;
}

kernel void gemm_q6k(
    device const uint   *weights [[buffer(0)]],
    device const float4 *x4      [[buffer(1)]],
    device float        *y       [[buffer(2)]],
    constant Params     &p       [[buffer(3)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    // Weights row-major [BM][BK]; activations row-major [BN][BK]. Reused at
    // the end as per-simdgroup scratch for edge tiles.
    threadgroup float sw[BM * BK];
    threadgroup float sx[BN * BK];

    const uint row0 = tg.x * BM;
    const uint tok0 = tg.y * BN;
    const uint blocks = p.in_dim / 256;
    const uint row_stride = blocks * 56;
    const uint x_stride = p.in_dim / 4;

    // Weight loader: one row, sixteen consecutive l values. Rows past
    // out_dim are clamped: never stored, and every load stays in bounds.
    const uint lr = tid / 2;
    const uint lh = (tid % 2) * 16;
    device const uint *w_row =
        weights + p.w_base + min(row0 + lr, p.out_dim - 1) * row_stride;

    // Activation loader: BN x BK floats = 256 float4, two per thread.
    // Tokens past n_tokens are clamped the same way.
    const uint xt_a = tid / 8, xt_b = xt_a + 16, xq = tid % 8;
    device const float4 *x_a = x4 + min(tok0 + xt_a, p.n_tokens - 1) * x_stride + xq;
    device const float4 *x_b = x4 + min(tok0 + xt_b, p.n_tokens - 1) * x_stride + xq;

    // This simdgroup's corner of the output tile.
    const uint sr = (sg % 2) * 32;   // rows
    const uint st = (sg / 2) * 16;   // tokens

    simdgroup_float8x8 acc[2][4];
    for (uint a = 0; a < 2; a++)
        for (uint b = 0; b < 4; b++)
            acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint n_steps = blocks * 8;
    for (uint step = 0; step < n_steps; step++) {
        const uint bi = step / 8, c = step % 8, n = c / 4, j = c % 4;

        device const uint *blk = w_row + bi * 56;
        const float d = float(as_type<half2>(blk[52]).x);
        const uint sc_i = n * 8 + j * 2 + lh / 16;
        const float s = d * float(as_type<char4>(blk[48 + sc_i / 4])[sc_i % 4]);
        threadgroup float4 *dst = (threadgroup float4 *)(sw + lr * BK + lh);
        dst[0] = q6k_4(blk, n, j, lh, s);
        dst[1] = q6k_4(blk, n, j, lh + 4, s);
        dst[2] = q6k_4(blk, n, j, lh + 8, s);
        dst[3] = q6k_4(blk, n, j, lh + 12, s);

        const uint ko = step * (BK / 4);
        ((threadgroup float4 *)(sx + xt_a * BK))[xq] = x_a[ko];
        ((threadgroup float4 *)(sx + xt_b * BK))[xq] = x_b[ko];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 ma[2], mb[4];
            simdgroup_load(ma[0], sx + (st + 0) * BK + kk, BK);
            simdgroup_load(ma[1], sx + (st + 8) * BK + kk, BK);
            // Transposed: mb[b] is k x rows, so acc is tokens x rows.
            for (uint b = 0; b < 4; b++)
                simdgroup_load(mb[b], sw + (sr + 8 * b) * BK + kk, BK, ulong2(0, 0), true);
            for (uint a = 0; a < 2; a++)
                for (uint b = 0; b < 4; b++)
                    simdgroup_multiply_accumulate(acc[a][b], ma[a], mb[b], acc[a][b]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // acc[a][b] covers tokens tok0+st+8a.. and rows row0+sr+8b.., and y is
    // token-major with stride out_dim, so interior tiles store directly.
    threadgroup float *scratch = sw + sg * 64;
    for (uint a = 0; a < 2; a++) {
        for (uint b = 0; b < 4; b++) {
            const uint t = tok0 + st + 8 * a;
            const uint r = row0 + sr + 8 * b;
            if (t + 8 <= p.n_tokens && r + 8 <= p.out_dim) {
                simdgroup_store(acc[a][b], y + (ulong)t * p.out_dim + r, p.out_dim);
            } else {
                simdgroup_store(acc[a][b], scratch, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = lane; i < 64; i += 32) {
                    const uint ti = t + i / 8, ri = r + i % 8;
                    if (ti < p.n_tokens && ri < p.out_dim) {
                        y[(ulong)ti * p.out_dim + ri] = scratch[i];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
