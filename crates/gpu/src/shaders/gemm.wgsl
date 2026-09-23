// Tiled quantized GEMM for prefill: Y[t, row] = sum_k W[row, k] * X[t, k].
//
// The matvec kernels in quant.wgsl reuse each weight row across only TILE
// tokens, so a prefill of T tokens streams the whole model T/TILE times. Here a
// workgroup owns a BM x BN tile of the output and walks K in BK-wide steps:
// each step dequantizes a BM x BK slab of weights and stages a BN x BK slab of
// activations in workgroup memory, then every thread accumulates an 8-row x
// 4-token register block from them. Weights are read once per BN tokens, and
// the inner loop is three vec4 loads and eight vec4 FMAs.
//
// Three things about the generated Metal decide this kernel's speed, and none
// of them is visible in the WGSL (`msl --gemm` prints it):
//
// * The accumulators are named vec4 variables, not an array. An array indexed
//   by a loop variable lands in per-thread scratch (see the TILE note in
//   quant.wgsl), and that alone would cap occupancy.
// * Both slabs are arrays of vec4 and every store writes a whole vec4. Naga's
//   Metal output drops dynamic-component stores into workgroup vectors —
//   `ws[i][c] = v` wrote component 0 only — and the scalar-array alternative
//   turns each inner-loop read into four scalar loads.
// * The K loop is unrolled by hand. Naga emits every loop as `while (true)`
//   with a first-iteration flag, which hides the constant trip count.
//
// Shares the bind group layout of quant.wgsl, so `QuantKernels::layout` and the
// same `MatvecParams` serve both.

struct Params {
    w_base: u32,
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
};

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// Shape, substituted by `gemm_shader_source`. Threads are laid out TH_R x TH_T,
// each owning RM4*4 rows x TN tokens; the loaders below need BM * BK = 4096
// (one 4x4 weight patch per thread) and BK dividing the 32-element quarter.
const RM4: u32 = @@RM4@@u;  // vec4s of rows per thread
const TN: u32 = @@TN@@u;    // tokens per thread, a multiple of 4
const TH_R: u32 = 16u;
const TH_T: u32 = 16u;
const WG: u32 = TH_R * TH_T;
const BM: u32 = TH_R * RM4 * 4u;
const BN: u32 = TH_T * TN;
const BK: u32 = 4096u / BM;
const BM4: u32 = BM / 4u;
const BN4: u32 = BN / 4u;
const TN4: u32 = TN / 4u;

// K-major: element [k][r4] is rows 4*r4..4*r4+4 at column k.
var<workgroup> ws: array<vec4<f32>, 1024>;  // BK * BM / 4, always 1024
var<workgroup> xs: array<vec4<f32>, @@XS@@>;

fn byte_at(base: u32, off: u32) -> u32 {
    return (weights[base + (off >> 2u)] >> ((off & 3u) * 8u)) & 0xFFu;
}

fn sc_i8(base: u32, i: u32) -> f32 {
    let b = i32(byte_at(base, i));
    return f32(select(b, b - 256, b > 127));
}

// ---------------------------------------------------------------- Q6_K ----
//
// Repacked blocks are 56 u32: ql at words 0..32, qh at 32..48, the sixteen i8
// scales at 48..52, d (f16) in the low half of word 52. Element e of a block
// is n*128 + j*32 + l (n = half, j = quarter, l < 32), and decodes as
//   ql = byte(n*64 + (j&1)*32 + l), nibble j>>1
//   qh = byte(n*32 + l), bits 2j..2j+2
//   q  = (nibble | qh << 4) - 32, scaled by d * sc[n*8 + j*2 + l/16]
// so a BK step lies inside one (n, j) quarter.

// Four consecutive l values of one row, dequantized.
fn q6k_4(blk: u32, n: u32, j: u32, l: u32) -> vec4<f32> {
    let lo = weights[blk + (n * 64u + (j & 1u) * 32u + l) / 4u] >> ((j >> 1u) * 4u);
    let hi = weights[blk + 32u + (n * 32u + l) / 4u] >> (j * 2u);
    let q4 = vec4<u32>(lo, lo >> 8u, lo >> 16u, lo >> 24u) & vec4<u32>(15u);
    let q2 = vec4<u32>(hi, hi >> 8u, hi >> 16u, hi >> 24u) & vec4<u32>(3u);
    let d = unpack2x16float(weights[blk + 52u]).x;
    let s = d * sc_i8(blk + 48u, n * 8u + j * 2u + l / 16u);
    return (vec4<f32>(q4 | (q2 << vec4<u32>(4u))) - 32.0) * s;
}

// Rows past out_dim are clamped to the last one: their results are never
// stored, and clamping keeps every load in bounds without a branch.
fn row_base(r: u32, stride: u32) -> u32 {
    return p.w_base + min(r, p.out_dim - 1u) * stride;
}

@compute @workgroup_size(WG)
fn gemm_q6k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let row0 = wg.x * BM;
    let tok0 = wg.y * BN;
    let blocks = p.in_dim / 256u;
    let row_stride = blocks * 56u;
    let x_stride = p.in_dim / 4u;

    // Weight loader: rows 4*wr..4*wr+4, l values wl..wl+4. Each thread
    // dequantizes a 4x4 patch and stores it transposed as four vec4 columns.
    let wr = tid % BM4;
    let wl = (tid / BM4) * 4u;
    let w0 = row_base(row0 + wr * 4u, row_stride);
    let w1 = row_base(row0 + wr * 4u + 1u, row_stride);
    let w2 = row_base(row0 + wr * 4u + 2u, row_stride);
    let w3 = row_base(row0 + wr * 4u + 3u, row_stride);

    // Activation loader (the first BN4 * BK/4 threads): tokens 4*xt..4*xt+4,
    // k quad xk, transposed the same way. Tokens past n_tokens are clamped
    // like rows are.
    let xt = tid % BN4;
    let xk = tid / BN4;
    let t_base = tok0 + xt * 4u;
    let last = p.n_tokens - 1u;
    let xt0 = min(t_base, last) * x_stride;
    let xt1 = min(t_base + 1u, last) * x_stride;
    let xt2 = min(t_base + 2u, last) * x_stride;
    let xt3 = min(t_base + 3u, last) * x_stride;

    // Compute roles.
    let tr = tid % TH_R;
    let tc = tid / TH_R;

    // @@ACC_DECL@@

    let per_block = 256u / BK;
    let n_steps = blocks * per_block;
    for (var step = 0u; step < n_steps; step = step + 1u) {
        let bi = step / per_block;
        let e0 = (step % per_block) * BK;
        let n = e0 / 128u;
        let j = (e0 % 128u) / 32u;
        let l = e0 % 32u + wl;

        // --- dequantize BM x BK weights -----------------------------------
        let bo = bi * 56u;
        let r0 = q6k_4(w0 + bo, n, j, l);
        let r1 = q6k_4(w1 + bo, n, j, l);
        let r2 = q6k_4(w2 + bo, n, j, l);
        let r3 = q6k_4(w3 + bo, n, j, l);
        let wbase = wl * BM4 + wr;
        ws[wbase] = vec4<f32>(r0.x, r1.x, r2.x, r3.x);
        ws[wbase + BM4] = vec4<f32>(r0.y, r1.y, r2.y, r3.y);
        ws[wbase + 2u * BM4] = vec4<f32>(r0.z, r1.z, r2.z, r3.z);
        ws[wbase + 3u * BM4] = vec4<f32>(r0.w, r1.w, r2.w, r3.w);

        // --- stage BN x BK activations -----------------------------------
        if (tid < BN4 * (BK / 4u)) {
            let ko = step * (BK / 4u) + xk;
            let v0 = x4[xt0 + ko];
            let v1 = x4[xt1 + ko];
            let v2 = x4[xt2 + ko];
            let v3 = x4[xt3 + ko];
            let xbase = xk * 4u * BN4 + xt;
            xs[xbase] = vec4<f32>(v0.x, v1.x, v2.x, v3.x);
            xs[xbase + BN4] = vec4<f32>(v0.y, v1.y, v2.y, v3.y);
            xs[xbase + 2u * BN4] = vec4<f32>(v0.z, v1.z, v2.z, v3.z);
            xs[xbase + 3u * BN4] = vec4<f32>(v0.w, v1.w, v2.w, v3.w);
        }
        workgroupBarrier();

        // --- accumulate ---------------------------------------------------
        // Four k per iteration, written out by `gemm_shader_source`.
        for (var k = 0u; k < BK; k = k + 4u) {
            // @@ACC_FMA@@
        }
        workgroupBarrier();
    }

    let r = row0 + tr * RM4 * 4u;
    let t = tok0 + tc * TN;
    // @@ACC_STORE@@
}

fn store4(r: u32, t: u32, v: vec4<f32>) {
    if (t >= p.n_tokens) { return; }
    let o = t * p.out_dim + r;
    if (r < p.out_dim) { y[o] = v.x; }
    if (r + 1u < p.out_dim) { y[o + 1u] = v.y; }
    if (r + 2u < p.out_dim) { y[o + 2u] = v.z; }
    if (r + 3u < p.out_dim) { y[o + 3u] = v.w; }
}
