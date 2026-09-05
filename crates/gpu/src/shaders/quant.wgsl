// Quantized matrix-vector kernels: k-quant blocks are decoded inside the dot
// product rather than materialized. Decode is memory-bound, so reading 4-bit
// weights and unpacking in registers beats reading dequantized f16.
//
// Weights live in one big storage buffer addressed by u32 index. Q4_K blocks
// are 144 bytes (36 u32) as on disk; Q6_K blocks are repacked at upload from
// 210 to 224 bytes (56 u32) so every block starts u32-aligned — 210 is not a
// multiple of 4, and unaligned block strides would cost a shift on every load.
//
// Each quant type has two entry points:
//
//   matvec_*    decode: one token, a scalar accumulator.
//   matvec_*_t  prefill: TILE tokens, sharing each weight row across the tile.
//
// `@@NAME@@` and `// @@NAME@@` are substituted by `quant_shader_source`.
//
// Two rewrites that look like clear wins here are not. Both were measured on
// an M4 Pro, and both are worth knowing about before trying them again:
//
// * Hoisting the unpack above the token loop, so a weight is decoded once per
//   tile rather than once per token. It loses: the token loop then reads x
//   with a stride instead of streaming it, and keeping the decoded weights
//   live costs enough registers to halve occupancy. Prefill got 1.6x slower.
// * Widening TILE. `acc` is indexed by a loop variable, which neither Naga nor
//   the Metal compiler promotes to registers, so it lives in per-thread
//   scratch and its size is what caps occupancy. Going from TILE 32 to 2 made
//   prefill 2.5x faster despite multiplying weight traffic by 16 — this kernel
//   is bound by occupancy, not by bandwidth.

struct Params {
    w_base: u32,      // u32 index of the weight tensor in `weights`
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
};

@group(0) @binding(0) var<storage, read> weights: array<u32>;
// Activations are read as vec4: every access below is 4-element aligned
// (in_dim is a multiple of 256, and all block offsets are multiples of 4), and
// scalar loads from storage were a large part of the kernel's cost.
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// Each workgroup computes ROWS output rows, LANES threads cooperating on each.
// LANES matches the hardware subgroup width so the per-row reduction is a
// single subgroupAdd; grouping ROWS of them per workgroup keeps occupancy up
// and lets the rows share the activation vector while it is hot in cache.
const LANES: u32 = @@LANES@@u;
const ROWS: u32 = @@ROWS@@u;
const WG: u32 = LANES * ROWS;
// Tokens per dispatch of the tiled kernels. Small on purpose — see the note
// on TILE at the top of this file.
const TILE: u32 = @@TILE@@u;

// @@REDUCE@@

// Reduce one row's lane group and write it. Called unconditionally so the
// barrier fallback's reduction stays in uniform control flow.
fn store_row(tid: u32, lane: u32, v: f32, row: u32, tok: u32) {
    let total = reduce_row(tid, lane, v);
    if (lane == 0u && row < p.out_dim && tok < p.n_tokens) {
        y[tok * p.out_dim + row] = total;
    }
}

fn byte_at(base: u32, off: u32) -> u32 {
    return (weights[base + (off >> 2u)] >> ((off & 3u) * 8u)) & 0xFFu;
}

fn sc_i8(base: u32, i: u32) -> f32 {
    let b = i32(byte_at(base, i));
    return f32(select(b, b - 256, b > 127));
}

// ---------------------------------------------------------------- Q4_K ----
//
// A block holds 256 weights as eight 32-element sub-blocks. Sub-block j (0..8)
// carries a 6-bit scale and a 6-bit min packed across 12 bytes. Sub-blocks 0..4
// hold both plainly; 4..8 borrow their top two bits from the high bits of the
// first eight bytes.
fn q4k_scale_min(sbase: u32, j: u32) -> vec2<f32> {
    var sc: u32;
    var m: u32;
    if (j < 4u) {
        sc = byte_at(sbase, j) & 63u;
        m = byte_at(sbase, j + 4u) & 63u;
    } else {
        let a = byte_at(sbase, j + 4u);
        sc = (a & 15u) | ((byte_at(sbase, j - 4u) >> 6u) << 4u);
        m = (a >> 4u) | ((byte_at(sbase, j) >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(m));
}

// The dequantized value is `d*scale*q - dmin*min`, an affine function of the
// 4-bit quant. Folding the block's d/dmin together with the sub-block's scale
// and min into (a, c) coefficients means the unpack below is one FMA per
// element and the token loop is a plain dot product — the earlier form carried
// the min term as a separate running sum of the activations, which every row
// recomputed identically.
//
// Returns (a, c) for the low-nibble sub-block in .xy and the high-nibble one
// in .zw. One unit of work is that 64-element *pair*: the two sub-blocks share
// 32 weight bytes, so splitting them across lanes would read those bytes twice.
fn q4k_affine(blk: u32, half: u32) -> vec4<f32> {
    let dm = unpack2x16float(weights[blk]);
    let sm0 = q4k_scale_min(blk + 1u, half * 2u);
    let sm1 = q4k_scale_min(blk + 1u, half * 2u + 1u);
    return vec4<f32>(
        dm.x * sm0.x, -dm.y * sm0.y,
        dm.x * sm1.x, -dm.y * sm1.y);
}

fn q4k_lo(packed: u32, ac: vec4<f32>) -> vec4<f32> {
    let n = vec4<u32>(packed, packed >> 8u, packed >> 16u, packed >> 24u) & vec4<u32>(15u);
    return vec4<f32>(n) * ac.x + ac.y;
}

fn q4k_hi(packed: u32, ac: vec4<f32>) -> vec4<f32> {
    let n = vec4<u32>(packed >> 4u, packed >> 12u, packed >> 20u, packed >> 28u) & vec4<u32>(15u);
    return vec4<f32>(n) * ac.z + ac.w;
}

@compute @workgroup_size(WG)
fn matvec_q4k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let blocks = p.in_dim / 256u;
    let n_pairs = blocks * 4u;        // 64-element sub-block pairs
    let row_stride = blocks * 36u;    // u32 per weight row

    // Rows are strided over the grid: the output projection has 262144 rows,
    // well past the 65535-per-dimension dispatch limit.
    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        var acc = 0.0;
        var pi = lane;
        loop {
            if (pi >= n_pairs) { break; }
            let b = pi / 4u;
            let half = pi % 4u;
            let blk = row_base + b * 36u;
            let ac = q4k_affine(blk, half);
            let qs_base = blk + 4u + half * 8u;
            let off = (b * 256u + half * 64u) / 4u;

            for (var w = 0u; w < 8u; w = w + 1u) {
                let packed = weights[qs_base + w];
                let base = off + w;
                acc = acc
                    + dot(q4k_lo(packed, ac), x4[base])
                    + dot(q4k_hi(packed, ac), x4[base + 8u]);
            }
            pi = pi + LANES;
        }

        let total = reduce_row(tid, lane, acc);
        if (lane == 0u && row < p.out_dim) {
            y[row] = total;
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

@compute @workgroup_size(WG)
fn matvec_q4k_t(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let tile_base = wg.y * TILE;
    let blocks = p.in_dim / 256u;
    let n_pairs = blocks * 4u;
    let row_stride = blocks * 36u;
    let x_stride = p.in_dim / 4u;
    // Uniform across the workgroup, so loops bounded by it may contain the
    // barrier fallback's reduction.
    let tile_n = min(TILE, p.n_tokens - min(tile_base, p.n_tokens));

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        var acc: array<f32, TILE>;
        for (var i = 0u; i < tile_n; i = i + 1u) { acc[i] = 0.0; }

        var pi = lane;
        loop {
            if (pi >= n_pairs) { break; }
            let b = pi / 4u;
            let half = pi % 4u;
            let blk = row_base + b * 36u;
            let ac = q4k_affine(blk, half);
            let qs_base = blk + 4u + half * 8u;
            let off = (b * 256u + half * 64u) / 4u;

            // Token outermost, so each token's 16 activation reads are
            // consecutive vec4s. Hoisting the unpack above this loop instead
            // trades that streaming order — and enough registers to halve
            // occupancy — for arithmetic that is not the binding constraint
            // here; measured, it costs more than it saves.
            for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
                let xb = (tile_base + tt) * x_stride + off;
                var s = 0.0;
                for (var w = 0u; w < 8u; w = w + 1u) {
                    let packed = weights[qs_base + w];
                    s = s
                        + dot(q4k_lo(packed, ac), x4[xb + w])
                        + dot(q4k_hi(packed, ac), x4[xb + w + 8u]);
                }
                acc[tt] = acc[tt] + s;
            }
            pi = pi + LANES;
        }

        for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
            store_row(tid, lane, acc[tt], row, tile_base + tt);
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

// ---------------------------------------------------------------- Q6_K ----
//
// One unit is eight consecutive `l` values within a (block, half), covering 32
// outputs across four sub-blocks. Reading whole u32s and hoisting the four
// sub-block scales cuts the load count ~6x versus per-byte access.
struct Q6Quad {
    q0: vec4<f32>,
    q1: vec4<f32>,
    q2: vec4<f32>,
    q3: vec4<f32>,
};

// Six-bit quants are split: four low bits in `ql`, two high bits in `qh`. Each
// (lo_a, lo_b, hh) triple yields four sub-block lanes of four values. `s`
// carries the four sub-block scales already multiplied by the block's d, so
// the token loop is a plain dot product.
fn q6k_quad(lo_a: u32, lo_b: u32, hh: u32, s: vec4<f32>) -> Q6Quad {
    let m8 = vec4<u32>(0xFFu);
    let a = vec4<u32>(lo_a, lo_a >> 8u, lo_a >> 16u, lo_a >> 24u) & m8;
    let b = vec4<u32>(lo_b, lo_b >> 8u, lo_b >> 16u, lo_b >> 24u) & m8;
    let h = vec4<u32>(hh, hh >> 8u, hh >> 16u, hh >> 24u) & m8;
    let m4 = vec4<u32>(15u);
    let m2 = vec4<u32>(3u);
    let sh2 = vec4<u32>(2u);
    let sh4 = vec4<u32>(4u);
    let sh6 = vec4<u32>(6u);

    var r: Q6Quad;
    r.q0 = (vec4<f32>((a & m4) | ((h & m2) << sh4)) - 32.0) * s.x;
    r.q1 = (vec4<f32>((b & m4) | (((h >> sh2) & m2) << sh4)) - 32.0) * s.y;
    r.q2 = (vec4<f32>((a >> sh4) | (((h >> sh4) & m2) << sh4)) - 32.0) * s.z;
    r.q3 = (vec4<f32>((b >> sh4) | (((h >> sh6) & m2) << sh4)) - 32.0) * s.w;
    return r;
}

// The four scales a unit needs: `l/16` is constant across its eight l values.
fn q6k_scales(blk: u32, n: u32, l0: u32) -> vec4<f32> {
    let sc_base = blk + 48u;
    let is = n * 8u + l0 / 16u;
    let d = unpack2x16float(weights[blk + 52u]).x;
    return vec4<f32>(
        sc_i8(sc_base, is),
        sc_i8(sc_base, is + 2u),
        sc_i8(sc_base, is + 4u),
        sc_i8(sc_base, is + 6u)) * d;
}

@compute @workgroup_size(WG)
fn matvec_q6k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let blocks = p.in_dim / 256u;
    let row_stride = blocks * 56u;    // 224 bytes per repacked block
    let n_units = blocks * 8u;

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        var acc = 0.0;
        var u = lane;
        loop {
            if (u >= n_units) { break; }
            let bi = u / 8u;
            let rem = u % 8u;
            let n = rem / 4u;
            let l0 = (rem % 4u) * 8u;

            let blk = row_base + bi * 56u;
            let ql_base = blk + n * 16u + l0 / 4u;
            let qh_base = blk + 32u + n * 8u + l0 / 4u;
            let s = q6k_scales(blk, n, l0);
            let off = (bi * 256u + n * 128u + l0) / 4u;

            for (var w = 0u; w < 2u; w = w + 1u) {
                let q = q6k_quad(
                    weights[ql_base + w],
                    weights[ql_base + 8u + w],
                    weights[qh_base + w],
                    s);
                let base = off + w;
                acc = acc
                    + dot(q.q0, x4[base])
                    + dot(q.q1, x4[base + 8u])
                    + dot(q.q2, x4[base + 16u])
                    + dot(q.q3, x4[base + 24u]);
            }
            u = u + LANES;
        }

        let total = reduce_row(tid, lane, acc);
        if (lane == 0u && row < p.out_dim) {
            y[row] = total;
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

@compute @workgroup_size(WG)
fn matvec_q6k_t(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let tile_base = wg.y * TILE;
    let blocks = p.in_dim / 256u;
    let row_stride = blocks * 56u;
    let n_units = blocks * 8u;
    let x_stride = p.in_dim / 4u;
    let tile_n = min(TILE, p.n_tokens - min(tile_base, p.n_tokens));

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        var acc: array<f32, TILE>;
        for (var i = 0u; i < tile_n; i = i + 1u) { acc[i] = 0.0; }

        var u = lane;
        loop {
            if (u >= n_units) { break; }
            let bi = u / 8u;
            let rem = u % 8u;
            let n = rem / 4u;
            let l0 = (rem % 4u) * 8u;

            let blk = row_base + bi * 56u;
            let ql_base = blk + n * 16u + l0 / 4u;
            let qh_base = blk + 32u + n * 8u + l0 / 4u;
            let s = q6k_scales(blk, n, l0);
            let off = (bi * 256u + n * 128u + l0) / 4u;

            for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
                let xb = (tile_base + tt) * x_stride + off;
                var sum = 0.0;
                for (var w = 0u; w < 2u; w = w + 1u) {
                    let q = q6k_quad(
                        weights[ql_base + w],
                        weights[ql_base + 8u + w],
                        weights[qh_base + w],
                        s);
                    sum = sum
                        + dot(q.q0, x4[xb + w])
                        + dot(q.q1, x4[xb + w + 8u])
                        + dot(q.q2, x4[xb + w + 16u])
                        + dot(q.q3, x4[xb + w + 24u]);
                }
                acc[tt] = acc[tt] + sum;
            }
            u = u + LANES;
        }

        for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
            store_row(tid, lane, acc[tt], row, tile_base + tt);
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

// ---------------------------------------------------------------- Q8_0 ----
//
// The simple legacy format: an f16 scale and 32 plain i8 quants per block,
// repacked at upload to 36 bytes so the scale word and each quant word are
// u32-aligned. One unit of work is one block: 8 quant words, 8 vec4s of x.
// Matches the CPU reference exactly: the block's sub-sum is accumulated first
// and multiplied by d once.

fn i8x4(w: u32) -> vec4<f32> {
    let v = vec4<u32>(w, w >> 8u, w >> 16u, w >> 24u) << vec4<u32>(24u);
    return vec4<f32>(bitcast<vec4<i32>>(v) >> vec4<u32>(24u));
}

@compute @workgroup_size(WG)
fn matvec_q8_0(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let blocks = p.in_dim / 32u;
    let row_stride = blocks * 9u;

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        var acc = 0.0;
        var b = lane;
        loop {
            if (b >= blocks) { break; }
            let blk = row_base + b * 9u;
            let d = unpack2x16float(weights[blk]).x;
            let off = b * 8u;
            var sub = 0.0;
            for (var w = 0u; w < 8u; w = w + 1u) {
                sub = sub + dot(i8x4(weights[blk + 1u + w]), x4[off + w]);
            }
            acc = acc + d * sub;
            b = b + LANES;
        }

        let total = reduce_row(tid, lane, acc);
        if (lane == 0u && row < p.out_dim) {
            y[row] = total;
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

@compute @workgroup_size(WG)
fn matvec_q8_0_t(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let tile_base = wg.y * TILE;
    let blocks = p.in_dim / 32u;
    let row_stride = blocks * 9u;
    let x_stride = p.in_dim / 4u;
    let tile_n = min(TILE, p.n_tokens - min(tile_base, p.n_tokens));

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        var acc: array<f32, TILE>;
        for (var i = 0u; i < tile_n; i = i + 1u) { acc[i] = 0.0; }

        var b = lane;
        loop {
            if (b >= blocks) { break; }
            let blk = row_base + b * 9u;
            let d = unpack2x16float(weights[blk]).x;
            let off = b * 8u;
            for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
                let xb = (tile_base + tt) * x_stride + off;
                var sub = 0.0;
                for (var w = 0u; w < 8u; w = w + 1u) {
                    sub = sub + dot(i8x4(weights[blk + 1u + w]), x4[xb + w]);
                }
                acc[tt] = acc[tt] + d * sub;
            }
            b = b + LANES;
        }

        for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
            store_row(tid, lane, acc[tt], row, tile_base + tt);
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

// ----------------------------------------------------------------- F32 ----
//
// Dense f32 weights, for the all-F32 synthetic validation models. Not tuned:
// real checkpoints quantize everything this would touch.

fn wf32x4(base: u32) -> vec4<f32> {
    return vec4<f32>(
        bitcast<f32>(weights[base]),
        bitcast<f32>(weights[base + 1u]),
        bitcast<f32>(weights[base + 2u]),
        bitcast<f32>(weights[base + 3u]));
}

@compute @workgroup_size(WG)
fn matvec_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let quads = p.in_dim / 4u;

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * p.in_dim;

        var acc = 0.0;
        var i = lane;
        loop {
            if (i >= quads) { break; }
            acc = acc + dot(wf32x4(row_base + i * 4u), x4[i]);
            i = i + LANES;
        }

        let total = reduce_row(tid, lane, acc);
        if (lane == 0u && row < p.out_dim) {
            y[row] = total;
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

@compute @workgroup_size(WG)
fn matvec_f32_t(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let tile_base = wg.y * TILE;
    let quads = p.in_dim / 4u;
    let x_stride = p.in_dim / 4u;
    let tile_n = min(TILE, p.n_tokens - min(tile_base, p.n_tokens));

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * p.in_dim;

        var acc: array<f32, TILE>;
        for (var i = 0u; i < tile_n; i = i + 1u) { acc[i] = 0.0; }

        var i = lane;
        loop {
            if (i >= quads) { break; }
            let w = wf32x4(row_base + i * 4u);
            for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
                acc[tt] = acc[tt] + dot(w, x4[(tile_base + tt) * x_stride + i]);
            }
            i = i + LANES;
        }

        for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
            store_row(tid, lane, acc[tt], row, tile_base + tt);
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

// Gather embedding rows. Q6_K here too, but decoding whole rows rather than
// contracting them, so it is a separate entry point.
@group(0) @binding(4) var<storage, read> tokens: array<u32>;

@compute @workgroup_size(WG)
fn embed_q6k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let t = wg.y;
    let row = tokens[t];
    let blocks = p.in_dim / 256u;
    let row_base = p.w_base + row * blocks * 56u;
    let out_base = t * p.in_dim;

    var u = lid.x + wg.x * WG;
    let n_units = blocks * 64u;
    loop {
        if (u >= n_units) { break; }
        let b = u / 64u;
        let rem = u % 64u;
        let n = rem / 32u;
        let l = rem % 32u;

        let blk = row_base + b * 56u;
        let d = unpack2x16float(weights[blk + 52u]).x;
        let ql_base = blk + n * 16u;
        let qh_base = blk + 32u + n * 8u;
        let sc_base = blk + 48u;

        let lo0 = byte_at(ql_base, l);
        let lo1 = byte_at(ql_base, l + 32u);
        let h = byte_at(qh_base, l);
        let is = n * 8u + l / 16u;

        let e = out_base + b * 256u + n * 128u + l;
        y[e]       = d * sc_i8(sc_base, is)      * f32(i32((lo0 & 15u) | ((h & 3u) << 4u)) - 32);
        y[e + 32u] = d * sc_i8(sc_base, is + 2u) * f32(i32((lo1 & 15u) | (((h >> 2u) & 3u) << 4u)) - 32);
        y[e + 64u] = d * sc_i8(sc_base, is + 4u) * f32(i32((lo0 >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32);
        y[e + 96u] = d * sc_i8(sc_base, is + 6u) * f32(i32((lo1 >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32);
        u = u + WG * nwg.x;
    }
}

// F32 embedding gather, for the synthetic validation models.
@compute @workgroup_size(WG)
fn embed_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let t = wg.y;
    let row_base = p.w_base + tokens[t] * p.in_dim;
    let out_base = t * p.in_dim;

    var i = lid.x + wg.x * WG;
    loop {
        if (i >= p.in_dim) { break; }
        y[out_base + i] = bitcast<f32>(weights[row_base + i]);
        i = i + WG * nwg.x;
    }
}

// ---------------------------------------------------------------- BF16 ----
//
// Gemma 4's E-series ships `per_layer_model_proj` as bf16 while everything
// around it is quantized. bf16 is just the top 16 bits of an f32, so widening
// is a shift rather than a decode — two weights per word, four per pair.

fn wbf16x4(base: u32) -> vec4<f32> {
    let w0 = weights[base];
    let w1 = weights[base + 1u];
    return vec4<f32>(
        bitcast<f32>(w0 << 16u),
        bitcast<f32>(w0 & 0xFFFF0000u),
        bitcast<f32>(w1 << 16u),
        bitcast<f32>(w1 & 0xFFFF0000u));
}

@compute @workgroup_size(WG)
fn matvec_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let quads = p.in_dim / 4u;
    // Two bf16 per word, so a row spans in_dim/2 words.
    let row_words = p.in_dim / 2u;

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_words;

        var acc = 0.0;
        var i = lane;
        loop {
            if (i >= quads) { break; }
            acc = acc + dot(wbf16x4(row_base + i * 2u), x4[i]);
            i = i + LANES;
        }

        let total = reduce_row(tid, lane, acc);
        if (lane == 0u && row < p.out_dim) {
            y[row] = total;
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

@compute @workgroup_size(WG)
fn matvec_bf16_t(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid % LANES;
    let row_in_wg = tid / LANES;
    let tile_base = wg.y * TILE;
    let quads = p.in_dim / 4u;
    let x_stride = p.in_dim / 4u;
    let row_words = p.in_dim / 2u;
    let tile_n = min(TILE, p.n_tokens - min(tile_base, p.n_tokens));

    // The loop bound is deliberately the workgroup's *base* row rather than
    // this thread's, so it mentions nothing derived from `lid`. Both forms
    // iterate the same number of times — `row - row_in_wg` is `row_block` —
    // but only this one is uniform to a compiler that will not do the algebra,
    // and the reduction below contains a `workgroupBarrier`. Naga accepts the
    // subtracted form; Tint rejects it, so WebGPU refuses to compile the whole
    // module. Do not fold `row_in_wg` back into the loop variable.
    var row_block = wg.x * ROWS;
    loop {
        if (row_block >= p.out_dim) { break; }
        let row = row_block + row_in_wg;
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_words;

        var acc: array<f32, TILE>;
        for (var i = 0u; i < tile_n; i = i + 1u) { acc[i] = 0.0; }

        var i = lane;
        loop {
            if (i >= quads) { break; }
            let w = wbf16x4(row_base + i * 2u);
            for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
                acc[tt] = acc[tt] + dot(w, x4[(tile_base + tt) * x_stride + i]);
            }
            i = i + LANES;
        }

        for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
            store_row(tid, lane, acc[tt], row, tile_base + tt);
        }
        row_block = row_block + nwg.x * ROWS;
    }
}

// Q8_0 embedding gather. E4B quantizes both `token_embd` and the much larger
// `per_layer_token_embd` to Q8_0, so this serves the ordinary token lookup and
// the per-layer table alike — the only difference is the row width.
@compute @workgroup_size(WG)
fn embed_q8_0(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let t = wg.y;
    let row = tokens[t];
    let blocks = p.in_dim / 32u;
    // 36-byte repacked blocks: one word of scale, eight of quants.
    let row_base = p.w_base + row * blocks * 9u;
    let out_base = t * p.in_dim;

    var b = lid.x + wg.x * WG;
    loop {
        if (b >= blocks) { break; }
        let blk = row_base + b * 9u;
        let d = unpack2x16float(weights[blk]).x;
        let e = out_base + b * 32u;
        for (var w = 0u; w < 8u; w = w + 1u) {
            let q = i8x4(weights[blk + 1u + w]) * d;
            let o = e + w * 4u;
            y[o] = q.x;
            y[o + 1u] = q.y;
            y[o + 2u] = q.z;
            y[o + 3u] = q.w;
        }
        b = b + WG * nwg.x;
    }
}
