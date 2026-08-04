// Quantized matrix-vector kernels: k-quant blocks are decoded inside the dot
// product rather than materialized. Decode is memory-bound, so reading 4-bit
// weights and unpacking in registers beats reading dequantized f16.
//
// Weights live in one big storage buffer addressed by u32 index. Q4_K blocks
// are 144 bytes (36 u32) as on disk; Q6_K blocks are repacked at upload from
// 210 to 224 bytes (56 u32) so every block starts u32-aligned — 210 is not a
// multiple of 4, and unaligned block strides would cost a shift on every load.

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
// Grouping rows this way amortizes the reduction barriers — one row per
// workgroup spends more time synchronizing than multiplying — and lets the
// rows share the activation vector while it is hot in cache.
const LANES: u32 = 32u;
const ROWS: u32 = 8u;
const WG: u32 = LANES * ROWS;
// Tokens handled per dispatch. Weights are the dominant traffic, so loading a
// block once and applying it to TILE activations cuts prefill bandwidth by the
// same factor. Decode passes n_tokens = 1 and simply leaves the rest idle.
const TILE: u32 = 32u;
var<workgroup> partial: array<f32, WG>;

fn byte_at(base: u32, off: u32) -> u32 {
    return (weights[base + (off >> 2u)] >> ((off & 3u) * 8u)) & 0xFFu;
}

// Q4_K sub-block j (0..8) carries a 6-bit scale and a 6-bit min packed across
// 12 bytes. Sub-blocks 0..4 hold both plainly; 4..8 borrow their top two bits
// from the high bits of the first eight bytes.
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

// Reduce within each row's lane group. All ROWS groups reduce in lockstep, so
// the barrier count is per workgroup rather than per row.
fn reduce_lanes(tid: u32, lane: u32, v: f32) -> f32 {
    partial[tid] = v;
    workgroupBarrier();
    var s = LANES / 2u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) { partial[tid] = partial[tid] + partial[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    return partial[tid - lane];
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
    let tile_base = wg.y * TILE;
    // Uniform across the workgroup, so loops bounded by it may contain barriers.
    let tile_n = min(TILE, p.n_tokens - min(tile_base, p.n_tokens));
    let blocks = p.in_dim / 256u;
    let n_pairs = blocks * 4u;        // 64-element sub-block pairs
    let row_stride = blocks * 36u;    // u32 per weight row

    // Rows are strided over the grid: the output projection has 262144 rows,
    // well past the 65535-per-dimension dispatch limit.
    var row = wg.x * ROWS + row_in_wg;
    loop {
        if (row - row_in_wg >= p.out_dim) { break; }
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        // One unit is a 64-element *pair* of sub-blocks sharing 32 weight
        // bytes: the low nibbles feed sub-block 2h, the high nibbles 2h+1.
        // Splitting them across lanes would read those bytes twice.
        var acc: array<f32, TILE>;
        for (var i = 0u; i < tile_n; i = i + 1u) { acc[i] = 0.0; }

        var pi = lane;
        loop {
            if (pi >= n_pairs) { break; }
            let b = pi / 4u;
            let half = pi % 4u;
            let blk = row_base + b * 36u;

            let dm = unpack2x16float(weights[blk]);
            let sm0 = q4k_scale_min(blk + 1u, half * 2u);
            let sm1 = q4k_scale_min(blk + 1u, half * 2u + 1u);
            let qs_base = blk + 4u + half * 8u;

            // Read the quants once, then reuse them for every token in the tile.
            var wq: array<u32, 8>;
            for (var w = 0u; w < 8u; w = w + 1u) { wq[w] = weights[qs_base + w]; }

            let off = b * 256u + half * 64u;
            for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
                let e0 = (tile_base + tt) * p.in_dim + off;
                // Four elements at a time: unpacking into vec4 and using dot()
                // lets the compiler issue vector FMAs. The scalar form is
                // ALU-bound, not bandwidth-bound, so this is the hot path.
                var v0 = vec4<f32>(0.0);
                var v1 = vec4<f32>(0.0);
                var xs0 = vec4<f32>(0.0);
                var xs1 = vec4<f32>(0.0);
                for (var w = 0u; w < 8u; w = w + 1u) {
                    let packed = wq[w];
                    let lo = vec4<f32>(
                        f32(packed & 15u),
                        f32((packed >> 8u) & 15u),
                        f32((packed >> 16u) & 15u),
                        f32((packed >> 24u) & 15u));
                    let hi = vec4<f32>(
                        f32((packed >> 4u) & 15u),
                        f32((packed >> 12u) & 15u),
                        f32((packed >> 20u) & 15u),
                        f32((packed >> 28u) & 15u));
                    let base = (e0 + w * 4u) / 4u;
                    let xa = x4[base];
                    let xb = x4[base + 8u];
                    v0 = v0 + lo * xa;
                    v1 = v1 + hi * xb;
                    xs0 = xs0 + xa;
                    xs1 = xs1 + xb;
                }
                let dot0 = v0.x + v0.y + v0.z + v0.w;
                let dot1 = v1.x + v1.y + v1.z + v1.w;
                let sum0 = xs0.x + xs0.y + xs0.z + xs0.w;
                let sum1 = xs1.x + xs1.y + xs1.z + xs1.w;
                // value = d*scale*q - dmin*min, so the min term factors out of
                // the sub-block and multiplies the plain sum of activations.
                acc[tt] = acc[tt] + dm.x * sm0.x * dot0 - dm.y * sm0.y * sum0
                                  + dm.x * sm1.x * dot1 - dm.y * sm1.y * sum1;
            }
            pi = pi + LANES;
        }

        // Only reduce the tile slots actually in use; decode passes one token
        // and must not pay for eight rounds of barriers.
        for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
            let total = reduce_lanes(tid, lane, acc[tt]);
            if (lane == 0u && row < p.out_dim) {
                y[(tile_base + tt) * p.out_dim + row] = total;
            }
            workgroupBarrier();
        }
        row = row + nwg.x * ROWS;
    }
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
    let tile_base = wg.y * TILE;
    // Uniform across the workgroup, so loops bounded by it may contain barriers.
    let tile_n = min(TILE, p.n_tokens - min(tile_base, p.n_tokens));
    let blocks = p.in_dim / 256u;
    let row_stride = blocks * 56u;    // 224 bytes per repacked block

    var row = wg.x * ROWS + row_in_wg;
    loop {
        if (row - row_in_wg >= p.out_dim) { break; }
        let row_base = p.w_base + min(row, p.out_dim - 1u) * row_stride;

        // One unit is eight consecutive `l` values within a (block, half),
        // covering 32 outputs. Reading whole u32s and hoisting the four
        // sub-block scales cuts the load count ~6x versus per-byte access.
        let n_units = blocks * 8u;
        var acc: array<f32, TILE>;
        for (var i = 0u; i < tile_n; i = i + 1u) { acc[i] = 0.0; }

        var u = lane;
        loop {
            if (u >= n_units) { break; }
            let bi = u / 8u;
            let rem = u % 8u;
            let n = rem / 4u;
            let quarter = rem % 4u;
            let l0 = quarter * 8u;

            let blk = row_base + bi * 56u;
            let d = unpack2x16float(weights[blk + 52u]).x;
            let ql_base = blk + n * 16u;
            let qh_base = blk + 32u + n * 8u;
            let sc_base = blk + 48u;

            // `l/16` is constant across the eight l values, so each unit needs
            // exactly four scales.
            let is = n * 8u + l0 / 16u;
            let s0 = sc_i8(sc_base, is);
            let s1 = sc_i8(sc_base, is + 2u);
            let s2 = sc_i8(sc_base, is + 4u);
            let s3 = sc_i8(sc_base, is + 6u);

            // Quants once per unit, reused across the tile.
            var wl: array<u32, 6>;
            for (var w = 0u; w < 2u; w = w + 1u) {
                wl[w] = weights[ql_base + l0 / 4u + w];
                wl[2u + w] = weights[ql_base + (l0 + 32u) / 4u + w];
                wl[4u + w] = weights[qh_base + l0 / 4u + w];
            }

            let off = bi * 256u + n * 128u + l0;
            for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
                let e = (tile_base + tt) * p.in_dim + off;
                var sub = 0.0;
                for (var w = 0u; w < 2u; w = w + 1u) {
                    let lo_a = wl[w];
                    let lo_b = wl[2u + w];
                    let hh = wl[4u + w];
                    var q0: vec4<f32>;
                    var q1: vec4<f32>;
                    var q2: vec4<f32>;
                    var q3: vec4<f32>;
                    for (var k = 0u; k < 4u; k = k + 1u) {
                        let sh = k * 8u;
                        let a0 = (lo_a >> sh) & 0xFFu;
                        let b0 = (lo_b >> sh) & 0xFFu;
                        let h = (hh >> sh) & 0xFFu;
                        q0[k] = f32(i32((a0 & 15u) | ((h & 3u) << 4u)) - 32);
                        q1[k] = f32(i32((b0 & 15u) | (((h >> 2u) & 3u) << 4u)) - 32);
                        q2[k] = f32(i32((a0 >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32);
                        q3[k] = f32(i32((b0 >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32);
                    }
                    let base = (e + w * 4u) / 4u;
                    sub = sub
                        + s0 * dot(q0, x4[base])
                        + s1 * dot(q1, x4[base + 8u])
                        + s2 * dot(q2, x4[base + 16u])
                        + s3 * dot(q3, x4[base + 24u]);
                }
                acc[tt] = acc[tt] + d * sub;
            }
            u = u + LANES;
        }

        // Only reduce the tile slots actually in use; decode passes one token
        // and must not pay for eight rounds of barriers.
        for (var tt = 0u; tt < tile_n; tt = tt + 1u) {
            let total = reduce_lanes(tid, lane, acc[tt]);
            if (lane == 0u && row < p.out_dim) {
                y[(tile_base + tt) * p.out_dim + row] = total;
            }
            workgroupBarrier();
        }
        row = row + nwg.x * ROWS;
    }
}

fn sc_i8(base: u32, i: u32) -> f32 {
    let b = i32(byte_at(base, i));
    return f32(select(b, b - 256, b > 127));
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
