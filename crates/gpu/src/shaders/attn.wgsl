// Attention, split into three passes over a scores scratch buffer.
//
// A fused flash-attention kernel would need a cross-thread reduction for every
// key position; splitting lets each pass parallelize along an axis that needs
// no reduction at all. Scores are bounded by the sliding window on 40 of the
// 48 layers, so the scratch buffer stays small.
//
// Note the softmax scale is 1.0 for this model — the attention temperature is
// baked into the QK-norm gains. See ARCHITECTURE.md.

struct Attn {
    n_tokens: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    base_pos: u32,     // position of the batch's first token
    window: u32,       // 0 = global (no wrap, unbounded history)
    max_vis: u32,      // scores stride per (token, head)
    scale: f32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> at: Attn;

const WG: u32 = 64u;
var<workgroup> red: array<f32, WG>;

// Oldest position visible from `pos`.
fn first_visible(pos: u32) -> u32 {
    if (at.window == 0u) { return 0u; }
    if (pos + 1u <= at.window) { return 0u; }
    return pos + 1u - at.window;
}

fn cache_slot(pos: u32) -> u32 {
    if (at.window == 0u) { return pos; }
    return pos % at.window;
}

// Pass 1: one workgroup per (token, head); each thread owns whole key
// positions, so a full dot product happens in-thread with no reduction.
@compute @workgroup_size(WG)
fn scores_pass(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = wg.y;
    let head = wg.x;
    if (token >= at.n_tokens || head >= at.n_heads) { return; }

    let pos = at.base_pos + token;
    let lo = first_visible(pos);
    let n_vis = pos - lo + 1u;

    let kv_head = head / (at.n_heads / at.n_kv_heads);
    let q_base = token * at.n_heads * at.head_dim + head * at.head_dim;
    let s_base = (token * at.n_heads + head) * at.max_vis;

    var j = lid.x;
    loop {
        if (j >= n_vis) { break; }
        let k_base = cache_slot(lo + j) * at.kv_dim + kv_head * at.head_dim;
        var dot = 0.0;
        for (var d = 0u; d < at.head_dim; d = d + 1u) {
            dot = dot + q[q_base + d] * k_cache[k_base + d];
        }
        scores[s_base + j] = dot * at.scale;
        j = j + WG;
    }
}

// Pass 2: softmax over the visible range, max-subtracted for stability.
@compute @workgroup_size(WG)
fn softmax_pass(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = wg.y;
    let head = wg.x;
    if (token >= at.n_tokens || head >= at.n_heads) { return; }

    let tid = lid.x;
    let pos = at.base_pos + token;
    let n_vis = pos - first_visible(pos) + 1u;
    let s_base = (token * at.n_heads + head) * at.max_vis;

    var m = -3.4028235e38;
    var j = tid;
    loop {
        if (j >= n_vis) { break; }
        m = max(m, scores[s_base + j]);
        j = j + WG;
    }
    red[tid] = m;
    workgroupBarrier();
    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
        workgroupBarrier();
        s = s / 2u;
    }
    let row_max = red[0];
    workgroupBarrier();

    var acc = 0.0;
    j = tid;
    loop {
        if (j >= n_vis) { break; }
        let e = exp(scores[s_base + j] - row_max);
        scores[s_base + j] = e;
        acc = acc + e;
        j = j + WG;
    }
    red[tid] = acc;
    workgroupBarrier();
    s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    let inv = 1.0 / red[0];

    j = tid;
    loop {
        if (j >= n_vis) { break; }
        scores[s_base + j] = scores[s_base + j] * inv;
        j = j + WG;
    }
}

// Pass 3: weighted sum of V. Positions are the outer loop so that neighbouring
// threads read neighbouring dimensions of the same cache row.
@compute @workgroup_size(WG)
fn weighted_v(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = wg.y;
    let head = wg.x;
    if (token >= at.n_tokens || head >= at.n_heads) { return; }

    let tid = lid.x;
    let pos = at.base_pos + token;
    let lo = first_visible(pos);
    let n_vis = pos - lo + 1u;

    let kv_head = head / (at.n_heads / at.n_kv_heads);
    let s_base = (token * at.n_heads + head) * at.max_vis;
    let o_base = token * at.n_heads * at.head_dim + head * at.head_dim;

    // Up to 512/64 = 8 accumulators per thread.
    var acc: array<f32, 8>;
    for (var r = 0u; r < 8u; r = r + 1u) { acc[r] = 0.0; }

    for (var j = 0u; j < n_vis; j = j + 1u) {
        let w = scores[s_base + j];
        let v_base = cache_slot(lo + j) * at.kv_dim + kv_head * at.head_dim;
        var r = 0u;
        var d = tid;
        loop {
            if (d >= at.head_dim) { break; }
            acc[r] = acc[r] + w * v_cache[v_base + d];
            r = r + 1u;
            d = d + WG;
        }
    }

    var r = 0u;
    var d = tid;
    loop {
        if (d >= at.head_dim) { break; }
        out[o_base + d] = acc[r];
        r = r + 1u;
        d = d + WG;
    }
}
