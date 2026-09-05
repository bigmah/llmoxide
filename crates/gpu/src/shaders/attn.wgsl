// Attention, split into three passes over a scores scratch buffer.
//
// A fused flash-attention kernel would need a cross-thread reduction for every
// key position; splitting lets each pass parallelize along an axis that needs
// no reduction at all. Scores are bounded by the sliding window on 40 of the
// 48 layers, so the scratch buffer stays small.
//
// Q, K and V are read as vec4. head_dim is 256 or 512 and kv_dim is a multiple
// of head_dim, so every base offset here is 4-element aligned. Scalar loads
// were most of what these passes did: the arithmetic is one FMA per element,
// so an instruction per element to fetch it doubles the work.
//
// At decode these passes have only n_heads workgroups between them, which is
// well short of filling the GPU. That is a property of the split — the work
// available without a cross-position reduction is n_heads * head_dim/4 lanes —
// and it is why decode cost still grows with context. Splitting the key range
// across workgroups and reducing the partials afterwards is the way out.
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

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k_cache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v_cache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> at: Attn;

const WG: u32 = 64u;
// head_dim/4, at the model's larger geometry. Threads cover it in WG-sized
// strides, so this is also the most vec4s one thread ever owns times WG.
const MAX_HD4: u32 = 128u;
var<workgroup> red: array<f32, WG>;
// This (token, head)'s query vector, read once instead of once per key.
var<workgroup> qs: array<vec4<f32>, MAX_HD4>;

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

// vec4 index of a key/value row's head slice.
fn kv_base4(pos: u32, kv_head: u32) -> u32 {
    return (cache_slot(pos) * at.kv_dim + kv_head * at.head_dim) / 4u;
}

fn kv_head_of(head: u32) -> u32 {
    return head / (at.n_heads / at.n_kv_heads);
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
    // Uniform across the workgroup, so the barrier below is still reached by
    // every thread that stays.
    if (token >= at.n_tokens || head >= at.n_heads) { return; }

    let pos = at.base_pos + token;
    let lo = first_visible(pos);
    let n_vis = pos - lo + 1u;
    let hd4 = at.head_dim / 4u;

    let q_base4 = (token * at.n_heads * at.head_dim + head * at.head_dim) / 4u;
    var i = lid.x;
    loop {
        if (i >= hd4) { break; }
        qs[i] = q[q_base4 + i];
        i = i + WG;
    }
    workgroupBarrier();

    let kv_head = kv_head_of(head);
    let s_base = (token * at.n_heads + head) * at.max_vis;
    var j = lid.x;
    loop {
        if (j >= n_vis) { break; }
        let k4 = kv_base4(lo + j, kv_head);
        var acc = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d = d + 1u) {
            acc = acc + qs[d] * k_cache[k4 + d];
        }
        scores[s_base + j] = (acc.x + acc.y + acc.z + acc.w) * at.scale;
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

    // f32::MIN, spelled by its bit pattern. The decimal form Rust prints for
    // it, -3.4028235e38, is not a legal f32 literal in WGSL: the spec parses
    // literals as abstract float first, and 3.4028235e38 is larger than
    // f32::MAX, so the conversion overflows. Naga rounds it to nearest and
    // accepts; Tint rejects, and the whole module fails to compile in a
    // browser. A bitcast has one meaning everywhere.
    var m = bitcast<f32>(0xff7fffffu);
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
//
// A thread owns one vec4 of the head, or two at head_dim 512. They are named
// rather than an array because a dynamically indexed local array does not
// reach registers — the same trap as the matvec tile, see quant.wgsl.
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
    let hd4 = at.head_dim / 4u;

    let kv_head = kv_head_of(head);
    let s_base = (token * at.n_heads + head) * at.max_vis;
    let o_base4 = (token * at.n_heads * at.head_dim + head * at.head_dim) / 4u;

    let d0 = tid;
    let d1 = tid + WG;
    let has1 = d1 < hd4;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    for (var j = 0u; j < n_vis; j = j + 1u) {
        let w = scores[s_base + j];
        let v4 = kv_base4(lo + j, kv_head);
        a0 = a0 + w * v_cache[v4 + d0];
        if (has1) { a1 = a1 + w * v_cache[v4 + d1]; }
    }

    if (d0 < hd4) { out[o_base4 + d0] = a0; }
    if (has1) { out[o_base4 + d1] = a1; }
}
