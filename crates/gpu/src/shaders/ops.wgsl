// Elementwise and normalization kernels.
//
// One bind group layout serves all of them so the driver can reuse pipelines
// without reshuffling bindings between dispatches:
//   0 params — every F32 model tensor (norm gains, rope factors), concatenated
//   1 a, 2 b — inputs
//   3 out    — output, and the *input* for the in-place kernels
//   4 op     — the uniform below
//
// wgpu rejects binding one buffer as both read-only and read-write within a
// dispatch, so anything that transforms a buffer in place reads `out` rather
// than aliasing `a` onto it. Use `copy` first when a distinct source is needed.
//
// Fields are reused across kernels; each entry point documents what it reads.

struct Op {
    n_rows: u32,
    dim: u32,
    off0: u32,    // index into `params`
    off1: u32,
    f0: f32,
    f1: f32,
    u0: u32,
    u1: u32,
};

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> op: Op;

const WG: u32 = 256u;
var<workgroup> red: array<f32, WG>;

fn wg_sum(tid: u32, v: f32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    return red[0];
}

// RMS norm over each row of `a`, optionally times a gain from `params`.
//   n_rows, dim  — row count and length
//   off0         — gain offset in `params`; u0 == 0 means no gain
//   f0           — epsilon
// Used for the whole-vector norms and, with dim = head_dim, the per-head QK
// and V norms — those are just rows of length head_dim.
@compute @workgroup_size(WG)
fn rms_norm(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    // Rows are strided over the grid so a batch with more rows than the
    // dispatch limit is covered rather than silently truncated.
    var r = wg.x;
    loop {
    if (r >= op.n_rows) { break; }
    let base = r * op.dim;

    var acc = 0.0;
    var i = tid;
    loop {
        if (i >= op.dim) { break; }
        let v = out[base + i];
        acc = acc + v * v;
        i = i + WG;
    }
    let total = wg_sum(tid, acc);
    let scale = inverseSqrt(total / f32(op.dim) + op.f0);

    i = tid;
    loop {
        if (i >= op.dim) { break; }
        var v = out[base + i] * scale;
        if (op.u0 != 0u) { v = v * params[op.off0 + i]; }
        out[base + i] = v;
        i = i + WG;
    }
    workgroupBarrier();
    r = r + nwg.x;
    }
}

// out = gelu(a) * b, the gated FFN activation. ggml's GEGLU uses the tanh
// approximation, not the erf form.
@compute @workgroup_size(WG)
fn geglu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
    let x = out[i];
    // Metal evaluates tanh via exp, so a large positive argument gives inf/inf
    // = NaN rather than saturating to 1. tanh is 1.0 to f32 precision well
    // before |x| = 30, so clamping is exact here, not an approximation.
    let inner = clamp(0.7978845608 * x * (1.0 + 0.044715 * x * x), -30.0, 30.0);
    out[i] = 0.5 * x * (1.0 + tanh(inner)) * b[i];
        i = i + stride;
    }
}

// out = (out + b) * f0 — the residual add fused with layer_output_scale.
@compute @workgroup_size(WG)
fn add_scale(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
    out[i] = (out[i] + b[i]) * op.f0;
        i = i + stride;
    }
}

// out = a — the explicit copy that lets the in-place kernels above run on a
// buffer distinct from their logical source.
@compute @workgroup_size(WG)
fn copy(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
    out[i] = a[i];
        i = i + stride;
    }
}

// out = out * f0
@compute @workgroup_size(WG)
fn scale(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
    out[i] = out[i] * op.f0;
        i = i + stride;
    }
}

// Rotary embedding, ggml NeoX pairing: element i pairs with i + head_dim/2,
// not with its neighbour.
//   n_rows = n_tokens * n_heads, dim = head_dim
//   u0 = heads per token, u1 = base position, f0 = rope base
//   off0 = frequency-factor offset in `params`; u1 flagged by off1 != 0
// Global layers divide each angle by its factor; the 1e30 sentinels there make
// those dimensions unrotated. SWA layers pass off1 == 0 and rotate everything.
@compute @workgroup_size(WG)
fn rope(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var row = wg.x;
    loop {
    if (row >= op.n_rows) { break; }
    let half = op.dim / 2u;
    let base = row * op.dim;
    let pos = f32(op.u1 + row / op.u0);
    let theta_scale = pow(op.f0, -2.0 / f32(op.dim));

    var i = lid.x;
    loop {
        if (i >= half) { break; }
        var factor = 1.0;
        if (op.off1 != 0u) { factor = params[op.off0 + i]; }
        let angle = pos * pow(theta_scale, f32(i)) / factor;
        let c = cos(angle);
        let s = sin(angle);
        let x0 = out[base + i];
        let x1 = out[base + i + half];
        out[base + i] = x0 * c - x1 * s;
        out[base + i + half] = x0 * s + x1 * c;
        i = i + WG;
    }
    row = row + nwg.x;
    }
}

// logits = f0 * tanh(logits / f0), then -inf at suppressed ids.
@compute @workgroup_size(WG)
fn soft_cap(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= op.dim) { break; }
        out[i] = op.f0 * tanh(clamp(out[i] / op.f0, -30.0, 30.0));
        i = i + stride;
    }
}

// Copy this batch's K/V rows into the cache.
//   n_rows = n_tokens, dim = kv_dim
//   u0 = window (0 = no wrap), u1 = base position
// Sliding-window layers wrap at `window`, so their cache stays 1024 rows
// instead of growing with the context.
@compute @workgroup_size(WG)
fn write_cache(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let token = i / op.dim;
        let lane = i % op.dim;
        let pos = op.u1 + token;
        var slot = pos;
        if (op.u0 != 0u) { slot = pos % op.u0; }
        out[slot * op.dim + lane] = a[i];
        i = i + stride;
    }
}
