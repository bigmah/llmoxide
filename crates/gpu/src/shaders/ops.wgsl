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

// out = gelu(out) * b, the gated FFN activation — in place on the gate, so
// bind the gate as `out` and the up-projection as `b`. ggml's GEGLU uses the
// tanh approximation, not the erf form.
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

// out = silu(out) * b, the SwiGLU FFN activation (qwen35).
@compute @workgroup_size(WG)
fn swiglu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let x = out[i];
        out[i] = x / (1.0 + exp(-x)) * b[i];
        i = i + stride;
    }
}

// out = out * silu(b) — the delta-net output gate.
@compute @workgroup_size(WG)
fn mul_silu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let x = b[i];
        out[i] = out[i] * (x / (1.0 + exp(-x)));
        i = i + stride;
    }
}

// out = out * sigmoid(b) — qwen35's per-head attention output gate.
@compute @workgroup_size(WG)
fn mul_sigmoid(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        out[i] = out[i] / (1.0 + exp(-b[i]));
        i = i + stride;
    }
}

// Copy one half of a fused per-head [query | gate] projection out of `a`.
//   n_rows = n_tokens * n_heads, dim = head_dim
//   u0 = source offset within the pair: 0 for the query half, dim for the gate
@compute @workgroup_size(WG)
fn split_half(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let row = i / op.dim;
        let d = i % op.dim;
        out[i] = a[row * 2u * op.dim + op.u0 + d];
        i = i + stride;
    }
}

// NeoX rotation over only the first off0 dims of each dim-wide head row; the
// rest are NoPE (qwen35: 64 of 256, base 1e7).
//   n_rows = n_tokens * heads, dim = head stride
//   off0 = n_rot, u0 = heads per token, u1 = base position, f0 = rope base
@compute @workgroup_size(WG)
fn rope_partial(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var row = wg.x;
    loop {
    if (row >= op.n_rows) { break; }
    let half = op.off0 / 2u;
    let base = row * op.dim;
    let pos = f32(op.u1 + row / op.u0);
    let theta_scale = pow(op.f0, -2.0 / f32(op.off0));

    var i = lid.x;
    loop {
        if (i >= half) { break; }
        let angle = pos * pow(theta_scale, f32(i));
        let cs = cos(angle);
        let sn = sin(angle);
        let x0 = out[base + i];
        let x1 = out[base + i + half];
        out[base + i] = x0 * cs - x1 * sn;
        out[base + i + half] = x0 * sn + x1 * cs;
        i = i + WG;
    }
    row = row + nwg.x;
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

// Gather one strided block of rows: out[r*dim + c] = a[u1 + r*u0 + c].
//   n_rows, dim — output shape
//   u0          — source stride between consecutive rows
//   u1          — offset of the first row in `a`
// Used to lift one layer's slice out of the per-layer-embedding table, whose
// rows are `n_layers * n_embd_per_layer` wide.
@compute @workgroup_size(WG)
fn copy_rows(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let r = i / op.dim;
        let c = i - r * op.dim;
        out[i] = a[op.u1 + r * op.u0 + c];
        i = i + stride;
    }
}

// 2-D rotary for the vision tower. Unlike `rope` above, the position is not
// the row index: each head's low half rotates by the patch's *column* and its
// high half by the patch's *row*, so one head carries both axes. NeoX pairing
// applies within each half independently — element i pairs with i + dim/4, not
// with i + dim/2.
//   n_rows = n_patches * n_heads, dim = head_dim
//   u0 = heads per patch, u1 = patches per grid row, f0 = rope base
@compute @workgroup_size(WG)
fn rope_2d(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var row = wg.x;
    loop {
    if (row >= op.n_rows) { break; }
    let half = op.dim / 2u;
    let quarter = half / 2u;
    let base = row * op.dim;
    // `patch` is a WGSL reserved keyword, hence `pidx`.
    let pidx = row / op.u0;
    let col = f32(pidx % op.u1);
    let py = f32(pidx / op.u1);
    // Each half is its own NeoX block of `half` dimensions.
    let theta_scale = pow(op.f0, -2.0 / f32(half));

    var i = lid.x;
    loop {
        if (i >= quarter) { break; }
        let step = pow(theta_scale, f32(i));

        let ac = col * step;
        let cc = cos(ac);
        let sc = sin(ac);
        let x0 = out[base + i];
        let x1 = out[base + i + quarter];
        out[base + i] = x0 * cc - x1 * sc;
        out[base + i + quarter] = x0 * sc + x1 * cc;

        let ar = py * step;
        let cr = cos(ar);
        let sr = sin(ar);
        let y0 = out[base + half + i];
        let y1 = out[base + half + i + quarter];
        out[base + half + i] = y0 * cr - y1 * sr;
        out[base + half + i + quarter] = y0 * sr + y1 * cr;

        i = i + WG;
    }
    row = row + nwg.x;
    }
}

// out = clamp(out, f0, f1) — the calibration ranges gemma4v's linears carry
// beside their weights. Dispatched only where a range actually exists.
@compute @workgroup_size(WG)
fn clamp_range(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        out[i] = clamp(out[i], op.f0, op.f1);
        i = i + stride;
    }
}

// Average-pool an [u1 wide] patch grid by u0 on each side, then scale by f0 —
// the pooler's `sqrt(n_embd)` folded in.
//   a = [nx*ny, dim] patches, out = [ox*oy, dim] tokens
//   n_rows = ox*oy, dim = width, u0 = kernel, u1 = nx
@compute @workgroup_size(WG)
fn pool_avg(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_rows * op.dim;
    let stride = nwg.x * WG;
    let k = op.u0;
    let nx = op.u1;
    let ox = nx / k;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let r = i / op.dim;
        let c = i % op.dim;
        let bx = r % ox;
        let by = r / ox;
        var acc = 0.0;
        for (var ky = 0u; ky < k; ky = ky + 1u) {
            for (var kx = 0u; kx < k; kx = kx + 1u) {
                acc = acc + a[((by * k + ky) * nx + bx * k + kx) * op.dim + c];
            }
        }
        out[i] = acc / f32(k * k) * op.f0;
        i = i + stride;
    }
}
