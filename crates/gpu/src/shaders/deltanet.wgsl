// Gated-delta-net kernels for the qwen35 recurrent layers.
//
// The layer's data flow (see model::qwen35::cpu, the validated reference):
//
//   fused = Wqkv x                      [t, conv_dim] = [t, q|k|v channels]
//   conv  = silu(causal_conv4(fused))   depthwise over the fused stream
//   q, k  = l2norm per s_dim head       first 2*key_dim channels of `conv`
//   per v-head: S <- g*S + b(v - (g*S)k) (x) k;   o = (S q) / sqrt(s_dim)
//
// One bind group layout serves every entry point; unused slots are bound to a
// dummy buffer, as in ops.wgsl.
//   0 params — F32 model tensors (conv taps, ssm_a, dt_bias), concatenated
//   1 a      — fused stream (conv kernels), scratch (commit), alpha (recur)
//   2 b      — beta projections (recur)
//   3 c      — conv output, i.e. the q|k|v channels (recur)
//   4 state  — conv window state, or S            (read_write)
//   5 out    — kernel output                      (read_write)
//   6 op     — the uniform below

struct DOp {
    n_tok: u32,
    conv_dim: u32,
    key_dim: u32,   // n_kh * s_dim
    d_inner: u32,   // n_vh * s_dim
    s_dim: u32,
    n_vh: u32,
    n_kh: u32,
    kernel: u32,    // conv width, taps per channel
    off_taps: u32,  // params offset of ssm_conv1d [conv_dim, kernel]
    off_a: u32,     // params offset of ssm_a [n_vh]
    off_dt: u32,    // params offset of ssm_dt.bias [n_vh]
    eps: f32,
    scale: f32,     // 1 / sqrt(s_dim)
};

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> op: DOp;

const WG: u32 = 256u;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// Depthwise causal conv over [state | fused batch], then SiLU. Position `p`
// relative to the batch start reads fused row `p` when p >= 0 and conv-state
// row `p + kernel-1` otherwise (state rows are time-major, oldest first).
// Reads state, never writes it — the shift below produces the new window.
@compute @workgroup_size(WG)
fn conv_silu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = op.n_tok * op.conv_dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let tok = i / op.conv_dim;
        let ch = i % op.conv_dim;
        let taps = op.off_taps + ch * op.kernel;

        var acc = a[i] * params[taps + op.kernel - 1u];
        for (var j = 0u; j < op.kernel - 1u; j = j + 1u) {
            let p = i32(tok + j) - i32(op.kernel - 1u);
            var v: f32;
            if (p >= 0) {
                v = a[u32(p) * op.conv_dim + ch];
            } else {
                v = state[u32(p + i32(op.kernel - 1u)) * op.conv_dim + ch];
            }
            acc = acc + v * params[taps + j];
        }
        out[i] = silu(acc);
        i = i + stride;
    }
}

// The conv window after this batch: the last kernel-1 fused rows, drawing from
// the old state when the batch is shorter than the window. Writes to `out` (a
// scratch buffer) so no thread reads a row another is overwriting; conv_commit
// copies it back.
@compute @workgroup_size(WG)
fn conv_shift(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = (op.kernel - 1u) * op.conv_dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let r = i / op.conv_dim;
        let ch = i % op.conv_dim;
        let p = i32(op.n_tok + r) - i32(op.kernel - 1u);
        if (p >= 0) {
            out[i] = a[u32(p) * op.conv_dim + ch];
        } else {
            out[i] = state[u32(p + i32(op.kernel - 1u)) * op.conv_dim + ch];
        }
        i = i + stride;
    }
}

@compute @workgroup_size(WG)
fn conv_commit(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = (op.kernel - 1u) * op.conv_dim;
    let stride = nwg.x * WG;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        state[i] = a[i];
        i = i + stride;
    }
}

var<workgroup> red: array<f32, WG>;

// L2-normalize the q and k heads of each token, in place on `out` (the conv
// output). Row r of n_tok * 2*n_kh covers channels [h*s_dim, (h+1)*s_dim) of
// its token, h = r % (2*n_kh) — the v channels beyond 2*key_dim are untouched.
// ggml semantics: eps floors the norm, it is not added under the root.
@compute @workgroup_size(WG)
fn l2norm_qk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let rows = op.n_tok * 2u * op.n_kh;
    var r = wg.x;
    loop {
        if (r >= rows) { break; }
        let base = (r / (2u * op.n_kh)) * op.conv_dim + (r % (2u * op.n_kh)) * op.s_dim;

        var acc = 0.0;
        var i = tid;
        loop {
            if (i >= op.s_dim) { break; }
            let v = out[base + i];
            acc = acc + v * v;
            i = i + WG;
        }
        red[tid] = acc;
        workgroupBarrier();
        var s = WG / 2u;
        loop {
            if (s == 0u) { break; }
            if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
            workgroupBarrier();
            s = s / 2u;
        }
        let inv = 1.0 / max(sqrt(red[0]), op.eps);
        workgroupBarrier();

        i = tid;
        loop {
            if (i >= op.s_dim) { break; }
            out[base + i] = out[base + i] * inv;
            i = i + WG;
        }
        workgroupBarrier();
        r = r + nwg.x;
    }
}

// The delta rule. One workgroup per v-head; thread j owns row j of that head's
// s_dim x s_dim state matrix (rows are value dims, matching the CPU layout).
// Tokens are inherently sequential, so the loop over them lives inside the
// kernel; heads never interact, so no cross-workgroup sync is needed.
//
// Bindings: a = alpha projections [t, n_vh], b = beta [t, n_vh],
// c = conv output (l2-normed q|k, v), state = S, out = [t, d_inner].

const MAX_S: u32 = 128u;
var<workgroup> q_s: array<f32, MAX_S>;
var<workgroup> k_s: array<f32, MAX_S>;
var<workgroup> gb: vec2<f32>;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(MAX_S)
fn delta_recur(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let hv = wg.x;
    let hk = (hv % op.n_kh) * op.s_dim;
    let own_row = tid < op.s_dim;

    for (var i = 0u; i < op.n_tok; i = i + 1u) {
        let tok_base = i * op.conv_dim;
        if (own_row) {
            q_s[tid] = c[tok_base + hk + tid];
            k_s[tid] = c[tok_base + op.key_dim + hk + tid];
        }
        if (tid == 0u) {
            let g = exp(params[op.off_a + hv]
                * softplus(a[i * op.n_vh + hv] + params[op.off_dt + hv]));
            gb = vec2<f32>(g, 1.0 / (1.0 + exp(-b[i * op.n_vh + hv])));
        }
        workgroupBarrier();

        if (own_row) {
            let g = gb.x;
            let v = c[tok_base + 2u * op.key_dim + hv * op.s_dim + tid];
            let srow = (hv * op.s_dim + tid) * op.s_dim;

            // Decay is folded into both passes: (g*S)k for the prediction,
            // then g*S + delta*k stored — term-for-term what the CPU does.
            var pred = 0.0;
            for (var d = 0u; d < op.s_dim; d = d + 1u) {
                pred = pred + (g * state[srow + d]) * k_s[d];
            }
            let delta = (v - pred) * gb.y;

            var o = 0.0;
            for (var d = 0u; d < op.s_dim; d = d + 1u) {
                let sv = g * state[srow + d] + delta * k_s[d];
                state[srow + d] = sv;
                o = o + sv * q_s[d];
            }
            out[i * op.d_inner + hv * op.s_dim + tid] = o * op.scale;
        }
        workgroupBarrier();
    }
}
