//! CPU reference forward pass for qwen35.
//!
//! Like the gemma4 path this favours transparency over speed and hands every
//! intermediate to a [`Trace`] hook under llama.cpp's `eval-callback` names,
//! so the two implementations can be diffed tensor by tensor.
//!
//! The block structure (no gemma-style post-norms, no residual scaling):
//!
//! ```text
//! h = embed(tok)                          // no sqrt(d) scale
//! per layer:
//!   x = rms(h) * attn_norm
//!   a = delta_net(x)  or  attention(x)    // by cfg.recurrent[il]
//!   h = h + a
//!   y = rms(h) * post_attention_norm      // pre-FFN norm despite the name
//!   h = h + Wd (silu(Wg y) * Wu y)
//! logits = output · (rms(h) * output_norm)
//! ```
//!
//! Gated delta net, per token and v-head `h` (`S = 128`, k-head `h % 16`):
//!
//! ```text
//! [q|k|v] = silu(causal_conv4(Wqkv x))    // depthwise over the fused stream
//! q, k    = l2norm per head
//! g = exp(a_h * softplus(alpha_h(x) + dt_h)),  b = sigmoid(beta_h(x))
//! S ← g·S;  S ← S + b(v − S k) ⊗ k;  o = (S q) / √128
//! out = Wout( rms(o)·norm_w · silu(z) ),  z = Wgate x
//! ```
//!
//! Full attention differs from gemma4 everywhere it could: the Q projection
//! emits `[query | gate]` per head and the gate sigmoid-scales the attention
//! output, Q/K norms are learned per-dim vectors, only the first 64 dims
//! rotate (base 1e7), the softmax scale is the standard `1/√head_dim`, and V
//! is used untouched.

use crate::cpu::Trace;
use crate::ops;
use crate::qwen35::config::Config;
use crate::qwen35::state::{LayerState, State};
use crate::qwen35::weights::{Attention, Weights};

fn no_trace(_: &str, _: &[f32]) {}

pub struct Cpu<'a> {
    pub cfg: &'a Config,
    pub w: &'a Weights<'a>,
    buf: Buffers,
}

/// Scratch reused across tokens and layers.
struct Buffers {
    h: Vec<f32>,
    x: Vec<f32>,
    /// Fused QKV stream (linear) — also holds the fused [q|gate] (attention).
    qkv: Vec<f32>,
    z: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv_out: Vec<f32>,
    /// L2-normalized q then k, one k-head grid each.
    qk: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
}

impl Buffers {
    fn new(cfg: &Config, max_tokens: usize) -> Self {
        let t = max_tokens;
        let qg_dim = (cfg.n_heads * cfg.head_dim * 2).max(cfg.conv_dim());
        let attn_dim = (cfg.n_heads * cfg.head_dim).max(cfg.d_inner);
        Self {
            h: vec![0.0; t * cfg.d_model],
            x: vec![0.0; t * cfg.d_model],
            qkv: vec![0.0; t * qg_dim],
            z: vec![0.0; t * cfg.d_inner],
            alpha: vec![0.0; t * cfg.n_v_heads],
            beta: vec![0.0; t * cfg.n_v_heads],
            conv_out: vec![0.0; cfg.conv_dim()],
            qk: vec![0.0; 2 * cfg.n_k_heads * cfg.lin_head_dim],
            k: vec![0.0; t * cfg.kv_dim()],
            v: vec![0.0; t * cfg.kv_dim()],
            attn: vec![0.0; t * attn_dim],
            proj: vec![0.0; t * cfg.d_model],
            gate: vec![0.0; t * cfg.ffn_dim],
            up: vec![0.0; t * cfg.ffn_dim],
            scores: Vec::new(),
        }
    }
}

impl<'a> Cpu<'a> {
    pub fn new(cfg: &'a Config, w: &'a Weights<'a>, max_batch: usize) -> Self {
        Self {
            cfg,
            w,
            buf: Buffers::new(cfg, max_batch),
        }
    }

    /// Run `tokens` starting at `state.len`; returns logits for the last one.
    pub fn forward(&mut self, tokens: &[u32], state: &mut State) -> Vec<f32> {
        self.forward_traced(tokens, state, &mut no_trace)
    }

    pub fn forward_traced(
        &mut self,
        tokens: &[u32],
        state: &mut State,
        trace: &mut Trace<'_>,
    ) -> Vec<f32> {
        let cfg = self.cfg;
        let t = tokens.len();
        let d = cfg.d_model;
        let base_pos = state.len;
        assert!(t > 0, "forward called with no tokens");
        assert!(
            base_pos + t <= state.capacity,
            "context overflow: {base_pos} + {t} > {}",
            state.capacity
        );

        let b = &mut self.buf;

        // --- embeddings (no scale) ----------------------------------------
        let h = &mut b.h[..t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            self.w
                .token_embd
                .dequant_row_into(tok as usize, &mut h[i * d..(i + 1) * d]);
        }
        trace("model.input_embed", h);

        for il in 0..cfg.n_layers {
            let lw = &self.w.layers[il];

            // --- pre-attention norm ---------------------------------------
            let x = &mut b.x[..t * d];
            x.copy_from_slice(h);
            for row in x.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, lw.attn_norm, cfg.rms_eps);
            }
            trace(&format!("attn_norm-{il}"), x);

            let proj = &mut b.proj[..t * d];
            match &lw.attn {
                Attention::Linear {
                    qkv,
                    gate,
                    conv,
                    alpha,
                    dt_bias,
                    a,
                    beta,
                    norm,
                    out,
                } => {
                    let s_dim = cfg.lin_head_dim;
                    let conv_dim = cfg.conv_dim();
                    let key_dim = cfg.n_k_heads * s_dim;
                    let kernel = cfg.conv_kernel;
                    let inv_sqrt_s = 1.0 / (s_dim as f32).sqrt();

                    let mixed = &mut b.qkv[..t * conv_dim];
                    ops::matmul(mixed, qkv, x, t);
                    trace(&format!("linear_attn_qkv_mixed-{il}"), mixed);

                    let z = &mut b.z[..t * cfg.d_inner];
                    ops::matmul(z, gate, x, t);
                    trace(&format!("z-{il}"), z);

                    let alpha_raw = &mut b.alpha[..t * cfg.n_v_heads];
                    ops::matmul(alpha_raw, alpha, x, t);
                    trace(&format!("alpha-{il}"), alpha_raw);
                    let beta_raw = &mut b.beta[..t * cfg.n_v_heads];
                    ops::matmul(beta_raw, beta, x, t);
                    trace(&format!("beta-{il}"), beta_raw);

                    let LayerState::Linear { conv: conv_state, s } = &mut state.layers[il]
                    else {
                        unreachable!("layer {il}: state kind disagrees with weights");
                    };

                    // The recurrence is inherently sequential over tokens.
                    let attn_out = &mut b.attn[..t * cfg.d_inner];
                    for i in 0..t {
                        let mixed_row = &mixed[i * conv_dim..(i + 1) * conv_dim];

                        // Depthwise causal conv over [state | current], then
                        // SiLU. Channel c's taps are conv[c*kernel..].
                        let conv_out = &mut b.conv_out;
                        for c in 0..conv_dim {
                            let taps = &conv[c * kernel..(c + 1) * kernel];
                            let mut acc = mixed_row[c] * taps[kernel - 1];
                            for (j, tap) in taps[..kernel - 1].iter().enumerate() {
                                acc += conv_state[j * conv_dim + c] * tap;
                            }
                            conv_out[c] = ops::silu(acc);
                        }
                        // Slide the window: drop the oldest row, append this one.
                        conv_state.copy_within(conv_dim.., 0);
                        conv_state[(kernel - 2) * conv_dim..].copy_from_slice(mixed_row);

                        // L2-normalize q and k per k-head, once each.
                        let qk = &mut b.qk;
                        qk.copy_from_slice(&conv_out[..2 * key_dim]);
                        for head in qk.chunks_exact_mut(s_dim) {
                            ops::l2_norm(head, cfg.rms_eps);
                        }
                        let (qn, kn) = qk.split_at(key_dim);
                        let vals = &conv_out[2 * key_dim..];

                        // Delta rule per v-head; k/q heads tile modulo.
                        let o = &mut attn_out[i * cfg.d_inner..(i + 1) * cfg.d_inner];
                        for hv in 0..cfg.n_v_heads {
                            let hk = (hv % cfg.n_k_heads) * s_dim;
                            let qh = &qn[hk..hk + s_dim];
                            let kh = &kn[hk..hk + s_dim];
                            let vh = &vals[hv * s_dim..(hv + 1) * s_dim];

                            let g = (a[hv]
                                * ops::softplus(alpha_raw[i * cfg.n_v_heads + hv] + dt_bias[hv]))
                            .exp();
                            let bmix = ops::sigmoid(beta_raw[i * cfg.n_v_heads + hv]);

                            let sh = &mut s[hv * s_dim * s_dim..(hv + 1) * s_dim * s_dim];
                            for x in sh.iter_mut() {
                                *x *= g;
                            }
                            let oh = &mut o[hv * s_dim..(hv + 1) * s_dim];
                            for j in 0..s_dim {
                                let row = &mut sh[j * s_dim..(j + 1) * s_dim];
                                let pred: f32 = row.iter().zip(kh).map(|(r, k)| r * k).sum();
                                let delta = (vh[j] - pred) * bmix;
                                for (r, &k) in row.iter_mut().zip(kh) {
                                    *r += delta * k;
                                }
                                let out: f32 = row.iter().zip(qh).map(|(r, q)| r * q).sum();
                                oh[j] = out * inv_sqrt_s;
                            }
                        }

                    }
                    trace(&format!("attn_output-{il}"), attn_out);

                    // Gated RMS norm: rms(o)·w, gated by silu(z), per head.
                    for (i, row) in attn_out.chunks_exact_mut(cfg.d_inner).enumerate() {
                        let zi = &z[i * cfg.d_inner..(i + 1) * cfg.d_inner];
                        for (hv, oh) in row.chunks_exact_mut(s_dim).enumerate() {
                            ops::rms_norm_mul(oh, norm, cfg.rms_eps);
                            for (j, ov) in oh.iter_mut().enumerate() {
                                *ov *= ops::silu(zi[hv * s_dim + j]);
                            }
                        }
                    }
                    trace(&format!("final_output-{il}"), attn_out);

                    ops::matmul(proj, out, attn_out, t);
                    trace(&format!("linear_attn_out-{il}"), proj);
                }

                Attention::Full {
                    q,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    output,
                } => {
                    let head_dim = cfg.head_dim;
                    let n_kv = cfg.n_kv_heads;
                    let kv_dim = cfg.kv_dim();
                    let group = cfg.gqa_group();
                    // Per-head [query | gate] stride in the fused projection.
                    let qg = 2 * head_dim;
                    let q_dim = cfg.n_heads * qg;

                    let qfull = &mut b.qkv[..t * q_dim];
                    ops::matmul(qfull, q, x, t);
                    trace(&format!("Qcur_full-{il}"), qfull);

                    let kc = &mut b.k[..t * kv_dim];
                    ops::matmul(kc, k, x, t);
                    trace(&format!("Kcur-{il}"), kc);
                    let vc = &mut b.v[..t * kv_dim];
                    ops::matmul(vc, v, x, t);
                    trace(&format!("Vcur-{il}"), vc);

                    // Query halves: norm then partial rotation, in place.
                    for (i, head) in qfull.chunks_exact_mut(qg).enumerate() {
                        let pos = base_pos + i / cfg.n_heads;
                        ops::rms_norm_mul(&mut head[..head_dim], q_norm, cfg.rms_eps);
                        ops::rope_neox(&mut head[..cfg.n_rot], pos, cfg.rope_base, &[]);
                    }
                    for (i, head) in kc.chunks_exact_mut(head_dim).enumerate() {
                        let pos = base_pos + i / n_kv;
                        ops::rms_norm_mul(head, k_norm, cfg.rms_eps);
                        ops::rope_neox(&mut head[..cfg.n_rot], pos, cfg.rope_base, &[]);
                    }
                    // Repacked query halves, matching llama.cpp's "Qcur" view.
                    let qonly: Vec<f32> = qfull
                        .chunks_exact(qg)
                        .flat_map(|h| &h[..head_dim])
                        .copied()
                        .collect();
                    trace(&format!("Qcur-{il}"), &qonly);
                    trace(&format!("Kcur_pos-{il}"), kc);

                    let LayerState::Attn(lcache) = &mut state.layers[il] else {
                        unreachable!("layer {il}: state kind disagrees with weights");
                    };
                    for i in 0..t {
                        lcache.store(
                            base_pos + i,
                            &kc[i * kv_dim..(i + 1) * kv_dim],
                            &vc[i * kv_dim..(i + 1) * kv_dim],
                        );
                    }

                    let scale = cfg.attn_scale();
                    let attn_dim = cfg.n_heads * head_dim;
                    let attn = &mut b.attn[..t * attn_dim];
                    for i in 0..t {
                        let pos = base_pos + i;
                        b.scores.resize(pos + 1, 0.0);

                        for hq in 0..cfg.n_heads {
                            let kv_head = hq / group;
                            let qh = &qfull[i * q_dim + hq * qg..][..head_dim];
                            let gate = &qfull[i * q_dim + hq * qg + head_dim..][..head_dim];

                            for (p, sc) in b.scores.iter_mut().enumerate() {
                                let kh = &lcache.k_at(p)[kv_head * head_dim..][..head_dim];
                                *sc = qh.iter().zip(kh).map(|(a, c)| a * c).sum::<f32>() * scale;
                            }
                            ops::softmax(&mut b.scores);

                            let out = &mut attn[i * attn_dim + hq * head_dim..][..head_dim];
                            out.fill(0.0);
                            for (p, &w) in b.scores.iter().enumerate() {
                                if w == 0.0 {
                                    continue;
                                }
                                let vh = &lcache.v_at(p)[kv_head * head_dim..][..head_dim];
                                for (o, &vv) in out.iter_mut().zip(vh) {
                                    *o += w * vv;
                                }
                            }
                            // The fused projection's second half gates the
                            // attention output before Wo.
                            for (o, &gv) in out.iter_mut().zip(gate) {
                                *o *= ops::sigmoid(gv);
                            }
                        }
                    }
                    trace(&format!("attn_gated-{il}"), attn);

                    ops::matmul(proj, output, attn, t);
                    trace(&format!("attn_output-{il}"), proj);
                }
            }

            // --- residual, pre-FFN norm, FFN, residual --------------------
            for (hv, &pv) in h.iter_mut().zip(proj.iter()) {
                *hv += pv;
            }
            trace(&format!("attn_residual-{il}"), h);

            let x = &mut b.x[..t * d];
            x.copy_from_slice(h);
            for row in x.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, lw.post_attention_norm, cfg.rms_eps);
            }
            trace(&format!("attn_post_norm-{il}"), x);

            let gate = &mut b.gate[..t * cfg.ffn_dim];
            let up = &mut b.up[..t * cfg.ffn_dim];
            ops::matmul(gate, &lw.ffn.gate, x, t);
            ops::matmul(up, &lw.ffn.up, x, t);
            for (g, &u) in gate.iter_mut().zip(up.iter()) {
                *g = ops::silu(*g) * u;
            }

            let proj = &mut b.proj[..t * d];
            ops::matmul(proj, &lw.ffn.down, gate, t);
            trace(&format!("ffn_out-{il}"), proj);

            for (hv, &pv) in h.iter_mut().zip(proj.iter()) {
                *hv += pv;
            }
            trace(&format!("l_out-{il}"), h);
        }

        state.len = base_pos + t;

        // --- output head ---------------------------------------------------
        for row in h.chunks_exact_mut(d) {
            ops::rms_norm_mul(row, self.w.output_norm, cfg.rms_eps);
        }
        let last = &h[(t - 1) * d..];
        trace("result_norm", last);

        let mut logits = vec![0.0; cfg.vocab];
        ops::matvec(&mut logits, &self.w.output, last);
        trace("result_output", &logits);
        logits
    }
}
