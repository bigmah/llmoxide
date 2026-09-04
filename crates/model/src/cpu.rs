//! CPU reference forward pass.
//!
//! This is the correctness oracle for the GPU path, so it favours transparency
//! over speed. Every intermediate is handed to a [`Trace`] hook under the same
//! name llama.cpp's `eval-callback` uses, which is what lets
//! `cargo run --bin validate` diff the two implementations tensor by tensor.
//!
//! The block, in the order the graph actually runs:
//!
//! ```text
//! h  = embed(tok) * sqrt(d_model)
//! x  = rms(h) * attn_norm
//! q  = rope(rms_head(Wq x) * q_norm)
//! k  = rope(rms_head(Wk x) * k_norm)
//! v  =      rms_head(Wv x)              // no weight, no rope; Wv := Wk on global layers
//! h  = h + rms(Wo attn(q,k,v)) * post_attention_norm
//! y  = rms(h) * ffn_norm
//! h  = h + rms(Wd (gelu(Wg y) * Wu y)) * post_ffw_norm
//! h  = h + rms(Wp (gelu(Wg_ple h) * ple[layer])) * post_norm   // E-series only
//! h  = h * layer_output_scale
//! ```
//!
//! The per-layer-embedding (PLE) line is what separates the E-series (E2B/E4B)
//! from the dense checkpoints. Each layer gets its own 256-wide embedding per
//! token, looked up from a table rather than computed, and gates the residual
//! stream through it. The dense 12B has `n_embd_per_layer == 0` and skips it.
//!
//! Layers past the KV-sharing boundary project Q only and attend into an
//! earlier layer's cache; see [`crate::config::LayerConfig::kv_source`].

use crate::cache::KvCache;
use crate::config::Config;
use crate::ops;
use crate::weights::Weights;

/// Receives every named intermediate. `&mut dyn` keeps `forward` monomorphic.
pub type Trace<'t> = dyn FnMut(&str, &[f32]) + 't;

fn no_trace(_: &str, _: &[f32]) {}

pub struct Cpu<'a> {
    pub cfg: &'a Config,
    pub w: &'a Weights<'a>,
    buf: Buffers,
}

/// Scratch reused across tokens and layers; sized for the widest layer.
struct Buffers {
    h: Vec<f32>,
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    /// `[t, n_layers, n_embd_per_layer]` — the blended per-layer inputs, held
    /// for the whole batch because every layer indexes its own slice. Empty
    /// when the checkpoint has no per-layer embeddings.
    ple: Vec<f32>,
    ple_proj: Vec<f32>,
    ple_gate: Vec<f32>,
}

impl Buffers {
    fn new(cfg: &Config, max_tokens: usize) -> Self {
        let d = cfg.d_model;
        let head_dim = cfg.layers.iter().map(|l| l.head_dim).max().unwrap_or(0);
        let q_dim = cfg.n_heads * head_dim;
        let kv_dim = cfg.layers.iter().map(|l| l.kv_dim()).max().unwrap_or(0);
        Self {
            h: vec![0.0; max_tokens * d],
            x: vec![0.0; max_tokens * d],
            q: vec![0.0; max_tokens * q_dim],
            k: vec![0.0; max_tokens * kv_dim],
            v: vec![0.0; max_tokens * kv_dim],
            attn: vec![0.0; max_tokens * q_dim],
            proj: vec![0.0; max_tokens * d],
            gate: vec![0.0; max_tokens * cfg.ffn_dim],
            up: vec![0.0; max_tokens * cfg.ffn_dim],
            scores: Vec::new(),
            ple: vec![0.0; max_tokens * cfg.n_embd_per_layer * cfg.n_layers],
            ple_proj: vec![0.0; max_tokens * cfg.n_embd_per_layer * cfg.n_layers],
            ple_gate: vec![0.0; max_tokens * cfg.n_embd_per_layer],
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

    /// Run `tokens` starting at `cache.len` and return the final logits for the
    /// last token only — the rest are never needed for autoregressive decoding.
    pub fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        self.forward_traced(tokens, cache, &mut no_trace)
    }

    pub fn forward_traced(
        &mut self,
        tokens: &[u32],
        cache: &mut KvCache,
        trace: &mut Trace<'_>,
    ) -> Vec<f32> {
        let cfg = self.cfg;
        let t = tokens.len();
        let d = cfg.d_model;
        let base_pos = cache.len;
        assert!(t > 0, "forward called with no tokens");
        assert!(
            base_pos + t <= cache.capacity,
            "context overflow: {} + {t} > {}",
            base_pos,
            cache.capacity
        );

        let b = &mut self.buf;

        // --- embeddings ----------------------------------------------------
        let h = &mut b.h[..t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            self.w
                .token_embd
                .dequant_row_into(tok as usize, &mut h[i * d..(i + 1) * d]);
        }
        trace("embd", h);
        for v in h.iter_mut() {
            *v *= cfg.embed_scale;
        }
        trace("inp_scaled", h);

        // --- per-layer embeddings ------------------------------------------
        // Built once for the whole batch: each token gets one small vector per
        // layer, half looked up from `per_layer_token_embd` and half projected
        // from the token embedding, averaged as `(a + b) / sqrt(2)`.
        let e = cfg.n_embd_per_layer;
        let span = e * cfg.n_layers;
        if let Some(ple) = &self.w.ple {
            let pl = &mut b.ple[..t * span];
            for (i, &tok) in tokens.iter().enumerate() {
                ple.token_embd
                    .dequant_row_into(tok as usize, &mut pl[i * span..(i + 1) * span]);
            }
            let lookup_scale = (e as f32).sqrt();
            for v in pl.iter_mut() {
                *v *= lookup_scale;
            }
            trace("inp_per_layer_selected", pl);

            let proj = &mut b.ple_proj[..t * span];
            ops::matmat(proj, &ple.model_proj, h, t);
            let proj_scale = 1.0 / (d as f32).sqrt();
            // The norm runs over each layer's slice independently, matching
            // llama.cpp normalizing dim0 of the [e, n_layers, t] view.
            for seg in proj.chunks_exact_mut(e) {
                for v in seg.iter_mut() {
                    *v *= proj_scale;
                }
                ops::rms_norm_mul(seg, ple.proj_norm, cfg.rms_eps);
            }
            trace("per_layer_proj", proj);

            let blend = 1.0 / 2f32.sqrt();
            for (o, &p) in pl.iter_mut().zip(proj.iter()) {
                *o = (*o + p) * blend;
            }
            trace("inp_per_layer", pl);
        }

        for il in 0..cfg.n_layers {
            let lc = &cfg.layers[il];
            let lw = &self.w.layers[il];
            let head_dim = lc.head_dim;
            let n_kv = lc.n_kv_heads;
            let q_dim = cfg.n_heads * head_dim;
            let kv_dim = lc.kv_dim();
            let group = cfg.n_heads / n_kv;
            // Only global layers divide their angles by the frequency factors;
            // SWA layers rotate every pair at full strength.
            let rope_factors: &[f32] = if lc.rope_factors {
                &cfg.rope_factors
            } else {
                &[]
            };

            // --- pre-attention norm ---------------------------------------
            let x = &mut b.x[..t * d];
            x.copy_from_slice(h);
            for row in x.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, lw.attn_norm, cfg.rms_eps);
            }
            trace(&format!("attn_norm-{il}"), x);

            // --- Q projection ---------------------------------------------
            // Q is projected on every layer, including the ones that own no KV
            // and only read an earlier layer's cache.
            let q = &mut b.q[..t * q_dim];
            ops::matmat(q, &lw.attn_q, x, t);
            trace(&format!("Qcur-{il}"), q);

            for head in q.chunks_exact_mut(head_dim) {
                ops::rms_norm_mul(head, lw.attn_q_norm, cfg.rms_eps);
            }
            trace(&format!("Qcur_normed-{il}"), q);
            for (i, head) in q.chunks_exact_mut(head_dim).enumerate() {
                let pos = base_pos + i / cfg.n_heads;
                ops::rope_neox(head, pos, lc.rope_base, rope_factors);
            }
            trace(&format!("Qcur_pos-{il}"), q);

            // --- K / V projections and cache write ------------------------
            // Skipped entirely on shared-KV layers: they neither compute nor
            // store, so the cache entry they read stays the one written by the
            // owning layer below them.
            if let Some(kvw) = &lw.kv {
                let k = &mut b.k[..t * kv_dim];
                let v = &mut b.v[..t * kv_dim];

                ops::matmat(k, &kvw.attn_k, x, t);
                trace(&format!("Kcur-{il}"), k);

                if lc.v_from_k {
                    // No attn_v in the file: V is the *same* projection as K,
                    // diverging only in the normalization applied below.
                    v.copy_from_slice(k);
                } else {
                    ops::matmat(v, &kvw.v_proj(), x, t);
                }
                trace(&format!("Vcur-{il}"), v);

                for head in k.chunks_exact_mut(head_dim) {
                    ops::rms_norm_mul(head, kvw.attn_k_norm, cfg.rms_eps);
                }
                trace(&format!("Kcur_normed-{il}"), k);
                for (i, head) in k.chunks_exact_mut(head_dim).enumerate() {
                    let pos = base_pos + i / n_kv;
                    ops::rope_neox(head, pos, lc.rope_base, rope_factors);
                }
                trace(&format!("Kcur_pos-{il}"), k);

                // V is normalized per head with no learned weight, never rotated.
                for head in v.chunks_exact_mut(head_dim) {
                    ops::rms_norm(head, cfg.rms_eps);
                }
                trace(&format!("Vcur_normed-{il}"), v);

                let lcache = cache.layer_mut(il);
                for i in 0..t {
                    lcache.store(
                        base_pos + i,
                        &k[i * kv_dim..(i + 1) * kv_dim],
                        &v[i * kv_dim..(i + 1) * kv_dim],
                    );
                }
            }

            // --- attention -------------------------------------------------
            let lcache = cache.layer(il);
            let scale = cfg.attn_scale(il);
            let attn = &mut b.attn[..t * q_dim];
            for i in 0..t {
                let pos = base_pos + i;
                let window = lcache.visible(pos);
                let n_vis = window.end() - window.start() + 1;
                b.scores.resize(n_vis, 0.0);

                for hq in 0..cfg.n_heads {
                    let kv_head = hq / group;
                    let qh = &q[i * q_dim + hq * head_dim..][..head_dim];

                    for (s, p) in window.clone().enumerate() {
                        let kh = &lcache.k_at(p)[kv_head * head_dim..][..head_dim];
                        b.scores[s] = qh.iter().zip(kh).map(|(a, c)| a * c).sum::<f32>() * scale;
                    }
                    ops::softmax(&mut b.scores);

                    let out = &mut attn[i * q_dim + hq * head_dim..][..head_dim];
                    out.fill(0.0);
                    for (s, p) in window.clone().enumerate() {
                        let w = b.scores[s];
                        if w == 0.0 {
                            continue;
                        }
                        let vh = &lcache.v_at(p)[kv_head * head_dim..][..head_dim];
                        for (o, &vv) in out.iter_mut().zip(vh) {
                            *o += w * vv;
                        }
                    }
                }
            }
            trace(&format!("kqv_out-{il}"), attn);

            // --- attention output projection + residual --------------------
            let proj = &mut b.proj[..t * d];
            ops::matmat(proj, &lw.attn_output, attn, t);
            for row in proj.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, lw.post_attention_norm, cfg.rms_eps);
            }
            trace(&format!("attn_post_norm-{il}"), proj);
            for (hv, &pv) in h.iter_mut().zip(proj.iter()) {
                *hv += pv;
            }
            trace(&format!("attn_out-{il}"), h);

            // --- feed-forward ----------------------------------------------
            let x = &mut b.x[..t * d];
            x.copy_from_slice(h);
            for row in x.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, lw.ffn_norm, cfg.rms_eps);
            }
            trace(&format!("ffn_norm-{il}"), x);

            let gate = &mut b.gate[..t * cfg.ffn_dim];
            let up = &mut b.up[..t * cfg.ffn_dim];
            ops::matmat(gate, &lw.ffn_gate, x, t);
            ops::matmat(up, &lw.ffn_up, x, t);
            let geglu = &mut b.gate[..t * cfg.ffn_dim];
            for i in 0..t * cfg.ffn_dim {
                geglu[i] = ops::gelu(geglu[i]) * up[i];
            }
            trace(&format!("ffn_geglu-{il}"), geglu);

            let proj = &mut b.proj[..t * d];
            ops::matmat(proj, &lw.ffn_down, geglu, t);
            for row in proj.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, lw.post_ffw_norm, cfg.rms_eps);
            }
            trace(&format!("ffn_post_norm-{il}"), proj);

            for (hv, &pv) in h.iter_mut().zip(proj.iter()) {
                *hv += pv;
            }
            trace(&format!("pe_in-{il}"), h);

            // --- per-layer embedding ---------------------------------------
            // Gate the residual stream down to the per-layer width, modulate it
            // by this layer's slice of the precomputed table, and project back.
            if let Some(pl) = &lw.per_layer {
                let gate = &mut b.ple_gate[..t * e];
                ops::matmat(gate, &pl.inp_gate, h, t);
                for (i, row) in gate.chunks_exact_mut(e).enumerate() {
                    let inp = &b.ple[i * span + il * e..][..e];
                    for (gv, &iv) in row.iter_mut().zip(inp) {
                        *gv = ops::gelu(*gv) * iv;
                    }
                }
                let proj = &mut b.proj[..t * d];
                ops::matmat(proj, &pl.proj, gate, t);
                for row in proj.chunks_exact_mut(d) {
                    ops::rms_norm_mul(row, pl.post_norm, cfg.rms_eps);
                }
                trace(&format!("per_layer_embd_out-{il}"), proj);
                for (hv, &pv) in h.iter_mut().zip(proj.iter()) {
                    *hv += pv;
                }
            }

            // The whole stream is rescaled by the layer's scalar.
            for hv in h.iter_mut() {
                *hv *= lw.output_scale;
            }
            trace(&format!("l_out-{il}"), h);
        }

        cache.len = base_pos + t;

        // --- output head ---------------------------------------------------
        for row in h.chunks_exact_mut(d) {
            ops::rms_norm_mul(row, self.w.output_norm, cfg.rms_eps);
        }
        trace("h_nextn", h);

        let last = &h[(t - 1) * d..];
        trace("result_norm", last);

        let mut logits = vec![0.0; cfg.vocab];
        ops::matvec(&mut logits, &self.w.output, last);
        if let Some(cap) = cfg.logit_softcap {
            ops::soft_cap(&mut logits, cap);
        }
        for &tok in &cfg.suppress_tokens {
            if let Some(l) = logits.get_mut(tok as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
        trace("result_output", &logits);
        logits
    }
}
