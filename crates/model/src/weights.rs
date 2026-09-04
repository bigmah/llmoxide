//! Borrowed handles to every tensor the forward pass needs.
//!
//! Nothing is copied: each field is a [`TensorView`] into the mmap'd GGUF.
//! Resolving all names up front turns a typo into a load-time error instead of
//! a panic 40 layers into a generation.

use anyhow::Context;
use gguf::{Gguf, TensorView};

use crate::config::Config;

/// The projections a layer needs to *produce* K and V.
///
/// Absent on layers that share an earlier layer's cache: those project Q only
/// and never write. Such layers may still carry `attn_k`/`attn_v` tensors in
/// the file — E4B ships them for all 42 blocks — but they are dead weight, so
/// modelling the absence here keeps the forward pass from reaching for them.
pub struct KvWeights<'a> {
    pub attn_k: TensorView<'a>,
    /// Absent where V reuses the K projection (the 12B's global layers).
    pub attn_v: Option<TensorView<'a>>,
    pub attn_k_norm: &'a [f32],
}

impl<'a> KvWeights<'a> {
    /// The projection that produces V. Where the file ships no `attn_v` this is
    /// the *K* weight; the two differ only in the normalization applied after.
    pub fn v_proj(&self) -> TensorView<'a> {
        self.attn_v.unwrap_or(self.attn_k)
    }
}

/// Per-layer-embedding (PLE) weights for one block.
pub struct PerLayerWeights<'a> {
    pub inp_gate: TensorView<'a>,
    pub proj: TensorView<'a>,
    pub post_norm: &'a [f32],
}

pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub attn_q: TensorView<'a>,
    pub attn_q_norm: &'a [f32],
    /// `None` on layers that attend into an earlier layer's cache.
    pub kv: Option<KvWeights<'a>>,
    pub attn_output: TensorView<'a>,
    pub post_attention_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub ffn_gate: TensorView<'a>,
    pub ffn_up: TensorView<'a>,
    pub ffn_down: TensorView<'a>,
    pub post_ffw_norm: &'a [f32],
    /// `None` on checkpoints without per-layer embeddings.
    pub per_layer: Option<PerLayerWeights<'a>>,
    /// Scalar applied to the whole residual stream at the end of the block.
    pub output_scale: f32,
}

/// Model-level per-layer-embedding tables.
pub struct PleWeights<'a> {
    /// `[n_embd_per_layer * n_layers, vocab]` — one small vector per layer per
    /// token, looked up rather than computed. This is where the E-series keeps
    /// the bulk of its parameters, and why "E4B" is 4.5B effective out of 8B.
    pub token_embd: TensorView<'a>,
    pub model_proj: TensorView<'a>,
    pub proj_norm: &'a [f32],
}

pub struct Weights<'a> {
    pub token_embd: TensorView<'a>,
    pub output_norm: &'a [f32],
    /// Tied to `token_embd` when the file ships no separate output matrix.
    pub output: TensorView<'a>,
    pub ple: Option<PleWeights<'a>>,
    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> Weights<'a> {
    pub fn load(g: &'a Gguf, cfg: &Config) -> anyhow::Result<Self> {
        let f32_tensor = |name: &str| -> anyhow::Result<&'a [f32]> {
            let t = g.tensor(name).with_context(|| format!("missing {name}"))?;
            t.as_f32()
                .with_context(|| format!("{name} is {}, expected F32", t.ty().name()))
        };

        let token_embd = g.tensor("token_embd.weight")?;
        let output = g.tensor_opt("output.weight").unwrap_or(token_embd);

        let ple = if cfg.n_embd_per_layer > 0 {
            Some(PleWeights {
                token_embd: g.tensor("per_layer_token_embd.weight")?,
                model_proj: g.tensor("per_layer_model_proj.weight")?,
                proj_norm: f32_tensor("per_layer_proj_norm.weight")?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let lc = &cfg.layers[i];
            let p = |s: &str| format!("blk.{i}.{s}");

            // Shared-KV layers project Q only; their K/V tensors, if the file
            // even ships them, are never read.
            let kv = if lc.owns_kv(i) {
                Some(KvWeights {
                    attn_k: g.tensor(&p("attn_k.weight"))?,
                    attn_v: g.tensor_opt(&p("attn_v.weight")),
                    attn_k_norm: f32_tensor(&p("attn_k_norm.weight"))?,
                })
            } else {
                None
            };

            let per_layer = if cfg.n_embd_per_layer > 0 {
                Some(PerLayerWeights {
                    inp_gate: g.tensor(&p("inp_gate.weight"))?,
                    proj: g.tensor(&p("proj.weight"))?,
                    post_norm: f32_tensor(&p("post_norm.weight"))?,
                })
            } else {
                None
            };

            let scale = f32_tensor(&p("layer_output_scale.weight"))?;
            anyhow::ensure!(scale.len() == 1, "blk.{i}: layer_output_scale must be scalar");

            layers.push(LayerWeights {
                attn_norm: f32_tensor(&p("attn_norm.weight"))?,
                attn_q: g.tensor(&p("attn_q.weight"))?,
                attn_q_norm: f32_tensor(&p("attn_q_norm.weight"))?,
                kv,
                attn_output: g.tensor(&p("attn_output.weight"))?,
                post_attention_norm: f32_tensor(&p("post_attention_norm.weight"))?,
                ffn_norm: f32_tensor(&p("ffn_norm.weight"))?,
                ffn_gate: g.tensor(&p("ffn_gate.weight"))?,
                ffn_up: g.tensor(&p("ffn_up.weight"))?,
                ffn_down: g.tensor(&p("ffn_down.weight"))?,
                post_ffw_norm: f32_tensor(&p("post_ffw_norm.weight"))?,
                per_layer,
                output_scale: scale[0],
            });
        }

        let w = Self {
            token_embd,
            output_norm: f32_tensor("output_norm.weight")?,
            output,
            ple,
            layers,
        };
        w.check_shapes(cfg)?;
        Ok(w)
    }

    /// Cross-check every weight against the config so a metadata/tensor
    /// mismatch surfaces here rather than as silent numerical garbage.
    fn check_shapes(&self, cfg: &Config) -> anyhow::Result<()> {
        let d = cfg.d_model;
        anyhow::ensure!(
            self.token_embd.in_dim() == d,
            "token_embd in_dim {} != d_model {d}",
            self.token_embd.in_dim()
        );

        if let Some(ple) = &self.ple {
            let per_tok = cfg.n_embd_per_layer * cfg.n_layers;
            anyhow::ensure!(
                ple.token_embd.in_dim() == per_tok,
                "per_layer_token_embd in_dim {} != n_embd_per_layer * n_layers {per_tok}",
                ple.token_embd.in_dim()
            );
            anyhow::ensure!(
                ple.model_proj.in_dim() == d && ple.model_proj.out_dim() == per_tok,
                "per_layer_model_proj is {}x{}, expected {d}x{per_tok}",
                ple.model_proj.in_dim(),
                ple.model_proj.out_dim()
            );
            anyhow::ensure!(
                ple.proj_norm.len() == cfg.n_embd_per_layer,
                "per_layer_proj_norm length {} != n_embd_per_layer {}",
                ple.proj_norm.len(),
                cfg.n_embd_per_layer
            );
        }

        for (i, (w, lc)) in self.layers.iter().zip(&cfg.layers).enumerate() {
            let q_dim = cfg.n_heads * lc.head_dim;
            let kv_dim = lc.kv_dim();
            let checks: [(&str, usize, usize); 5] = [
                ("attn_q.out", w.attn_q.out_dim(), q_dim),
                ("attn_output.in", w.attn_output.in_dim(), q_dim),
                ("attn_output.out", w.attn_output.out_dim(), d),
                ("ffn_gate.out", w.ffn_gate.out_dim(), cfg.ffn_dim),
                ("ffn_down.out", w.ffn_down.out_dim(), d),
            ];
            for (what, got, want) in checks {
                anyhow::ensure!(got == want, "blk.{i}.{what}: {got} != expected {want}");
            }
            anyhow::ensure!(
                w.attn_q_norm.len() == lc.head_dim,
                "blk.{i}: q-norm length {} != head_dim {}",
                w.attn_q_norm.len(),
                lc.head_dim
            );

            if let Some(kv) = &w.kv {
                anyhow::ensure!(
                    kv.attn_k.out_dim() == kv_dim,
                    "blk.{i}.attn_k.out: {} != {kv_dim}",
                    kv.attn_k.out_dim()
                );
                anyhow::ensure!(
                    kv.attn_k_norm.len() == lc.head_dim,
                    "blk.{i}: k-norm length {} != head_dim {}",
                    kv.attn_k_norm.len(),
                    lc.head_dim
                );
                if let Some(v) = kv.attn_v {
                    anyhow::ensure!(
                        v.out_dim() == kv_dim,
                        "blk.{i}.attn_v.out: {} != {kv_dim}",
                        v.out_dim()
                    );
                }
                anyhow::ensure!(
                    kv.attn_v.is_none() == lc.v_from_k,
                    "blk.{i}: attn_v presence disagrees with config v_from_k"
                );
            }

            if let Some(pl) = &w.per_layer {
                let e = cfg.n_embd_per_layer;
                anyhow::ensure!(
                    pl.inp_gate.in_dim() == d && pl.inp_gate.out_dim() == e,
                    "blk.{i}.inp_gate is {}x{}, expected {d}x{e}",
                    pl.inp_gate.in_dim(),
                    pl.inp_gate.out_dim()
                );
                anyhow::ensure!(
                    pl.proj.in_dim() == e && pl.proj.out_dim() == d,
                    "blk.{i}.proj is {}x{}, expected {e}x{d}",
                    pl.proj.in_dim(),
                    pl.proj.out_dim()
                );
                anyhow::ensure!(
                    pl.post_norm.len() == d,
                    "blk.{i}.post_norm length {} != d_model {d}",
                    pl.post_norm.len()
                );
            }
        }
        Ok(())
    }
}
