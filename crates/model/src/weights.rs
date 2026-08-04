//! Borrowed handles to every tensor the forward pass needs.
//!
//! Nothing is copied: each field is a [`TensorView`] into the mmap'd GGUF.
//! Resolving all names up front turns a typo into a load-time error instead of
//! a panic 40 layers into a generation.

use anyhow::Context;
use gguf::{Gguf, TensorView};

use crate::config::Config;

pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub attn_q: TensorView<'a>,
    pub attn_k: TensorView<'a>,
    /// Absent on global layers, where V reuses the K projection.
    pub attn_v: Option<TensorView<'a>>,
    pub attn_q_norm: &'a [f32],
    pub attn_k_norm: &'a [f32],
    pub attn_output: TensorView<'a>,
    pub post_attention_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub ffn_gate: TensorView<'a>,
    pub ffn_up: TensorView<'a>,
    pub ffn_down: TensorView<'a>,
    pub post_ffw_norm: &'a [f32],
    /// Scalar applied to the whole residual stream at the end of the block.
    pub output_scale: f32,
}

impl<'a> LayerWeights<'a> {
    /// The projection that produces V. On global layers this is the *K* weight;
    /// the two differ only in the normalization applied afterwards.
    pub fn v_proj(&self) -> TensorView<'a> {
        self.attn_v.unwrap_or(self.attn_k)
    }
}

pub struct Weights<'a> {
    pub token_embd: TensorView<'a>,
    pub output_norm: &'a [f32],
    /// Tied to `token_embd` when the file ships no separate output matrix.
    pub output: TensorView<'a>,
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

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let lc = &cfg.layers[i];
            let p = |s: &str| format!("blk.{i}.{s}");

            let attn_v = g.tensor_opt(&p("attn_v.weight"));
            anyhow::ensure!(
                attn_v.is_some() != lc.v_from_k,
                "blk.{i}: attn_v presence ({}) contradicts sliding_window_pattern \
                 (expected v_from_k={})",
                attn_v.is_some(),
                lc.v_from_k
            );

            let scale = f32_tensor(&p("layer_output_scale.weight"))?;
            anyhow::ensure!(scale.len() == 1, "blk.{i}: layer_output_scale must be scalar");

            layers.push(LayerWeights {
                attn_norm: f32_tensor(&p("attn_norm.weight"))?,
                attn_q: g.tensor(&p("attn_q.weight"))?,
                attn_k: g.tensor(&p("attn_k.weight"))?,
                attn_v,
                attn_q_norm: f32_tensor(&p("attn_q_norm.weight"))?,
                attn_k_norm: f32_tensor(&p("attn_k_norm.weight"))?,
                attn_output: g.tensor(&p("attn_output.weight"))?,
                post_attention_norm: f32_tensor(&p("post_attention_norm.weight"))?,
                ffn_norm: f32_tensor(&p("ffn_norm.weight"))?,
                ffn_gate: g.tensor(&p("ffn_gate.weight"))?,
                ffn_up: g.tensor(&p("ffn_up.weight"))?,
                ffn_down: g.tensor(&p("ffn_down.weight"))?,
                post_ffw_norm: f32_tensor(&p("post_ffw_norm.weight"))?,
                output_scale: scale[0],
            });
        }

        let w = Self {
            token_embd,
            output_norm: f32_tensor("output_norm.weight")?,
            output,
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

        for (i, (w, lc)) in self.layers.iter().zip(&cfg.layers).enumerate() {
            let q_dim = cfg.n_heads * lc.head_dim;
            let kv_dim = lc.kv_dim();
            let checks: [(&str, usize, usize); 6] = [
                ("attn_q.out", w.attn_q.out_dim(), q_dim),
                ("attn_k.out", w.attn_k.out_dim(), kv_dim),
                ("attn_output.in", w.attn_output.in_dim(), q_dim),
                ("attn_output.out", w.attn_output.out_dim(), d),
                ("ffn_gate.out", w.ffn_gate.out_dim(), cfg.ffn_dim),
                ("ffn_down.out", w.ffn_down.out_dim(), d),
            ];
            for (what, got, want) in checks {
                anyhow::ensure!(got == want, "blk.{i}.{what}: {got} != expected {want}");
            }
            anyhow::ensure!(
                w.attn_q_norm.len() == lc.head_dim && w.attn_k_norm.len() == lc.head_dim,
                "blk.{i}: qk-norm length != head_dim {}",
                lc.head_dim
            );
            if let Some(v) = w.attn_v {
                anyhow::ensure!(
                    v.out_dim() == kv_dim,
                    "blk.{i}.attn_v.out: {} != {kv_dim}",
                    v.out_dim()
                );
            }
        }
        Ok(())
    }
}
