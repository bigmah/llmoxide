//! Borrowed tensor handles for the qwen35 forward pass.
//!
//! Same philosophy as the gemma4 loader: resolve and shape-check every name up
//! front so a converter quirk fails at load time, not 60 layers into a
//! generation. The trailing NextN/MTP block (`blk.{n_layers}`) is deliberately
//! not loaded — the main pass never touches it.

use anyhow::Context;
use gguf::{Gguf, TensorView};

use super::config::Config;

/// FFN weights are identical on both layer kinds.
pub struct FfnWeights<'a> {
    pub gate: TensorView<'a>,
    pub up: TensorView<'a>,
    pub down: TensorView<'a>,
}

pub enum Attention<'a> {
    /// Gated delta net ("linear attention" / recurrent) layer.
    Linear {
        /// Fused Q|K|V projection over `conv_dim()` outputs.
        qkv: TensorView<'a>,
        /// The output gate `z`, applied after the recurrence as `silu(z)`.
        gate: TensorView<'a>,
        /// Depthwise causal conv, `[kernel, conv_dim]`, dense f32.
        conv: &'a [f32],
        /// Per-v-head decay projection and its bias.
        alpha: TensorView<'a>,
        dt_bias: &'a [f32],
        /// Per-v-head `-exp(A_log)`, baked negative by the converter.
        a: &'a [f32],
        /// Per-v-head mixing strength projection (sigmoid applied at runtime).
        beta: TensorView<'a>,
        /// Gated RMS-norm weight over one v-head, shared across heads.
        norm: &'a [f32],
        out: TensorView<'a>,
    },
    /// Full-attention layer.
    Full {
        /// Fused per-head `[query | output-gate]` projection: head `h` owns
        /// rows `h*2*head_dim .. (h+1)*2*head_dim`, query first.
        q: TensorView<'a>,
        k: TensorView<'a>,
        v: TensorView<'a>,
        /// Learned per-dim RMS gains over one head (unlike gemma4's scalars).
        q_norm: &'a [f32],
        k_norm: &'a [f32],
        output: TensorView<'a>,
    },
}

pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    /// Pre-FFN norm. The name is the converter's; functionally this is the
    /// classic `ffn_norm` — the FFN residual branches *before* it.
    pub post_attention_norm: &'a [f32],
    pub attn: Attention<'a>,
    pub ffn: FfnWeights<'a>,
}

pub struct Weights<'a> {
    pub token_embd: TensorView<'a>,
    pub output_norm: &'a [f32],
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
            let p = |s: &str| format!("blk.{i}.{s}");

            let attn = if cfg.recurrent[i] {
                Attention::Linear {
                    qkv: g.tensor(&p("attn_qkv.weight"))?,
                    gate: g.tensor(&p("attn_gate.weight"))?,
                    conv: f32_tensor(&p("ssm_conv1d.weight"))?,
                    alpha: g.tensor(&p("ssm_alpha.weight"))?,
                    dt_bias: f32_tensor(&p("ssm_dt.bias"))?,
                    a: f32_tensor(&p("ssm_a"))?,
                    beta: g.tensor(&p("ssm_beta.weight"))?,
                    norm: f32_tensor(&p("ssm_norm.weight"))?,
                    out: g.tensor(&p("ssm_out.weight"))?,
                }
            } else {
                Attention::Full {
                    q: g.tensor(&p("attn_q.weight"))?,
                    k: g.tensor(&p("attn_k.weight"))?,
                    v: g.tensor(&p("attn_v.weight"))?,
                    q_norm: f32_tensor(&p("attn_q_norm.weight"))?,
                    k_norm: f32_tensor(&p("attn_k_norm.weight"))?,
                    output: g.tensor(&p("attn_output.weight"))?,
                }
            };

            layers.push(LayerWeights {
                attn_norm: f32_tensor(&p("attn_norm.weight"))?,
                post_attention_norm: f32_tensor(&p("post_attention_norm.weight"))?,
                attn,
                ffn: FfnWeights {
                    gate: g.tensor(&p("ffn_gate.weight"))?,
                    up: g.tensor(&p("ffn_up.weight"))?,
                    down: g.tensor(&p("ffn_down.weight"))?,
                },
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

        for (i, lw) in self.layers.iter().enumerate() {
            let mut checks: Vec<(&str, usize, usize)> = vec![
                ("attn_norm.len", lw.attn_norm.len(), d),
                ("post_attention_norm.len", lw.post_attention_norm.len(), d),
                ("ffn_gate.out", lw.ffn.gate.out_dim(), cfg.ffn_dim),
                ("ffn_up.out", lw.ffn.up.out_dim(), cfg.ffn_dim),
                ("ffn_down.in", lw.ffn.down.in_dim(), cfg.ffn_dim),
                ("ffn_down.out", lw.ffn.down.out_dim(), d),
            ];
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
                    checks.extend([
                        ("attn_qkv.out", qkv.out_dim(), cfg.conv_dim()),
                        ("attn_gate.out", gate.out_dim(), cfg.d_inner),
                        ("ssm_conv1d.len", conv.len(), cfg.conv_kernel * cfg.conv_dim()),
                        ("ssm_alpha.out", alpha.out_dim(), cfg.n_v_heads),
                        ("ssm_dt.bias.len", dt_bias.len(), cfg.n_v_heads),
                        ("ssm_a.len", a.len(), cfg.n_v_heads),
                        ("ssm_beta.out", beta.out_dim(), cfg.n_v_heads),
                        ("ssm_norm.len", norm.len(), cfg.lin_head_dim),
                        ("ssm_out.in", out.in_dim(), cfg.d_inner),
                        ("ssm_out.out", out.out_dim(), d),
                    ]);
                }
                Attention::Full {
                    q,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    output,
                } => {
                    checks.extend([
                        // Q carries query and gate: two head_dim blocks per head.
                        ("attn_q.out", q.out_dim(), cfg.n_heads * cfg.head_dim * 2),
                        ("attn_k.out", k.out_dim(), cfg.kv_dim()),
                        ("attn_v.out", v.out_dim(), cfg.kv_dim()),
                        ("attn_q_norm.len", q_norm.len(), cfg.head_dim),
                        ("attn_k_norm.len", k_norm.len(), cfg.head_dim),
                        ("attn_output.in", output.in_dim(), cfg.n_heads * cfg.head_dim),
                        ("attn_output.out", output.out_dim(), d),
                    ]);
                }
            }
            for (what, got, want) in checks {
                anyhow::ensure!(got == want, "blk.{i}.{what}: {got} != expected {want}");
            }
        }
        Ok(())
    }
}
