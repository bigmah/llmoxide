//! qwen35 and qwen3 architecture description, derived from GGUF metadata.
//!
//! qwen35 (Qwen3.5 27B) is a *hybrid* stack: most layers are gated-delta-net
//! linear attention ("recurrent" in llama.cpp's terms), with a full-attention
//! layer every `full_attention_interval`-th position. The checkpoint also
//! appends one NextN/MTP block past the main stack (`nextn_predict_layers = 1`);
//! it is only used for speculative decoding, so the main forward pass skips it
//! and `n_layers` here counts main layers only.
//!
//! Plain **qwen3** (Qwen3 0.6B and friends) is the same description with two
//! things switched off, which is why it lives here rather than in a module of
//! its own: every layer is full attention ([`Config::recurrent`] all false),
//! and the query projection carries no fused output gate
//! ([`Config::query_gate`]). Everything else — RMSNorm, GQA, per-dim query and
//! key norms, NeoX RoPE, SwiGLU — is bit-for-bit the same code path. The
//! alternative was a second copy of an attention block this file already has
//! working and validated.

use gguf::Gguf;

/// Resolve the end-of-generation token set by spelling. `<|im_end|>` is the
/// assistant's turn terminator; `<|endoftext|>` ends pretraining documents.
fn eog_tokens(g: &Gguf) -> Vec<u32> {
    let names = ["<|im_end|>", "<|endoftext|>"];
    let tokens = g.string_array("tokenizer.ggml.tokens").unwrap_or(&[]);
    let mut out: Vec<u32> = names
        .iter()
        .filter_map(|n| tokens.iter().position(|t| t == n).map(|i| i as u32))
        .collect();
    if let Ok(eos) = g.u64("tokenizer.ggml.eos_token_id") {
        out.push(eos as u32);
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Main layers only — the trailing NextN/MTP block is not counted and
    /// never executed.
    pub n_layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub context_length: usize,

    // -- full-attention layers ---------------------------------------------
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Per-head width (`attention.key_length` == `attention.value_length`).
    pub head_dim: usize,
    /// Only the first `n_rot` dims of each head rotate; the rest are NoPE.
    /// The checkpoint declares M-RoPE sections for vision, but with text-only
    /// input every section sees the same position, which reduces to plain
    /// NeoX RoPE — verified against llama.cpp tensor-by-tensor.
    pub n_rot: usize,
    pub rope_base: f32,

    /// Whether `attn_q` emits a fused `[query | output-gate]` pair per head.
    ///
    /// qwen35 does; plain qwen3 does not, and its `attn_q` is `head_dim` wide
    /// per head rather than `2 * head_dim`. This is the only difference in the
    /// attention block between the two.
    pub query_gate: bool,

    // -- gated-delta-net layers (all zero when `recurrent` is all false) ----
    /// Depthwise causal conv width over the fused QKV stream (`ssm.conv_kernel`).
    pub conv_kernel: usize,
    /// Head width for both K and V sides of the state (`ssm.state_size`).
    pub lin_head_dim: usize,
    /// K/Q head count (`ssm.group_count`); V heads tile over these mod-wise.
    pub n_k_heads: usize,
    /// V head count (`ssm.time_step_rank` — the converter's name, not ours).
    pub n_v_heads: usize,
    /// `n_v_heads * lin_head_dim`, the value stream width.
    pub d_inner: usize,

    /// `true` = gated delta net, `false` = full attention, per main layer.
    pub recurrent: Vec<bool>,

    pub bos: u32,
    pub eos: u32,
    pub eog: Vec<u32>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let arch = g.str("general.architecture")?.to_string();
        // The hybrid carries an `ssm.*` block and a gated query; the dense one
        // carries neither, and reading those keys would fail rather than
        // default.
        let hybrid = match arch.as_str() {
            "qwen35" => true,
            "qwen3" => false,
            other => anyhow::bail!("unsupported architecture {other:?}"),
        };
        let k = |s: &str| format!("{arch}.{s}");

        let n_layers_all = g.usize(&k("block_count"))?;
        let n_nextn = g.usize(&k("nextn_predict_layers")).unwrap_or(0);
        anyhow::ensure!(n_nextn < n_layers_all, "nextn_predict_layers >= block_count");
        let n_layers = n_layers_all - n_nextn;

        let d_model = g.usize(&k("embedding_length"))?;
        let head_dim = g.usize(&k("attention.key_length"))?;
        anyhow::ensure!(
            g.usize(&k("attention.value_length"))? == head_dim,
            "key_length != value_length"
        );

        let (lin_head_dim, n_k_heads, n_v_heads, d_inner) = if hybrid {
            let lin_head_dim = g.usize(&k("ssm.state_size"))?;
            let n_k_heads = g.usize(&k("ssm.group_count"))?;
            let n_v_heads = g.usize(&k("ssm.time_step_rank"))?;
            let d_inner = g.usize(&k("ssm.inner_size"))?;
            anyhow::ensure!(
                d_inner == n_v_heads * lin_head_dim,
                "ssm.inner_size {d_inner} != n_v_heads {n_v_heads} * head_dim {lin_head_dim}"
            );
            anyhow::ensure!(
                n_v_heads % n_k_heads == 0,
                "v-head count {n_v_heads} not a multiple of k-head count {n_k_heads}"
            );
            (lin_head_dim, n_k_heads, n_v_heads, d_inner)
        } else {
            (0, 0, 0, 0)
        };

        // Which layers are recurrent: an explicit per-layer array wins, else
        // every `interval`-th layer (1-based) is full attention. A dense stack
        // has none.
        let recurrent = if !hybrid {
            vec![false; n_layers]
        } else {
            match g.bool_array(&k("attention.recurrent_layers")) {
                Ok(v) => {
                    anyhow::ensure!(v.len() >= n_layers, "recurrent_layers array too short");
                    v[..n_layers].to_vec()
                }
                Err(_) => {
                    let interval = g.usize(&k("full_attention_interval")).unwrap_or(4);
                    (0..n_layers).map(|i| (i + 1) % interval != 0).collect()
                }
            }
        };

        // The tensor table carries the shape; reading the payload for it would
        // mean holding the whole embedding just to learn `ne[1]`, which the
        // browser build cannot do.
        let vocab = g.info("token_embd.weight")?.out_dim();

        Ok(Self {
            n_layers,
            d_model,
            ffn_dim: g.usize(&k("feed_forward_length"))?,
            vocab,
            rms_eps: g.f32(&k("attention.layer_norm_rms_epsilon"))?,
            context_length: g.usize(&k("context_length")).unwrap_or(8192),
            n_heads: g.usize(&k("attention.head_count"))?,
            n_kv_heads: g.usize(&k("attention.head_count_kv"))?,
            head_dim,
            n_rot: g.usize(&k("rope.dimension_count")).unwrap_or(head_dim),
            rope_base: g.f32(&k("rope.freq_base"))?,
            query_gate: hybrid,
            conv_kernel: if hybrid { g.usize(&k("ssm.conv_kernel"))? } else { 0 },
            lin_head_dim,
            n_k_heads,
            n_v_heads,
            d_inner,
            recurrent,
            bos: g.u64("tokenizer.ggml.bos_token_id").unwrap_or(0) as u32,
            eos: g.u64("tokenizer.ggml.eos_token_id").unwrap_or(0) as u32,
            eog: eog_tokens(g),
        })
    }

    /// Width of the fused QKV stream the conv runs over: Q and K at
    /// `n_k_heads` heads each, V at `n_v_heads`.
    pub fn conv_dim(&self) -> usize {
        2 * self.n_k_heads * self.lin_head_dim + self.d_inner
    }

    /// Query heads per KV head on full-attention layers.
    pub fn gqa_group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Softmax scale on full-attention layers — the standard `1/√head_dim`
    /// (gemma4's folded-scale surprise does not apply here).
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Per-head stride of the `attn_q` projection: query, plus the output gate
    /// when the checkpoint fuses one in.
    pub fn q_stride(&self) -> usize {
        if self.query_gate {
            2 * self.head_dim
        } else {
            self.head_dim
        }
    }

    pub fn summary(&self) -> String {
        let recr = self.recurrent.iter().filter(|r| **r).count();
        if recr == 0 {
            return format!(
                "qwen3: {}L dense @ {} heads/{} kv, head_dim {} rope {}/{} base {:.0} \
                 d_model {} ffn {} vocab {}",
                self.n_layers,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.n_rot,
                self.head_dim,
                self.rope_base,
                self.d_model,
                self.ffn_dim,
                self.vocab,
            );
        }
        format!(
            "qwen35: {}L ({recr} gated-delta-net @ {} v-heads/{} k-heads x{}, \
             {} full-attn @ {} heads/{} kv, head_dim {} rope {}/{} base {:.0}) \
             d_model {} ffn {} vocab {}",
            self.n_layers,
            self.n_v_heads,
            self.n_k_heads,
            self.lin_head_dim,
            self.n_layers - recr,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.n_rot,
            self.head_dim,
            self.rope_base,
            self.d_model,
            self.ffn_dim,
            self.vocab,
        )
    }
}
