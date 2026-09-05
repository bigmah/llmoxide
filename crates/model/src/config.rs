//! Gemma4 architecture description, derived entirely from GGUF metadata.
//!
//! Everything here was read out of the model file rather than hard-coded, so a
//! differently-shaped gemma4 checkpoint should load without edits. The handful
//! of genuinely fixed conventions (partial-RoPE sentinel, softcap form) are
//! called out where they appear.

use anyhow::Context;
use gguf::Gguf;

/// Frequency factors at or above this are the converter's "do not rotate this
/// dimension" sentinel (`1e30`); dividing by them zeroes the angle.
const NO_ROPE_SENTINEL: f32 = 1e20;

/// Resolve the end-of-generation token set by spelling, falling back to just
/// `<eos>` if a checkpoint names its control tokens differently.
fn eog_tokens(g: &Gguf) -> Vec<u32> {
    let names = ["<eos>", "<turn|>", "<|tool_response>"];
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

/// Read a per-layer integer that a checkpoint may store either as one scalar
/// covering every layer or as an explicit per-layer array.
///
/// Both spellings are live: the 12B writes `head_count_kv` as a 48-entry array,
/// E4B writes it as the scalar `2`. llama.cpp papers over this with
/// `get_key_or_arr`; this is the same accommodation.
fn per_layer_u32(g: &Gguf, key: &str, n_layers: usize) -> anyhow::Result<Vec<u32>> {
    if let Ok(v) = g.u32_array(key) {
        anyhow::ensure!(
            v.len() == n_layers,
            "{key}: {} entries for {n_layers} layers",
            v.len()
        );
        return Ok(v);
    }
    let scalar = g.u64(key)? as u32;
    Ok(vec![scalar; n_layers])
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Sliding-window attention layer. Global (full-attention) layers are the
    /// every-6th exception.
    pub swa: bool,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    /// `Some(w)` limits attention to the last `w` positions inclusive of self.
    pub window: Option<usize>,
    /// Whether this layer divides its RoPE angles by [`Config::rope_factors`].
    ///
    /// Only global layers do. Verified against llama.cpp: on a sliding-window
    /// layer the highest head dimensions still rotate by a small angle, while
    /// on a global layer they are bit-identical before and after RoPE. The
    /// effect is that local layers keep full positional resolution across their
    /// 1024-token window, and global layers retain only the lowest 64
    /// frequencies with the remaining dimensions left unrotated (NoPE).
    pub rope_factors: bool,
    /// Global layers ship no `attn_v` tensor: V is projected with `attn_k`
    /// and then normalized differently. See [`crate::weights`].
    pub v_from_k: bool,
    /// Which layer's KV cache this layer reads. Equal to the layer's own index
    /// unless the checkpoint shares KV across the tail of the stack, in which
    /// case several layers project only Q and attend into an earlier layer's
    /// cache. See [`Config::from_gguf`].
    pub kv_source: usize,
}

impl LayerConfig {
    /// Whether this layer computes and stores its own K/V. Shared layers do
    /// neither: their `attn_k`/`attn_v` weights are dead metadata.
    #[inline]
    pub fn owns_kv(&self, index: usize) -> bool {
        self.kv_source == index
    }
}

impl LayerConfig {
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    /// Logits are squashed as `cap * tanh(x / cap)`.
    pub logit_softcap: Option<f32>,
    /// Embeddings are multiplied by this on the way in (`sqrt(d_model)`).
    pub embed_scale: f32,
    /// Per-rotary-pair frequency divisors. Entries at the `1e30` sentinel make
    /// that pair's angle zero, i.e. those dimensions are effectively NoPE.
    /// Indexed by pair, so `rope_factors[ic]` covers dims `ic` and
    /// `ic + head_dim/2` under ggml's NeoX pairing.
    pub rope_factors: Vec<f32>,
    /// How many leading pairs actually rotate — derived, for reporting only.
    pub rope_pairs: usize,
    pub context_length: usize,
    /// Width of the per-layer embedding (PLE) vectors, or 0 when the checkpoint
    /// has none. The E-series (E2B/E4B) spends most of its parameter budget on
    /// a per-layer embedding table and keeps the residual stream narrow; the
    /// dense 12B/31B set this to 0 and skip the whole path.
    pub n_embd_per_layer: usize,
    pub layers: Vec<LayerConfig>,
    /// Token ids the model must never emit (image/audio sentinels).
    pub suppress_tokens: Vec<u32>,
    pub bos: u32,
    pub eos: u32,
    /// Tokens that end a generation.
    ///
    /// Beyond `<eos>` this includes `<turn|>` (the model closing its turn) and
    /// `<|tool_response>` — after emitting a tool call the model opens a
    /// response block and stops, waiting for the harness to supply the result.
    /// Treating that as end-of-generation is what makes agentic tool use work.
    pub eog: Vec<u32>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let arch = g.str("general.architecture")?.to_string();
        anyhow::ensure!(arch == "gemma4", "unsupported architecture {arch:?}");
        let k = |s: &str| format!("{arch}.{s}");

        let n_layers = g.usize(&k("block_count"))?;
        let d_model = g.usize(&k("embedding_length"))?;
        let n_heads = g.usize(&k("attention.head_count"))?;
        let ffn_dim = g.usize(&k("feed_forward_length"))?;
        let rms_eps = g.f32(&k("attention.layer_norm_rms_epsilon"))?;

        // Per-layer arrays: KV head count and the SWA pattern.
        let kv_heads = per_layer_u32(g, &k("attention.head_count_kv"), n_layers)?;
        let swa_pattern = g.bool_array(&k("attention.sliding_window_pattern"))?;
        anyhow::ensure!(
            swa_pattern.len() == n_layers,
            "sliding_window_pattern has {} entries for {n_layers} layers",
            swa_pattern.len()
        );

        // Per-layer embeddings. Absent (or 0) on the dense checkpoints.
        let n_embd_per_layer = g.usize(&k("embedding_length_per_layer_input")).unwrap_or(0);

        // KV sharing across the tail of the stack. `shared_kv_layers = n` means
        // the last `n` layers project Q only and attend into the cache of the
        // last *preceding* layer of the same attention type — the sliding ones
        // reuse `boundary - 2`, the global ones `boundary - 1`, which is exactly
        // llama.cpp's reuse rule (llama-model.cpp, `n_layer_kv_from_start`).
        let n_shared = g.usize(&k("attention.shared_kv_layers")).unwrap_or(0);
        let boundary = n_layers
            .checked_sub(n_shared)
            .with_context(|| format!("shared_kv_layers {n_shared} exceeds block_count {n_layers}"))?;
        anyhow::ensure!(
            n_shared == 0 || boundary >= 2,
            "shared_kv_layers {n_shared} leaves only {boundary} owning layers; \
             need at least 2 so both attention types have a source"
        );
        if n_shared > 0 {
            // The reuse rule assumes the two layers below the boundary are one
            // of each type. Verify rather than trust the pattern.
            anyhow::ensure!(
                swa_pattern[boundary - 2] != swa_pattern[boundary - 1],
                "layers {} and {} below the KV-sharing boundary are both swa={}; \
                 the reuse rule needs one sliding and one global",
                boundary - 2,
                boundary - 1,
                swa_pattern[boundary - 1]
            );
        }

        let head_dim_global = g.usize(&k("attention.key_length"))?;
        let head_dim_swa = g.usize(&k("attention.key_length_swa"))?;
        let rope_base_global = g.f32(&k("rope.freq_base"))?;
        let rope_base_swa = g.f32(&k("rope.freq_base_swa"))?;
        let window = g.usize(&k("attention.sliding_window"))?;

        let layers = (0..n_layers)
            .map(|i| {
                let swa = swa_pattern[i];
                LayerConfig {
                    swa,
                    n_kv_heads: kv_heads[i] as usize,
                    head_dim: if swa { head_dim_swa } else { head_dim_global },
                    rope_base: if swa { rope_base_swa } else { rope_base_global },
                    window: swa.then_some(window),
                    rope_factors: !swa,
                    // Driven by what the file actually ships rather than by the
                    // SWA pattern: the 12B omits `attn_v` on global layers and
                    // folds V into the K projection, while E4B ships a real
                    // `attn_v` everywhere.
                    v_from_k: !g.has_tensor(&format!("blk.{i}.attn_v.weight")),
                    kv_source: if i < boundary {
                        i
                    } else if swa {
                        boundary - 2
                    } else {
                        boundary - 1
                    },
                }
            })
            .collect();

        // ggml rotates every pair but divides each angle by its frequency
        // factor, so the `1e30` entries are how this checkpoint expresses
        // partial RoPE. An absent tensor means "rotate everything".
        let rope_factors = g
            .tensor_opt("rope_freqs.weight")
            .map(|t| t.to_f32())
            .unwrap_or_default();
        let rope_pairs = rope_factors
            .iter()
            .position(|&f| f >= NO_ROPE_SENTINEL)
            .unwrap_or(rope_factors.len().max(head_dim_global / 2));

        // The tensor *table* carries the shape; reading the payload for it
        // would mean holding an 800 MB embedding just to learn `ne[1]`.
        let vocab = g.info("token_embd.weight")?.out_dim();

        Ok(Self {
            n_layers,
            d_model,
            n_heads,
            ffn_dim,
            vocab,
            rms_eps,
            logit_softcap: g.f32(&k("final_logit_softcapping")).ok().filter(|c| *c > 0.0),
            embed_scale: (d_model as f32).sqrt(),
            rope_factors,
            rope_pairs,
            context_length: g.usize(&k("context_length")).unwrap_or(8192),
            n_embd_per_layer,
            layers,
            suppress_tokens: g.u32_array("tokenizer.ggml.suppress_tokens").unwrap_or_default(),
            bos: g.u64("tokenizer.ggml.bos_token_id").unwrap_or(2) as u32,
            eos: g.u64("tokenizer.ggml.eos_token_id").unwrap_or(1) as u32,
            eog: eog_tokens(g),
        })
    }

    /// Attention softmax scale — deliberately `1.0`, not `1/sqrt(head_dim)`.
    ///
    /// This checkpoint folds the attention temperature into the QK-norm gains.
    /// `attn_q_norm` and `attn_k_norm` are *uniform scalars* rather than learned
    /// per-dimension vectors, and `k_norm` tracks `32/head_dim` (0.125 for the
    /// 256-wide SWA layers, exactly 0.0625 for the 512-wide global ones) — an
    /// inverse-dimension law, not the inverse-square-root a plain gain would
    /// show. Applying `1/sqrt(head_dim)` on top flattens the softmax badly
    /// enough that all positions collapse to the same hidden state within a few
    /// layers. Verified against llama.cpp on the per-head attention outputs.
    pub fn attn_scale(&self, layer: usize) -> f32 {
        let _ = layer;
        std::env::var("LLMOXIDE_ATTN_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0)
    }

    /// Query heads per KV head (GQA group size).
    pub fn gqa_group(&self, layer: usize) -> usize {
        self.n_heads / self.layers[layer].n_kv_heads
    }

    /// How many KV slots a layer needs to serve `n_ctx` tokens. SWA layers only
    /// ever look back `window` positions, so their cache is a small ring.
    pub fn kv_slots(&self, layer: usize, n_ctx: usize) -> usize {
        match self.layers[layer].window {
            Some(w) => w.min(n_ctx),
            None => n_ctx,
        }
    }

    /// Total KV cache bytes at f16 for a given context length. Layers that
    /// share an earlier layer's cache allocate nothing.
    pub fn kv_bytes(&self, n_ctx: usize) -> usize {
        (0..self.n_layers)
            .filter(|&i| self.layers[i].owns_kv(i))
            .map(|i| self.kv_slots(i, n_ctx) * self.layers[i].kv_dim() * 2 * 2)
            .sum()
    }

    /// Layers that actually hold a cache, in order.
    pub fn kv_owning_layers(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n_layers).filter(|&i| self.layers[i].owns_kv(i))
    }

    pub fn summary(&self) -> String {
        let swa = self.layers.iter().filter(|l| l.swa).count();
        let shared = self.n_layers - self.kv_owning_layers().count();
        let extra = format!(
            "{}{}",
            if self.n_embd_per_layer > 0 {
                format!("\n  per-layer embeddings: width {}", self.n_embd_per_layer)
            } else {
                String::new()
            },
            if shared > 0 {
                format!("\n  KV sharing: {shared} of {} layers reuse an earlier cache", self.n_layers)
            } else {
                String::new()
            },
        );
        format!(
            "gemma4: {}L (
  {swa} sliding-window @ window {:?}, {} kv-heads, head_dim {}, rope_base {:.0}
  {} global,           full attention, {} kv-heads, head_dim {}, rope_base {:.0}{}
) d_model {} ffn {} heads {} vocab {} rope_dim {} (rest NoPE) softcap {:?}",
            self.n_layers,
            self.layers.iter().find(|l| l.swa).and_then(|l| l.window),
            self.layers.iter().find(|l| l.swa).map_or(0, |l| l.n_kv_heads),
            self.layers.iter().find(|l| l.swa).map_or(0, |l| l.head_dim),
            self.layers.iter().find(|l| l.swa).map_or(0.0, |l| l.rope_base),
            self.n_layers - swa,
            self.layers.iter().find(|l| !l.swa).map_or(0, |l| l.n_kv_heads),
            self.layers.iter().find(|l| !l.swa).map_or(0, |l| l.head_dim),
            self.layers.iter().find(|l| !l.swa).map_or(0.0, |l| l.rope_base),
            if self.layers.iter().any(|l| !l.swa && l.v_from_k) {
                ", V shares K proj"
            } else {
                ""
            },
            self.d_model,
            self.ffn_dim,
            self.n_heads,
            self.vocab,
            self.rope_pairs * 2,
            self.logit_softcap,
        ) + &extra
    }
}
