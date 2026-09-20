//! Gemma 4 vision tower description, read from the `mmproj` GGUF.
//!
//! The encoder is a ViT in shape only. Every block is the *text* stack's block
//! with the sequence axis swapped for patches: RMSNorm rather than LayerNorm,
//! per-head Q/K norms, a gated GELU feed-forward, and a post-norm on both
//! residual branches. What is genuinely its own are the two positional
//! mechanisms — a learned (x, y) lookup added to the patch embedding *and* a
//! 2-D RoPE applied inside attention — and the clamped linears, which carry
//! calibration ranges next to their weights.
//!
//! A handful of constants are not in the file. llama.cpp hard-codes them per
//! projector type (`clip.cpp`, `PROJECTOR_TYPE_GEMMA4V`), so they are spelled
//! out here with the same values rather than defaulted to something plausible.

use anyhow::Context;
use gguf::Gguf;

/// Pooling kernel, and so the patch-to-token ratio on each side. Not in the
/// GGUF; llama.cpp sets it per projector type and reads the optional override
/// below.
const N_MERGE: usize = 3;

/// RoPE base for the in-attention 2-D rotation. Deliberately tiny next to a
/// text model's 10 000 — the grid is tens of patches across, not thousands of
/// tokens.
const ROPE_THETA: f32 = 100.0;

/// Token budget per image. The floor is not a memory concern but a quality
/// one: llama.cpp raised it to 40 because the tower does visibly badly on
/// very small inputs.
const MIN_IMAGE_TOKENS: usize = 40;
/// llama.cpp allows 280. 256 here so that one image always fits inside the
/// default `max_batch`, which it must: the span prefills in a single
/// bidirectional batch. The difference is a few percent of resolution on the
/// largest images and nothing else.
const MAX_IMAGE_TOKENS: usize = 256;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    /// Average-pool kernel applied to the patch grid before projection.
    pub n_merge: usize,
    /// Width of the text model's residual stream — what the projector emits.
    pub proj_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// Resize bounds in pixels, derived from the token budget.
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
    /// Rows in each of the two positional lookup tables.
    pub pos_table_len: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        anyhow::ensure!(
            g.str("general.architecture").unwrap_or_default() == "clip",
            "not an mmproj file: general.architecture is not \"clip\""
        );
        anyhow::ensure!(
            g.get("clip.has_vision_encoder")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "mmproj carries no vision encoder"
        );
        let proj = g.str("clip.vision.projector_type").unwrap_or_default();
        anyhow::ensure!(
            proj == "gemma4v",
            "unsupported vision projector {proj:?}; this path implements gemma4v"
        );

        let d_model = g.usize("clip.vision.embedding_length")?;
        let n_heads = g.usize("clip.vision.attention.head_count")?;
        anyhow::ensure!(
            d_model % n_heads == 0,
            "embedding_length {d_model} not divisible by head_count {n_heads}"
        );
        let head_dim = d_model / n_heads;
        anyhow::ensure!(
            head_dim % 2 == 0,
            "head_dim {head_dim} is odd; the 2-D rotation splits it in half"
        );

        let patch_size = g.usize("clip.vision.patch_size")?;
        // An override exists but is absent on every published checkpoint.
        let n_merge = g
            .usize("clip.vision.projector_scale_factor")
            .unwrap_or(N_MERGE);
        anyhow::ensure!(n_merge > 0, "projector scale factor must be positive");

        // Both bounds count *pooled* tokens, so one token is an n_merge-square
        // block of patches.
        let patch_area = patch_size * patch_size * n_merge * n_merge;

        let pos = g
            .info("v.position_embd.weight")
            .context("mmproj has no v.position_embd.weight")?;
        let pos_table_len = pos.dims.get(1).copied().unwrap_or(0) as usize;
        anyhow::ensure!(
            pos.dims.len() == 3 && pos.dims[2] == 2,
            "v.position_embd.weight should be [d, n, 2] (an x table and a y table), got {:?}",
            pos.dims
        );

        let proj_dim = g.info("mm.input_projection.weight")?.out_dim();

        Ok(Self {
            n_layers: g.usize("clip.vision.block_count")?,
            d_model,
            ffn_dim: g.usize("clip.vision.feed_forward_length")?,
            n_heads,
            head_dim,
            patch_size,
            n_merge,
            proj_dim,
            eps: g.f32("clip.vision.attention.layer_norm_epsilon")?,
            rope_theta: ROPE_THETA,
            image_min_pixels: MIN_IMAGE_TOKENS * patch_area,
            image_max_pixels: MAX_IMAGE_TOKENS * patch_area,
            pos_table_len,
        })
    }

    /// Side of one pooled token in pixels — every resized image is a whole
    /// number of these on both axes.
    pub fn align(&self) -> usize {
        self.patch_size * self.n_merge
    }

    pub fn summary(&self) -> String {
        format!(
            "gemma4v: {}L d {} ffn {} heads {} head_dim {} patch {} pool {}x{} -> {} \
             ({}..{} px/image, pos table {})",
            self.n_layers,
            self.d_model,
            self.ffn_dim,
            self.n_heads,
            self.head_dim,
            self.patch_size,
            self.n_merge,
            self.n_merge,
            self.proj_dim,
            self.image_min_pixels,
            self.image_max_pixels,
            self.pos_table_len,
        )
    }
}
