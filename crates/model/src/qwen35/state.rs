//! Per-layer inference state for the hybrid stack.
//!
//! Full-attention layers keep an ordinary KV cache ([`LayerCache`] with no
//! window — they see the whole context). Gated-delta-net layers keep two small
//! fixed-size buffers instead, which is the whole point of the architecture:
//!
//! * the conv window — the last `kernel-1` fused-QKV vectors, and
//! * the recurrent state — one `head_dim × head_dim` matrix per v-head.
//!
//! Unlike a KV cache, recurrent state cannot rewind: position `p`'s state has
//! already absorbed everything before it. Prefix reuse therefore only works
//! append-only, which [`crate::cache::KvCache`]-style truncation would break —
//! so this type only grows or resets.

use crate::cache::LayerCache;
use crate::qwen35::config::Config;

pub enum LayerState {
    Linear {
        /// `(kernel-1) * conv_dim`, time-major, oldest row first.
        conv: Vec<f32>,
        /// `n_v_heads * head_dim * head_dim`; within a head, row `j` (a value
        /// dim) holds that dim's weights over the key dims — rows are what the
        /// recurrence reads and writes contiguously.
        s: Vec<f32>,
    },
    Attn(LayerCache),
}

pub struct State {
    pub layers: Vec<LayerState>,
    /// Number of positions absorbed so far.
    pub len: usize,
    pub capacity: usize,
}

impl State {
    pub fn new(cfg: &Config, n_ctx: usize) -> Self {
        let layers = (0..cfg.n_layers)
            .map(|i| {
                if cfg.recurrent[i] {
                    LayerState::Linear {
                        conv: vec![0.0; (cfg.conv_kernel - 1) * cfg.conv_dim()],
                        s: vec![0.0; cfg.n_v_heads * cfg.lin_head_dim * cfg.lin_head_dim],
                    }
                } else {
                    let kv_dim = cfg.kv_dim();
                    LayerState::Attn(LayerCache {
                        k: vec![0.0; n_ctx * kv_dim],
                        v: vec![0.0; n_ctx * kv_dim],
                        slots: n_ctx,
                        kv_dim,
                        window: None,
                    })
                }
            })
            .collect();
        Self {
            layers,
            len: 0,
            capacity: n_ctx,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        for l in &mut self.layers {
            if let LayerState::Linear { conv, s } = l {
                conv.fill(0.0);
                s.fill(0.0);
            }
        }
    }

    /// Overwrite every layer's state, attention KV included.
    ///
    /// [`Self::clear`] zeroes the recurrent buffers because the recurrence is
    /// cumulative and would otherwise be wrong; it deliberately leaves the
    /// attention caches alone, since nothing reads past `len`. This clears
    /// both, for when the goal is that the conversation stops existing.
    pub fn wipe(&mut self) {
        self.len = 0;
        for l in &mut self.layers {
            match l {
                LayerState::Linear { conv, s } => {
                    secret::zero_slice(conv);
                    secret::zero_slice(s);
                }
                LayerState::Attn(c) => c.wipe(),
            }
        }
    }

    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Linear { conv, s } => conv.len() + s.len(),
                LayerState::Attn(c) => c.k.len() + c.v.len(),
            })
            .sum::<usize>()
            * std::mem::size_of::<f32>()
    }
}
