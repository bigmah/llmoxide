//! KV cache.
//!
//! Sliding-window layers only ever read the last `window` positions, so they
//! get a ring buffer of exactly that size instead of a full-context allocation.
//! With 40 of the 48 layers windowed at 1024, this is the difference between
//! ~2 GB and ~90 GB at the model's full 262 144-token context.

use crate::config::Config;

pub struct LayerCache {
    /// `slots * kv_dim`, row-major by slot.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub slots: usize,
    pub kv_dim: usize,
    /// `Some(w)` for a ring of size `w`; `None` for a linear full-context cache.
    pub window: Option<usize>,
}

impl LayerCache {
    /// Slot holding position `pos`. Ring layers wrap; global layers are direct.
    #[inline]
    pub fn slot(&self, pos: usize) -> usize {
        match self.window {
            Some(w) => pos % w,
            None => pos,
        }
    }

    /// The inclusive range of positions visible from `pos`.
    #[inline]
    pub fn visible(&self, pos: usize) -> std::ops::RangeInclusive<usize> {
        let first = match self.window {
            Some(w) => pos.saturating_sub(w - 1),
            None => 0,
        };
        first..=pos
    }

    #[inline]
    pub fn k_at(&self, pos: usize) -> &[f32] {
        let s = self.slot(pos) * self.kv_dim;
        &self.k[s..s + self.kv_dim]
    }

    #[inline]
    pub fn v_at(&self, pos: usize) -> &[f32] {
        let s = self.slot(pos) * self.kv_dim;
        &self.v[s..s + self.kv_dim]
    }

    /// Overwrite this layer's keys and values in full.
    pub fn wipe(&mut self) {
        secret::zero_slice(&mut self.k);
        secret::zero_slice(&mut self.v);
    }

    pub fn store(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let s = self.slot(pos) * self.kv_dim;
        self.k[s..s + self.kv_dim].copy_from_slice(k);
        self.v[s..s + self.kv_dim].copy_from_slice(v);
    }
}

pub struct KvCache {
    /// One entry per *owning* layer, not per layer. Checkpoints that share KV
    /// across the tail of the stack allocate far fewer of these than there are
    /// blocks — on E4B, 24 caches for 42 layers.
    pub layers: Vec<LayerCache>,
    /// Layer index -> position in [`Self::layers`]. Shared layers point at the
    /// entry owned by an earlier layer.
    index: Vec<usize>,
    /// Number of positions written so far; the next token lands at this index.
    pub len: usize,
    pub capacity: usize,
}

impl KvCache {
    pub fn new(cfg: &Config, n_ctx: usize) -> Self {
        let mut layers = Vec::new();
        let mut index = vec![usize::MAX; cfg.n_layers];
        for i in 0..cfg.n_layers {
            if !cfg.layers[i].owns_kv(i) {
                continue;
            }
            let lc = &cfg.layers[i];
            let slots = cfg.kv_slots(i, n_ctx);
            let kv_dim = lc.kv_dim();
            index[i] = layers.len();
            layers.push(LayerCache {
                k: vec![0.0; slots * kv_dim],
                v: vec![0.0; slots * kv_dim],
                slots,
                kv_dim,
                window: lc.window.map(|w| w.min(n_ctx)),
            });
        }
        // Second pass: point the sharing layers at their source, which is
        // always below them and therefore already placed.
        for i in 0..cfg.n_layers {
            if index[i] == usize::MAX {
                index[i] = index[cfg.layers[i].kv_source];
                debug_assert_ne!(index[i], usize::MAX, "layer {i} shares a layer with no cache");
            }
        }
        Self {
            layers,
            index,
            len: 0,
            capacity: n_ctx,
        }
    }

    /// The cache layer `il` reads from — its own, or an earlier layer's.
    #[inline]
    pub fn layer(&self, il: usize) -> &LayerCache {
        &self.layers[self.index[il]]
    }

    #[inline]
    pub fn layer_mut(&mut self, il: usize) -> &mut LayerCache {
        &mut self.layers[self.index[il]]
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Overwrite the cache, not just rewind it.
    ///
    /// [`Self::clear`] resets the write position and leaves the keys and values
    /// of the previous conversation in place — correct for inference, since
    /// nothing reads past `len`, but they are still there to be read by anything
    /// else. The wipe is unconditional across the full allocation: a shorter
    /// follow-up conversation must not leave the tail of a longer one exposed.
    pub fn wipe(&mut self) {
        self.len = 0;
        for l in &mut self.layers {
            l.wipe();
        }
    }

    /// Drop everything from `pos` onward, e.g. when a prefix is reused.
    pub fn truncate(&mut self, pos: usize) {
        self.len = self.len.min(pos);
    }

    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.k.len() + l.v.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(window: Option<usize>, slots: usize) -> LayerCache {
        LayerCache {
            k: vec![0.0; slots],
            v: vec![0.0; slots],
            slots,
            kv_dim: 1,
            window,
        }
    }

    #[test]
    fn ring_wraps_and_window_slides() {
        let c = ring(Some(4), 4);
        assert_eq!(c.slot(0), 0);
        assert_eq!(c.slot(5), 1);
        // At position 5 with window 4 we see 2..=5 — exactly 4 positions.
        assert_eq!(c.visible(5), 2..=5);
        assert_eq!(c.visible(5).count(), 4);
    }

    #[test]
    fn window_clamps_at_sequence_start() {
        let c = ring(Some(4), 4);
        assert_eq!(c.visible(1), 0..=1);
    }

    #[test]
    fn global_layer_sees_everything() {
        let c = ring(None, 16);
        assert_eq!(c.visible(9), 0..=9);
        assert_eq!(c.slot(9), 9);
    }

    #[test]
    fn windowed_positions_never_collide_within_view() {
        // Any two distinct positions visible at once must map to distinct slots,
        // otherwise the ring would overwrite live entries.
        let c = ring(Some(4), 4);
        let pos = 11;
        let slots: Vec<_> = c.visible(pos).map(|p| c.slot(p)).collect();
        let mut uniq = slots.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(slots.len(), uniq.len(), "slot collision in {slots:?}");
    }
}
