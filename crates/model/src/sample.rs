//! Token sampling.
//!
//! Defaults come from the checkpoint's own `general.sampling.*` metadata
//! (top-k 64, top-p 0.95, temperature 1.0) rather than from convention.

#[derive(Debug, Clone)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Penalty applied to tokens in the recent window; 1.0 disables it.
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            repeat_penalty: 1.0,
            repeat_window: 64,
            seed: 0,
        }
    }
}

impl Sampling {
    pub fn from_gguf(g: &gguf::Gguf) -> Self {
        let d = Self::default();
        Self {
            temperature: g.f32("general.sampling.temp").unwrap_or(d.temperature),
            top_k: g
                .u64("general.sampling.top_k")
                .map(|v| v as usize)
                .unwrap_or(d.top_k),
            top_p: g.f32("general.sampling.top_p").unwrap_or(d.top_p),
            ..d
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// xorshift64*, so a run is reproducible from its seed without pulling in a
/// random-number crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Any nonzero state works; 0 would be a fixed point.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        // Top 24 bits give a uniform value in [0, 1).
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u32 << 24) as f32
    }
}

pub struct Sampler {
    cfg: Sampling,
    rng: Rng,
    /// Scratch, reused across steps to keep sampling allocation-free.
    buf: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(cfg: Sampling) -> Self {
        Self {
            rng: Rng::new(cfg.seed),
            cfg,
            buf: Vec::new(),
        }
    }

    pub fn config(&self) -> &Sampling {
        &self.cfg
    }

    /// Pick the next token. `recent` is the tail of the generated sequence,
    /// used only for the repetition penalty.
    pub fn sample(&mut self, logits: &mut [f32], recent: &[u32]) -> u32 {
        if self.cfg.repeat_penalty != 1.0 {
            let start = recent.len().saturating_sub(self.cfg.repeat_window);
            for &t in &recent[start..] {
                if let Some(l) = logits.get_mut(t as usize) {
                    // Divide positives, multiply negatives, so the penalty
                    // always moves a logit toward less likely.
                    *l = if *l > 0.0 {
                        *l / self.cfg.repeat_penalty
                    } else {
                        *l * self.cfg.repeat_penalty
                    };
                }
            }
        }

        if self.cfg.is_greedy() {
            return argmax(logits);
        }

        // Top-k first: a full sort of 262144 logits per token would dominate
        // the sampling cost.
        let k = self.cfg.top_k.clamp(1, logits.len());
        self.buf.clear();
        self.buf
            .extend(logits.iter().copied().enumerate().filter_map(|(i, l)| {
                l.is_finite().then_some((i as u32, l))
            }));
        if self.buf.is_empty() {
            return argmax(logits);
        }
        let k = k.min(self.buf.len());
        self.buf.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        self.buf.truncate(k);
        self.buf.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let max = self.buf[0].1;
        let inv_t = 1.0 / self.cfg.temperature;
        let mut sum = 0.0;
        for (_, l) in self.buf.iter_mut() {
            *l = ((*l - max) * inv_t).exp();
            sum += *l;
        }

        // Nucleus: keep the shortest prefix whose mass reaches top_p.
        let mut cutoff = self.buf.len();
        if self.cfg.top_p < 1.0 {
            let target = self.cfg.top_p * sum;
            let mut acc = 0.0;
            for (i, (_, p)) in self.buf.iter().enumerate() {
                acc += *p;
                if acc >= target {
                    cutoff = i + 1;
                    break;
                }
            }
            sum = self.buf[..cutoff].iter().map(|(_, p)| *p).sum();
        }

        let mut r = self.rng.next_f32() * sum;
        for (id, p) in &self.buf[..cutoff] {
            r -= *p;
            if r <= 0.0 {
                return *id;
            }
        }
        self.buf[cutoff - 1].0
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_max() {
        let mut s = Sampler::new(Sampling {
            temperature: 0.0,
            ..Default::default()
        });
        let mut l = vec![0.1, 5.0, 2.0];
        assert_eq!(s.sample(&mut l, &[]), 1);
    }

    #[test]
    fn top_k_one_is_deterministic() {
        let mut s = Sampler::new(Sampling {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            ..Default::default()
        });
        let mut l = vec![0.1, 5.0, 2.0];
        for _ in 0..10 {
            assert_eq!(s.sample(&mut l.clone(), &[]), 1);
        }
    }

    #[test]
    fn never_returns_a_suppressed_token() {
        // -inf logits must be unreachable regardless of temperature.
        let mut s = Sampler::new(Sampling {
            temperature: 2.0,
            ..Default::default()
        });
        let mut l = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        for _ in 0..50 {
            assert_eq!(s.sample(&mut l.clone(), &[]), 1);
        }
    }

    #[test]
    fn repeat_penalty_demotes_recent_tokens() {
        let mut s = Sampler::new(Sampling {
            temperature: 0.0,
            repeat_penalty: 10.0,
            ..Default::default()
        });
        let mut l = vec![1.0, 1.2, 0.9];
        // Without the penalty token 1 wins; penalizing it hands over to 0.
        assert_eq!(s.sample(&mut l, &[1]), 0);
    }

    #[test]
    fn sampling_stays_in_the_top_k_set() {
        let mut s = Sampler::new(Sampling {
            temperature: 5.0,
            top_k: 2,
            top_p: 1.0,
            ..Default::default()
        });
        let l = vec![10.0, 9.0, 1.0, 0.5];
        for _ in 0..200 {
            assert!(s.sample(&mut l.clone(), &[]) < 2);
        }
    }

    #[test]
    fn rng_is_uniform_enough_and_in_range() {
        let mut r = Rng::new(42);
        let mut sum = 0.0;
        for _ in 0..10_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
            sum += v;
        }
        let mean = sum / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
    }
}
