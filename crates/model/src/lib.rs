pub mod cache;
pub mod config;
pub mod cpu;
pub mod ops;
pub mod qwen35;
pub mod sample;
pub mod vision;
pub mod weights;

pub use cache::KvCache;
pub use config::{Config, LayerConfig};
pub use cpu::Cpu;
pub use sample::{Sampler, Sampling};
pub use weights::Weights;

/// Overwrite a buffer that held conversation-derived state.
///
/// With the `private` feature this is `llmoxide-secret`'s `memset_s` plus a
/// compiler fence, which the optimizer is not permitted to elide. Without it,
/// an ordinary `fill` — which in practice zeroes too, but carries no such
/// guarantee, and does not drag a crate full of `mlock`/`ptrace` declarations
/// into a build that only wants inference.
///
/// The distinction only affects [`cache::KvCache::wipe`] and
/// [`qwen35::State::wipe`]. Neither `clear` nor the correctness of prefix
/// reuse depends on it: those reset a length, and this overwrites what the
/// length used to cover.
#[inline]
pub fn zero_slice<T: Copy + Default>(s: &mut [T]) {
    #[cfg(feature = "private")]
    secret::zero_slice(s);
    #[cfg(not(feature = "private"))]
    s.fill(T::default());
}

/// Which architecture a GGUF file carries, for dispatch at the entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Gemma4,
    /// Both the hybrid 27B and plain dense Qwen3, which
    /// [`qwen35::Config`] describes as the same stack with the delta-net
    /// layers and the query gate switched off.
    Qwen35,
}

impl Arch {
    pub fn detect(g: &gguf::Gguf) -> anyhow::Result<Self> {
        match g.str("general.architecture")? {
            "gemma4" => Ok(Self::Gemma4),
            "qwen35" | "qwen3" => Ok(Self::Qwen35),
            other => anyhow::bail!("unsupported architecture {other:?}"),
        }
    }
}
