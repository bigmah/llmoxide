pub mod cache;
pub mod config;
pub mod cpu;
pub mod ops;
pub mod qwen35;
pub mod sample;
pub mod weights;

pub use cache::KvCache;
pub use config::{Config, LayerConfig};
pub use cpu::Cpu;
pub use sample::{Sampler, Sampling};
pub use weights::Weights;

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
