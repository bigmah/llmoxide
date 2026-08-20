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
    Qwen35,
}

impl Arch {
    pub fn detect(g: &gguf::Gguf) -> anyhow::Result<Self> {
        match g.str("general.architecture")? {
            "gemma4" => Ok(Self::Gemma4),
            "qwen35" => Ok(Self::Qwen35),
            other => anyhow::bail!("unsupported architecture {other:?}"),
        }
    }
}
