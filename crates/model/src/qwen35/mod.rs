//! qwen35 (Qwen3.5): hybrid gated-delta-net + full-attention stack.
//!
//! Everything architecture-specific lives here; the k-quant decoders, scalar
//! kernels, and sampling in the parent crate are shared with gemma4.

pub mod config;
pub mod cpu;
pub mod state;
pub mod weights;

pub use config::Config;
pub use cpu::Cpu;
pub use state::State;
pub use weights::Weights;
