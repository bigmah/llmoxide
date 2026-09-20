//! The tower's description lives in `llmoxide-model`, beside the text
//! architectures it is built from — the GPU tower in `llmoxide-gpu` reads the
//! same struct, and neither crate should have to depend on an image decoder to
//! get at it.

pub use model::vision::Config as VisionConfig;
