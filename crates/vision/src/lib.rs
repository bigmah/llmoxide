//! Gemma 4's vision tower: an image in, residual-stream rows out.
//!
//! The rows this produces are what [`llmoxide::Segment::Embeds`] carries —
//! already in the text model's embedding space, occupying positions in the
//! cache exactly as tokens do. Nothing downstream of the prefill loop has to
//! know which an position came from.
//!
//! The encoder lives in a *separate* GGUF (`mmproj-…`) from the text model.
//! That is llama.cpp's convention and worth keeping: the text checkpoint is
//! useful on its own, and the tower is a fifth of a gigabyte that a text-only
//! run should not pay for.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let v = llmoxide_vision::Vision::open("models/mmproj-gemma-4-E4B-it-BF16.gguf")?;
//! let img = llmoxide_vision::prepare(&std::fs::read("photo.jpg")?, &v.cfg)?;
//! let embeds = v.encode(&img)?;
//! println!("{} rows of {}", embeds.n, embeds.proj_dim);
//! # Ok(()) }
//! ```

pub mod config;
pub mod cpu;
pub mod preprocess;

pub use config::VisionConfig;
pub use cpu::{ImageEmbeds, Vision};
pub use preprocess::{prepare, prepare_rgb8, smart_resize, Planar};
