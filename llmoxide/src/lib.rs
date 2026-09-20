//! Local LLM inference, from the GGUF file down to the compute kernels, with
//! no ML dependencies.
//!
//! This is the crate to depend on. It re-exports every other crate in the
//! workspace, so a consumer adds one dependency and gets versions that are
//! guaranteed to agree:
//!
//! ```no_run
//! # fn main() -> llmoxide::Result<()> {
//! use llmoxide::{LoadOptions, Request, Session};
//!
//! let mut s = Session::load("models/Qwen3-0.6B-Q8_0.gguf", &LoadOptions::default())?;
//! let out = s.complete(Request::user("name three prime numbers"))?;
//! println!("{}", out.completion.content);
//! # Ok(()) }
//! ```
//!
//! Streaming is the same call with a callback:
//!
//! ```no_run
//! # fn main() -> llmoxide::Result<()> {
//! # use llmoxide::{Flow, LoadOptions, Request, Session};
//! # let mut s = Session::load("m.gguf", &LoadOptions::default())?;
//! s.generate(Request::user("hello"), |piece| {
//!     print!("{piece}");
//!     Flow::Continue
//! })?;
//! # Ok(()) }
//! ```
//!
//! # Layers
//!
//! Reach past [`Session`] when you want less of it:
//!
//! * [`backend`] — [`Backend`], one trait over gemma4 and qwen35 on CPU or
//!   GPU. Token ids in, logits out, no chat template and no sampling. This is
//!   also where a new architecture is added, and where the multimodal seam is
//!   ([`Backend::forward_embeds`]).
//! * [`session`] — [`Session`], which adds the tokenizer, the chat dialect,
//!   sampling, prefix reuse, and tool-call handling.
//!
//! # Features
//!
//! * `gpu` *(default)* — the wgpu backends. Without it the crate still reads
//!   checkpoints and runs the CPU reference paths, and does not build wgpu.
//! * `hub` — resumable, hash-verified checkpoint downloads.
//! * `private` — keep the resident conversation in locked, self-zeroing
//!   memory. Off by default: it is what `llmoxide-private` is built on, not
//!   something an embedding caller needs, and it writes to stderr whenever
//!   `mlock` is refused. [`PRIVATE_MEMORY`] reports whether it is on. See
//!   [`store`] for what does and does not change.
//!
//! # What this crate does not do
//!
//! One [`Session`] is one conversation on one GPU context: there is no
//! batching and no sharing a loaded model across threads. Serving several
//! callers means a queue in front of one session — see `llmoxide-server`.

pub mod backend;
mod error;
#[cfg(feature = "vision")]
pub mod image_input;
pub mod session;
pub mod store;

pub use backend::{Backend, Device, DevicePref, Info, LoadOptions};
pub use error::{Error, Result};
pub use session::{
    ChatFormat, Event, FinishReason, Flow, Outcome, Prompt, Request, Segment, Session, Sink,
};

// The workspace, re-exported. Consumers reach types like `Message` and
// `Sampling` through these rather than adding `llmoxide-chat` and
// `llmoxide-model` themselves and risking a version skew.
pub use chat;
#[cfg(feature = "vision")]
pub use vision;

pub use gguf;
pub use model;
pub use tokenizer;

#[cfg(feature = "gpu")]
pub use gpu;
#[cfg(feature = "hub")]
pub use hub;
#[cfg(feature = "private")]
pub use secret;

/// Whether this build holds the resident conversation in locked, self-zeroing
/// memory — that is, whether the `private` feature is on.
///
/// A caller that depends on the guarantee should assert it rather than assume
/// it, since a feature is easy to leave off by accident and nothing else about
/// the build changes visibly:
///
/// ```
/// # fn main() {
/// assert!(llmoxide::PRIVATE_MEMORY || !llmoxide::PRIVATE_MEMORY);
/// # }
/// ```
pub const PRIVATE_MEMORY: bool = cfg!(feature = "private");

/// The most-used types, for `use llmoxide::prelude::*;`.
pub mod prelude {
    pub use crate::backend::{Backend, DevicePref, LoadOptions};
    pub use crate::session::{Flow, Outcome, Request, Session};
    pub use chat::{Message, Tool};
    pub use model::sample::Sampling;
    pub use model::Arch;
}
