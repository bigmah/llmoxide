//! One error type for the whole surface.
//!
//! The crates underneath return `anyhow::Result`, which is the right choice
//! for a binary and the wrong one for a library: a caller cannot match on it.
//! Everything a consumer is plausibly going to branch on gets its own variant
//! here; the rest arrives as [`Error::Other`] with the original chain intact.

/// Errors from loading a checkpoint or running a generation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The GGUF declares an architecture this build does not implement.
    #[error("unsupported architecture {0:?}")]
    UnsupportedArch(String),

    /// The loaded backend has no implementation for this input or operation —
    /// notably [`Backend::forward_embeds`], which only gemma4 implements.
    /// See [`crate::backend`].
    ///
    /// [`Backend::forward_embeds`]: crate::backend::Backend::forward_embeds
    #[error("{0} is not supported by this backend")]
    Unsupported(&'static str),

    /// The prompt does not fit, and the cache cannot rewind to make room.
    #[error("prompt is {tokens} tokens but the context is {context}")]
    ContextOverflow { tokens: usize, context: usize },

    /// A GPU backend was requested from a build compiled without the `gpu`
    /// feature.
    #[error("this build has no GPU backend: enable the `gpu` feature of `llmoxide`")]
    NoGpuBackend,

    /// No GPU adapter, or the checkpoint does not fit on the one present.
    #[error("no usable GPU: {0}")]
    NoDevice(String),

    /// An image could not be read, decoded, or encoded.
    #[error("image input: {0}")]
    Image(String),

    #[error(transparent)]
    Gguf(#[from] gguf::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
