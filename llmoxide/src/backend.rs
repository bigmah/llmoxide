//! The execution seam: one trait over every architecture and device.
//!
//! Before this existed, each entry point — the CLI, the server, the browser
//! build — wrote its own `enum Backend` and its own match arms to dispatch
//! `forward` across gemma4 and qwen35, CPU and GPU. They have all been folded
//! into [`Backend`], so adding an architecture is one more impl here instead
//! of a new arm in three places.

use gguf::Gguf;
use model::Arch;

use crate::{Error, Result};

/// Where a loaded model runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// The validated reference path. Correct, and roughly two orders of
    /// magnitude slower than the GPU — it exists to check the GPU against.
    Cpu,
    Gpu {
        adapter: String,
    },
}

/// Which device [`load`] should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePref {
    #[default]
    Gpu,
    Cpu,
}

/// What a loaded model is, without reaching for the architecture-specific
/// config type behind it.
#[derive(Debug, Clone)]
pub struct Info {
    pub arch: Arch,
    pub device: Device,
    /// The cap this model was loaded with, not the checkpoint's maximum.
    pub context_len: usize,
    /// Largest number of tokens one `forward` call may carry.
    pub max_batch: usize,
    /// Residual stream width. An encoder producing rows for
    /// [`Backend::forward_embeds`] must match it.
    pub d_model: usize,
    pub vocab: usize,
    /// Token ids that end a generation.
    pub eog: Vec<u32>,
}

/// A loaded model that can run forward passes.
///
/// One implementation per (architecture, device) pair. The trait is
/// object-safe: [`load`] hands back a `Box<dyn Backend>` and callers never
/// name the concrete type.
///
/// `Send` but deliberately not `Sync`: a backend owns a wgpu queue and a
/// single KV cache, so it is moved onto one thread and stays there — which is
/// exactly what `llmoxide-server` does. (Were the browser build ever to come
/// through this trait, the bound would have to become conditional: wgpu's
/// types are not `Send` on wasm32.)
pub trait Backend: Send {
    /// Run `tokens` and return the logits for the last position.
    ///
    /// Appends to whatever state the model already holds — a KV cache for the
    /// attention layers, a recurrent state for qwen35's delta-net ones. It is
    /// never a fresh pass over the whole conversation; [`Backend::wipe`] is
    /// the only way back to an empty state.
    ///
    /// `tokens.len()` must not exceed [`Info::max_batch`].
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Run precomputed embedding rows in place of token ids — the seam a
    /// vision or audio encoder enters through.
    ///
    /// `rows` is row-major, `n * d_model` floats, already in the residual
    /// stream's space: what gemma4's `mmproj` encoder emits for an image, or
    /// what any other projector would. They occupy positions in the cache
    /// exactly as tokens do, so a multimodal prompt is an interleaving of
    /// [`Segment::Tokens`] and [`Segment::Embeds`] and nothing downstream of
    /// the prefill loop has to care which it was.
    ///
    /// [`Segment::Tokens`]: crate::session::Segment::Tokens
    /// [`Segment::Embeds`]: crate::session::Segment::Embeds
    ///
    /// gemma4 implements this; qwen35 does not, and its default returns
    /// [`Error::Unsupported`].
    ///
    /// Implementations must treat the rows as a span that attends to itself
    /// in **both** directions — an image is not a causal sequence — and must
    /// not apply the input scale that token embeddings get.
    fn forward_embeds(&mut self, _rows: &[f32]) -> Result<Vec<f32>> {
        Err(Error::Unsupported("embedding input"))
    }

    fn info(&self) -> &Info;

    /// Overwrite the resident conversation: the KV cache, the recurrent
    /// state, and every device buffer derived from them.
    ///
    /// This is not a reset-to-zero for tidiness. It is the operation private
    /// mode is built on, so implementations overwrite rather than merely
    /// dropping or rewinding.
    fn wipe(&mut self);
}

/// How the model is loaded.
///
/// `n_ctx` and `max_batch` are sized at load time because every scratch
/// buffer and the whole KV cache are allocated against them; neither can grow
/// afterwards.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub n_ctx: usize,
    /// Prefill chunk size. Larger is faster and costs scratch memory
    /// proportional to `max_batch * d_model`.
    ///
    /// It is also the ceiling on one image: an encoded image prefills as a
    /// single bidirectional batch, so `max_batch` must be at least as large
    /// as the tower's token budget for that image.
    pub max_batch: usize,
    pub device: DevicePref,
    /// The `mmproj-*.gguf` carrying the vision tower, when image input is
    /// wanted. The text checkpoint is complete without it.
    pub mmproj: Option<std::path::PathBuf>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            n_ctx: 16384,
            // 256 tokens is also the largest image the vision tower will
            // produce, so one fits in a batch by construction.
            max_batch: 256,
            device: DevicePref::Gpu,
            mmproj: None,
        }
    }
}

impl LoadOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_ctx(mut self, n: usize) -> Self {
        self.n_ctx = n;
        self
    }

    pub fn max_batch(mut self, n: usize) -> Self {
        self.max_batch = n;
        self
    }

    pub fn device(mut self, d: DevicePref) -> Self {
        self.device = d;
        self
    }

    /// Load a vision tower alongside the text model, enabling image input.
    pub fn mmproj(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.mmproj = Some(path.into());
        self
    }
}

/// Detect the architecture in `g` and load it onto the requested device.
///
/// Takes the [`Gguf`] by value: the GPU backends upload every weight and drop
/// the mapping, while the CPU ones borrow it for the life of the model (see
/// the note on [`DevicePref::Cpu`] below).
pub fn load(g: Gguf, opts: &LoadOptions) -> Result<Box<dyn Backend>> {
    let arch = Arch::detect(&g).map_err(|_| {
        Error::UnsupportedArch(
            g.str("general.architecture")
                .unwrap_or("<missing>")
                .to_string(),
        )
    })?;
    match (arch, opts.device) {
        #[cfg(feature = "gpu")]
        (Arch::Gemma4, DevicePref::Gpu) => gemma4_gpu(g, opts),
        #[cfg(feature = "gpu")]
        (Arch::Qwen35, DevicePref::Gpu) => qwen35_gpu(g, opts),
        #[cfg(not(feature = "gpu"))]
        (_, DevicePref::Gpu) => Err(Error::NoGpuBackend),
        (Arch::Gemma4, DevicePref::Cpu) => gemma4_cpu(g, opts),
        (Arch::Qwen35, DevicePref::Cpu) => qwen35_cpu(g, opts),
    }
}

// ---------------------------------------------------------------- GPU paths

#[cfg(feature = "gpu")]
fn device() -> Result<gpu::Gpu> {
    gpu::Gpu::blocking_new().map_err(|e| Error::NoDevice(e.to_string()))
}

#[cfg(feature = "gpu")]
struct Gemma4Gpu {
    inner: gpu::forward::GpuModel,
    info: Info,
}

#[cfg(feature = "gpu")]
fn gemma4_gpu(g: Gguf, opts: &LoadOptions) -> Result<Box<dyn Backend>> {
    let cfg = model::Config::from_gguf(&g)?;
    let dev = device()?;
    let adapter = dev.adapter_name.clone();
    tracing::info!(%adapter, "gpu");
    let (d_model, vocab, eog) = (cfg.d_model, cfg.vocab, cfg.eog.clone());
    let inner = gpu::forward::GpuModel::load(dev, &g, cfg, opts.n_ctx, opts.max_batch.max(1))?;
    let info = Info {
        arch: Arch::Gemma4,
        device: Device::Gpu { adapter },
        context_len: inner.context_len(),
        max_batch: inner.max_batch(),
        d_model,
        vocab,
        eog,
    };
    Ok(Box::new(Gemma4Gpu { inner, info }))
}

#[cfg(feature = "gpu")]
impl Backend for Gemma4Gpu {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.inner.forward(tokens)?)
    }
    fn forward_embeds(&mut self, rows: &[f32]) -> Result<Vec<f32>> {
        Ok(self.inner.forward_embeds(rows)?)
    }
    fn info(&self) -> &Info {
        &self.info
    }
    fn wipe(&mut self) {
        self.inner.wipe();
    }
}

#[cfg(feature = "gpu")]
struct Qwen35Gpu {
    inner: gpu::qwen35::Qwen35Gpu,
    info: Info,
}

#[cfg(feature = "gpu")]
fn qwen35_gpu(g: Gguf, opts: &LoadOptions) -> Result<Box<dyn Backend>> {
    let cfg = model::qwen35::Config::from_gguf(&g)?;
    tracing::info!("{}", cfg.summary());
    let dev = device()?;
    let adapter = dev.adapter_name.clone();
    tracing::info!(%adapter, "gpu");
    let (d_model, vocab, eog) = (cfg.d_model, cfg.vocab, cfg.eog.clone());
    let inner = gpu::qwen35::Qwen35Gpu::load(dev, &g, cfg, opts.n_ctx, opts.max_batch.max(1))?;
    let info = Info {
        arch: Arch::Qwen35,
        device: Device::Gpu { adapter },
        context_len: inner.context_len(),
        max_batch: inner.max_batch(),
        d_model,
        vocab,
        eog,
    };
    Ok(Box::new(Qwen35Gpu { inner, info }))
}

#[cfg(feature = "gpu")]
impl Backend for Qwen35Gpu {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.inner.forward(tokens)?)
    }
    fn info(&self) -> &Info {
        &self.info
    }
    fn wipe(&mut self) {
        self.inner.wipe();
    }
}

// ---------------------------------------------------------------- CPU paths
//
// Both reference paths keep the weights borrowed from the mmap'd file, so the
// `Gguf`, its config and its weights are leaked to `'static` rather than
// threaded through a self-referential struct. That caps a process at one CPU
// model for its lifetime, which is what these are for — validating the GPU
// against, not serving. The GPU backends above upload and drop, and have no
// such limit.

struct Gemma4Cpu {
    cpu: model::Cpu<'static>,
    cache: model::KvCache,
    info: Info,
}

fn gemma4_cpu(g: Gguf, opts: &LoadOptions) -> Result<Box<dyn Backend>> {
    // The CPU path re-reads every weight per token; fault the mapping in up
    // front rather than one page at a time.
    g.prefault();
    let g: &'static Gguf = Box::leak(Box::new(g));
    let cfg: &'static model::Config = Box::leak(Box::new(model::Config::from_gguf(g)?));
    let w: &'static model::Weights<'static> = Box::leak(Box::new(model::Weights::load(g, cfg)?));
    let info = Info {
        arch: Arch::Gemma4,
        device: Device::Cpu,
        context_len: opts.n_ctx,
        max_batch: opts.max_batch.max(1),
        d_model: cfg.d_model,
        vocab: cfg.vocab,
        eog: cfg.eog.clone(),
    };
    Ok(Box::new(Gemma4Cpu {
        cpu: model::Cpu::new(cfg, w, opts.max_batch.max(1)),
        cache: model::KvCache::new(cfg, opts.n_ctx),
        info,
    }))
}

impl Backend for Gemma4Cpu {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.cpu.forward(tokens, &mut self.cache))
    }
    fn forward_embeds(&mut self, rows: &[f32]) -> Result<Vec<f32>> {
        let d = self.info.d_model;
        if rows.is_empty() || rows.len() % d != 0 {
            return Err(Error::Unsupported("embedding rows are not a multiple of d_model"));
        }
        Ok(self.cpu.forward_embeds(rows, rows.len() / d, &mut self.cache))
    }
    fn info(&self) -> &Info {
        &self.info
    }
    fn wipe(&mut self) {
        self.cache.wipe();
    }
}

struct Qwen35Cpu {
    cpu: model::qwen35::Cpu<'static>,
    state: model::qwen35::State,
    info: Info,
}

fn qwen35_cpu(g: Gguf, opts: &LoadOptions) -> Result<Box<dyn Backend>> {
    g.prefault();
    let g: &'static Gguf = Box::leak(Box::new(g));
    let cfg: &'static model::qwen35::Config =
        Box::leak(Box::new(model::qwen35::Config::from_gguf(g)?));
    tracing::info!("{} (CPU reference path)", cfg.summary());
    let w: &'static model::qwen35::Weights<'static> =
        Box::leak(Box::new(model::qwen35::Weights::load(g, cfg)?));
    let info = Info {
        arch: Arch::Qwen35,
        device: Device::Cpu,
        context_len: opts.n_ctx,
        max_batch: opts.max_batch.max(1),
        d_model: cfg.d_model,
        vocab: cfg.vocab,
        eog: cfg.eog.clone(),
    };
    Ok(Box::new(Qwen35Cpu {
        cpu: model::qwen35::Cpu::new(cfg, w, opts.max_batch.max(1)),
        state: model::qwen35::State::new(cfg, opts.n_ctx),
        info,
    }))
}

impl Backend for Qwen35Cpu {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.cpu.forward(tokens, &mut self.state))
    }
    fn info(&self) -> &Info {
        &self.info
    }
    fn wipe(&mut self) {
        self.state.wipe();
    }
}
