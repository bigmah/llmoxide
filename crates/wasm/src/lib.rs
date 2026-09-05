//! gemma4 E4B in a browser tab.
//!
//! The same loader, tokenizer, chat format and WGSL kernels the native build
//! uses, compiled to wasm32 and pointed at WebGPU instead of Metal. The shaders
//! port for free — WebGPU consumes WGSL directly — so what this crate adds is
//! everything *around* them that assumed a filesystem and a thread that may
//! block.
//!
//! Two constraints shape it, and both come from the size of the checkpoint:
//!
//! * **wasm32 addresses 4 GB and E4B Q4_K_M is 5.3.** So the weights never
//!   enter linear memory as a whole. [`gguf::Header`] is parsed from a prefix
//!   of the file, and [`gpu::Weights::upload_streaming`] then walks the tensor
//!   table pulling one chunk at a time out of the JS `File` and writing it
//!   straight to a GPU buffer. Peak host usage is one 32 MB chunk. A CPU
//!   forward pass is not merely slow here, it is impossible — there is nowhere
//!   to put the weights.
//! * **The main thread may not block.** Every logit readback is a `map_async`
//!   whose completion the browser delivers on its own event loop, so
//!   [`gpu::forward::GpuModel::forward_async`] is the only usable entry point.
//!   That it suspends is also what keeps the page responsive and lets tokens
//!   paint as they arrive.
//!
//! What is *not* claimed: the private-mode guarantees do not survive the port.
//! The heap is still zeroed on free, but `mlock` has no meaning in a sandbox
//! whose memory the host may move or snapshot at will — see `secret::sys`.

use std::cell::Cell;
use std::collections::HashMap;

use chat::Message;
use model::sample::{Sampler, Sampling};
use tokenizer::Tokenizer;
use wasm_bindgen::prelude::*;

/// Overwrite every heap block as it is freed, once armed.
///
/// Worth installing for the same reason the native private mode does: prompt
/// text is a `String`, then a token `Vec`, then decoded pieces, and any of
/// those intermediates can be the copy that outlives the conversation. It
/// covers strictly less here than it does natively — see the module docs — but
/// what it covers, it covers.
#[global_allocator]
static ALLOC: secret::ZeroizingAlloc<std::alloc::System> =
    secret::ZeroizingAlloc(std::alloc::System);

thread_local! {
    /// Set by [`request_stop`], read between decode steps.
    ///
    /// A `thread_local` rather than a field on [`Session`] because it has to
    /// be writable *while* `reply` is running, and `reply` holds the session's
    /// only `&mut`. wasm-bindgen would reject a second method call on a
    /// borrowed object with "recursive use of an object", so the flag lives
    /// outside it.
    static STOP: Cell<bool> = const { Cell::new(false) };
}

/// Ask the running generation to stop after the current token.
#[wasm_bindgen]
pub fn request_stop() {
    STOP.with(|s| s.set(true));
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// Reading a JS File
// ---------------------------------------------------------------------------

/// Read `[start, end)` out of the checkpoint.
///
/// A [`web_sys::Blob`] rather than a `File` so the page decides where the
/// bytes come from — a file the user picked, or a download it has already
/// fetched and cached — without this crate knowing which. A `File` *is* a
/// `Blob`, so the picker still works unchanged.
///
/// `Blob::slice` **must** be the `f64` form. The `i32` overload silently
/// saturates past 2 GB — a third of the way into a 5 GB checkpoint — and the
/// tensors after that point would load as whatever happened to be at the
/// clamped offset, with no error anywhere.
async fn read_range(file: &web_sys::Blob, start: u64, end: u64) -> anyhow::Result<Vec<u8>> {
    let blob = file
        .slice_with_f64_and_f64(start as f64, end as f64)
        .map_err(|e| anyhow::anyhow!("slice {start}..{end} failed: {}", js_text(&e)))?;
    let buf = wasm_bindgen_futures::JsFuture::from(blob.array_buffer())
        .await
        .map_err(|e| anyhow::anyhow!("read {start}..{end} failed: {}", js_text(&e)))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

fn js_text(v: &JsValue) -> String {
    v.as_string()
        .or_else(|| js_sys::Reflect::get(v, &"message".into()).ok()?.as_string())
        .unwrap_or_else(|| format!("{v:?}"))
}

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Report progress to JS as `(stage, done, total)`. A callback that throws is
/// the page's problem, not ours; loading continues.
fn report(cb: &js_sys::Function, stage: &str, done: f64, total: f64) {
    let _ = cb.call3(
        &JsValue::NULL,
        &JsValue::from_str(stage),
        &JsValue::from_f64(done),
        &JsValue::from_f64(total),
    );
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Which model is loaded. Mirrors `server::engine::Backend`, minus the CPU
/// reference path — there is no memory for that here.
enum Backend {
    Gemma4(gpu::forward::GpuModel),
    /// The hybrid 27B and plain dense Qwen3 share this; see
    /// `model::qwen35::Config`.
    Qwen(gpu::qwen35::Qwen35Gpu),
}

impl Backend {
    fn max_batch(&self) -> usize {
        match self {
            Self::Gemma4(m) => m.max_batch(),
            Self::Qwen(m) => m.max_batch(),
        }
    }

    fn eog(&self) -> Vec<u32> {
        match self {
            Self::Gemma4(m) => m.config().eog.clone(),
            Self::Qwen(m) => m.config().eog.clone(),
        }
    }

    fn wipe(&mut self) {
        match self {
            Self::Gemma4(m) => m.wipe(),
            Self::Qwen(m) => m.wipe(),
        }
    }

    async fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Gemma4(m) => m.forward_async(tokens).await,
            Self::Qwen(m) => m.forward_async(tokens).await,
        }
    }

    fn summary(&self) -> String {
        match self {
            Self::Gemma4(m) => {
                let c = m.config();
                format!(
                    "gemma4 · {} layers, d_model {}, vocab {}",
                    c.n_layers, c.d_model, c.vocab
                )
            }
            Self::Qwen(m) => m.config().summary(),
        }
    }
}

/// Each architecture speaks its own chat dialect.
enum Format {
    Gemma4(chat::Special),
    Qwen(chat::qwen::Special),
}

/// A loaded model and the conversation in front of it.
#[wasm_bindgen]
pub struct Session {
    backend: Backend,
    tok: Tokenizer,
    format: Format,
    sampling: Sampling,
    /// The exact token sequence resident in the KV cache, kept between turns so
    /// a conversation that only appends re-prefills nothing. Locked and zeroed
    /// on the same terms as the native engine — see [`secret::SecretVec`].
    cached: secret::SecretVec<u32>,
    history: Vec<Message>,
    n_ctx: usize,
    adapter: String,
    weight_bytes: u64,
}

/// The parsed config, held between architecture detection and model
/// construction (the weight upload happens in between).
enum Build {
    Gemma4(model::Config),
    Qwen(model::qwen35::Config),
}

/// Largest resident non-arena payload accepted, as a guard rather than a limit.
///
/// The F32 norms of an E4B come to a few megabytes. A checkpoint that wants
/// hundreds is one whose embedding or output projection was left dense, which
/// will not fit here — better to say so at load than to die inside an
/// allocation.
const MAX_RESIDENT_BYTES: u64 = 512 << 20;

/// Load a checkpoint the user picked with a file input.
///
/// `on_progress(stage, done, total)` is called throughout; `done`/`total` are
/// bytes during the upload and zero elsewhere.
#[wasm_bindgen]
pub async fn load(
    file: web_sys::Blob,
    n_ctx: usize,
    max_batch: usize,
    on_progress: js_sys::Function,
) -> Result<Session, JsValue> {
    let size = file.size() as u64;
    report(&on_progress, "reading header", 0.0, 0.0);

    // The tensor table sits behind the metadata, and gemma4 carries a 262 144
    // entry tokenizer vocabulary there — several megabytes before the first
    // tensor is even described. A short prefix and a corrupt file are
    // indistinguishable from inside the parser, so grow rather than guess.
    let mut want: u64 = 16 << 20;
    let header = loop {
        let n = want.min(size);
        let prefix = read_range(&file, 0, n).await.map_err(js_err)?;
        match gguf::Header::parse(&prefix) {
            Ok(h) => break h,
            Err(gguf::Error::Eof(_)) if n < size => want *= 2,
            Err(e) => return Err(js_err(e)),
        }
    };
    // Catches a truncated file before any of it reaches the GPU.
    header.check_bounds(size).map_err(js_err)?;

    // Split the tensor table in two. The quantized weights are streamed
    // straight to the GPU and never made resident; everything else — the F32
    // norms, the layer output scales, the rope frequencies — is small, is read
    // by name during model construction, and is loaded now.
    //
    // The rule is deliberately "not quantized" rather than either
    // architecture's arena list, because those lists need a `Config` and a
    // `Config` needs a `Gguf` — this breaks that circle. It is a superset of
    // what either needs: anything quantized is uploaded, never read here.
    let small: Vec<&gguf::TensorInfo> = header
        .tensors
        .iter()
        .filter(|t| !t.ty.is_quantized() && t.ty != gguf::GgmlType::BF16)
        .collect();
    let resident_bytes: u64 = small.iter().map(|t| t.byte_len() as u64).sum();
    if resident_bytes > MAX_RESIDENT_BYTES {
        return Err(js_err(format!(
            "this checkpoint keeps {resident_bytes} bytes of unquantized tensors, \
             more than the {MAX_RESIDENT_BYTES}-byte budget wasm32 leaves for them"
        )));
    }

    report(&on_progress, "reading parameters", 0.0, 0.0);
    let mut resident: HashMap<String, Vec<u8>> = HashMap::with_capacity(small.len());
    for t in &small {
        let r = header.byte_range(t);
        resident.insert(
            t.name.clone(),
            read_range(&file, r.start, r.end).await.map_err(js_err)?,
        );
    }
    let g = gguf::Gguf::sparse(header, resident).map_err(js_err)?;

    let tok = Tokenizer::from_gguf(&g).map_err(js_err)?;
    let sampling = Sampling::from_gguf(&g);
    let arch = model::Arch::detect(&g).map_err(js_err)?;

    // Config and chat dialect are per-architecture; so is the arena list, which
    // is why it is computed here rather than above.
    let (format, names, build): (Format, Vec<String>, Build) = match arch {
        model::Arch::Gemma4 => {
            let cfg = model::Config::from_gguf(&g).map_err(js_err)?;
            let sp = chat::Special::new(&tok, g.str("tokenizer.chat_template").ok())
                .map_err(js_err)?;
            (
                Format::Gemma4(sp),
                gpu::forward::GpuModel::arena_tensors(g.header()),
                Build::Gemma4(cfg),
            )
        }
        model::Arch::Qwen35 => {
            let cfg = model::qwen35::Config::from_gguf(&g).map_err(js_err)?;
            let sp = chat::qwen::Special::new(&tok).map_err(js_err)?;
            let names = gpu::qwen35::Qwen35Gpu::arena_tensors(g.header(), &cfg);
            (Format::Qwen(sp), names, Build::Qwen(cfg))
        }
    };

    report(&on_progress, "requesting adapter", 0.0, 0.0);
    let device = gpu::Gpu::new().await.map_err(js_err)?;
    let adapter = device.adapter_name.clone();

    // Before the weights move, not after: see `Gpu::check_shaders`.
    device.check_shaders().await.map_err(js_err)?;

    // Each call gets its own clone of the File handle so the returned futures
    // own what they read from, which keeps `upload_streaming` free of any
    // lifetime tie to this scope.
    let src = file.clone();
    let progress = on_progress.clone();
    let weights = gpu::Weights::upload_streaming(
        &device,
        g.header(),
        names,
        move |a, b| {
            let f = src.clone();
            async move { read_range(&f, a, b).await }
        },
        |done, total| report(&progress, "uploading weights", done as f64, total as f64),
    )
    .await
    .map_err(js_err)?;
    let weight_bytes = weights.bytes;

    report(&on_progress, "building pipelines", 0.0, 0.0);
    let backend = match build {
        Build::Gemma4(cfg) => Backend::Gemma4(
            gpu::forward::GpuModel::load_with_weights(device, &g, cfg, weights, n_ctx, max_batch)
                .map_err(js_err)?,
        ),
        Build::Qwen(cfg) => Backend::Qwen(
            gpu::qwen35::Qwen35Gpu::load_with_weights(device, &g, cfg, weights, n_ctx, max_batch)
                .map_err(js_err)?,
        ),
    };

    // Arm only now. Loading moves gigabytes through the heap and zeroing those
    // on the way out is pure cost — they are not secret, and nothing has been
    // typed yet.
    drop(g);
    secret::arm();

    report(&on_progress, "ready", 0.0, 0.0);
    Ok(Session {
        backend,
        tok,
        format,
        sampling,
        cached: secret::SecretVec::new(),
        history: Vec::new(),
        n_ctx,
        adapter,
        weight_bytes,
    })
}

#[wasm_bindgen]
impl Session {
    /// One line about what is loaded, for the banner.
    #[wasm_bindgen(getter)]
    pub fn info(&self) -> String {
        format!(
            "{} · {} · {:.2} GB of weights resident · context {}",
            self.adapter,
            self.backend.summary(),
            self.weight_bytes as f64 / 1e9,
            self.n_ctx,
        )
    }

    #[wasm_bindgen(getter)]
    pub fn turns(&self) -> usize {
        self.history.len()
    }

    /// Forget the conversation: overwrite the resident token ids, every device
    /// buffer derived from them, and the message history.
    pub fn wipe(&mut self) {
        self.backend.wipe();
        self.cached.wipe();
        // `clear` alone would leave the plaintext in the freed blocks; the
        // armed allocator overwrites them as the `String`s drop.
        self.history.clear();
        self.history.shrink_to_fit();
    }

    /// Answer `text`, streaming each decoded piece to `on_token(text)`.
    ///
    /// Returns the reply with any thought channel already stripped. The await
    /// inside every forward pass is what keeps the page responsive: each token
    /// yields to the browser's event loop before the next one starts.
    pub async fn reply(
        &mut self,
        text: String,
        max_tokens: usize,
        on_token: js_sys::Function,
    ) -> Result<String, JsValue> {
        STOP.with(|s| s.set(false));
        self.history.push(Message {
            role: "user".into(),
            content: Some(serde_json::Value::String(text)),
            ..Default::default()
        });
        match self.run(max_tokens, &on_token).await {
            Ok(reply) => {
                self.history.push(Message {
                    role: "assistant".into(),
                    content: Some(serde_json::Value::String(reply.clone())),
                    ..Default::default()
                });
                Ok(reply)
            }
            Err(e) => {
                // Do not leave a half-answered turn in the history; the next
                // prompt would replay it and diverge from the cache anyway.
                self.history.pop();
                Err(e)
            }
        }
    }
}

impl Session {
    /// How much of `prompt` the cache can serve.
    ///
    /// Only a strict prefix of what is already resident counts. Rewinding is
    /// not an option: the sliding-window layers keep a ring of 1024 slots, so
    /// positions before a rewind point have already been overwritten.
    fn reusable_prefix(&self, prompt: &[u32]) -> usize {
        let n = self
            .cached
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        if n == self.cached.len() && n < prompt.len() {
            n
        } else {
            0
        }
    }

    async fn run(
        &mut self,
        max_tokens: usize,
        on_token: &js_sys::Function,
    ) -> Result<String, JsValue> {
        // Thinking off. For gemma4 that is `Special::closes_empty_thought`,
        // read off the checkpoint rather than assumed; for Qwen it means the
        // generation prompt closes the think block immediately.
        let prompt = match &self.format {
            Format::Gemma4(sp) => chat::build_prompt(&self.tok, sp, &self.history, &[], false),
            Format::Qwen(sp) => chat::qwen::build_prompt(&self.tok, sp, &self.history, &[], false),
        };
        if prompt.len() + 8 >= self.n_ctx {
            return Err(js_err(format!(
                "prompt is {} tokens but the context is {}",
                prompt.len(),
                self.n_ctx
            )));
        }

        let reuse = self.reusable_prefix(&prompt);
        if reuse == 0 {
            // Nothing of the resident conversation is reachable from here, so
            // overwrite it now rather than leaving it to be shadowed.
            self.backend.wipe();
            self.cached.wipe();
        }

        let batch = self.backend.max_batch();
        let mut logits = Vec::new();
        let mut i = reuse;
        while i < prompt.len() {
            let end = (i + batch).min(prompt.len());
            logits = self
                .backend
                .forward(&prompt[i..end])
                .await
                .map_err(js_err)?;
            i = end;
        }
        self.cached.replace(&prompt);

        let eog = self.backend.eog();
        let max_new = max_tokens.min(self.n_ctx.saturating_sub(self.cached.len()).saturating_sub(1));
        let mut sampler = Sampler::new(self.sampling.clone());
        let mut generated: Vec<u32> = Vec::new();
        let mut decoder = tokenizer::Decoder::new(&self.tok);

        for _ in 0..max_new {
            if STOP.with(Cell::get) {
                break;
            }
            let next = sampler.sample(&mut logits, &generated);
            generated.push(next);
            if eog.contains(&next) {
                break;
            }

            let piece = decoder.push(next);
            if !piece.is_empty() {
                let _ = on_token.call1(&JsValue::NULL, &JsValue::from_str(&piece));
            }

            self.cached.push(next);
            logits = self.backend.forward(&[next]).await.map_err(js_err)?;
        }

        let completion = match &self.format {
            Format::Gemma4(sp) => chat::parse_completion(&self.tok, sp, &generated),
            // With thinking off the generation prompt has already closed the
            // think block, so the stream starts in content.
            Format::Qwen(sp) => chat::qwen::parse_completion(&self.tok, sp, &generated, false),
        };
        Ok(completion.content)
    }
}
