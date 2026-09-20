//! A loaded model plus the conversation resident in its cache.
//!
//! This is the layer most consumers want: it owns the tokenizer, the chat
//! dialect for the architecture, and the token ids currently in the KV cache,
//! and it turns [`chat::Message`]s into generated text. It used to live in
//! `llmoxide-server`, which meant embedding inference pulled in axum and
//! tokio; nothing here is async or aware of HTTP.
//!
//! One [`Session`] is one conversation: the backends hold a single KV cache
//! and a single GPU context, so there is no sharing a loaded model across
//! concurrent requests. Put a queue in front of it, as the server does.

use std::sync::mpsc;

use chat::{Message, Tool};
use model::sample::{Sampler, Sampling};
use model::Arch;
use tokenizer::{Decoder, Tokenizer};

use crate::backend::{self, Backend, Info, LoadOptions};
use crate::store::{TextStore, TokenStore};
use crate::{Error, Result};

/// A run of prompt that occupies consecutive positions in the cache.
///
/// Text prompts are a single [`Segment::Tokens`]. The variant beside it is the
/// multimodal seam: see [`Backend::forward_embeds`].
#[derive(Debug, Clone)]
pub enum Segment {
    Tokens(Vec<u32>),
    /// `n` rows of `d_model` floats, row-major — an encoded image or audio
    /// span, already projected into the residual stream.
    Embeds { rows: Vec<f32>, n: usize },
}

impl Segment {
    /// How many positions this segment occupies.
    pub fn len(&self) -> usize {
        match self {
            Self::Tokens(t) => t.len(),
            Self::Embeds { n, .. } => *n,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A prompt as the backends consume it.
#[derive(Debug, Clone, Default)]
pub struct Prompt(pub Vec<Segment>);

impl Prompt {
    pub fn tokens(ids: Vec<u32>) -> Self {
        Self(vec![Segment::Tokens(ids)])
    }

    pub fn len(&self) -> usize {
        self.0.iter().map(Segment::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The leading run of token ids, which is all that prefix reuse can
    /// compare. Empty when the prompt opens with an embedding span.
    fn leading_tokens(&self) -> &[u32] {
        match self.0.first() {
            Some(Segment::Tokens(t)) => t,
            _ => &[],
        }
    }

    fn has_embeds(&self) -> bool {
        self.0.iter().any(|s| matches!(s, Segment::Embeds { .. }))
    }
}

/// What to generate.
#[derive(Debug, Clone)]
pub struct Request {
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    /// `None` uses the checkpoint's own recommended settings, read from its
    /// GGUF metadata at load time.
    pub sampling: Option<Sampling>,
    pub max_tokens: usize,
    /// Let the model use its thought channel. Off by default: it costs tokens
    /// and most clients do not surface it.
    pub enable_thinking: bool,
    /// Extra stop strings, checked against decoded text.
    pub stop: Vec<String>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
            sampling: None,
            max_tokens: 512,
            enable_thinking: false,
            stop: Vec::new(),
        }
    }
}

impl Request {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            ..Default::default()
        }
    }

    /// A single user turn — the one-liner case.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new([Message::user(text)])
    }

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn sampling(mut self, s: Sampling) -> Self {
        self.sampling = Some(s);
        self
    }

    pub fn tools(mut self, t: impl IntoIterator<Item = Tool>) -> Self {
        self.tools = t.into_iter().collect();
        self
    }

    pub fn thinking(mut self, on: bool) -> Self {
        self.enable_thinking = on;
        self
    }

    pub fn stop(mut self, s: impl IntoIterator<Item = String>) -> Self {
        self.stop = s.into_iter().collect();
        self
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    /// The caller returned [`Flow::Stop`].
    Cancelled,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Whether to keep generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// The result of a finished generation.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub completion: chat::Completion,
    pub prompt_tokens: usize,
    /// How many prompt positions the cache served without recomputing.
    pub cached_tokens: usize,
    pub generated: usize,
    pub reason: FinishReason,
}

/// Streaming callbacks. Every method has a default, so implement only what
/// you need; `()` implements it as a no-op for non-streaming callers.
pub trait Sink {
    fn prefill(&mut self, tokens: usize, cached: usize) {
        let _ = (tokens, cached);
    }

    /// Called once per generated token, with the text it decoded to — which
    /// is often empty, because a multi-byte character spans several tokens.
    fn token(&mut self, id: u32, text: &str) -> Flow {
        let _ = (id, text);
        Flow::Continue
    }
}

impl Sink for () {}

struct FnSink<F>(F);

impl<F: FnMut(&str) -> Flow> Sink for FnSink<F> {
    fn token(&mut self, _id: u32, text: &str) -> Flow {
        (self.0)(text)
    }
}

/// Streamed progress, for callers that would rather have a channel than a
/// callback. See [`Session::generate_to`].
#[derive(Debug, Clone)]
pub enum Event {
    /// Prompt accepted: total positions, and how many were served from cache.
    Prefill { tokens: usize, cached: usize },
    Token { id: u32, text: String },
    Done {
        completion: chat::Completion,
        generated: usize,
        reason: FinishReason,
    },
    Error(String),
}

/// Each architecture speaks its own chat dialect.
pub enum ChatFormat {
    Gemma4(chat::Special),
    Qwen35(chat::qwen::Special),
}

impl ChatFormat {
    /// Build the dialect for `arch`, reading whatever the checkpoint declares.
    pub fn detect(arch: Arch, tok: &Tokenizer, template: Option<&str>) -> Result<Self> {
        Ok(match arch {
            Arch::Gemma4 => Self::Gemma4(chat::Special::new(tok, template)?),
            Arch::Qwen35 => Self::Qwen35(chat::qwen::Special::new(tok)?),
        })
    }

    /// Assemble a prompt, encoding any images in place.
    ///
    /// With `images` supplied, an image-carrying [`Message`] emits a
    /// [`Segment::Embeds`] between the text runs. Without one, an image is an
    /// error rather than something quietly dropped — answering about an image
    /// the model never saw is the worst available outcome.
    pub fn build(
        &self,
        tok: &Tokenizer,
        messages: &[Message],
        tools: &[Tool],
        thinking: bool,
        images: Option<&dyn chat::ImageEncoder>,
    ) -> Result<Prompt> {
        let carries_image = messages.iter().any(chat::has_image);
        match self {
            Self::Gemma4(sp) => {
                if carries_image && images.is_none() {
                    return Err(Error::Image(
                        "this request carries an image but no vision tower is loaded;                          pass LoadOptions::mmproj(\"mmproj-….gguf\")"
                            .into(),
                    ));
                }
                let pieces = chat::build_prompt_mm(tok, sp, messages, tools, thinking, images)?;
                Ok(Prompt(
                    pieces
                        .into_iter()
                        .map(|p| match p {
                            chat::Piece::Tokens(ids) => Segment::Tokens(ids),
                            chat::Piece::Embeds { rows, n } => Segment::Embeds { rows, n },
                        })
                        .collect(),
                ))
            }
            Self::Qwen35(sp) => {
                if carries_image {
                    return Err(Error::Unsupported("image input on qwen35"));
                }
                Ok(Prompt::tokens(chat::qwen::build_prompt(
                    tok, sp, messages, tools, thinking,
                )))
            }
        }
    }

    fn parse(&self, tok: &Tokenizer, generated: &[u32], thinking: bool) -> chat::Completion {
        match self {
            Self::Gemma4(sp) => chat::parse_completion(tok, sp, generated),
            // With thinking on, the generation prompt leaves a `<think>` block
            // open, so the stream starts inside reasoning.
            Self::Qwen35(sp) => chat::qwen::parse_completion(tok, sp, generated, thinking),
        }
    }
}

/// A loaded model and the conversation resident in its cache.
pub struct Session {
    backend: Box<dyn Backend>,
    tok: Tokenizer,
    format: ChatFormat,
    /// The exact token sequence currently in the KV cache / recurrent state.
    ///
    /// This is the single most sensitive allocation in the process: not a
    /// derived activation but the conversation itself, decodable back to
    /// plaintext with the tokenizer sitting right next to it, and kept alive
    /// between turns on purpose so prefix reuse works.
    ///
    /// With the `private` feature it is locked into RAM so it cannot reach
    /// swap or a hibernation image, and zeroed on wipe and on drop; without
    /// it, an ordinary `Vec`. See [`crate::store`].
    cached: TokenStore,
    /// False once a prompt carrying embedding spans has been prefilled:
    /// `cached` no longer describes the whole resident sequence, so comparing
    /// against it would reuse a prefix that is not really there.
    cache_comparable: bool,
    /// What the checkpoint's own metadata recommends.
    pub default_sampling: Sampling,
    /// The vision tower, when one was loaded. Image input needs it; text does
    /// not, and a text-only process should not pay a gigabyte for it.
    #[cfg(feature = "vision")]
    vision: Option<crate::image_input::Tower>,
}

impl Session {
    /// Open a GGUF checkpoint and load it.
    ///
    /// ```no_run
    /// # fn main() -> llmoxide::Result<()> {
    /// use llmoxide::{LoadOptions, Request, Session};
    ///
    /// let mut s = Session::load("models/Qwen3-0.6B-Q8_0.gguf", &LoadOptions::default())?;
    /// let out = s.complete(Request::user("what is the capital of France?"))?;
    /// println!("{}", out.completion.content);
    /// # Ok(()) }
    /// ```
    pub fn load(path: impl AsRef<std::path::Path>, opts: &LoadOptions) -> Result<Self> {
        let g = gguf::Gguf::open(path.as_ref())?;
        let arch = Arch::detect(&g).map_err(|_| {
            Error::UnsupportedArch(
                g.str("general.architecture")
                    .unwrap_or("<missing>")
                    .to_string(),
            )
        })?;
        let tok = Tokenizer::from_gguf(&g)?;
        let default_sampling = Sampling::from_gguf(&g);
        let format = ChatFormat::detect(arch, &tok, g.str("tokenizer.chat_template").ok())?;
        let backend = backend::load(g, opts)?;
        let session = Self::new(backend, tok, format, default_sampling);
        match &opts.mmproj {
            None => Ok(session),
            #[cfg(feature = "vision")]
            Some(path) => Ok(session.with_vision(crate::image_input::open_tower(
                path,
                opts.device,
            )?)),
            #[cfg(not(feature = "vision"))]
            Some(_) => Err(Error::Unsupported(
                "image input: this build has no `vision` feature",
            )),
        }
    }

    /// Wrap a backend built some other way — the seam the browser build enters
    /// through, since it streams weights in itself rather than mapping a file.
    pub fn new(
        backend: Box<dyn Backend>,
        tok: Tokenizer,
        format: ChatFormat,
        default_sampling: Sampling,
    ) -> Self {
        Self {
            backend,
            tok,
            format,
            cached: TokenStore::new(),
            cache_comparable: true,
            default_sampling,
            #[cfg(feature = "vision")]
            vision: None,
        }
    }

    /// Attach a vision tower to a session built with [`Session::new`].
    #[cfg(feature = "vision")]
    pub fn with_vision(mut self, v: crate::image_input::Tower) -> Self {
        self.vision = Some(v);
        self
    }

    /// Whether this session can accept image input.
    pub fn supports_images(&self) -> bool {
        #[cfg(feature = "vision")]
        {
            self.vision.is_some()
        }
        #[cfg(not(feature = "vision"))]
        {
            false
        }
    }

    pub fn info(&self) -> &Info {
        self.backend.info()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn format(&self) -> &ChatFormat {
        &self.format
    }

    pub fn context_len(&self) -> usize {
        self.backend.info().context_len
    }

    /// Forget the current conversation: overwrite the resident token ids and
    /// every buffer derived from them.
    ///
    /// Cheap enough to call between turns — it clears device buffers and
    /// locked pages, not the weights, so the model stays resident and the next
    /// prompt only pays a full prefill.
    pub fn wipe(&mut self) {
        self.backend.wipe();
        self.cached.wipe();
        self.cache_comparable = true;
    }

    /// Generate, with no streaming.
    pub fn complete(&mut self, req: Request) -> Result<Outcome> {
        self.generate_with(req, &mut ())
    }

    /// Generate, calling `on_token` with each piece of decoded text as it
    /// arrives. Return [`Flow::Stop`] from it to cut the generation short.
    pub fn generate(
        &mut self,
        req: Request,
        on_token: impl FnMut(&str) -> Flow,
    ) -> Result<Outcome> {
        self.generate_with(req, &mut FnSink(on_token))
    }

    /// Generate, reporting to `sink`.
    pub fn generate_with(&mut self, req: Request, sink: &mut dyn Sink) -> Result<Outcome> {
        self.run(req, sink)
    }

    /// Generate, streaming [`Event`]s down a channel and reporting failure as
    /// [`Event::Error`] rather than returning it.
    ///
    /// For callers that already own a channel — the HTTP server runs the
    /// engine on its own thread, because the wgpu resources a backend holds
    /// are not `Sync`.
    pub fn generate_to(&mut self, req: Request, out: &mpsc::Sender<Event>) {
        struct ChannelSink<'a>(&'a mpsc::Sender<Event>);
        impl Sink for ChannelSink<'_> {
            fn prefill(&mut self, tokens: usize, cached: usize) {
                let _ = self.0.send(Event::Prefill { tokens, cached });
            }
            fn token(&mut self, id: u32, text: &str) -> Flow {
                match self.0.send(Event::Token {
                    id,
                    text: text.to_string(),
                }) {
                    Ok(()) => Flow::Continue,
                    // Client hung up.
                    Err(_) => Flow::Stop,
                }
            }
        }
        match self.run(req, &mut ChannelSink(out)) {
            Ok(o) => {
                let _ = out.send(Event::Done {
                    completion: o.completion,
                    generated: o.generated,
                    reason: o.reason,
                });
            }
            Err(e) => {
                let _ = out.send(Event::Error(e.to_string()));
            }
        }
    }

    /// How much of `prompt` the cache can serve.
    ///
    /// Only a strict prefix of what is already resident counts. Rewinding is
    /// not an option: the sliding-window layers keep a ring of 1024 slots, so
    /// positions before the rewind point have already been overwritten by
    /// later ones. Conversations that only append — which is the agentic case
    /// — hit this path every turn.
    fn reusable_prefix(&self, prompt: &Prompt) -> usize {
        if !self.cache_comparable {
            return 0;
        }
        let head = prompt.leading_tokens();
        let n = self
            .cached
            .iter()
            .zip(head)
            .take_while(|(a, b)| a == b)
            .count();
        if n == self.cached.len() && n < prompt.len() {
            n
        } else {
            0
        }
    }

    fn run(&mut self, req: Request, sink: &mut dyn Sink) -> Result<Outcome> {
        // Scoped so the immutable borrow of `self.vision` ends before the
        // prefill below needs `&mut self`.
        let prompt = {
            #[cfg(feature = "vision")]
            let encoder = self.vision.as_ref().map(|v| crate::image_input::Encoder {
                vision: v,
                max_tokens: self.backend.info().max_batch,
            });
            #[cfg(feature = "vision")]
            let images = encoder.as_ref().map(|e| e as &dyn chat::ImageEncoder);
            #[cfg(not(feature = "vision"))]
            let images: Option<&dyn chat::ImageEncoder> = None;

            self.format.build(
                &self.tok,
                &req.messages,
                &req.tools,
                req.enable_thinking,
                images,
            )?
        };

        let n_ctx = self.context_len();
        if prompt.len() + 8 >= n_ctx {
            return Err(Error::ContextOverflow {
                tokens: prompt.len(),
                context: n_ctx,
            });
        }

        let reuse = self.reusable_prefix(&prompt);
        if reuse == 0 {
            // Nothing of the resident conversation is reachable from here on,
            // so overwrite it now rather than leaving it to be shadowed by
            // whatever the new one happens to write.
            self.backend.wipe();
            self.cached.wipe();
            self.cache_comparable = true;
        }
        tracing::info!(
            prompt = prompt.len(),
            cached = reuse,
            tools = req.tools.len(),
            "prefill"
        );
        sink.prefill(prompt.len(), reuse);

        let mut logits = self.prefill(&prompt, reuse)?;

        // Record what is now resident. With embedding spans in the prompt the
        // token ids no longer describe the whole sequence, so the next turn
        // must not try to reuse it.
        self.cached.replace(prompt.leading_tokens());
        if prompt.has_embeds() {
            self.cache_comparable = false;
        }

        let eog = self.backend.info().eog.clone();
        let max_new = req
            .max_tokens
            .min(n_ctx.saturating_sub(prompt.len()).saturating_sub(1));
        let mut sampler = Sampler::new(req.sampling.unwrap_or_else(|| self.default_sampling.clone()));
        let mut generated: Vec<u32> = Vec::new();
        let mut decoder = Decoder::new(&self.tok);
        // Accumulates the whole response for stop-string matching. Locked
        // under the `private` feature, so a long generation cannot be paged
        // out mid-answer.
        let mut text = TextStore::new();
        let mut reason = FinishReason::Length;

        for _ in 0..max_new {
            let next = sampler.sample(&mut logits, &generated);
            if eog.contains(&next) {
                reason = FinishReason::Stop;
                generated.push(next);
                break;
            }
            generated.push(next);

            let piece = decoder.push(next);
            text.push_str(&piece);
            if sink.token(next, &piece) == Flow::Stop {
                reason = FinishReason::Cancelled;
                break;
            }

            if req.stop.iter().any(|s| !s.is_empty() && text.contains(s)) {
                reason = FinishReason::Stop;
                break;
            }

            if self.cache_comparable {
                self.cached.push(next);
            }
            logits = self.backend.forward(&[next])?;
        }

        let mut completion = self.format.parse(&self.tok, &generated, req.enable_thinking);
        self.filter_tool_calls(&mut completion, &req.tools);

        if !completion.tool_calls.is_empty() {
            reason = FinishReason::ToolCalls;
        }
        tracing::info!(
            generated = generated.len(),
            reason = reason.as_str(),
            calls = completion.tool_calls.len(),
            chars = completion.content.len(),
            "done"
        );
        Ok(Outcome {
            completion,
            prompt_tokens: prompt.len(),
            cached_tokens: reuse,
            generated: generated.len(),
            reason,
        })
    }

    /// Run the uncached tail of `prompt`, in chunks the scratch buffers hold.
    fn prefill(&mut self, prompt: &Prompt, reuse: usize) -> Result<Vec<f32>> {
        let batch = self.backend.info().max_batch;
        let d_model = self.backend.info().d_model;
        let mut logits = Vec::new();
        let mut pos = 0usize;

        for seg in &prompt.0 {
            let len = seg.len();
            if pos + len <= reuse {
                pos += len;
                continue;
            }
            match seg {
                Segment::Tokens(ids) => {
                    // `reuse` is a prefix, so this trims only the first
                    // partially-cached segment and is zero for every later one.
                    let start = reuse.saturating_sub(pos);
                    for chunk in ids[start..].chunks(batch) {
                        logits = self.backend.forward(chunk)?;
                    }
                }
                Segment::Embeds { rows, n } => {
                    // Deliberately not chunked. The span attends to itself in
                    // both directions, so splitting it would leave the first
                    // half unable to see the second and silently change what
                    // the model is looking at. An image that does not fit is
                    // an error the caller can act on.
                    if *n > batch {
                        return Err(Error::Unsupported(
                            "embedding span longer than max_batch; raise LoadOptions::max_batch",
                        ));
                    }
                    debug_assert_eq!(rows.len(), n * d_model);
                    logits = self.backend.forward_embeds(rows)?;
                }
            }
            pos += len;
        }
        Ok(logits)
    }

    /// Drop calls to tools the request never declared.
    ///
    /// The model will occasionally invent one — asked about a directory with
    /// only `list_files` available, it has been seen answering correctly and
    /// then calling `read_file` anyway — and a client that dispatched it would
    /// either error out or, worse, run something unintended.
    fn filter_tool_calls(&self, completion: &mut chat::Completion, tools: &[Tool]) {
        let declared: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();

        // gemma4 also emits its call DSL unwrapped sometimes, in which case it
        // lands in the content. Promote those to real calls when the tool was
        // declared, and excise them either way — raw DSL in the visible reply
        // is never what the client wants. (qwen35's tool markers are control
        // tokens, so its calls cannot land in content this way.)
        if matches!(self.format, ChatFormat::Gemma4(_)) {
            let bare = chat::dsl::find_bare_calls(&completion.content);
            if !bare.is_empty() {
                let mut text = completion.content.clone();
                for (range, call) in bare.into_iter().rev() {
                    text.replace_range(range, "");
                    if declared.contains(&call.name.as_str()) {
                        completion.tool_calls.push(call);
                    } else {
                        tracing::warn!(tool = %call.name, "discarding bare call to undeclared tool");
                    }
                }
                completion.content = text.trim().to_string();
            }
        }
        completion.tool_calls.retain(|c| {
            let ok = declared.contains(&c.name.as_str());
            if !ok {
                tracing::warn!(tool = %c.name, "dropping call to undeclared tool");
            }
            ok
        });
    }
}
