//! The single-slot inference engine.
//!
//! One model, one request at a time — this is a personal server, so a queue in
//! front of one GPU context beats the complexity of batching. Generation runs
//! on a dedicated thread because `GpuModel` owns wgpu resources that are not
//! `Sync`, and it streams events back over a channel.
//!
//! Both architectures run on the wgpu backend; `LLMOXIDE_CPU=1` selects the
//! validated qwen35 CPU reference path instead. Each architecture brings its
//! own chat format; the generation loop between them is shared.

use std::sync::mpsc;

use chat::{Message, Special, Tool};
use model::sample::{Sampler, Sampling};
use model::{qwen35, Arch};
use tokenizer::Tokenizer;

pub struct GenerateRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub sampling: Sampling,
    pub max_tokens: usize,
    pub enable_thinking: bool,
    /// Extra stop strings from the client, checked against decoded text.
    pub stop: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Event {
    /// Prompt accepted: total tokens, and how many were served from cache.
    Prefill { tokens: usize, cached: usize },
    /// A newly generated token and the text it decoded to (possibly empty).
    Token { id: u32, text: String },
    Done {
        completion: chat::Completion,
        generated: usize,
        reason: FinishReason,
    },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }
}

/// Which model is loaded, with its execution backend.
enum Backend {
    Gemma4Gpu(gpu::forward::GpuModel),
    Qwen35Gpu(gpu::qwen35::Qwen35Gpu),
    /// The qwen35 CPU reference path (`LLMOXIDE_CPU=1`). Weights borrow the
    /// mmap'd file, so both are leaked to `'static` — one model per process.
    Qwen35Cpu {
        cpu: qwen35::Cpu<'static>,
        state: qwen35::State,
        n_ctx: usize,
        max_batch: usize,
        eog: Vec<u32>,
    },
}

impl Backend {
    fn context_len(&self) -> usize {
        match self {
            Self::Gemma4Gpu(m) => m.context_len(),
            Self::Qwen35Gpu(m) => m.context_len(),
            Self::Qwen35Cpu { n_ctx, .. } => *n_ctx,
        }
    }

    fn max_batch(&self) -> usize {
        match self {
            Self::Gemma4Gpu(m) => m.max_batch(),
            Self::Qwen35Gpu(m) => m.max_batch(),
            Self::Qwen35Cpu { max_batch, .. } => *max_batch,
        }
    }

    fn eog(&self) -> &[u32] {
        match self {
            Self::Gemma4Gpu(m) => &m.config().eog,
            Self::Qwen35Gpu(m) => &m.config().eog,
            Self::Qwen35Cpu { eog, .. } => eog,
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Gemma4Gpu(m) => m.reset(),
            Self::Qwen35Gpu(m) => m.reset(),
            Self::Qwen35Cpu { state, .. } => state.clear(),
        }
    }

    fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Gemma4Gpu(m) => m.forward(tokens),
            Self::Qwen35Gpu(m) => m.forward(tokens),
            Self::Qwen35Cpu { cpu, state, .. } => Ok(cpu.forward(tokens, state)),
        }
    }
}

/// Each architecture speaks its own chat dialect.
enum ChatFormat {
    Gemma4(Special),
    Qwen35(chat::qwen::Special),
}

pub struct Engine {
    backend: Backend,
    tok: Tokenizer,
    chat: ChatFormat,
    /// The exact token sequence currently in the KV cache / recurrent state.
    cached: Vec<u32>,
    pub default_sampling: Sampling,
}

impl Engine {
    pub fn load(path: &str, n_ctx: usize, max_batch: usize) -> anyhow::Result<Self> {
        let g = gguf::Gguf::open(path)?;
        let tok = Tokenizer::from_gguf(&g)?;
        let default_sampling = Sampling::from_gguf(&g);

        let (backend, chat) = match Arch::detect(&g)? {
            Arch::Gemma4 => {
                let cfg = model::Config::from_gguf(&g)?;
                let chat = ChatFormat::Gemma4(Special::new(&tok)?);
                let device = gpu::Gpu::blocking_new()?;
                tracing::info!(adapter = %device.adapter_name, "gpu");
                let m = gpu::forward::GpuModel::load(device, &g, cfg, n_ctx, max_batch)?;
                (Backend::Gemma4Gpu(m), chat)
            }
            Arch::Qwen35 => {
                let chat = ChatFormat::Qwen35(chat::qwen::Special::new(&tok)?);
                if std::env::var_os("LLMOXIDE_CPU").is_some() {
                    let g: &'static gguf::Gguf = Box::leak(Box::new(g));
                    g.prefault();
                    let cfg: &'static qwen35::Config =
                        Box::leak(Box::new(qwen35::Config::from_gguf(g)?));
                    tracing::info!("{} (CPU reference path)", cfg.summary());
                    let w: &'static qwen35::Weights<'static> =
                        Box::leak(Box::new(qwen35::Weights::load(g, cfg)?));
                    let eog = cfg.eog.clone();
                    (
                        Backend::Qwen35Cpu {
                            cpu: qwen35::Cpu::new(cfg, w, max_batch),
                            state: qwen35::State::new(cfg, n_ctx),
                            n_ctx,
                            max_batch,
                            eog,
                        },
                        chat,
                    )
                } else {
                    let cfg = qwen35::Config::from_gguf(&g)?;
                    let device = gpu::Gpu::blocking_new()?;
                    tracing::info!(adapter = %device.adapter_name, "gpu");
                    tracing::info!("{}", cfg.summary());
                    let m = gpu::qwen35::Qwen35Gpu::load(device, &g, cfg, n_ctx, max_batch)?;
                    (Backend::Qwen35Gpu(m), chat)
                }
            }
        };

        Ok(Self {
            backend,
            tok,
            chat,
            cached: Vec::new(),
            default_sampling,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn context_len(&self) -> usize {
        self.backend.context_len()
    }

    /// How much of `prompt` the cache can serve.
    ///
    /// Only a strict prefix of what is already resident counts. Rewinding is
    /// not an option: the sliding-window layers keep a ring of 1024 slots, so
    /// positions before the rewind point have already been overwritten by
    /// later ones. Conversations that only append — which is the agentic case —
    /// hit this path every turn.
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

    pub fn generate(&mut self, req: GenerateRequest, out: &mpsc::Sender<Event>) {
        if let Err(e) = self.run(req, out) {
            let _ = out.send(Event::Error(e.to_string()));
        }
    }

    fn run(&mut self, req: GenerateRequest, out: &mpsc::Sender<Event>) -> anyhow::Result<()> {
        let prompt = match &self.chat {
            ChatFormat::Gemma4(sp) => chat::build_prompt(
                &self.tok,
                sp,
                &req.messages,
                &req.tools,
                req.enable_thinking,
            ),
            ChatFormat::Qwen35(sp) => chat::qwen::build_prompt(
                &self.tok,
                sp,
                &req.messages,
                &req.tools,
                req.enable_thinking,
            ),
        };
        let n_ctx = self.context_len();
        anyhow::ensure!(
            prompt.len() + 8 < n_ctx,
            "prompt is {} tokens but the context is {n_ctx}",
            prompt.len()
        );

        let reuse = self.reusable_prefix(&prompt);
        if reuse == 0 {
            self.backend.reset();
            self.cached.clear();
        }
        tracing::info!(
            prompt = prompt.len(),
            cached = reuse,
            tools = req.tools.len(),
            "prefill"
        );
        let _ = out.send(Event::Prefill {
            tokens: prompt.len(),
            cached: reuse,
        });

        // Prefill the uncached tail in chunks the scratch buffers can hold.
        let mut logits = Vec::new();
        let batch = self.backend.max_batch();
        let mut i = reuse;
        while i < prompt.len() {
            let end = (i + batch).min(prompt.len());
            logits = self.backend.forward(&prompt[i..end])?;
            i = end;
        }
        self.cached = prompt;

        let eog = self.backend.eog().to_vec();
        let max_new = req
            .max_tokens
            .min(n_ctx.saturating_sub(self.cached.len()).saturating_sub(1));
        let mut sampler = Sampler::new(req.sampling);
        let mut generated: Vec<u32> = Vec::new();
        let mut decoder = tokenizer::Decoder::new(&self.tok);
        let mut text = String::new();
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
            if out.send(Event::Token { id: next, text: piece }).is_err() {
                // Client hung up.
                return Ok(());
            }

            if req.stop.iter().any(|s| !s.is_empty() && text.contains(s)) {
                reason = FinishReason::Stop;
                break;
            }

            self.cached.push(next);
            logits = self.backend.forward(&[next])?;
        }

        let mut completion = match &self.chat {
            ChatFormat::Gemma4(sp) => chat::parse_completion(&self.tok, sp, &generated),
            // With thinking on, the generation prompt leaves a `<think>` block
            // open, so the stream starts inside reasoning.
            ChatFormat::Qwen35(sp) => {
                chat::qwen::parse_completion(&self.tok, sp, &generated, req.enable_thinking)
            }
        };

        // Drop calls to tools the request never declared. The model will
        // occasionally invent one — asked about a directory with only
        // `list_files` available, it has been seen answering correctly and then
        // calling `read_file` anyway — and a client that dispatched it would
        // either error out or, worse, run something unintended.
        let declared: Vec<&str> = req.tools.iter().map(|t| t.function.name.as_str()).collect();

        // gemma4 also emits its call DSL unwrapped sometimes, in which case it
        // lands in the content. Promote those to real calls when the tool was
        // declared, and excise them either way — raw DSL in the visible reply
        // is never what the client wants. (qwen35's tool markers are control
        // tokens, so its calls cannot land in content this way.)
        if matches!(self.chat, ChatFormat::Gemma4(_)) {
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
        let _ = out.send(Event::Done {
            completion,
            generated: generated.len(),
            reason,
        });
        Ok(())
    }
}
