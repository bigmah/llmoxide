//! The engine thread, and the handle the UI talks to it through.
//!
//! Same shape as `llmoxide-private`: the session owns wgpu resources that are
//! not `Sync`, so it is built on its own thread and never leaves it. The
//! difference is the far end — the UI is async, so replies come back on
//! `futures` channels the Dioxus executor can await instead of blocking the
//! event loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use chat::Message;
use futures::channel::{mpsc as amp, oneshot};
use llmoxide::{DevicePref, Flow, LoadOptions, Request, Session, Sink};

/// Streamed progress for one turn.
pub enum Reply {
    Token(String),
    /// The finished text, as the chat dialect parsed it — which can differ
    /// from the concatenated tokens (a stripped thinking channel, say).
    Done(String),
    Error(String),
}

/// What loaded — a one-line description — or why it did not.
pub type Loaded = oneshot::Receiver<Result<String, String>>;

enum Cmd {
    Load(String, oneshot::Sender<Result<String, String>>),
    Gen(Box<Request>, amp::UnboundedSender<Reply>),
    Wipe(oneshot::Sender<()>),
}

/// Cheap to clone; every clone reaches the same engine thread.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::Sender<Cmd>,
    /// The load started at launch, if any. Taken by whichever component
    /// awaits it.
    initial: Arc<Mutex<Option<Loaded>>>,
    /// Set while a checkpoint is loading. By then the previous model has been
    /// wiped and dropped, and nothing can be sent until the load finishes, so
    /// there is nothing for [`Engine::shutdown`] to wait for.
    loading: Arc<AtomicBool>,
    /// Set by [`Engine::shutdown`]: the running turn stops at its next token
    /// and no new one starts, so the final wipe is not queued behind a reply
    /// nobody will read.
    closing: Arc<AtomicBool>,
}

pub struct Settings {
    pub n_ctx: usize,
    pub batch: usize,
    pub cpu: bool,
}

impl Engine {
    /// Start the engine thread, and `model` loading on it if given. Returns
    /// at once, so the window can come up and show progress instead of the
    /// app hanging for the seconds it takes to page a checkpoint in.
    pub fn spawn(s: Settings, model: Option<String>) -> Self {
        let (tx, rx) = mpsc::channel::<Cmd>();
        let closing = Arc::new(AtomicBool::new(false));
        let loading = Arc::new(AtomicBool::new(false));
        let flags = (closing.clone(), loading.clone());
        std::thread::Builder::new()
            .name("llmoxide-engine".into())
            .spawn(move || run(s, rx, &flags.0, &flags.1))
            .expect("spawn engine thread");
        let engine = Self {
            tx,
            initial: Arc::new(Mutex::new(None)),
            closing,
            loading,
        };
        if let Some(path) = model {
            *engine.initial.lock().unwrap() = Some(engine.load(path));
        }
        engine
    }

    pub fn take_initial(&self) -> Option<Loaded> {
        self.initial.lock().unwrap().take()
    }

    /// Swap in the checkpoint at `path`. The current one is wiped and dropped
    /// first — two large models rarely fit side by side — so a failed load
    /// leaves no model at all. Queued behind any generation still running.
    pub fn load(&self, path: String) -> Loaded {
        let (tx, rx) = oneshot::channel();
        if let Err(mpsc::SendError(Cmd::Load(_, tx))) = self.tx.send(Cmd::Load(path, tx)) {
            let _ = tx.send(Err("engine thread gone".into()));
        }
        rx
    }

    /// Start a turn over `history`. Dropping the receiver cancels it at the
    /// next token.
    pub fn generate(&self, history: Vec<Message>) -> amp::UnboundedReceiver<Reply> {
        let (tx, rx) = amp::unbounded();
        let req = Request {
            messages: history,
            tools: Vec::new(),
            sampling: Some(Default::default()),
            max_tokens: 2048,
            enable_thinking: false,
            stop: Vec::new(),
        };
        if self.tx.send(Cmd::Gen(Box::new(req), tx.clone())).is_err() {
            let _ = tx.unbounded_send(Reply::Error("engine thread gone".into()));
        }
        rx
    }

    /// Overwrite the resident conversation, device buffers and locked pages.
    /// Resolves once that has actually happened — a wipe you did not wait for
    /// is not a wipe. Queued behind any generation still running, so cancel
    /// that first or this waits for it to finish.
    pub async fn wipe(&self) {
        let (done, wait) = oneshot::channel();
        if self.tx.send(Cmd::Wipe(done)).is_ok() {
            let _ = wait.await;
        }
    }

    /// Stop whatever is running and wipe, blocking until done. For the exit
    /// paths, which cannot await. Only the first call does anything, and
    /// says so by returning `true`.
    pub fn shutdown(&self) -> bool {
        if self.closing.swap(true, Ordering::SeqCst) {
            return false;
        }
        // Mid-load there is nothing to wipe, and waiting would mean waiting
        // out the load. `closing` is already set, so nothing runs after it.
        if self.loading.load(Ordering::SeqCst) {
            return true;
        }
        futures::executor::block_on(self.wipe());
        true
    }
}

fn run(s: Settings, rx: mpsc::Receiver<Cmd>, closing: &AtomicBool, loading: &AtomicBool) {
    let opts = LoadOptions::new()
        .n_ctx(s.n_ctx)
        .max_batch(s.batch)
        .device(if s.cpu {
            DevicePref::Cpu
        } else {
            DevicePref::Gpu
        });
    let mut session: Option<Session> = None;

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Load(_, _) | Cmd::Gen(_, _) if closing.load(Ordering::SeqCst) => {}
            Cmd::Load(path, done) => {
                if let Some(mut old) = session.take() {
                    old.wipe();
                }
                loading.store(true, Ordering::SeqCst);
                let result = Session::load(&path, &opts);
                loading.store(false, Ordering::SeqCst);
                let _ = done.send(match result {
                    Ok(new) => {
                        // Arm once the first model is resident: loading
                        // stages gigabytes of weights through the heap, and
                        // zeroing those on the way out is pure cost — they
                        // are not secret, and nothing typed has been read
                        // yet. Later loads run armed, and pay for it.
                        secret::arm();
                        let desc = describe(&path, &new);
                        session = Some(new);
                        Ok(desc)
                    }
                    // The picker lists every .gguf, and a vision projector
                    // sits right beside its model under a similar name.
                    Err(e) if e.to_string().contains("\"clip\"") => Err(format!(
                        "{path} is a vision projector (mmproj), not a chat model"
                    )),
                    Err(e) => Err(format!("{path}: {e}")),
                });
            }
            Cmd::Gen(req, tx) => {
                let reply = match session.as_mut() {
                    None => Reply::Error("no model loaded".into()),
                    Some(session) => {
                        let mut sink = ChannelSink(&tx, closing);
                        match session.generate_with(*req, &mut sink) {
                            Ok(o) => Reply::Done(o.completion.content),
                            Err(e) => Reply::Error(e.to_string()),
                        }
                    }
                };
                let _ = tx.unbounded_send(reply);
            }
            Cmd::Wipe(done) => {
                if let Some(session) = session.as_mut() {
                    session.wipe();
                }
                let _ = done.send(());
            }
        }
    }
    // Every handle is gone. Wipe before the buffers are torn down.
    if let Some(mut session) = session {
        session.wipe();
    }
}

/// One line for the header: file, device, context.
fn describe(path: &str, session: &Session) -> String {
    let info = session.info();
    let name = std::path::Path::new(path)
        .file_name()
        .map_or(path.to_string(), |n| n.to_string_lossy().into_owned());
    let device = match &info.device {
        llmoxide::Device::Cpu => "CPU",
        llmoxide::Device::Gpu { adapter } => adapter.as_str(),
    };
    format!("{name} · {device} · ctx {}", info.context_len)
}

struct ChannelSink<'a>(&'a amp::UnboundedSender<Reply>, &'a AtomicBool);

impl Sink for ChannelSink<'_> {
    fn token(&mut self, _id: u32, text: &str) -> Flow {
        if self.1.load(Ordering::Relaxed) {
            return Flow::Stop;
        }
        if text.is_empty() {
            return Flow::Continue;
        }
        match self.0.unbounded_send(Reply::Token(text.to_string())) {
            Ok(()) => Flow::Continue,
            // The UI dropped the receiver: stop was pressed, or the
            // conversation was wiped out from under this turn.
            Err(_) => Flow::Stop,
        }
    }
}
