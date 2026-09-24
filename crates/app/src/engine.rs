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
    Gen(Box<Request>, amp::UnboundedSender<Reply>),
    Wipe(oneshot::Sender<()>),
}

/// Cheap to clone; every clone reaches the same engine thread.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::Sender<Cmd>,
    /// Resolves once, when loading finishes. Taken by whichever component
    /// awaits it.
    ready: Arc<Mutex<Option<Loaded>>>,
    /// Set by [`Engine::shutdown`]: the running turn stops at its next token
    /// and no new one starts, so the final wipe is not queued behind a reply
    /// nobody will read.
    closing: Arc<AtomicBool>,
}

pub struct Settings {
    pub model: String,
    pub n_ctx: usize,
    pub batch: usize,
    pub cpu: bool,
}

impl Engine {
    /// Start loading in the background and return at once, so the window can
    /// come up and show progress instead of the app hanging for the seconds
    /// it takes to page a checkpoint in.
    pub fn spawn(s: Settings) -> Self {
        let (tx, rx) = mpsc::channel::<Cmd>();
        let (tx_ready, rx_ready) = oneshot::channel();
        let closing = Arc::new(AtomicBool::new(false));
        let flag = closing.clone();
        std::thread::Builder::new()
            .name("llmoxide-engine".into())
            .spawn(move || run(s, rx, tx_ready, &flag))
            .expect("spawn engine thread");
        Self {
            tx,
            ready: Arc::new(Mutex::new(Some(rx_ready))),
            closing,
        }
    }

    pub fn take_ready(&self) -> Option<Loaded> {
        self.ready.lock().unwrap().take()
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
        futures::executor::block_on(self.wipe());
        true
    }
}

fn run(
    s: Settings,
    rx: mpsc::Receiver<Cmd>,
    ready: oneshot::Sender<Result<String, String>>,
    closing: &AtomicBool,
) {
    let opts = LoadOptions::new()
        .n_ctx(s.n_ctx)
        .max_batch(s.batch)
        .device(if s.cpu { DevicePref::Cpu } else { DevicePref::Gpu });
    let mut session = match Session::load(&s.model, &opts) {
        Ok(session) => session,
        Err(e) => {
            let _ = ready.send(Err(format!("{}: {e}", s.model)));
            return;
        }
    };
    // Arm only now: loading stages gigabytes of weights through the heap, and
    // zeroing those on the way out is pure cost — they are not secret, and
    // nothing typed has been read yet.
    secret::arm();
    let info = session.info();
    let name = std::path::Path::new(&s.model)
        .file_name()
        .map_or(s.model.clone(), |n| n.to_string_lossy().into_owned());
    let device = match &info.device {
        llmoxide::Device::Cpu => "CPU",
        llmoxide::Device::Gpu { adapter } => adapter.as_str(),
    };
    let _ = ready.send(Ok(format!(
        "{name} · {device} · ctx {}",
        info.context_len
    )));

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Gen(_, _) if closing.load(Ordering::SeqCst) => {}
            Cmd::Gen(req, tx) => {
                let mut sink = ChannelSink(&tx, closing);
                let reply = match session.generate_with(*req, &mut sink) {
                    Ok(o) => Reply::Done(o.completion.content),
                    Err(e) => Reply::Error(e.to_string()),
                };
                let _ = tx.unbounded_send(reply);
            }
            Cmd::Wipe(done) => {
                session.wipe();
                let _ = done.send(());
            }
        }
    }
    // Every handle is gone. Wipe before the buffers are torn down.
    session.wipe();
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
