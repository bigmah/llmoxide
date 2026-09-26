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

use chat::{ApiToolCall, FunctionCall, Message};
use futures::channel::{mpsc as amp, oneshot};
use llmoxide::{DevicePref, Flow, LoadOptions, Request, Session, Sink};

use crate::read::{self, Grant};

/// Rounds of tool calls one turn may make before the model is asked to
/// answer with what it has.
const MAX_STEPS: usize = 8;

/// Streamed progress for one turn.
pub enum Reply {
    Token(String),
    /// A round of tool calls ran. `assistant` carries the calls and
    /// `results` the `tool` messages answering them; both belong in the
    /// history, ahead of the final text, so the next turn replays them.
    /// Tokens streamed before this were the calls being written, not reply.
    Tools {
        assistant: Message,
        results: Vec<Message>,
        rows: Vec<ToolRow>,
    },
    /// The finished text, as the chat dialect parsed it — which can differ
    /// from the concatenated tokens (a stripped thinking channel, say).
    Done(String),
    Error(String),
}

/// One tool call, as the chat window shows it.
#[derive(Clone, PartialEq)]
pub struct ToolRow {
    pub summary: String,
    pub ok: bool,
    /// What the model was given, for when the row is opened.
    pub detail: String,
}

/// What loaded — a one-line description — or why it did not.
pub type Loaded = oneshot::Receiver<Result<String, String>>;

enum Cmd {
    Load(String, oneshot::Sender<Result<String, String>>),
    Gen(Box<Request>, Option<Arc<Grant>>, amp::UnboundedSender<Reply>),
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

    /// Start a turn over `history`, with the read tools if a folder is
    /// granted. Dropping the receiver cancels it at the next token or tool
    /// call.
    pub fn generate(
        &self,
        history: Vec<Message>,
        grant: Option<Arc<Grant>>,
    ) -> amp::UnboundedReceiver<Reply> {
        let (tx, rx) = amp::unbounded();
        let req = Request {
            messages: history,
            tools: Vec::new(),
            sampling: Some(Default::default()),
            max_tokens: 2048,
            enable_thinking: false,
            stop: Vec::new(),
        };
        if self
            .tx
            .send(Cmd::Gen(Box::new(req), grant, tx.clone()))
            .is_err()
        {
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
            Cmd::Load(_, _) | Cmd::Gen(_, _, _) if closing.load(Ordering::SeqCst) => {}
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
            Cmd::Gen(req, grant, tx) => {
                let reply = match session.as_mut() {
                    None => Reply::Error("no model loaded".into()),
                    Some(session) => turn(session, *req, grant.as_deref(), &tx, closing),
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

/// Generate, run any tool calls, feed the results back, and generate again,
/// until the model answers without calling anything. The tools are quick,
/// bounded reads, so they run here on the engine thread, behind the same
/// `closing` flag as generation.
fn turn(
    session: &mut Session,
    mut req: Request,
    grant: Option<&Grant>,
    tx: &amp::UnboundedSender<Reply>,
    closing: &AtomicBool,
) -> Reply {
    if let Some(g) = grant {
        req.tools = read::tools();
        if req.messages.first().is_none_or(|m| m.role != "system") {
            req.messages.insert(0, Message::system(read::system_prompt(g)));
        }
    }
    let mut budget = read::Budget::new();
    for step in 0..=MAX_STEPS {
        // Out of steps: take the tools away, so the model has to answer.
        if step == MAX_STEPS {
            req.tools.clear();
        }
        let mut sink = ChannelSink(tx, closing);
        let o = match session.generate_with(req.clone(), &mut sink) {
            Ok(o) => o,
            Err(e) => return Reply::Error(e.to_string()),
        };
        let c = o.completion;
        let Some(g) = grant.filter(|_| !c.tool_calls.is_empty()) else {
            return Reply::Done(c.content);
        };
        // Stop was pressed, or the window is closing: read nothing more.
        if tx.is_closed() || closing.load(Ordering::SeqCst) {
            return Reply::Done(c.content);
        }
        let calls: Vec<ApiToolCall> = c
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, call)| ApiToolCall {
                id: format!("call_{step}_{i}"),
                kind: "function".into(),
                function: FunctionCall {
                    name: call.name.clone(),
                    arguments: call.arguments.to_string(),
                },
            })
            .collect();
        let mut results = Vec::new();
        let mut rows = Vec::new();
        for (call, api) in c.tool_calls.iter().zip(&calls) {
            let out = read::call(g, &call.name, &call.arguments, &mut budget);
            rows.push(ToolRow {
                summary: out.summary,
                ok: out.ok,
                detail: out.text.clone(),
            });
            results.push(Message {
                role: "tool".into(),
                content: Some(serde_json::Value::String(out.text)),
                tool_call_id: Some(api.id.clone()),
                name: Some(call.name.clone()),
                ..Default::default()
            });
        }
        let assistant = Message {
            role: "assistant".into(),
            content: Some(serde_json::Value::String(c.content)),
            tool_calls: calls,
            ..Default::default()
        };
        req.messages.push(assistant.clone());
        req.messages.extend(results.iter().cloned());
        let sent = tx.unbounded_send(Reply::Tools {
            assistant,
            results,
            rows,
        });
        if sent.is_err() {
            return Reply::Done(String::new());
        }
    }
    unreachable!("the last step has no tools, so it returns")
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use futures::StreamExt;

    use super::*;

    /// The tool loop against a real model: grant a folder, ask about a file
    /// in it, and check the answer came from reading it — then ask again, so
    /// the history with its tool messages is replayed.
    ///
    ///   LLMOXIDE_TEST_MODEL=models/gemma-4-E4B-it-Q4_K_M.gguf \
    ///     cargo test -p llmoxide-app --release -- --ignored --nocapture
    #[test]
    #[ignore = "needs LLMOXIDE_TEST_MODEL"]
    fn answers_from_a_granted_folder() {
        let model = std::env::var("LLMOXIDE_TEST_MODEL").expect("LLMOXIDE_TEST_MODEL");
        let dir = std::env::temp_dir().join(format!("llmoxide-tool-loop-{}.noindex", std::process::id()));
        std::fs::create_dir_all(dir.join("docs")).unwrap();
        std::fs::write(dir.join("docs/notes.txt"), "Meeting notes.\nThe launch code is PERIWINKLE-42.\n").unwrap();
        std::fs::write(dir.join("README.md"), "# Demo\nSee docs/ for notes.\n").unwrap();

        let engine = Engine::spawn(
            Settings { n_ctx: 8192, batch: 256, cpu: false },
            Some(model),
        );
        let loaded = futures::executor::block_on(engine.take_initial().unwrap()).unwrap();
        println!("loaded: {loaded:?}");
        let grant = Arc::new(Grant::open(dir.to_str().unwrap(), false).unwrap());

        let ask = |history: Vec<Message>| {
            let rx = engine.generate(history, Some(grant.clone()));
            let mut context = Vec::new();
            let mut rows = Vec::new();
            let mut done = None;
            for r in futures::executor::block_on(rx.collect::<Vec<_>>()) {
                match r {
                    Reply::Token(_) => {}
                    Reply::Tools { assistant, results, rows: r } => {
                        context.push(assistant);
                        context.extend(results);
                        rows.extend(r);
                    }
                    Reply::Done(t) => done = Some(t),
                    Reply::Error(e) => panic!("error: {e}"),
                }
            }
            for r in &rows {
                println!("  tool: {} (ok={})", r.summary, r.ok);
            }
            let text = done.expect("a final reply");
            println!("  reply: {text}");
            (context, rows, text)
        };

        let q1 = Message::user("What is the launch code? It is written somewhere in the shared folder.");
        let (context, rows, a1) = ask(vec![q1.clone()]);
        assert!(rows.iter().any(|r| r.ok), "no tool call succeeded");
        assert!(a1.contains("PERIWINKLE-42"), "answer did not use the file: {a1}");

        let mut history = vec![q1];
        history.extend(context);
        history.push(Message::assistant(a1));
        history.push(Message::user("Which file was it in? Answer with just the path."));
        let (_, _, a2) = ask(history);
        assert!(a2.contains("notes.txt"), "follow-up lost the tool context: {a2}");

        engine.shutdown();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
