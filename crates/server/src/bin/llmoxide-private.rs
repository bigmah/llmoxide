//! An inference mode that leaves nothing behind.
//!
//! The engine already writes nothing to disk, but a session through it is still
//! recoverable afterwards from three places, none of which a `reset` touches:
//! the resident prompt ids the server keeps for prefix reuse, the KV cache and
//! activation buffers on the GPU, and the transient heap copies that prompt
//! text passes through on its way between them. This binary closes all three
//! and gives you a `/wipe` that means it.
//!
//!   llmoxide-private <model.gguf>
//!
//! What it does differently to `llmoxide` and `llmoxide-serve`:
//!
//! * **Prompts are typed, never passed as arguments.** `llmoxide model "..."`
//!   writes the prompt verbatim into `~/.zsh_history`, which on this machine is
//!   configured to keep a million timestamped lines forever. Reading from stdin
//!   sidesteps the shell entirely.
//! * **No client, so no client-side archive.** opencode keeps every message in
//!   plaintext in `~/.local/share/opencode/opencode.db`. This mode has no
//!   storage of any kind: the conversation lives in locked memory and ends when
//!   you wipe it or the process exits.
//! * **The heap is zeroed as it is freed** ([`secret::ZeroizingAlloc`]), so the
//!   copies no wipe could chase — the chat template's `String`s, decoded token
//!   pieces, per-token logit vectors — do not outlive their allocation.
//! * **Sensitive buffers are locked into RAM**, so they cannot reach swap or
//!   `/var/vm/sleepimage`. This matters more than the wipe itself: zeroing a
//!   page *after* it has been written to a hibernation image does not unwrite
//!   it, and this machine has `hibernatemode 3` with a 2 GB sleep image.
//!
//! What it does not do: your terminal keeps its scrollback (`/wipe` asks it to
//! clear, which most emulators honour, but that is a request, not a guarantee),
//! the GGUF's access time still shows that inference ran, and root on a live
//! machine can still read this process's memory.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use chat::Message;
use server::engine::{Engine, Event, GenerateRequest};

/// Overwrite every heap block as it is freed, once armed. Installed process-wide
/// because prompt text does not stay in one place: it is a `Bytes`, then a
/// `String`, then a token `Vec`, then decoded pieces, and any of those
/// intermediates can be the copy that survives.
#[global_allocator]
static ALLOC: secret::ZeroizingAlloc<std::alloc::System> =
    secret::ZeroizingAlloc(std::alloc::System);

/// Set from the SIGINT handler. Generation stops at the next token rather than
/// the process dying mid-answer with the cache still populated.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(sig: i32, handler: usize) -> usize;
}
const SIGINT: i32 = 2;

extern "C" fn on_sigint(_: i32) {
    // Only an atomic store, which is safe to do from a signal handler.
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// Work for the engine thread; the engine owns wgpu resources that are not
/// `Sync`, so it stays put and everything reaches it down this channel.
enum Cmd {
    Gen(Box<GenerateRequest>, mpsc::Sender<Event>),
    Wipe(mpsc::Sender<()>),
}

fn main() -> anyhow::Result<()> {
    // Before anything reads a prompt: no core file on a crash, no debugger
    // attaching to a live session.
    secret::harden();
    unsafe { signal(SIGINT, on_sigint as *const () as usize) };

    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gemma4-v2-Q4_K_M.gguf".to_string());
    let n_ctx: usize = env_usize("LLMOXIDE_CTX", 16384);
    let batch: usize = env_usize("LLMOXIDE_BATCH", 256);

    eprintln!("loading {model}…");

    let (tx_cmd, rx_cmd) = mpsc::channel::<Cmd>();
    let (tx_ready, rx_ready) = mpsc::channel::<anyhow::Result<()>>();

    // The engine is built on its own thread and never leaves it.
    let engine_thread = std::thread::Builder::new()
        .name("llmoxide-engine".into())
        .spawn(move || {
            let mut engine = match Engine::load(&model, n_ctx, batch) {
                Ok(e) => {
                    let _ = tx_ready.send(Ok(()));
                    e
                }
                Err(e) => {
                    let _ = tx_ready.send(Err(e));
                    return;
                }
            };
            // Arm only now: loading stages gigabytes of weights through the
            // heap, and zeroing those on the way out is pure cost — they are
            // not secret, and nothing typed has been read yet.
            secret::arm();

            while let Ok(cmd) = rx_cmd.recv() {
                match cmd {
                    Cmd::Gen(req, tx) => engine.generate(*req, &tx),
                    Cmd::Wipe(done) => {
                        engine.wipe();
                        let _ = done.send(());
                    }
                }
            }
            // Channel closed: the REPL is gone. Wipe before the weights and
            // buffers are torn down.
            engine.wipe();
        })
        .expect("spawn engine thread");

    rx_ready.recv().expect("engine thread died")?;
    banner(n_ctx);

    let mut history: Vec<Message> = Vec::new();
    let stdin = std::io::stdin();
    let mut line = String::new();

    loop {
        print!("\n\x1b[1m»\x1b[0m ");
        std::io::stdout().flush().ok();

        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            // EOF (ctrl-D).
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/quit" | "/exit" => break,
            "/help" => {
                help();
                continue;
            }
            "/wipe" => {
                history.clear();
                wipe(&tx_cmd);
                // Ask the terminal to drop its scrollback too. Honoured by
                // Terminal.app and iTerm2; harmless where it is not.
                print!("\x1b[3J\x1b[H\x1b[2J");
                std::io::stdout().flush().ok();
                println!("wiped: conversation, device buffers, locked pages.");
                continue;
            }
            "/new" => {
                history.clear();
                wipe(&tx_cmd);
                println!("new conversation.");
                continue;
            }
            _ => {}
        }

        history.push(Message {
            role: "user".into(),
            content: Some(serde_json::Value::String(input.to_string())),
            ..Default::default()
        });

        INTERRUPTED.store(false, Ordering::SeqCst);
        let reply = turn(&tx_cmd, &history);
        match reply {
            Ok(text) => history.push(Message {
                role: "assistant".into(),
                content: Some(serde_json::Value::String(text)),
                ..Default::default()
            }),
            Err(e) => {
                eprintln!("\nerror: {e}");
                // Do not leave a half-answered turn in the history; the next
                // prompt would replay it and diverge from the cache anyway.
                history.pop();
            }
        }
    }

    // Wiping before the history is dropped keeps the order obvious: device and
    // locked memory first, then the `String`s, which the armed allocator zeroes
    // as they are freed.
    wipe(&tx_cmd);
    history.clear();
    drop(tx_cmd);
    let _ = engine_thread.join();
    println!("\nwiped and exited.");
    Ok(())
}

/// Run one turn, streaming to stdout, and return the reply text.
fn turn(tx_cmd: &mpsc::Sender<Cmd>, history: &[Message]) -> anyhow::Result<String> {
    let (tx_ev, rx_ev) = mpsc::channel();
    let req = GenerateRequest {
        messages: history.to_vec(),
        tools: Vec::new(),
        sampling: Default::default(),
        max_tokens: 2048,
        enable_thinking: false,
        stop: Vec::new(),
    };
    tx_cmd
        .send(Cmd::Gen(Box::new(req), tx_ev))
        .map_err(|_| anyhow::anyhow!("engine thread gone"))?;

    let mut out = String::new();
    println!();
    for ev in rx_ev {
        if INTERRUPTED.load(Ordering::Relaxed) {
            // Dropping the receiver makes the engine's next `send` fail, which
            // it already treats as "client hung up" and stops on.
            println!("\n^C");
            break;
        }
        match ev {
            Event::Token { text, .. } => {
                print!("{text}");
                std::io::stdout().flush().ok();
                out.push_str(&text);
            }
            Event::Done { completion, .. } => {
                out = completion.content;
                break;
            }
            Event::Error(e) => return Err(anyhow::anyhow!(e)),
            Event::Prefill { .. } => {}
        }
    }
    println!();
    Ok(out)
}

fn wipe(tx_cmd: &mpsc::Sender<Cmd>) {
    let (done, wait) = mpsc::channel();
    if tx_cmd.send(Cmd::Wipe(done)).is_ok() {
        // Block until it has actually run — a wipe you did not wait for is not
        // a wipe.
        let _ = wait.recv();
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn banner(n_ctx: usize) {
    println!("\nllmoxide private mode — nothing is written to disk.");
    println!("context {n_ctx}; heap zeroed on free; prompt and cache locked into RAM.");
    help();
}

fn help() {
    println!("  /wipe   overwrite the conversation, device buffers and scrollback");
    println!("  /new    same, but stay in the session");
    println!("  /quit   wipe and exit  (ctrl-D also works, ctrl-C stops a reply)");
}
