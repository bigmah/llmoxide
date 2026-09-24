//! A private chat window over the llmoxide engine.
//!
//!   llmoxide-app [model.gguf]
//!
//! Without a model it opens on the picker; **Model…** swaps one at any time.
//!
//! The same guarantees as `llmoxide-private`, with a window instead of a
//! terminal — which also removes the terminal's scrollback, the one residue
//! the REPL could only ask to have cleared.
//!
//! * **The UI is rendered in-process.** Dioxus's native renderer (Blitz)
//!   lays out and paints with wgpu inside this process. There is no webview:
//!   no WebKit helper processes holding copies of the conversation outside
//!   this process's zeroed heap, and no WebKit caches on disk. No JavaScript
//!   runs, and the renderer is built without its HTTP client, devserver
//!   socket, clipboard or accessibility bridge (see `Cargo.toml`).
//! * **The heap is zeroed as it is freed** ([`secret::ZeroizingAlloc`]), so
//!   the text in signals, DOM nodes and text layout does not outlive its
//!   allocation once it is wiped.
//! * **The model's copy of the conversation is locked into RAM** and
//!   overwritten on Wipe, on window close, and on exit.
//!
//! What it does not do: the UI's own copy of the conversation is zeroed when
//! freed but not mlocked (as with the REPL's history), glyphs drawn on screen
//! are in the compositor's buffers until repainted, the GGUF's access time
//! still shows that inference ran, and root on a live machine can read this
//! process's memory.

mod engine;
mod ui;

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use dioxus_native::{Config, LogicalSize, WindowAttributes};
use engine::{Engine, Settings};

/// This binary exists for the guarantee, so a build without it must fail
/// rather than ship claiming it.
const _: () = assert!(
    llmoxide::PRIVATE_MEMORY,
    "llmoxide-app needs llmoxide's `private` feature"
);

/// Overwrite every heap block as it is freed, once armed. Process-wide because
/// the UI copies text at every layer — signal, DOM node, shaped layout.
#[global_allocator]
static ALLOC: secret::ZeroizingAlloc<std::alloc::System> =
    secret::ZeroizingAlloc(std::alloc::System);

/// The exit paths that never return to `main` — Quit from the app menu calls
/// `exit()` directly, and a signal ends the process — reach the engine here.
static ENGINE: OnceLock<Engine> = OnceLock::new();

/// Set from the signal handler, which may do nothing but an atomic store; a
/// watcher thread does the actual wipe.
static SIGNALLED: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn atexit(cb: extern "C" fn()) -> i32;
    fn _exit(code: i32) -> !;
    fn signal(sig: i32, handler: usize) -> usize;
}
const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;

extern "C" fn on_signal(_: i32) {
    SIGNALLED.store(true, Ordering::SeqCst);
}

extern "C" fn on_exit() {
    if let Some(engine) = ENGINE.get() {
        if engine.shutdown() {
            eprintln!("wiped and exited.");
        }
    }
}

fn main() {
    // Before any window exists: no core file on a crash, no debugger
    // attaching to a live session.
    secret::harden();
    // A debug build links Dioxus's devserver client, and it dials whatever
    // port this names and applies code patches from it. Nothing here wants
    // that; see `Cargo.toml` for why the client is linked at all.
    std::env::remove_var("DIOXUS_DEVSERVER_PORT");

    // A model named on the command line or in the environment is loaded
    // even if missing, so the error says why. The default only if present:
    // without it the window opens on the model picker instead.
    let model = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("LLMOXIDE_MODEL").ok())
        .or_else(|| {
            let default = "models/gemma-4-E4B-it-Q4_K_M.gguf";
            std::path::Path::new(default)
                .exists()
                .then(|| default.to_string())
        });
    let engine = Engine::spawn(
        Settings {
            n_ctx: env_usize("LLMOXIDE_CTX", 16384),
            batch: env_usize("LLMOXIDE_BATCH", 256),
            cpu: std::env::var_os("LLMOXIDE_CPU").is_some(),
        },
        model,
    );
    let _ = ENGINE.set(engine.clone());

    // A panic that reaches abort becomes SIGABRT, and macOS's ReportCrash
    // then writes a report to ~/Library/Logs/DiagnosticReports — no memory
    // contents, but a dated record that this ran, which `RLIMIT_CORE = 0`
    // does not prevent. Print, wipe, and leave by `_exit` instead, which is
    // not a crash. Not on the engine thread: its wipe would wait on itself.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        if std::thread::current().name() != Some("llmoxide-engine") {
            if let Some(engine) = ENGINE.get() {
                engine.shutdown();
            }
        }
        unsafe { _exit(101) }
    }));
    unsafe {
        atexit(on_exit);
        signal(SIGINT, on_signal as *const () as usize);
        signal(SIGTERM, on_signal as *const () as usize);
    }
    std::thread::spawn(|| loop {
        if SIGNALLED.load(Ordering::SeqCst) {
            // Wipe here, then exit. Leaving it to `on_exit` would run it on
            // this thread after its thread-locals are destroyed, and waiting
            // for the engine needs them.
            if let Some(engine) = ENGINE.get() {
                if engine.shutdown() {
                    eprintln!("wiped and exited.");
                }
            }
            std::process::exit(130);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    });

    let ctx = engine.clone();
    dioxus_native::launch_cfg(
        ui::app,
        vec![Box::new(move || Box::new(ctx.clone()) as Box<dyn Any>)],
        vec![Box::new(
            Config::new().with_window_attributes(
                WindowAttributes::default()
                    .with_title("llmoxide")
                    .with_inner_size(LogicalSize::new(820.0, 900.0)),
            ),
        )],
    );

    // The window is closed. Closing is the only way out that returns here;
    // Quit and signals go through `on_exit` instead, and all of them end in
    // the same `shutdown`.
    if engine.shutdown() {
        eprintln!("wiped and exited.");
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
