//! The chat window.
//!
//! Everything shown here lives in this process: Blitz lays out and paints the
//! DOM itself, so a message's text goes from the engine's channel into a
//! signal, into a DOM text node, into glyphs — every copy on the heap the
//! armed allocator zeroes on free. Wiping clears the signals, which drops the
//! nodes, which frees the text.

use std::sync::Arc;

use chat::Message;
use dioxus_native::prelude::dioxus_core::Task;
use dioxus_native::prelude::*;
use futures::StreamExt;

use crate::engine::{Engine, Reply, ToolRow};
use crate::md::Markdown;
use crate::read::Grant;

const CSS: &str = r#"
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #0e0e10;
    color: #ececf1;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 15px;
}
/* A faint oxide glow from the top edge, over near-black. */
.app {
    display: flex; flex-direction: column; height: 100vh;
    background: radial-gradient(ellipse 90% 55% at 50% -12%, #2b1810 0%, #16100e 45%, #0e0e10 80%);
}
button {
    border: none; background: transparent; font-family: inherit; padding: 0; color: inherit;
    transition: background-color 160ms ease, color 160ms ease, border-color 160ms ease,
                box-shadow 200ms ease, transform 120ms ease, opacity 160ms ease;
}
button:active { transform: scale(0.96); }

/* Motion. Everything eases in on mount; nothing loops except the thinking dots. */
@keyframes rise {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0px); }
}
@keyframes drop {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0px); }
}
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes blink {
    0% { opacity: 0.25; transform: translateY(0px); }
    30% { opacity: 1; transform: translateY(-3px); }
    60% { opacity: 0.25; transform: translateY(0px); }
    100% { opacity: 0.25; transform: translateY(0px); }
}

/* Header: model switcher on the left, New chat on the right. */
.bar {
    display: flex; align-items: center; gap: 6px;
    padding: 10px 14px;
}
.model {
    display: flex; align-items: center; gap: 7px;
    padding: 7px 12px; border-radius: 10px;
    font-size: 15px; font-weight: 600; color: #ececf1; letter-spacing: -0.01em;
}
.model:hover { background: rgba(255, 255, 255, 0.06); }
.model, .ghost, .badge { white-space: nowrap; flex-shrink: 0; }
.detail { white-space: nowrap; overflow: hidden; min-width: 0; color: #6e6e76; font-size: 12px; }
.model .chev { color: #6e6e76; font-size: 11px; }
.model.cta {
    background: #e8743b; color: #140a05; font-size: 14px;
    box-shadow: 0 4px 18px rgba(232, 116, 59, 0.35);
}
.model.cta:hover { background: #f08449; }
.spacer { flex: 1; }
.badge {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 11px 5px 9px; border-radius: 999px;
    background: rgba(232, 116, 59, 0.09); border: 1px solid rgba(232, 116, 59, 0.22);
    color: #f0a27a; font-size: 12px; font-weight: 500;
}
/* The padlock, in oxide orange. Drawn with boxes because Blitz paints an
   RSX-built <svg> empty; sized in em, so font-size sets its size. */
.padlock { position: relative; width: 1em; height: 1.2em; flex-shrink: 0; }
.padlock .shackle {
    position: absolute; left: 0.2em; top: 0; width: 0.6em; height: 0.66em;
    border: 0.13em solid #c65a2e; border-bottom: none;
    border-radius: 0.3em 0.3em 0 0;
}
.padlock .body {
    position: absolute; left: 0; bottom: 0; width: 1em; height: 0.7em;
    border-radius: 0.16em;
    background: linear-gradient(160deg, #e5733a, #a8431c);
    display: flex; flex-direction: column; align-items: center; padding-top: 0.17em;
}
.padlock .hole { width: 0.19em; height: 0.19em; border-radius: 0.1em; background: #2a160d; }
.padlock .slot { width: 0.08em; height: 0.17em; background: #2a160d; }
.ghost {
    display: flex; align-items: center; gap: 6px;
    padding: 7px 11px; border-radius: 10px;
    color: #b4b4bc; font-size: 13px; font-weight: 500;
}
.ghost:hover { background: rgba(255, 255, 255, 0.06); color: #ececf1; }
.glyph { font-size: 17px; line-height: 14px; font-weight: 400; }

/* The folder grant: a card under the header, and a chip in it once granted. */
.grant {
    display: flex; align-items: center; gap: 4px;
    padding: 4px 4px 4px 11px; border-radius: 999px;
    background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.08);
    color: #b4b4bc; font-size: 12px;
    white-space: nowrap; overflow: hidden; min-width: 0;
    animation: fade 220ms ease both;
}
.grant .x { color: #6e6e76; padding: 0 6px; font-size: 14px; border-radius: 999px; }
.grant .x:hover { color: #ececf1; background: rgba(255, 255, 255, 0.1); }
.folder {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    margin: 2px 14px 6px 14px; padding: 14px 16px; border-radius: 16px;
    background: #17171a; border: 1px solid rgba(255, 255, 255, 0.07);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    font-size: 13px;
    animation: drop 240ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.folder .label { color: #ececf1; font-weight: 500; }
.folder input {
    flex: 1; min-width: 200px; padding: 8px 12px; border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1); background: #0f0f11; color: #ececf1;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13px;
    overflow: hidden; white-space: nowrap; outline: none;
}
.folder .opt { color: #8b8b93; display: flex; align-items: center; gap: 7px; }
.folder .opt:hover { color: #ececf1; }
.folder .box {
    width: 15px; height: 15px; border-radius: 4px; border: 1px solid #55555c;
    display: flex; align-items: center; justify-content: center; font-size: 11px; color: #140a05;
}
.folder .box.on { background: #e8743b; border-color: #e8743b; }
.folder .go {
    padding: 8px 14px; border-radius: 10px; background: #e8743b; color: #140a05;
    font-size: 13px; font-weight: 600;
}
.folder .go:hover { background: #f08449; }
.folder .note { width: 100%; color: #6e6e76; font-size: 12px; line-height: 1.5; }
.folder .err { width: 100%; color: #f28b7d; font-size: 12px; }
.tools {
    display: flex; flex-direction: column; gap: 2px; margin-bottom: 10px;
    padding-left: 10px; border-left: 2px solid rgba(232, 116, 59, 0.3);
}
.tool { color: #7c7c85; font-size: 13px; display: flex; gap: 7px; padding: 3px 0; animation: fade 200ms ease both; }
.tool:hover { color: #c8c8d0; }
.tool.bad { color: #d98b7f; }
.tool .icode, .grant .icode {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12.5px;
}
.tooldetail {
    margin: 2px 0 6px 0; padding: 10px 12px; border-radius: 10px; background: #0b0b0d;
    border: 1px solid rgba(255, 255, 255, 0.06); color: #b4b4bc;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px;
    white-space: pre-wrap; max-height: 240px; overflow-y: auto;
    animation: drop 200ms ease both;
}
.banner {
    margin: 2px 14px 6px 14px; padding: 10px 14px; border-radius: 12px; font-size: 13px;
    background: rgba(242, 139, 125, 0.1); border: 1px solid rgba(242, 139, 125, 0.25); color: #f5b1a8;
    animation: drop 240ms ease both;
}

/* column-reverse pins the scroll position to the newest message, which is
   what a chat wants and what Blitz offers no scroll API for. */
.log {
    flex: 1; overflow-y: auto;
    display: flex; flex-direction: column-reverse;
}
.col { width: 100%; max-width: 740px; margin: 0 auto; padding: 0 24px; }
.turns { display: flex; flex-direction: column; gap: 28px; padding-top: 20px; padding-bottom: 28px; }
/* The empty state fills the log so the greeting sits mid-window. */
.col.fill { flex: 1; display: flex; flex-direction: column; justify-content: center; }
.welcome {
    display: flex; flex-direction: column; align-items: center; text-align: center;
    padding-bottom: 48px; animation: rise 420ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.hero {
    width: 76px; height: 76px; border-radius: 24px; margin-bottom: 26px;
    display: flex; align-items: center; justify-content: center; font-size: 34px;
    background: linear-gradient(160deg, #241612, #140e0c);
    border: 1px solid rgba(232, 116, 59, 0.28);
    box-shadow: 0 0 60px rgba(232, 116, 59, 0.22), 0 18px 40px rgba(0, 0, 0, 0.5);
}
.welcome .title { font-size: 30px; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 10px; }
.welcome .sub { color: #7c7c85; font-size: 14px; line-height: 1.6; max-width: 440px; }

.user {
    align-self: flex-end; max-width: 78%;
    padding: 11px 17px; border-radius: 22px 22px 6px 22px;
    background: #232327; border: 1px solid rgba(255, 255, 255, 0.05);
    white-space: pre-wrap; line-height: 1.55;
    animation: rise 300ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
/* Replies take the full width rather than shrink-wrapping: Blitz measures an
   auto-width box's text too narrow and breaks lines that fit. */
.assistant {
    display: flex; gap: 14px; align-items: flex-start;
    animation: rise 340ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.avatar {
    width: 30px; height: 30px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 13px;
    background: linear-gradient(160deg, #241612, #140e0c);
    border: 1px solid rgba(232, 116, 59, 0.28);
    box-shadow: 0 0 18px rgba(232, 116, 59, 0.14);
}
.reply { flex: 1; min-width: 0; line-height: 1.65; padding-top: 3px; color: #e2e2e8; }
.thinking { display: flex; gap: 5px; align-items: center; height: 24px; }
.thinking .dot {
    width: 7px; height: 7px; border-radius: 4px; background: #e8743b;
    animation: blink 1.2s ease-in-out infinite;
}
.thinking .d2 { animation-delay: 0.15s; }
.thinking .d3 { animation-delay: 0.3s; }
.actions { display: flex; margin-top: -2px; animation: fade 300ms ease both; }
.ghost.small { font-size: 12px; color: #6e6e76; padding: 4px 8px; margin-left: -8px; border-radius: 8px; }
.ghost.small:hover { color: #ececf1; }
.error {
    align-self: center; color: #f28b7d; font-size: 13px;
    padding: 8px 14px; border-radius: 999px; background: rgba(242, 139, 125, 0.08);
    animation: rise 260ms ease both;
}

/* Markdown in replies (md.rs). */
.md p { margin: 0 0 14px 0; }
.md .h { font-weight: 600; margin: 22px 0 10px 0; letter-spacing: -0.01em; color: #f4f4f8; }
.md .h1 { font-size: 22px; }
.md .h2 { font-size: 19px; }
.md .h3 { font-size: 17px; }
.md .h4 { font-size: 15px; }
.md .list { margin: 0 0 14px 0; display: flex; flex-direction: column; gap: 6px; }
.md .li { display: flex; gap: 10px; }
.md .marker { color: #e8743b; min-width: 18px; text-align: right; flex-shrink: 0; }
.md .libody { flex: 1; min-width: 0; }
.md .libody p { margin: 0 0 4px 0; }
.md .libody .list { margin: 4px 0 4px 0; }
.md .quote {
    border-left: 3px solid rgba(232, 116, 59, 0.45); padding: 2px 0 2px 14px;
    color: #b4b4bc; margin: 0 0 14px 0;
}
.md .icode {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13px;
    background: rgba(255, 255, 255, 0.07); color: #f3b48e; padding: 1px 6px; border-radius: 6px;
}
.md .codeblock {
    margin: 0 0 14px 0; border-radius: 14px; background: #0b0b0d;
    border: 1px solid rgba(255, 255, 255, 0.07);
}
.md .codehead {
    padding: 6px 14px; font-size: 11.5px; line-height: 1.4; color: #6e6e76; letter-spacing: 0.03em;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.md pre {
    margin: 0; padding: 14px; overflow-x: auto; color: #dcdce4;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13px;
    line-height: 1.55; white-space: pre;
}
.md .hr { height: 1px; background: rgba(255, 255, 255, 0.08); margin: 20px 0; }
.md .link { color: #f0a27a; text-decoration: underline; }
.md .del { text-decoration: line-through; }
.md .table {
    display: flex; flex-direction: column; margin: 0 0 14px 0;
    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px;
}
.md .tr { display: flex; border-top: 1px solid rgba(255, 255, 255, 0.06); }
.md .tr:first-child { border-top: none; }
.md .th { font-weight: 600; background: rgba(255, 255, 255, 0.03); border-radius: 12px 12px 0 0; }
.md .td { flex: 1; min-width: 0; padding: 8px 12px; }
.md strong { font-weight: 600; color: #f4f4f8; }

/* Composer: one floating rounded box with the send button inside it. */
.foot { padding: 4px 0 12px 0; }
.composer {
    display: flex; align-items: flex-end; gap: 8px;
    padding: 9px 9px 9px 20px;
    border-radius: 28px; background: #1a1a1d;
    border: 1px solid rgba(255, 255, 255, 0.09);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 1px 0 rgba(255, 255, 255, 0.04);
    transition: border-color 200ms ease, box-shadow 200ms ease;
}
.composer.live {
    border-color: rgba(232, 116, 59, 0.35);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 28px rgba(232, 116, 59, 0.12);
}
.field { flex: 1; position: relative; display: flex; }
.ph { position: absolute; left: 0; top: 7px; color: #5f5f67; line-height: 22px; }
textarea {
    flex: 1; position: relative; resize: none; padding: 7px 0;
    border: none; outline: none; background: transparent; color: #ececf1;
    font-family: inherit; font-size: 15px; line-height: 22px;
}
.round {
    width: 36px; height: 36px; border-radius: 18px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    border: none; background: #e8743b; color: #140a05; padding: 0;
    font-size: 18px; font-weight: 700;
    box-shadow: 0 4px 16px rgba(232, 116, 59, 0.4);
}
.round:hover { background: #f08449; }
.round.off { background: #2a2a2e; color: #5f5f67; box-shadow: none; }
.round.halt { background: #ececf1; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4); }
.stop { width: 11px; height: 11px; border-radius: 3px; background: #140a05; }
.hint { text-align: center; color: #4f4f56; font-size: 11.5px; padding-top: 10px; }
"#;

#[derive(Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
}

#[derive(Clone)]
struct Turn {
    role: Role,
    text: String,
    /// Tool calls made while answering, for the rows above the reply.
    tools: Vec<ToolRow>,
    /// The calls and their results as messages, replayed ahead of `text`.
    context: Vec<Message>,
}

impl Turn {
    fn new(role: Role, text: String) -> Self {
        Self {
            role,
            text,
            tools: Vec::new(),
            context: Vec::new(),
        }
    }
}

#[derive(Clone, PartialEq)]
enum Status {
    NoModel,
    Loading(String),
    Ready(String),
    Failed(String),
}

pub fn app() -> Element {
    let engine = use_context::<Engine>();
    let mut status = use_signal(|| Status::NoModel);
    let mut turns = use_signal(Vec::<Turn>::new);
    let mut draft = use_signal(String::new);
    let mut error = use_signal(|| None::<String>);
    let mut running = use_signal(|| None::<Task>);
    // Bumped to give the input box focus back. Blitz has no focus API, but it
    // honours `autofocus` on mount, and a new key is a new mount.
    let mut focus = use_signal(|| 0u32);
    // The folder the model may read, only ever in memory. `None` is no tools.
    let mut grant = use_signal(|| None::<Arc<Grant>>);
    let mut folder_open = use_signal(|| false);
    let mut folder_draft = use_signal(String::new);
    let mut allow_protected = use_signal(|| false);
    let mut folder_error = use_signal(|| None::<String>);

    use_hook(crate::icon::set_dock_icon);

    let e = engine.clone();
    use_future(move || {
        let e = e.clone();
        async move {
            if let Some(loaded) = e.take_initial() {
                status.set(Status::Loading("model".into()));
                status.set(finish_load(loaded).await);
            }
        }
    });

    let e = engine.clone();
    // A `Callback` is `Copy`, so both the Enter key and the button can hold it.
    let send = use_callback(move |()| {
        if running.read().is_some() || !matches!(*status.read(), Status::Ready(_)) {
            return;
        }
        if draft.read().trim().is_empty() {
            return;
        }
        let text = draft.take();
        error.set(None);
        turns.write().push(Turn::new(Role::User, text.trim().to_string()));
        let history = turns.read().iter().flat_map(to_messages).collect();
        turns
            .write()
            .push(Turn::new(Role::Assistant, String::new()));

        let mut rx = e.generate(history, grant.read().clone());
        let task = spawn(async move {
            while let Some(reply) = rx.next().await {
                match reply {
                    Reply::Token(t) => {
                        if let Some(last) = turns.write().last_mut() {
                            last.text.push_str(&t);
                        }
                    }
                    Reply::Tools {
                        assistant,
                        results,
                        rows,
                    } => {
                        if let Some(last) = turns.write().last_mut() {
                            last.context.push(assistant);
                            last.context.extend(results);
                            last.tools.extend(rows);
                            last.text.clear();
                        }
                    }
                    Reply::Done(full) => {
                        if let Some(last) = turns.write().last_mut() {
                            last.text = full;
                        }
                        break;
                    }
                    Reply::Error(msg) => {
                        // Do not leave a half-answered turn in the history —
                        // the next prompt would replay it. Hand the prompt
                        // back instead so it can be retried.
                        let mut t = turns.write();
                        t.pop();
                        if let Some(user) = t.pop() {
                            draft.set(user.text);
                        }
                        error.set(Some(msg));
                        break;
                    }
                }
            }
            running.set(None);
        });
        running.set(Some(task));
    });

    // Dropping the task drops its receiver, and the engine stops at the next
    // token. What was generated so far stays, as in the REPL.
    let mut stop = move || {
        if let Some(task) = running.take() {
            task.cancel();
        }
    };

    // The conversation survives a switch: it is text on this side, and the
    // next turn replays it into the new model.
    let e = engine.clone();
    let pick = move |_| {
        if matches!(*status.read(), Status::Loading(_)) {
            return;
        }
        let e = e.clone();
        spawn(async move {
            let mut dialog = rfd::AsyncFileDialog::new()
                .set_title("Choose a model")
                .add_filter("GGUF model", &["gguf"]);
            if let Ok(dir) = std::path::Path::new("models").canonicalize() {
                dialog = dialog.set_directory(dir);
            }
            let Some(file) = dialog.pick_file().await else {
                return;
            };
            stop();
            let path = file.path().to_string_lossy().into_owned();
            status.set(Status::Loading(file.file_name()));
            status.set(finish_load(e.load(path)).await);
            focus += 1;
        });
    };

    let e = engine.clone();
    let wipe = move |_| {
        stop();
        crate::clip::forget();
        turns.write().clear();
        draft.set(String::new());
        error.set(None);
        // The grant goes with the conversation: nothing remembers a folder.
        grant.set(None);
        folder_open.set(false);
        folder_draft.set(String::new());
        folder_error.set(None);
        allow_protected.set(false);
        let e = e.clone();
        spawn(async move { e.wipe().await });
        focus += 1;
    };

    let mut grant_folder = move || {
        let opened = Grant::open(&folder_draft.read(), allow_protected());
        match opened {
            Ok(g) => {
                grant.set(Some(Arc::new(g)));
                folder_open.set(false);
                folder_draft.set(String::new());
                folder_error.set(None);
                focus += 1;
            }
            Err(e) => folder_error.set(Some(e)),
        }
    };

    let busy = running.read().is_some();
    let granted = grant.read().as_ref().map(|g| g.display());
    // The chip shows the folder's own name; the full path is in the welcome
    // text, where it has room.
    let granted_name = granted
        .as_deref()
        .map(|p| p.rsplit('/').find(|s| !s.is_empty()).unwrap_or(p).to_string());
    let n_turns = turns.read().len();
    let loading = matches!(*status.read(), Status::Loading(_));
    let ready = matches!(*status.read(), Status::Ready(_));
    let no_model = matches!(*status.read(), Status::NoModel | Status::Failed(_));
    let (model_name, model_detail) = match &*status.read() {
        Status::NoModel | Status::Failed(_) => ("Choose a model".to_string(), String::new()),
        Status::Loading(name) => (short_name(name), "loading…".to_string()),
        Status::Ready(d) => {
            let (name, rest) = d.split_once(" · ").unwrap_or((d.as_str(), ""));
            (short_name(name), rest.to_string())
        }
    };
    let failed = match &*status.read() {
        Status::Failed(e) => Some(e.clone()),
        _ => None,
    };
    let can_send = ready && !draft.read().trim().is_empty();
    // The box grows with the draft, up to eight lines, then scrolls.
    let lines = draft.read().lines().count().max(1) + draft.read().ends_with('\n') as usize;
    let input_height = lines.clamp(1, 8) * 22 + 14;

    rsx! {
        style { {CSS} }
        div { class: "app",
            div { class: "bar",
                button {
                    class: if no_model { "model cta" } else { "model" },
                    disabled: loading,
                    onclick: pick,
                    "{model_name}"
                    if !no_model {
                        span { class: "chev", "▾" }
                    }
                }
                span { class: "detail", "{model_detail}" }
                div { class: "spacer" }
                if let Some(dir) = granted_name {
                    span { class: "grant",
                        "Reading "
                        span { class: "icode", "{dir}" }
                        // Revoking mid-reply would pull the folder out from
                        // under a tool call; the engine holds its own handle
                        // until the turn ends either way.
                        button { class: "x", onclick: move |_| grant.set(None), "×" }
                    }
                } else if !no_model {
                    button {
                        class: "ghost",
                        onclick: move |_| {
                            let open = !folder_open();
                            folder_open.set(open);
                            folder_error.set(None);
                        },
                        "Folder…"
                    }
                }
                span { class: "badge", span { style: "font-size: 11px", Padlock {} } "Private" }
                button { class: "ghost", onclick: wipe, span { class: "glyph", "+" } "New chat" }
            }
            if folder_open() && granted.is_none() {
                div { class: "folder",
                    span { class: "label", "Let the model read files in" }
                    // Typed, not picked: the system folder dialog saves the
                    // last folder it showed to a preferences file.
                    input {
                        r#type: "text",
                        autofocus: true,
                        value: "{folder_draft}",
                        oninput: move |ev| folder_draft.set(ev.value()),
                        onkeydown: move |ev| {
                            if ev.key() == Key::Enter {
                                ev.prevent_default();
                                grant_folder();
                            } else if ev.key() == Key::Escape {
                                folder_open.set(false);
                            }
                        },
                    }
                    button { class: "go", onclick: move |_| grant_folder(), "Allow reading" }
                    button {
                        class: "opt",
                        onclick: move |_| allow_protected.set(!allow_protected()),
                        span { class: if allow_protected() { "box on" } else { "box" },
                            if allow_protected() { "✓" }
                        }
                        "Allow protected folders"
                    }
                    if allow_protected() {
                        div { class: "note",
                            "Documents, Desktop, Downloads, Library and external drives are protected by macOS, which keeps a lasting record that this app was allowed in (the folder, not the files)."
                        }
                    }
                    div { class: "note",
                        "Type a full path, such as ~/projects/foo. Read-only: the model can list, read and search files there, and nothing is written or sent. Reads leave no trace on disk, and New chat forgets the folder."
                    }
                    if let Some(e) = folder_error.read().as_ref() {
                        div { class: "err", "{e}" }
                    }
                }
            }
            if let Some(e) = failed {
                div { class: "banner", "Couldn't load the model: {e}" }
            }
            div { class: "log",
                div { class: if turns.read().is_empty() { "col fill" } else { "col" },
                    div { class: "turns",
                        if turns.read().is_empty() {
                            div { class: "welcome",
                                div { class: "hero", Padlock {} }
                                div { class: "title",
                                    if loading {
                                        "Loading model…"
                                    } else if no_model {
                                        "Choose a model to start"
                                    } else {
                                        "What can I help with?"
                                    }
                                }
                                div { class: "sub",
                                    if no_model {
                                        "Pick a .gguf file with the button in the top left."
                                    } else if let Some(dir) = granted.as_ref() {
                                        "The model can read files in {dir}. Reading leaves no trace on disk; nothing is written or sent."
                                    } else {
                                        "Runs entirely on this machine. Nothing is sent anywhere or written to disk."
                                    }
                                }
                            }
                        }
                        for (i, t) in turns.read().iter().enumerate() {
                            if t.role == Role::User {
                                div { key: "{i}", class: "user", "{t.text}" }
                            } else {
                                div { key: "{i}", class: "assistant",
                                    div { class: "avatar", Padlock {} }
                                    div { class: "reply",
                                        if !t.tools.is_empty() {
                                            div { class: "tools",
                                                for (j, row) in t.tools.iter().enumerate() {
                                                    ToolLine { key: "{j}", row: row.clone() }
                                                }
                                            }
                                        }
                                        if t.text.is_empty() {
                                            div { class: "thinking",
                                                span { class: "dot" }
                                                span { class: "dot d2" }
                                                span { class: "dot d3" }
                                            }
                                        } else {
                                            Markdown { text: t.text.clone() }
                                            // Not while it is still being written.
                                            if !(busy && i + 1 == n_turns) {
                                                CopyButton { text: t.text.clone() }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(msg) = error.read().as_ref() {
                            div { class: "error", "Something went wrong: {msg}" }
                        }
                    }
                }
            }
            div { class: "foot",
                div { class: "col",
                    div { class: if can_send { "composer live" } else { "composer" },
                        div { class: "field",
                        // Blitz does not paint `placeholder`; this sits under
                        // the transparent textarea, so clicks still reach it.
                        if draft.read().is_empty() {
                            span { class: "ph",
                                if ready { "Message" } else { "Load a model to start chatting" }
                            }
                        }
                        for epoch in std::iter::once(focus()) {
                            textarea {
                                key: "{epoch}",
                                autofocus: true,
                                style: "height: {input_height}px",
                                value: "{draft}",
                                oninput: move |ev| draft.set(ev.value()),
                                onkeydown: move |ev| {
                                    if ev.key() == Key::Enter && !ev.modifiers().shift() {
                                        ev.prevent_default();
                                        send.call(());
                                    }
                                },
                            }
                        }
                        }
                        if busy {
                            button { class: "round halt", onclick: move |_| stop(), span { class: "stop" } }
                        } else {
                            button {
                                // Blitz does not match `:disabled`, so the dimmed look is a class.
                                class: if can_send { "round" } else { "round off" },
                                disabled: !can_send,
                                onclick: move |_| { send.call(()); focus += 1; },
                                "↑"
                            }
                        }
                    }
                    div { class: "hint",
                        "Private: New chat or closing the window erases this conversation from memory."
                    }
                }
            }
        }
    }
}

/// One tool call under a reply: its summary, and what the model was given
/// when clicked.
#[component]
fn ToolLine(row: ToolRow) -> Element {
    let mut open = use_signal(|| false);
    rsx! {
        button {
            class: if row.ok { "tool" } else { "tool bad" },
            onclick: move |_| open.set(!open()),
            span { if open() { "▾" } else { "▸" } }
            span {
                // `name` in a summary is a path or pattern: show it as code.
                for (k, part) in row.summary.split('`').enumerate() {
                    if k % 2 == 1 {
                        span { class: "icode", "{part}" }
                    } else {
                        "{part}"
                    }
                }
            }
        }
        if open() {
            div { class: "tooldetail", "{row.detail}" }
        }
    }
}

/// Copies a reply's Markdown source, and says so for a moment.
#[component]
fn CopyButton(text: String) -> Element {
    let mut copied = use_signal(|| None::<bool>);
    rsx! {
        div { class: "actions",
            button {
                class: "ghost small",
                onclick: move |_| {
                    copied.set(Some(crate::clip::copy(&text)));
                    spawn(async move {
                        sleep_ms(1500).await;
                        copied.set(None);
                    });
                },
                match copied() {
                    None => "Copy",
                    Some(true) => "Copied",
                    Some(false) => "Clipboard unavailable",
                }
            }
        }
    }
}

/// Resolves after `ms`. The UI's executor has no timers of its own.
async fn sleep_ms(ms: u64) {
    let (tx, rx) = futures::channel::oneshot::channel::<()>();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(ms));
        let _ = tx.send(());
    });
    let _ = rx.await;
}

/// The app's mark: a padlock in oxide orange.
#[component]
fn Padlock() -> Element {
    rsx! {
        div { class: "padlock",
            div { class: "shackle" }
            div { class: "body",
                div { class: "hole" }
                div { class: "slot" }
            }
        }
    }
}

/// The file name without `.gguf`, which is all anyone needs to recognise it.
fn short_name(file: &str) -> String {
    file.strip_suffix(".gguf").unwrap_or(file).to_string()
}

/// A turn as the model sees it: any tool calls and results, then the text.
fn to_messages(t: &Turn) -> Vec<Message> {
    let mut v = t.context.clone();
    v.push(Message {
        role: match t.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
        .into(),
        content: Some(serde_json::Value::String(t.text.clone())),
        ..Default::default()
    });
    v
}

async fn finish_load(loaded: crate::engine::Loaded) -> Status {
    match loaded.await {
        Ok(Ok(desc)) => Status::Ready(desc),
        Ok(Err(e)) => Status::Failed(e),
        Err(_) => Status::Failed("engine thread died while loading".into()),
    }
}
