//! The chat window.
//!
//! Everything shown here lives in this process: Blitz lays out and paints the
//! DOM itself, so a message's text goes from the engine's channel into a
//! signal, into a DOM text node, into glyphs — every copy on the heap the
//! armed allocator zeroes on free. Wiping clears the signals, which drops the
//! nodes, which frees the text.

use chat::Message;
use dioxus_native::prelude::dioxus_core::Task;
use dioxus_native::prelude::*;
use futures::StreamExt;

use crate::engine::{Engine, Reply};
use crate::md::Markdown;

const CSS: &str = r#"
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #212121;
    color: #ececec;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 15px;
}
.app { display: flex; flex-direction: column; height: 100vh; }

/* Header: model switcher on the left, New chat on the right. */
.bar {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px;
    border-bottom: 1px solid #2c2c2c;
}
.model {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 10px; border-radius: 8px;
    font-size: 16px; font-weight: 600; color: #ececec;
}
.model:hover { background: #2f2f2f; }
.model .chev { color: #8e8e8e; font-size: 12px; }
.model.cta { background: #ececec; color: #212121; font-size: 14px; }
.detail { color: #8e8e8e; font-size: 12px; }
.spacer { flex: 1; }
.badge {
    display: flex; align-items: center; gap: 5px;
    padding: 4px 9px; border-radius: 999px;
    border: 1px solid #2f4a3a; color: #7fcf9a; font-size: 12px;
}
.ghost {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 10px; border-radius: 8px;
    color: #ececec; font-size: 14px;
}
.ghost:hover { background: #2f2f2f; }
.banner {
    padding: 8px 16px; font-size: 13px;
    background: #3a2020; color: #f5b1a8;
}

/* column-reverse pins the scroll position to the newest message, which is
   what a chat wants and what Blitz offers no scroll API for. */
.log {
    flex: 1; overflow-y: auto;
    display: flex; flex-direction: column-reverse;
}
.col { width: 100%; max-width: 760px; margin: 0 auto; padding: 0 20px; }
.turns { display: flex; flex-direction: column; gap: 24px; padding-top: 24px; padding-bottom: 24px; }
/* The empty state fills the log so the greeting sits mid-window. */
.col.fill { flex: 1; display: flex; flex-direction: column; justify-content: center; }
.welcome { text-align: center; padding-bottom: 40px; }
.welcome .title { font-size: 28px; font-weight: 600; margin-bottom: 10px; }
.welcome .sub { color: #8e8e8e; font-size: 14px; line-height: 1.5; }

.user {
    align-self: flex-end; max-width: 75%;
    padding: 10px 16px; border-radius: 20px;
    background: #303030;
    white-space: pre-wrap; line-height: 1.5;
}
/* Replies take the full width rather than shrink-wrapping: Blitz measures an
   auto-width box's text too narrow and breaks lines that fit. */
.assistant { display: flex; gap: 14px; align-items: flex-start; }
.avatar {
    width: 28px; height: 28px; border-radius: 14px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    border: 1px solid #3a3a3a; color: #bdbdbd; font-size: 10px; font-weight: 600;
}
.reply { flex: 1; min-width: 0; line-height: 1.6; padding-top: 2px; }
.thinking { color: #8e8e8e; }
.actions { display: flex; margin-top: -4px; }
.ghost.small { font-size: 12px; color: #8e8e8e; padding: 4px 8px; margin-left: -8px; }
.ghost.small:hover { color: #ececec; }
.error { align-self: center; color: #f08a7e; font-size: 13px; }

/* Markdown in replies (md.rs). */
.md p { margin: 0 0 12px 0; }
.md .h { font-weight: 600; margin: 18px 0 8px 0; }
.md .h1 { font-size: 22px; }
.md .h2 { font-size: 19px; }
.md .h3 { font-size: 17px; }
.md .h4 { font-size: 15px; }
.md .list { margin: 0 0 12px 0; display: flex; flex-direction: column; gap: 4px; }
.md .li { display: flex; gap: 8px; }
.md .marker { color: #8e8e8e; min-width: 18px; text-align: right; flex-shrink: 0; }
.md .libody { flex: 1; min-width: 0; }
.md .libody p { margin: 0 0 4px 0; }
.md .libody .list { margin: 4px 0 4px 0; }
.md .quote { border-left: 3px solid #4a4a4a; padding-left: 12px; color: #c5c5c5; margin: 0 0 12px 0; }
.md .icode {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13.5px;
    background: #353535; padding: 1px 5px; border-radius: 5px;
}
.md .codeblock { margin: 0 0 12px 0; border-radius: 10px; background: #171717; border: 1px solid #2c2c2c; }
.md .codehead {
    padding: 6px 12px; font-size: 12px; color: #8e8e8e;
    background: #262626; border-radius: 10px 10px 0 0;
}
.md pre {
    margin: 0; padding: 12px; overflow-x: auto;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13.5px;
    line-height: 1.5; white-space: pre;
}
.md .hr { height: 1px; background: #3a3a3a; margin: 16px 0; }
.md .link { color: #7ab7ff; text-decoration: underline; }
.md .del { text-decoration: line-through; }
.md .table { display: flex; flex-direction: column; margin: 0 0 12px 0; border: 1px solid #3a3a3a; border-radius: 8px; }
.md .tr { display: flex; border-top: 1px solid #3a3a3a; }
.md .tr:first-child { border-top: none; }
.md .th { font-weight: 600; background: #2a2a2a; }
.md .td { flex: 1; min-width: 0; padding: 6px 10px; }
.md strong { font-weight: 600; }

/* Composer: one rounded box with the send button inside it. */
.foot { padding: 0 0 10px 0; }
.composer {
    display: flex; align-items: flex-end; gap: 8px;
    padding: 8px 8px 8px 18px;
    border-radius: 26px; background: #303030; border: 1px solid #3a3a3a;
}
.field { flex: 1; position: relative; display: flex; }
.ph { position: absolute; left: 0; top: 7px; color: #8e8e8e; line-height: 22px; }
textarea {
    flex: 1; position: relative; resize: none; padding: 7px 0;
    border: none; outline: none; background: transparent; color: #ececec;
    font-family: inherit; font-size: 15px; line-height: 22px;
}
.round {
    width: 36px; height: 36px; border-radius: 18px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    border: none; background: #ececec; color: #212121; padding: 0;
}
.round:hover { background: #ffffff; }
.round.off { background: #4a4a4a; color: #8e8e8e; }
.round { font-size: 19px; font-weight: 700; }
.stop { width: 12px; height: 12px; border-radius: 2px; background: #212121; }
.glyph { font-size: 18px; line-height: 16px; }
.hint { text-align: center; color: #7a7a7a; font-size: 12px; padding-top: 8px; }
button { border: none; background: transparent; font-family: inherit; padding: 0; }
"#;

#[derive(Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
}

#[derive(Clone, PartialEq)]
struct Turn {
    role: Role,
    text: String,
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
        turns.write().push(Turn {
            role: Role::User,
            text: text.trim().to_string(),
        });
        let history = turns.read().iter().map(to_message).collect();
        turns.write().push(Turn {
            role: Role::Assistant,
            text: String::new(),
        });

        let mut rx = e.generate(history);
        let task = spawn(async move {
            while let Some(reply) = rx.next().await {
                match reply {
                    Reply::Token(t) => {
                        if let Some(last) = turns.write().last_mut() {
                            last.text.push_str(&t);
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
        let e = e.clone();
        spawn(async move { e.wipe().await });
        focus += 1;
    };

    let busy = running.read().is_some();
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
                span { class: "badge", "● Private" }
                button { class: "ghost", onclick: wipe, span { class: "glyph", "+" } "New chat" }
            }
            if let Some(e) = failed {
                div { class: "banner", "Couldn't load the model: {e}" }
            }
            div { class: "log",
                div { class: if turns.read().is_empty() { "col fill" } else { "col" },
                    div { class: "turns",
                        if turns.read().is_empty() {
                            div { class: "welcome",
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
                                    div { class: "avatar", "AI" }
                                    div { class: "reply",
                                        if t.text.is_empty() {
                                            span { class: "thinking", "Thinking…" }
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
                    div { class: "composer",
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
                            button { class: "round", onclick: move |_| stop(), span { class: "stop" } }
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

/// The file name without `.gguf`, which is all anyone needs to recognise it.
fn short_name(file: &str) -> String {
    file.strip_suffix(".gguf").unwrap_or(file).to_string()
}

fn to_message(t: &Turn) -> Message {
    Message {
        role: match t.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
        .into(),
        content: Some(serde_json::Value::String(t.text.clone())),
        ..Default::default()
    }
}

async fn finish_load(loaded: crate::engine::Loaded) -> Status {
    match loaded.await {
        Ok(Ok(desc)) => Status::Ready(desc),
        Ok(Err(e)) => Status::Failed(e),
        Err(_) => Status::Failed("engine thread died while loading".into()),
    }
}
