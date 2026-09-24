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

const CSS: &str = r#"
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #16171b;
    color: #e6e6e9;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 15px;
}
.app { display: flex; flex-direction: column; height: 100vh; }
.bar {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px;
    border-bottom: 1px solid #2a2c33;
    background: #1c1d22;
}
.status { flex: 1; color: #9a9ca5; font-size: 13px; }
.status.err { color: #f08a7e; }
.lock { color: #7fcf9a; font-size: 13px; }
/* column-reverse pins the scroll position to the newest message, which is
   what a chat wants and what Blitz offers no scroll API for. */
.log {
    flex: 1; overflow-y: auto;
    display: flex; flex-direction: column-reverse;
    padding: 16px;
}
.turns { display: flex; flex-direction: column; gap: 12px; }
.empty { color: #6d6f78; text-align: center; margin: auto; padding: 40px; }
.msg { white-space: pre-wrap; line-height: 1.45; }
/* Replies take the full width rather than shrink-wrapping: Blitz measures an
   auto-width box's pre-wrapped text too narrow and breaks lines that fit. */
.user {
    align-self: flex-end; max-width: 85%;
    padding: 10px 14px; border-radius: 12px; background: #2d4a7a;
}
.assistant { padding: 4px 2px; }
.error { align-self: center; color: #f08a7e; font-size: 13px; }
.compose {
    display: flex; gap: 8px; padding: 12px 16px;
    border-top: 1px solid #2a2c33; background: #1c1d22;
}
textarea {
    flex: 1; height: 72px; resize: none;
    padding: 10px 12px; border-radius: 10px;
    border: 1px solid #353841; background: #121316; color: #e6e6e9;
    font-family: inherit; font-size: 15px;
}
button {
    align-self: flex-end;
    padding: 8px 14px; border-radius: 10px; border: 1px solid #353841;
    background: #2a2c33; color: #e6e6e9; font-size: 14px;
}
button:hover { background: #33363f; }
button:disabled { opacity: 0.5; }
button.primary { background: #3d6bb3; border-color: #3d6bb3; }
button.primary:hover { background: #4a78c0; }
button.danger { color: #f08a7e; }
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
        turns.write().clear();
        draft.set(String::new());
        error.set(None);
        let e = e.clone();
        spawn(async move { e.wipe().await });
        focus += 1;
    };

    let busy = running.read().is_some();
    let (status_class, status_text) = match &*status.read() {
        Status::NoModel => ("status", "no model loaded".to_string()),
        Status::Loading(name) => ("status", format!("loading {name}…")),
        Status::Ready(d) => ("status", d.clone()),
        Status::Failed(e) => ("status err", format!("failed to load: {e}")),
    };
    let loading = matches!(*status.read(), Status::Loading(_));
    let no_model = matches!(*status.read(), Status::NoModel | Status::Failed(_));

    rsx! {
        style { {CSS} }
        div { class: "app",
            div { class: "bar",
                span { class: "lock", "● private" }
                span { class: "{status_class}", "{status_text}" }
                button {
                    class: if no_model { "primary" } else { "" },
                    disabled: loading,
                    onclick: pick,
                    "Model…"
                }
                button { class: "danger", onclick: wipe, "Wipe" }
            }
            div { class: "log",
                div { class: "turns",
                    if turns.read().is_empty() {
                        div { class: "empty",
                            if no_model {
                                "Choose a model (.gguf) to start."
                            } else {
                                "Nothing here is written to disk. Wipe overwrites the conversation, "
                                "the model's cache and device buffers; closing the window does the same."
                            }
                        }
                    }
                    for (i, t) in turns.read().iter().enumerate() {
                        div {
                            key: "{i}",
                            class: if t.role == Role::User { "msg user" } else { "msg assistant" },
                            if t.text.is_empty() { "…" } else { "{t.text}" }
                        }
                    }
                    if let Some(msg) = error.read().as_ref() {
                        div { class: "error", "error: {msg}" }
                    }
                }
            }
            div { class: "compose",
                for epoch in std::iter::once(focus()) {
                    textarea {
                        key: "{epoch}",
                        autofocus: true,
                        placeholder: "Message  (Enter to send, Shift+Enter for a new line)",
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
                if busy {
                    button { onclick: move |_| stop(), "Stop" }
                } else {
                    button { class: "primary", onclick: move |_| { send.call(()); focus += 1; }, "Send" }
                }
            }
        }
    }
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
