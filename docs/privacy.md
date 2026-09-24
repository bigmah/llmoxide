# Private mode

`llmoxide-private` exists because an engine that writes nothing to disk is not
the same thing as a session being unrecoverable afterwards. Three things hold
the conversation, and `reset` touches none of them: the resident prompt ids kept
for prefix reuse, the device buffers holding everything derived from them, and
the heap copies prompt text passes through in between.

```sh
./target/release/llmoxide-private models/Qwen3.8-27B-Q6_K.gguf
```

```
  /wipe   overwrite the conversation, device buffers and scrollback
  /new    same, but stay in the session
  /image <path> [question]   ask about an image (needs --mmproj)
  /quit   wipe and exit  (ctrl-D also works, ctrl-C stops a reply)
```

With `--mmproj models/mmproj-gemma-4-E4B-it-BF16.gguf`, `/image` takes a local
path and the engine thread reads it, so the pixels never cross into the REPL's
own heap. See [Images](images.md).

What it does that the other entry points do not:

- **Prompts are typed, never passed as arguments.** `llmoxide model "..."` puts
  the prompt verbatim into your shell history — and a history configured with
  `SAVEHIST` raised and `EXTENDED_HISTORY` set keeps it, timestamped,
  indefinitely. Reading stdin skips the shell entirely.
- **No client, so no client-side archive.** This is the one that matters most in
  practice, and the reason this mode exists at all rather than a flag on the
  server.
- **The heap is zeroed as it is freed.** `secret::ZeroizingAlloc` is installed
  as the global allocator, so the copies no wipe could chase — the chat
  template's strings, decoded token pieces, per-token logit vectors — never
  outlive their allocation. `realloc` deliberately falls through to
  alloc + copy + dealloc rather than the system's, which would hand back a
  growing `String`'s old block with the plaintext intact.
- **The prompt ids and the response are `mlock`ed.** This matters more than the
  wipe itself: zeroing a page *after* it has reached swap or
  `/var/vm/sleepimage` does not unwrite it. Locked pages never go there.
- **`secret::harden()`** drops `RLIMIT_CORE` to zero and sets `PT_DENY_ATTACH`,
  closing the two ways this memory is read without touching disk at all.

`wipe` clears device memory as well as host memory, and the engine now wipes
rather than resets whenever a prompt misses the cache — that costs a buffer
clear per miss, tens of milliseconds against a prefill measured in seconds.

## Verifying it

`wipe` is exactly the kind of claim that looks true and isn't: `reset` appears
to clear the KV cache and does not, and a `clear_buffer` that was queued but
never submitted is indistinguishable from the host side. So it is checked:

```sh
./target/release/wipe_check <model.gguf> [prompt]   # exits non-zero on any residue
```

It prefills a prompt, confirms the device buffers are full of it, wipes, and
reads every buffer back. Measured: **4 019 403 non-zero words across 108 buffers
on gemma4, and 41 487 330 across 147 on the 27B — 0 after the wipe on both.** It
refuses to pass vacuously if nothing was resident to begin with.

## What this does not cover

- **Terminal scrollback.** `/wipe` asks the emulator to clear it, which
  Terminal.app and iTerm2 honour, but that is a request, not a guarantee.
  Closing the window is the reliable version.
- **That inference happened.** The GGUF's access time, the GPU at full tilt for
  twenty minutes, the process-launch record in the unified log. What was asked
  can be made unrecoverable; that something was asked cannot.
- **Root on a live machine**, which can read process memory regardless.
- **A `SIGKILL` before the wipe runs** — though `mlock` covers the disk side of
  that case, and the kernel zeroes freed physical pages before reissuing them.
- **Any client you put in front of the server.** See below.

## The desktop app

`llmoxide-app` is the same session with a chat window instead of a terminal,
which also removes the one residue the REPL could only ask about: scrollback.

```sh
cargo run --release -- models/Qwen3.8-27B-Q6_K.gguf  # or LLMOXIDE_MODEL=...
```

It is `default-members`, which is what makes a bare `cargo run` open it — and
also means a bare `cargo build`, `cargo test` or `cargo clippy` covers only the
app. Add `--workspace` for the rest, or `-p` for one package.

One side effect of sharing the lockfile: blitz-dom pins `image = "=0.25.6"`
exactly, so the vision crate decodes with it too (it had 0.25.10, and zune-jpeg
0.5 rather than 0.4). Measured on `test-image.jpg`: 108 of 936 960 channel
values differ, none by more than 2 levels; PNG decodes identically. The
newspaper still reads "MEN WALK ON MOON", and `vision_check` still passes.

Enter sends, Shift+Enter is a new line, **Stop** cuts a reply short, **Wipe**
overwrites the conversation, the model's cache and the device buffers.
**Model…** opens the platform's file dialog (NSOpenPanel, the Windows common
dialog, the XDG portal) on `models/`. Switching wipes and drops the current
model before loading the next, since two large ones rarely fit side by side,
and keeps the conversation: it is text on the UI side, and the next turn
replays it into the new model. With no model argument and no
`models/gemma-4-E4B-it-Q4_K_M.gguf`, the window opens on the picker.
Closing the window, Cmd+Q, Ctrl-C and `SIGTERM` all wipe before the process
exits, and a panic wipes and leaves by `_exit` so it never becomes a macOS
crash report. `LLMOXIDE_CTX`, `LLMOXIDE_BATCH` and `LLMOXIDE_CPU` work as in the
REPL.

The UI is Dioxus with its **native** renderer (Blitz: winit + vello on wgpu),
not the webview one. That choice is the privacy argument. A webview renders in
WebKit's own helper processes, so every message would have a copy outside this
process's zeroed heap, and WebKit keeps its own caches under `~/Library`. Blitz
lays out and paints inside this process, so the text goes from the engine's
channel into a signal, then a DOM node, then shaped glyphs, and each copy sits
on the heap `ZeroizingAlloc` zeroes on free. No JavaScript runs.

The renderer is built with most of its default features **off**, and each one
is a capability the binary does not have. `nm` on the release build finds no
`reqwest`, `hyper`, `tungstenite`, `arboard` or `accesskit`:

| feature | why it is off |
|---|---|
| `net` | an HTTP client for fetching remote resources |
| `accessibility` | publishes every message to the OS accessibility tree, readable by any app granted accessibility access |
| `clipboard` | copied text lands on the shared pasteboard, which clipboard managers keep. Opt back in with `--features clipboard` |
| `file_dialog` | nothing here opens files |
| `hot-reload` | **on**, because dioxus-native 0.7.10 does not compile without it. Its devserver client is compiled only with `debug_assertions`, so release builds have none; debug builds only dial out when `DIOXUS_DEVSERVER_PORT` is set, and `main` unsets it |

On top of [the REPL's list](#what-this-does-not-cover), the app does not cover:

- **The UI's copy of the conversation is zeroed when freed but not `mlock`ed**,
  the same as the REPL's history `Vec`. The engine's copy is locked.
- **Pixels.** Rendered text is in the window's GPU surfaces and the
  compositor's buffers until it is repainted. Wipe repaints.
- **Native crashes** (a GPU driver fault, say) still produce a report in
  `~/Library/Logs/DiagnosticReports`. The report holds a stack and machine
  details, not memory contents, but it is a dated record that the app ran.
- **Input methods.** Keystrokes pass through the OS text input system like
  any app's.
- **The file dialog remembers the last folder.** macOS stores it as a bookmark
  under `NSOSPLastRootDirectory` in
  `~/Library/Preferences/com.apple.ViewBridge.masquerading-service-lacks-host-bundle-identifier.plist`,
  shared by every app without a bundle id. The folder, not the file, and
  nothing from a conversation — but a record of where the models live.
  `defaults delete com.apple.ViewBridge.masquerading-service-lacks-host-bundle-identifier NSOSPLastRootDirectory`
  removes it; passing the model as an argument avoids the dialog entirely.
