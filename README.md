# llmoxide

A private, local chat app for open-weight LLMs. The conversation never leaves
the machine and never touches the disk. When you close the window, it is
overwritten.

```sh
cargo run --release -p llmoxide-hub --bin llmoxide-fetch -- gemma4-e4b-q4   # 5.3 GB, once
cargo run --release                                                        # opens the chat window
```

Everything is written from scratch in Rust with no ML dependencies: the GGUF
loader, k-quant decoders, tokenizers, and wgpu compute kernels. It runs
Gemma 4 (12B and E4B, with image input), Qwen 3.8 27B (hybrid delta-net), and
dense Qwen3. The same engine drives a terminal REPL, a library crate, and a
single-file browser build.

## The chat window

`llmoxide-app` is a native window over the inference engine. Everything runs
in one process, and nothing in it talks to the network.

- **Enter** sends. **Shift+Enter** adds a new line. **Stop** cuts a reply short.
- **Wipe** overwrites the conversation, the model's cache and the GPU buffers.
- **Model…** opens the system file dialog to switch checkpoints. The current
  model is wiped and dropped before the next one loads. The conversation
  carries over and is replayed into the new model.
- With no argument, it opens `models/gemma-4-E4B-it-Q4_K_M.gguf`. If that file
  isn't there, it opens on the picker. You can pass any other checkpoint:

  ```sh
  cargo run --release -- models/Qwen3.8-27B-Q6_K.gguf   # or LLMOXIDE_MODEL=...
  ```

## Privacy

Most local LLM setups are private only in the sense that inference is local.
The conversation still ends up somewhere: in a client's SQLite history, in
shell history, in terminal scrollback, in a webview cache, or in swap.
llmoxide is built so that none of those copies exist in the first place.

| where a conversation usually persists | here |
|---|---|
| a chat client's history database | there is no client; the UI is the engine's own process |
| a webview's helper processes and caches under `~/Library` | the UI is Dioxus's **native** renderer (Blitz on wgpu), not a webview. It lays out and paints in-process, and no JavaScript runs |
| swap and `/var/vm/sleepimage` | the prompt ids and the reply are `mlock`ed, so these pages are never swapped out |
| freed heap blocks | a zeroing global allocator overwrites every allocation when it is freed, including `realloc`'s old block |
| GPU memory | Wipe clears every device buffer. `wipe_check` reads them back to prove it: 41 million non-zero words on the 27B before a wipe, **0 after** |
| shell history | prompts are typed into the window, never passed as arguments |
| core dumps, debuggers, crash reports | `RLIMIT_CORE=0` and `PT_DENY_ATTACH` are set. A panic wipes, then calls `_exit`, so macOS never writes a crash report |
| the pasteboard, the accessibility tree | the renderer's clipboard, accessibility and network features are compiled out. `nm` on the release binary finds no `reqwest`, `hyper`, `tungstenite`, `arboard` or `accesskit` |

Every exit path wipes before the process ends. That covers closing the window,
Cmd+Q, Ctrl-C, `SIGTERM`, and a panic.

Some things are out of scope. The window cannot hide that inference happened:
the model file's access time and the process launch are still recorded. It
cannot protect against root on a live machine. Pixels stay in the compositor
until they are repainted. The file dialog remembers the last folder it opened.
[docs/privacy.md](docs/privacy.md) has the full threat model, what is not
covered, and how each claim is verified.

## Portability

- **A single binary.** The release build of the app is one ~24 MB executable.
  It links only the system frameworks: no Python, no CUDA, no runtime to
  install.
- **A single model file.** Each checkpoint is a single GGUF.
  [`llmoxide-fetch`](docs/models.md) downloads it with resume, checks it
  against its SHA-256, and parses it before the download counts as complete.
  After that, nothing needs the network.
- **A GPU through wgpu.** The WGSL kernels are standard compute shaders. On
  Apple GPUs, a hand-written Metal GEMM speeds up prefill, and the WGSL
  version is used everywhere else.
- **A browser tab.** The same engine compiles to wasm and runs on WebGPU in
  Chrome, Edge and Safari. `scripts/build-web.sh --embed <model>` bakes the
  checkpoint into **one HTML file** that works offline from `file://`, with no
  server. The browser build cannot `mlock`, so it zeroes memory but does not
  lock it. See [docs/browser.md](docs/browser.md).
- **A library.** `llmoxide` is one dependency that re-exports the whole
  workspace. The `private` feature adds the same locked, self-zeroing memory
  the app uses. See [docs/library.md](docs/library.md).

The desktop builds are developed and verified on macOS (Apple M4 Pro, 48 GB).
The privacy layer calls `memset_s` and `PT_DENY_ATTACH` directly, so Linux and
Windows need ports of those calls before the app builds there.

## Models

| alias | checkpoint | size | notes |
|---|---|---|---|
| `gemma4-e4b-q4` | Gemma 4 E4B, Q4_K_M | 5.3 GB | the app's default |
| `gemma4-e4b` | Gemma 4 E4B, Q8_0 | 8.0 GB | |
| `gemma4-e4b-mmproj` | Gemma 4 vision tower | 1.0 GB | image input, see [docs/images.md](docs/images.md) |
| `gemma4` | Gemma 4 12B, Q4_K_M | 7.4 GB | |
| `qwen35` | Qwen 3.8 27B, Q6_K | 22.4 GB | hybrid delta-net / attention |
| `qwen3-0.6b` | Qwen3 0.6B, Q8_0 | 0.6 GB | small enough for a browser download |

Each model is checked tensor by tensor against llama.cpp. The GPU path matches
the CPU reference at every checkpoint, and greedy output is byte-identical to
`llama-completion`. See [docs/correctness.md](docs/correctness.md). On the M4
Pro, gemma4 decodes at 22 tok/s. The 27B prefills at 79 tok/s and decodes at
9.4 tok/s, within ~10% of the memory-bandwidth limit. See
[docs/performance.md](docs/performance.md).

## Other ways in

The window is the workspace's only default member, so a bare `cargo build` or
`cargo test` covers only the app. Add `--workspace` to build everything else:

```sh
cargo build --release --workspace
./target/release/llmoxide-fetch                  # all models, into models/
./target/release/llmoxide-private <model.gguf>   # the same private session, in a terminal
./target/release/wipe_check <model.gguf>         # prove a wipe leaves no residue
./target/release/llmoxide-serve <model.gguf>     # OpenAI-compatible API — not private
```

The server is a side feature, and it is **not private**. Any client you point
at it keeps its own transcript. See [docs/serving.md](docs/serving.md).

## Docs

- [privacy.md](docs/privacy.md): private mode, the desktop app, and what they do not cover
- [models.md](docs/models.md): checkpoints, `llmoxide-fetch`, and per-architecture traps
- [correctness.md](docs/correctness.md): validation against llama.cpp, and the tools that reproduce it
- [performance.md](docs/performance.md): measurements, the GEMM, and KV cache reuse
- [images.md](docs/images.md): the gemma4 vision tower
- [browser.md](docs/browser.md): the wasm/WebGPU build and the single-file page
- [library.md](docs/library.md): `Session`, `Backend`, and feature flags
- [serving.md](docs/serving.md): the HTTP API, tool calls, and opencode
- [layout.md](docs/layout.md): crates and their dependencies
- [limitations.md](docs/limitations.md): known limitations
- [ARCHITECTURE.md](ARCHITECTURE.md) and [ARCHITECTURE-qwen35.md](ARCHITECTURE-qwen35.md): the architectures' traps
