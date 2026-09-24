# Workspace layout

Every package is `llmoxide-*`; the directory keeps the short name, and so does
the code, because each dependency is renamed back at the `Cargo.toml` line that
declares it. Generic package names like `model` or `gpu` would squat the
namespace of anything that depends on this.

```
directory         package             what it is
llmoxide/         llmoxide            the facade: Session, Backend, and the CLI
crates/gguf       llmoxide-gguf       GGUF v3 reader, mmap'd; Q4_K / Q6_K / Q8_0 decoders
crates/tokenizer  llmoxide-tokenizer  gemma4 BPE (262144 tokens) + qwen35 byte-level BPE (248320)
crates/model      llmoxide-model      architecture configs, CPU reference forward passes, sampling
crates/gpu        llmoxide-gpu        wgpu device, weight arena, WGSL kernels, GPU forwards + vision tower
crates/chat       llmoxide-chat       prompt assembly: gemma4's tool DSL + qwen's ChatML/XML
crates/vision     llmoxide-vision     gemma4v: image preprocessing and the CPU reference tower
crates/secret     llmoxide-secret     locked, self-zeroing memory; the zeroing global allocator
crates/hub        llmoxide-hub        resumable, verified Hugging Face downloads
crates/server     llmoxide-server     the private REPL, plus the axum OpenAI-compatible API
crates/app        llmoxide-app        the private chat window: Dioxus, native renderer, in-process
crates/wasm       llmoxide-web        the browser build: WebGPU, streamed weights, chat REPL
web/              —                   the page shell; build-web.sh emits llmoxide.html into it
```

Dependencies point one way: `gguf` and `secret` at the bottom, `llmoxide` at
the top, and `llmoxide-server` only on `llmoxide`. Nothing below the facade
knows about HTTP, and the generation loop no longer lives in the server crate,
so embedding inference does not compile axum.

`crates/wasm` is deliberately **not** a workspace member — it only ever builds
for `wasm32-unknown-unknown`, and membership would pull wasm-bindgen and
web-sys into every native `cargo build`. It is also the one entry point that
does not go through `llmoxide::Session`: its forward pass is `async`, because a
browser's main thread may not block on a buffer map, and `Backend` is a
blocking trait.
