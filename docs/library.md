# Using llmoxide as a library

The workspace is consumable as a library. `llmoxide` is the crate to depend on:
it re-exports every other one, so a consumer adds a single dependency and gets
versions that cannot drift apart.

```toml
[dependencies]
# Inference, and nothing else.
llmoxide = { git = "https://github.com/bigmah/llmoxide" }

# ...or with the privacy guarantees `llmoxide-private` is built on.
llmoxide = { git = "https://github.com/bigmah/llmoxide", features = ["private"] }
```

```rust
use llmoxide::{LoadOptions, Request, Session};

let mut session = Session::load("models/Qwen3-0.6B-Q8_0.gguf", &LoadOptions::default())?;
let out = session.complete(Request::user("what is the capital of France?"))?;
println!("{}", out.completion.content);
```

Streaming is the same call with a callback; return `Flow::Stop` to cut a
generation short:

```rust
session.generate(Request::user("hello"), |piece| {
    print!("{piece}");
    Flow::Continue
})?;
```

Two runnable examples, both of which work against the 0.6B checkpoint:

```sh
cargo run --release -p llmoxide --example generate    -- models/Qwen3-0.6B-Q8_0.gguf "hello"
cargo run --release -p llmoxide --example chat_stream -- models/Qwen3-0.6B-Q8_0.gguf
```

## The layers

| | |
|---|---|
| `llmoxide::Session` | tokenizer, chat dialect, sampling, prefix reuse, tool-call filtering |
| `llmoxide::backend::Backend` | one trait per (architecture, device): token ids in, logits out |

Reach for `Backend` directly when the chat layer is in the way — that is what
the `llmoxide` binary does, since a sampler between you and the logits defeats
the point of a reference check. It is also where a new architecture is added,
and where multimodal input will arrive: `Backend::forward_embeds` takes
already-projected embedding rows, and `Session` prefills a `Prompt` of
interleaved `Segment::Tokens` and `Segment::Embeds`, so an `mmproj` encoder
plugs in without touching the generation loop. No architecture implements it
yet; the default returns `Error::Unsupported`.

## Features

| feature | | |
|---|---|---|
| `gpu` | default | the wgpu backends. Off, the crate still reads checkpoints and runs the CPU reference paths, and does not build wgpu at all (9 fewer crates). |
| `hub` | | resumable, hash-verified checkpoint downloads. |
| `private` | | locked, self-zeroing conversation memory. |
| `vision` | | image input: the gemma4v tower plus `image` for decoding. Off by default — a text-only caller should not pay for it, and the browser build cannot use it. |

**`private` is off by default, and a plain dependency gets none of the privacy
machinery** — `llmoxide-secret` does not appear in the tree at all, not even
transitively. That is deliberate rather than an oversight about what this
project is for. Locked memory is not free for a caller who only wants
inference: `SecretVec` allocates page-aligned, zeroes on every reallocation,
and prints a warning to stderr every time `mlock` is refused, which is every
time under a low `RLIMIT_MEMLOCK` or in a container without `IPC_LOCK`. A
library has no business writing to stderr on a machine that never asked for
the guarantee — and an inference dependency that declares `mlock`, `ptrace`
and `PT_DENY_ATTACH` is a bad surprise in someone else's audit.

What the feature does *not* gate: `Backend::wipe` always clears the KV cache,
the recurrent state and the device buffers, because prefix reuse depends on
it. `private` decides whether those overwrites are `memset_s`-with-a-fence
rather than an ordinary `fill`, and whether the resident token ids — the
conversation itself, decodable straight back to plaintext — are locked into
RAM. `llmoxide::PRIVATE_MEMORY` reports which build you have, so a caller that
needs the guarantee can assert it instead of assuming it. `llmoxide-private`
does exactly that, as a `const` assertion: the build fails rather than ship a
banner promising locked memory it does not have.

## What it will not do

One `Session` is one conversation on one GPU context: no batching, and no
sharing a loaded model between threads. A backend is `Send` but not `Sync`, so
serving several callers means a queue in front of one session — which is
exactly what `llmoxide-server` is.

The CPU reference backends leak their `Gguf`, config and weights to `'static`
rather than threading a self-referential borrow through three types, so a
process gets one CPU model for its lifetime. The GPU backends upload and drop
the mapping, and have no such limit.
