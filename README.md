# llmoxide

A local inference CLI that leaves nothing behind — GGUF loader, k-quant
decoders, tokenizers, and wgpu compute kernels written from scratch in Rust with
no ML dependencies. Three architectures: gemma4, qwen35's hybrid delta-net
stack, and plain dense Qwen3. The same code also compiles to wasm and runs in a
browser on WebGPU — see [In a browser](#in-a-browser).

The point of it is `llmoxide-private`: a session that writes nothing to disk,
keeps the conversation in locked memory, and can overwrite every trace of it on
demand. There is an OpenAI-compatible server too, but it is a side road and it
is **not private** — see [Serving](#serving-afterthought-and-not-private).

Built for one user on one machine (Apple M4 Pro, 48 GB), so it runs one request
at a time against one GPU context.

```sh
cargo build --release
./target/release/llmoxide-fetch                                # get the models
./target/release/llmoxide-private models/Qwen3.8-27B-Q6_K.gguf # use them
```

```
»  what is the capital of France?
Paris
»  /wipe
wiped: conversation, device buffers, locked pages.
```

## Models

Four checkpoints, each validated against tensor-by-tensor. All run on the GPU;
the CPU forward passes stay in the tree as the reference every kernel is checked
against, not as a fallback.

| alias | file | arch | size | sha256 |
|---|---|---|---|---|
| `gemma4` | `gemma4-v2-Q4_K_M.gguf` | gemma4, 12B, 48 layers | 7.38 GB | `0b9506ca…` |
| `gemma4-e4b` | `gemma-4-E4B-it-Q8_0.gguf` | gemma4, E4B, 42 layers | 8.03 GB | `34be82b1…` |
| `gemma4-e4b-q4` | `gemma-4-E4B-it-Q4_K_M.gguf` | gemma4, E4B, 42 layers | 5.34 GB | `d35a3aa7…` |
| `qwen35` | `Qwen3.8-27B-OBLITERATED-Q6_K.gguf` | qwen35, 27B hybrid, 64 layers | 22.43 GB | `3535d4a1…` |
| `qwen3-0.6b` | `Qwen3-0.6B-Q8_0.gguf` | qwen3, 0.6B dense, 28 layers | 0.64 GB | `e150ed54…` |

From [`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
and [`OBLITERATUS/Qwen3.8-27B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
respectively, and `gemma4-e4b` from
[`ggml-org/gemma-4-E4B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF).
`qwen3-0.6b` is the one small enough to *serve*: 0.64 GB, from
[`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF).
It is Q8_0 rather than a Q4 because at 0.6B the quantization error is the first
thing you notice, and 0.64 GB is already inside any sane download budget. There
is nothing smaller worth having in either family — see
[Why not a smaller Gemma](#why-not-a-smaller-gemma).

`gemma4-e4b-q4` is the same E4B at a size a browser tab will actually allocate,
from [`lmstudio-community`](https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF) —
that repo rather than the obvious ones because a GGUF is only loadable here if
*every* tensor is a type the decoders handle. ggml-org publishes E4B as Q8_0 or
Q4_0, and there is no Q4_0 decoder; bartowski's Q4_K_M mixes in 84 Q5_K
tensors, which there is also no decoder for. Both are rejected at load rather
than part-way through a forward pass, which is what the structure check in
`llmoxide-fetch` is for — it caught the second one after the download had
already passed its SHA-256.
`qwen35` is the interesting one architecturally: a hybrid stack where three
quarters of the layers are gated delta-net rather than attention.

Plain **qwen3** is not a separate implementation. Read as a config, it *is*
qwen35 with two things switched off: every layer is full attention rather than
delta-net, and `attn_q` carries no fused output gate, so it is `head_dim` wide
per head instead of `2 * head_dim`. Everything else — RMSNorm, GQA, per-dim
query and key norms, NeoX RoPE, SwiGLU — is the same code, so `qwen3` costs one
`Config::query_gate` flag and three branches rather than a second copy of an
attention block that already worked. The 27B is re-checked against the CPU path
at all 771 checkpoints after every change to it.

Two Qwen traps came with that. Its vocabulary uses the **classic `qwen2`
pre-tokenizer split**, which is the qwen35 one with every `\p{M}` term removed —
so combining marks no longer travel with the letters they modify. Picking the
wrong split does not fail, it just tokenizes subtly differently everywhere;
`pre::Marks` makes it one function with marks switched off rather than two
hand-ports that could drift. And Qwen3 ships **no `general.sampling.*`
metadata**, so a single hard-coded fallback would sample it with Gemma's
temperature of 1.0 against Qwen's published 0.7 — which on a 0.6B model reads
as the model being weak rather than the sampler being wrong.
`Sampling::family_default` picks per family.

One chat-format trap comes with it. The two checkpoints disagree on how to turn
*off* reasoning: the 12B's template suppresses it by making the generation prompt
open and immediately close an empty `thought` channel, and E4B's template has no
such suppressor. Emitting the 12B's form to E4B does not disable thinking — the
model finds the channel already closed, never opens another, and writes its
reasoning into the visible answer. `Special::closes_empty_thought` reads which
convention a checkpoint uses off its embedded template rather than assuming.

`gemma4-e4b` is the same architecture as the 12B only in name. The E-series
keeps a narrow 2560-wide residual stream and spends its parameters on a
*per-layer embedding* table instead — a 256-wide vector looked up per token per
layer, gated into the stream at the end of every block. That is what "E4B"
means: 4.5B effective parameters out of 8B on disk. It also shares KV across
the top of the stack, so layers 24–41 project Q only and attend into the cache
of layer 22 (sliding) or 23 (global), and it writes `head_count_kv` as a scalar
where the 12B writes a 48-entry array. Config reads all of this out of the GGUF;
see [`crates/model/src/config.rs`](crates/model/src/config.rs).

```sh
./target/release/llmoxide-fetch            # all three, into models/
./target/release/llmoxide-fetch gemma4     # or one
./target/release/llmoxide-fetch hf:owner/repo/file.gguf
./target/release/llmoxide-fetch https://huggingface.co/owner/repo/blob/main/f.gguf
```

Transfers **resume** — interrupt one, run the same command again, and it picks
up from the byte it stopped at. Everything reports progress, the SHA-256 passes
included, because at 21 GB a verification with no output is indistinguishable
from a hang.

Nothing is called finished until three things agree: the size matches
`x-linked-size`, the SHA-256 matches `x-linked-etag`, and `gguf::Gguf::open`
parses the result — which walks the tensor table and rejects any tensor running
past the end of the file. Only then does the `.part` file take its real name, so
an interrupted fetch can never be mistaken for a complete one. This repo already
lost time to a download that stopped at 17.8 of 20.9 GiB and wrote a
plausible-looking file; that is the check which would have caught it.

If the same checkpoint is already in the directory under a different name it is
hashed and hard-linked rather than downloaded again (`--no-adopt` to disable).
Re-uploads get renamed constantly, and re-fetching 21 GB you already have is the
most expensive mistake available here.

Two traps worth knowing if you touch `crates/hub`:

- A `/blob/` URL — what the website's copy button gives you — returns HTML. It
  is rewritten to `/resolve/`, otherwise you download a few kilobytes of markup
  that looks like a corrupt model.
- A resolve URL 302s to a CDN, and `x-linked-size` / `x-linked-etag` live on
  *that 302*, not on what it points at. Follow the redirect and you get the
  CDN's `etag` instead — the xet content hash, which is also 64 hex characters
  and is not the SHA-256 of the file. Verifying against it fails every honest
  download, at the end, after 21 GB. `probe` stops at the redirect for exactly
  this reason.

## Private mode

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
  /quit   wipe and exit  (ctrl-D also works, ctrl-C stops a reply)
```

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

### Verifying it

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

### What this does not cover

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

## Correctness

Both architectures are genuinely unusual — see [ARCHITECTURE.md](ARCHITECTURE.md)
for the four things that will silently produce garbage if you assume the
Gemma 2/3 shape, and [ARCHITECTURE-qwen35.md](ARCHITECTURE-qwen35.md) for the
hybrid stack's own traps (head tiling, l2-norm eps, the fused query gate).
Everything is checked against llama.cpp rather than asserted:

| what | check |
|---|---|
| gemma4 tokenizer | exact id-for-id match with `llama-tokenize` on 13 cases + a 3547-token file |
| gemma4 CPU forward | **byte-identical** greedy output to `llama-completion --temp 0` |
| gemma4 GPU kernels | every matvec within ~1e-7 of the CPU dequant-dot, on real weights |
| gemma4 GPU forward | all 773 intermediate tensors match the CPU path across 48 layers (~1e-6) |
| E4B CPU forward | 674 tensors across all 42 layers traced against `llama-eval-callback`; KV-sharing boundary matches exactly (llama.cpp emits `Kcur` for layers 0–23 only) |
| E4B GPU forward | **471/471** checkpoints match the CPU path across 42 layers (~1e-6), per-layer embeddings included |
| E4B chat format | `reasoning_content` / `content` split matches `llama-server --jinja` on the same request, thinking on and off |
| qwen35 tokenizer | exact id-for-id match with `llama-tokenize` on 18 cases + 3 files (~11k tokens) |
| qwen35 CPU forward | 567 tensors across all 64 layers match `llama-eval-callback` within 4e-4 on a real 27B; **byte-identical** greedy output to `llama-completion --temp 0` |
| qwen35 GPU forward | **771/771** checkpoints match the CPU path on the 27B (logits rel 1.3e-6, same argmax) |
| qwen35 chat format | prompt ids match a jinja2 rendering of the embedded template, with and without tools |
| qwen3 tokenizer | exact id-for-id match with `llama-tokenize` on 21 cases and 6 files (30 610 tokens), with `--no-escape`; the only divergences are literal `<think>`/`<tool_call>` spellings in raw text, which are deliberately not matched as control tokens and which the validated qwen35 path treats the same way |
| qwen3 CPU forward | **byte-identical** greedy output to `llama-completion --temp 0` on 6 prompts / 288 tokens, code and prose |
| qwen3 GPU forward | **339/339** checkpoints match the CPU path across 28 layers, same argmax |
| qwen3 chat format | single-turn prompt ids match a jinja2 rendering of the embedded template exactly; multi-turn deliberately differs, see below |
| browser kernels | the WGSL rewrites for Tint leave `bisect` at **all checkpoints match** on E4B Q4_K_M, and `kernels` matching the CPU dequant-dot on *both* reduction paths (`LLMOXIDE_NO_SUBGROUP=1` is the one the browser takes) |

Tools that reproduce this:

```sh
./target/release/kernels        models/gemma4-v2-Q4_K_M.gguf   # per-kernel vs CPU
./target/release/bisect         models/gemma4-v2-Q4_K_M.gguf   # per-layer GPU vs CPU
./target/release/bisect_qwen35  <model> [ids]                  # per-checkpoint GPU vs CPU
./target/release/validate_qwen35 <model> <refs.json> <ids>     # vs llama-eval-callback
./target/release/upload_check   <model>                        # weight arena readback
./target/release/wipe_check     <model> [prompt]               # wipe leaves no residue
./target/release/tok            <model> [text]                 # ids, vs llama-tokenize
./target/release/prompt         <model> < messages.json        # chat ids, vs a jinja render
```

`bisect` reports the *first* diverging checkpoint, which is how the NaN in
GeGLU and the attention-scale error were both found. The qwen35 workflow
(including a synthetic-checkpoint generator for fast whole-graph checks) is in
[ARCHITECTURE-qwen35.md](ARCHITECTURE-qwen35.md) and `scripts/`.

One caveat on provenance: the tensor-level qwen35 numbers above were measured
against `Qwen3.8-27B-Uncensored-Cyber-Q6_K.gguf`, which carries no NextN block.
The checkpoint in `models/` is SHA-256 identical to the OBLITERATED build in the
table above — same architecture plus one NextN block, which the loader skips —
but has not been put back through `validate_qwen35` since.

## Performance

On an M4 Pro (20 GPU cores), measured, not projected. Decode is quoted with the
context it ran at, because attention cost grows with it:

| gemma4 | before | now | llama.cpp (Metal) |
|---|---|---|---|
| decode @ 128 ctx | 12.7 tok/s | **22.3 tok/s** | 32.6 tok/s |
| decode @ 512 ctx | 8.6 tok/s | **18.4 tok/s** | |
| decode @ 2048 ctx | 5.0 tok/s | **12.9 tok/s** | |
| prefill, 374 tokens | 23.5 tok/s | **35.0 tok/s** | much higher |
| prefill, 1490 tokens | 20.7 tok/s | **32.7 tok/s** | |

qwen35 27B decodes at **~9.4 tok/s** on the GPU, against ~1.1 tok/s on the CPU
reference path — about 9x, and the difference between patient chat and
correctness work only.

Greedy output is unchanged token-for-token, and `bisect` still matches the CPU
path at every checkpoint.

Four things got gemma4 there, in rough order of how much they were worth:

- **Naga's loop-termination counters are off** (`Gpu::shader`). Naga wraps every
  loop in a decrementing 64-bit guard, which costs ALU in the innermost loop and
  hides constant trip counts from the Metal compiler. Bounds checks stay on;
  only the guard is dropped, and every loop here is bounded by a tensor
  dimension.
- **The token tile is small, not large.** `acc` is indexed by a loop variable,
  so it lives in per-thread scratch rather than registers, and its footprint is
  what caps occupancy. Going from 32 tokens to 2 made prefill 2.5x faster while
  multiplying weight traffic by 16 — so the matvec is bound by occupancy, not
  bandwidth, which is the opposite of what this file used to claim.
- **Attention reads Q/K/V as `vec4`**, with Q staged in workgroup memory instead
  of re-read per key. Those passes are one FMA per element; a scalar load per
  element doubled the work. Worth ~2x on decode at 2048 context.
- **`subgroupAdd` replaces the barrier tree** in the matvec row reduction. Worth
  a few percent on its own — the earlier note that Naga can't do this is wrong.
  Naga rejects `enable subgroups;`, but the builtins compile without the
  directive and the Metal backend emits `simd_sum`. `Gpu::new` still probes the
  hardware for the subgroup width and falls back to the barrier tree if it
  isn't 32; `LLMOXIDE_NO_SUBGROUP=1` forces that path so it stays tested.

What is still on the table: decode cost continues to grow with context because
the three attention passes only have `n_heads` workgroups between them, which
cannot fill the GPU — splitting the key range across workgroups and reducing the
partials afterwards is the fix. Prefill wants a real GEMM that stages the
activation tile in workgroup memory, rather than the current one-row-per-lane
matvec reused across a small token tile.

Two dead ends are written up in `crates/gpu/src/shaders/quant.wgsl` so they
don't get retried: hoisting the dequantization above the token loop, and
widening the tile. Both look like obvious wins and both lose.

Most of that was settled by reading the generated Metal rather than guessing —
whether an accumulator reaches a register is not visible in the WGSL:

```sh
./target/release/msl              # quant kernels as MSL, as wgpu compiles them
./target/release/msl --barrier    # ...with the non-subgroup reduction
```

### Two silent failures at 27B scale

Both return all-zero buffers with no error, and both are guarded now — details
in [ARCHITECTURE-qwen35.md](ARCHITECTURE-qwen35.md):

- **Uploads past the working set.** Mapping every arena buffer at once doubles a
  25 GB model with staging shadows and the writes are simply dropped.
  `Weights::upload` fills and flushes one buffer at a time and reads sampled
  spans back; `upload_check` sweeps every tensor head.
- **Over-long command buffers.** A whole forward pass is ~1500 dispatches, which
  trips Metal's GPU watchdog on cold pipelines and long prefills.
  `Qwen35Gpu::run` chunks at 64 dispatches per submit, and a device-lost
  callback in `Gpu::new` turns any future watchdog kill into a printed message.

### KV cache and prefix reuse

Sliding-window layers get a 1024-slot ring instead of a full-context
allocation — with 40 of 48 layers windowed, that is the difference between
~2 GB and ~90 GB at gemma4's full 262144-token context. qwen35's recurrent
layers carry state of a fixed size instead: a conv window plus one s_dim ×
s_dim matrix per v-head, regardless of context length.

The cache is reused when a new prompt strictly extends what is resident.
Rewinding is deliberately not attempted: the ring has already overwritten the
positions a rewind would need, and the delta-net state has no history to roll
back to at all. To make the common agentic case hit this path, replayed
assistant turns reproduce the empty thought channel the generation prompt
emits — otherwise every turn diverges from the cache at the first assistant
message and re-prefills the whole conversation.

This is the one place the prompt deliberately departs from the checkpoints'
own templates, and it applies to Qwen too. Qwen3's template strips the think
block from assistant turns before the last user message; the 27B's emits none
at all. Both would diverge from what the model actually generated, because the
generation prompt that produced those turns *ended* with
`<think>\n\n</think>\n\n`. Replaying it is both closer to the model's real
context and the only version that keeps the cache. Single-turn prompts, where
the question does not arise, match the template id-for-id.

## One-shot generation

```sh
./target/release/llmoxide models/gemma4-v2-Q4_K_M.gguf "The capital of France is" 8
./target/release/llmoxide models/Qwen3.8-27B-Q6_K.gguf "..." 8
./target/release/llmoxide models/Qwen3.8-27B-Q6_K.gguf "..." 8 --cpu   # reference path
```

Useful for diffing against `llama-completion`, which is what it is there for.
Note that the prompt is an argument, so it lands in your shell history —
`llmoxide-private` if that matters.

## Serving (afterthought, and not private)

```sh
./target/release/llmoxide-serve models/gemma4-v2-Q4_K_M.gguf   # http://127.0.0.1:8080
```

`LLMOXIDE_CTX` (default 16384), `LLMOXIDE_PORT` (8080), `LLMOXIDE_BATCH` (256).
`LLMOXIDE_CPU=1` puts qwen35 on its CPU reference path; `LLMOXIDE_NO_SUBGROUP=1`
forces the barrier-tree row reduction.

**This is not a private session, and cannot be made into one from this side of
the socket.** The server keeps its own hands clean — locked and zeroed prompt
ids, the zeroing allocator (`LLMOXIDE_NO_ZEROIZE=1` to disable), no core dumps,
`POST /v1/wipe` to overwrite the resident conversation, and no request body in
the log. But whatever you point at it usually keeps a transcript, and that is
where the conversation actually persists. opencode writes every message in
plaintext to `~/.local/share/opencode/opencode.db`, has no option to turn that
off, and builds `export` / `import` / `stats` on top of it.

If you want the server anyway, put the client's store somewhere that does not
survive a reboot — opencode honours `XDG_DATA_HOME`, so a RAM disk works:

```sh
DISK=$(hdiutil attach -nomount ram://1048576)      # 512 MB
newfs_hfs -v ocram "$DISK" && mkdir -p /tmp/ocram
mount -t hfs "$DISK" /tmp/ocram
XDG_DATA_HOME=/tmp/ocram opencode
```

`hdiutil detach "$DISK"` ends it. This is a weaker guarantee than private mode
gives: those are ordinary pages, not `mlock`ed, so they can still reach
(encrypted) swap.

Two smaller notes. `PT_DENY_ATTACH` is opt-in here (`LLMOXIDE_PRIVATE=1`) rather
than automatic, because it blocks profilers and a server is the thing you
profile. And removing the request-body log was not enough on its own:
`serde_json` quotes the offending value inside its own error message, so
`ApiError` carries a detailed message for the client and a sanitized one —
category, line, column — for the log.

### opencode

`opencode.json` in this repo points opencode at the local server (port 8081 —
start the server with `LLMOXIDE_PORT=8081` or edit the `baseURL`). It needs the
provider package once:

```sh
cd ~/.config/opencode && npm install @ai-sdk/openai-compatible
```

Then `opencode run --model llmoxide/Qwen3.8-27B-Q6_K "..."`, or copy the
`provider` block into `~/.config/opencode/opencode.json` to use it anywhere.

### API

`GET /v1/models`, `POST /v1/chat/completions` (streaming and not), `POST /v1/wipe`,
`GET /health`.

Supports `tools`, `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`,
`stop`, and a non-standard `enable_thinking` for the model's thought channel
(off by default).

Tool calls are the interesting part, and each model has its own wire format.
gemma4 does not emit JSON — it uses a custom DSL where strings are delimited
by the single token `<|"|>`:

```
<|tool_call>call:read_file{path:<|"|>src/main.rs<|"|>,limit:20}<tool_call|>
```

qwen35 wraps an XML-ish block in `<tool_call>` control tokens
(`<function=read_file>` / `<parameter=path>`); `crates/chat/src/qwen.rs`
translates that, `crates/chat/src/dsl.rs` the gemma4 DSL. Two deliberate
behaviours shared by both:

- Parsing works on **token ids**, not decoded text, because the quote marker and
  channel markers are control tokens a text decoder drops.
- Calls naming a tool the request never declared are **rejected**. The model
  will occasionally invent one, and a client that dispatched it would either
  error out or run something unintended.

## In a browser

The same engine, compiled to wasm32 and pointed at WebGPU instead of Metal.
One self-contained HTML file — the wasm module is baked into it — and the
checkpoint either downloads once and caches, or is read off your own disk.
Either way it goes straight into GPU memory and is never uploaded anywhere.

```sh
scripts/build-web.sh              # -> web/llmoxide.html, ~1.2 MB
open web/llmoxide.html            # or serve it; both work
```

To host it, put `llmoxide.html` and a `.gguf` on any static host and point
`MODEL_URL` at the checkpoint (it defaults to `./Qwen3-0.6B-Q8_0.gguf`, i.e.
the file sitting next to the page). The download is one ordinary GET — no Range
support needed — so a plain bucket or CDN is enough, and it lands in the Cache
API so a repeat visit starts instantly. `?model=<url>` overrides it for
testing.

| checkpoint | delivery | load | decode |
|---|---|---|---|
| `qwen3-0.6b` | 0.64 GB download | 3.0 s | 32–115 tok/s |
| `gemma4-e4b-q4` | 5.34 GB local file | 5.0 s | 24–30 tok/s |

Verified in **Chrome 141 and Safari 26.6** on the M4 Pro. Safari matters
architecturally, not just as a checkbox: it reports `maxBufferSize` of 2.15 GB
against Chrome's 4.29, which is exactly why `Gpu::arena_buffer_bytes` takes the
number from the device rather than a constant. Its WGSL compiler is also a
third implementation after Naga and Tint, and it accepts the kernels unchanged.

### One file, model included

```sh
scripts/build-web.sh --embed models/Qwen3-0.6B-Q8_0.gguf
```

Bakes the checkpoint into the page: **854 MB of HTML**, opens from `file://`,
no server and no network. Ready in 2.1 s in Chrome and comparable in Safari.

The base64 has to arrive in pieces. V8 caps a single string at 536,870,888
characters and this model's base64 is 852,596,992 — so one blob is not slow,
it is unbuildable. `build-web.sh` emits 48 MB chunks and the page drops each
from the DOM as it decodes, since those strings are the largest objects on it.

**Compressing the model buys nothing.** Quantized weights are close to random:
measured on this Q8_0, `gzip -9` gets 4.5% off and `zstd -19` 4.8%, which does
not pay for a decompressor in the page. What *is* worth doing is serving the
page with `Content-Encoding: gzip`, which takes the embedded build from 854 MB
to **638 MB over the wire** — the base64 tax refunded almost exactly, for one
line of server config and no code.

Even so, prefer two files for a website. The embedded build cannot show
download progress (nothing runs until the whole document is parsed), re-parses
854 MB on every visit, and puts the model outside the Cache API. It is for
handing someone a single file that works offline.

```
»  My favourite colour is teal. Just acknowledge that briefly.
Teal is a lovely colour!
»  What is my favourite colour?
Your favourite colour is teal.
»  /wipe
wiped: conversation, device buffers, locked pages.
»  What is my favourite colour?
I do not know your favorite color.
```

Measured in Chrome on the M4 Pro. `gemma4-e4b-q4` puts **5.54 GB of weights
resident in 5.0 s** — 1.1–1.2 GB/s from disk through the browser to the GPU —
and decodes at the same order as the native build's 22.3 tok/s, which is less
surprising than it sounds: the kernels are identical. WGSL is WebGPU's own
shading language, so `crates/gpu/src/shaders` ships to the browser unchanged
rather than being translated.

Needs WebGPU: Chrome or Edge 113+, or Safari 26+. There is no fallback, and
that is not laziness — see below.

### Why not a smaller Gemma

There isn't one. The smallest Gemma 4 is E2B, and its smallest GGUF at any
quantization is 2.29 GB; **66% of that file is embeddings**, because the vocab
is 262144 and the E-series multiplies it by depth — its per-layer embedding
table alone is 262144 x 256 x 35 = 2.35 B parameters. Push every weight in E2B
to two bits and the floor is still 1.16 GB. The E-series spends bytes to save
FLOPs, which is the right trade for a phone and the wrong one for a download.

Qwen3 0.6B is the way under a gigabyte, and it is why `qwen3` is supported at
all. Do not expect much of it — it is a 0.6B model, fine for short exchanges
and visibly limited beyond that — but it is coherent, it streams fast, and it
fits.

### Why the weights never enter wasm memory

wasm32 addresses 4 GB. The checkpoint is 5.3. So a CPU forward pass in the
browser is not slow, it is *impossible* — there is nowhere to put the weights,
at any quantization that leaves the model worth running.

The loader is therefore split. `gguf::Header` is everything but the tensor
payload, and parses from a **prefix** of the file: a few megabytes gives every
tensor's type, shape and offset, which is enough to plan the whole upload
before a weight byte has moved. `Weights::upload_streaming` then walks that
table pulling 32 MB at a time out of the JS `File` and writing each chunk
straight into a GPU buffer. Peak host usage is one chunk. `gguf::Source::Sparse`
holds the handful of F32 norms that model construction actually reads by name,
and returns a *typed error* for anything else rather than a wrong slice — which
is how the one place that read an 800 MB tensor merely to learn its `ne[1]`
turned up on the first run.

Two things follow from that split, and both are load-bearing:

- `Blob::slice` must be the `f64` overload. The `i32` one saturates past 2 GB —
  a third of the way in — and every tensor after that point would load from the
  clamped offset with no error anywhere.
- Buffer sizes come from the device, not from a constant. Native adapters here
  report a 30 GB `max_buffer_size`; WebGPU's *default* is 256 MB, and the
  adapter maximum is commonly 2 GB. (This M4 Pro offers 4.29 GB, so the model
  lands in two buffers.)

### Two shaders that compile natively and not in a browser

Both were found by running it, both produced no error at the point of failure,
and both are now caught up front by `Gpu::check_shaders` — which asks for
compilation messages *before* the 5.5 GB upload rather than after.

- **Naga and Tint disagree about uniformity.** The matvec kernels stride rows
  across the grid with `if (row - row_in_wg >= p.out_dim) { break; }`. The
  `row_in_wg` cancels, so every thread runs the same number of iterations and
  reaches the reduction's `workgroupBarrier` together — which is what makes the
  barrier legal. Naga accepts this. Tint will not do the algebra, sees a bound
  derived from `local_invocation_id`, and rejects the whole module. Keeping the
  loop variable as the workgroup's *base* row fixes it and changes nothing.
  Native never noticed because native takes the `subgroupAdd` path, which has
  no barrier at all; the barrier fallback is only reached with
  `LLMOXIDE_NO_SUBGROUP=1`.
- **`-3.4028235e38` is not a valid f32 literal in WGSL.** It is what Rust
  prints for `f32::MIN`, and it round-trips in Rust. WGSL parses literals as
  abstract float first, where `3.4028235e38` is larger than `f32::MAX`, so the
  conversion overflows. `bitcast<f32>(0xff7fffffu)` means one thing everywhere.

The symptom in both cases was the same and is worth recognising: WebGPU does
not fail `create_shader_module`, it reports asynchronously. So pipelines are
created invalid, every dispatch against them is silently dropped, and the first
visible sign is a reply made of `<unused12><unused35>` ninety seconds later.

### What does not survive the port

- **`mlock` does not.** A wasm module's memory is a JS `ArrayBuffer` the host
  may move, page or snapshot at will, and nothing inside the sandbox can pin
  it. `secret::sys`'s wasm shims say so rather than quietly returning success
  for a lock that never happened. The zeroing allocator *is* real — `memset_s`
  becomes a volatile write loop, which is the same guarantee by another route —
  so `/wipe` still overwrites the conversation, the heap blocks it passed
  through, and every device buffer. It cannot reach the tab's own heap
  snapshot, and devtools can read this memory regardless.
- **The 27B does not** — not for any code reason, it simply will not fit. The
  qwen35 module itself compiles to wasm and is what runs Qwen3 0.6B there.
- **The CPU reference path does not** — see above. It still compiles (rayon
  swapped for a serial shim, since real wasm threads need `SharedArrayBuffer`
  and so COOP/COEP headers, which a local file has no way to set), but there is
  no memory for it to run in.
- **Checkpoint capture does not.** `bisect` and `wipe_check` read buffers back
  synchronously, and a browser's main thread may not block on a buffer map.
  Everything on the hot path went async instead: `GpuModel::forward_async` is
  the same dispatch as `forward`, differing only in how it waits — which is
  also what keeps the page responsive and lets tokens paint as they arrive.

## Layout

```
crates/gguf       GGUF v3 reader, mmap'd; Q4_K / Q6_K / Q8_0 decoders
crates/tokenizer  gemma4 BPE (262144 tokens) + qwen35 byte-level BPE (248320)
crates/model      architecture configs, CPU reference forward passes, sampling
crates/gpu        wgpu device, weight arena, WGSL kernels, both GPU forwards
crates/chat       prompt assembly: gemma4's tool DSL + qwen's ChatML/XML
crates/secret     locked, self-zeroing memory; the zeroing global allocator
crates/hub        resumable, verified Hugging Face downloads
crates/server     the private REPL, plus the axum OpenAI-compatible API
crates/wasm       the browser build: WebGPU, streamed weights, chat REPL
web/              the page shell; build-web.sh emits llmoxide.html into it
```

`crates/wasm` is deliberately **not** a workspace member — it only ever builds
for `wasm32-unknown-unknown`, and membership would pull wasm-bindgen and
web-sys into every native `cargo build`.

## Known limitations

- gemma4 decode is ~1.5x slower than llama.cpp, and the gap widens with
  context (see above).
- qwen35 decode at ~9.4 tok/s is usable but unoptimised. The delta-net
  recurrence is sequential by construction: `delta_recur` loops the whole token
  range inside one dispatch of `n_v_heads` workgroups, so those layers get no
  token parallelism during prefill and cannot fill the GPU. A chunked
  formulation is the obvious next thing to attack.
- No architecture here can rewind its cache, so prompt reuse is append-only:
  gemma4's ring buffers have overwritten the positions, and qwen35's recurrent
  state was never a history to begin with.
- No speculative decoding, though two of the checkpoints ship a drafter for it:
  qwen35's MTP/NextN head (`blk.64`) is skipped at load, and Gemma 4 publishes a
  separate MTP drafter (`google/gemma-4-E4B-it-assistant`). Wiring either up
  needs a KV cache that can rewind on a rejected draft, which is the same gap as
  the entry above.
- Single request at a time, one GPU context; no batching across clients.
- Text only — the vocabularies carry image/audio/video tokens and E4B ships an
  `mmproj` encoder, but no multimodal path is implemented.
- The context is capped at `LLMOXIDE_CTX`, well below the models' 262144,
  since attention scratch scales with it.
- Private mode covers this process, not the machine, and not the server: see
  "What this does not cover" and "Serving" above.
- The embedded (`--embed`) build is an 854 MB HTML file with no download
  progress and no separate caching of the model. It exists for offline
  distribution; hosting wants the two-file form.
- The browser build needs WebGPU and has no CPU fallback — wasm32's 4 GB
  address space cannot hold the weights at any useful quantization. It also
  cannot `mlock`, so it is the one entry point where "leaves nothing behind" is
  a weaker claim than elsewhere. See "In a browser".
- Qwen3 0.6B is the only checkpoint here small enough to serve over the web,
  and it is a 0.6B model: fine for short exchanges, visibly limited past that,
  and prone to answering *about* your question rather than answering it. There
  is no larger option under a gigabyte in either family — see "Why not a
  smaller Gemma".
- `qwen3` support is dense-attention only. The delta-net path it shares a module
  with is exercised by the 27B, not by any small checkpoint, so a regression
  there needs the 22 GB file to catch.
- The browser build's correctness rests on the native `bisect`, not on a check
  that runs in a browser: comparing per-checkpoint tensors there would mean
  shipping the CPU reference path, which is exactly what does not fit. The
  kernels are byte-identical WGSL and `Gpu::check_shaders` proves they compiled,
  but nothing verifies the browser's *numerics* against the CPU the way
  `bisect` does natively.
