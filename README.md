# llmoxide

Hand-rolled inference for two checkpoints — `gemma4-v2-Q4_K_M.gguf` and
`Qwen3.8-27B-Q6_K.gguf` (arch `qwen35`, a hybrid gated-delta-net/attention
stack) — GGUF loader, k-quant decoders, tokenizers, wgpu compute kernels, and
an OpenAI-compatible server, in Rust with no ML dependencies.

Both architectures run on the GPU. The CPU forward passes stay in the tree as
the reference every kernel is checked against, not as a fallback.

Built for one user on one machine (Apple M4 Pro, 48 GB), so it runs one request
at a time against one GPU context.

```
crates/gguf       GGUF v3 reader, mmap'd; Q4_K / Q6_K / Q8_0 decoders
crates/tokenizer  gemma4 BPE (262144 tokens) + qwen35 byte-level BPE (248320)
crates/model      architecture configs, CPU reference forward passes, sampling
crates/gpu        wgpu device, weight arena, WGSL kernels, both GPU forwards
crates/chat       prompt assembly: gemma4's tool DSL + qwen35's ChatML/XML
crates/server     axum OpenAI-compatible API, dispatching on architecture
```

## Running

```sh
cargo build --release
./target/release/llmoxide-serve models/gemma4-v2-Q4_K_M.gguf   # http://127.0.0.1:8080
./target/release/llmoxide-serve models/Qwen3.8-27B-Q6_K.gguf   # same API
```

`LLMOXIDE_CTX` (default 16384), `LLMOXIDE_PORT` (8080), `LLMOXIDE_BATCH` (256).
`LLMOXIDE_CPU=1` puts qwen35 on its CPU reference path instead of the GPU;
`LLMOXIDE_NO_SUBGROUP=1` forces the barrier-tree row reduction.

One-shot generation, either architecture, either backend:

```sh
./target/release/llmoxide models/gemma4-v2-Q4_K_M.gguf "The capital of France is" 8
./target/release/llmoxide models/Qwen3.8-27B-Q6_K.gguf "..." 8
./target/release/llmoxide models/Qwen3.8-27B-Q6_K.gguf "..." 8 --cpu   # reference path
```

### opencode

`opencode.json` in this repo points opencode at the local server (port 8081 —
start the server with `LLMOXIDE_PORT=8081` or edit the `baseURL`). It needs the
provider package once:

```sh
cd ~/.config/opencode && npm install @ai-sdk/openai-compatible
```

Then `opencode run --model llmoxide/Qwen3.8-27B-Q6_K "..."`, or copy the
`provider` block into `~/.config/opencode/opencode.json` to use it anywhere.

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
| qwen35 tokenizer | exact id-for-id match with `llama-tokenize` on 18 cases + 3 files (~11k tokens) |
| qwen35 CPU forward | 567 tensors across all 64 layers match `llama-eval-callback` within 4e-4 on a real 27B; **byte-identical** greedy output to `llama-completion --temp 0` |
| qwen35 GPU forward | **771/771** checkpoints match the CPU path on the 27B (logits rel 1.3e-6, same argmax) |
| qwen35 chat format | prompt ids match a jinja2 rendering of the embedded template, with and without tools |

Tools that reproduce this:

```sh
./target/release/kernels        models/gemma4-v2-Q4_K_M.gguf   # per-kernel vs CPU
./target/release/bisect         models/gemma4-v2-Q4_K_M.gguf   # per-layer GPU vs CPU
./target/release/bisect_qwen35  <model> [ids]                  # per-checkpoint GPU vs CPU
./target/release/validate_qwen35 <model> <refs.json> <ids>     # vs llama-eval-callback
./target/release/upload_check   <model>                        # weight arena readback
```

`bisect` reports the *first* diverging checkpoint, which is how the NaN in
GeGLU and the attention-scale error were both found. The qwen35 workflow
(including a synthetic-checkpoint generator for fast whole-graph checks) is in
[ARCHITECTURE-qwen35.md](ARCHITECTURE-qwen35.md) and `scripts/`.

The tensor-level qwen35 numbers above were measured against
`Qwen3.8-27B-Uncensored-Cyber-Q6_K.gguf`, which carries no NextN block. The
checkpoint this repo is pointed at, `models/Qwen3.8-27B-Q6_K.gguf`, was a
truncated download when those notes were written and has since been re-fetched
complete — 20.9 GiB, 866 tensors, 65 blocks, i.e. the same architecture plus
one NextN block, which the loader skips. An earlier complete merge of it
loaded and decoded on the GPU path with greedy output matching the validated
checkpoint; the build currently in `models/` is a newer merge and has not been
put back through `validate_qwen35`.

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

## API

`GET /v1/models`, `POST /v1/chat/completions` (streaming and not), `GET /health`.

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

## Known limitations

- gemma4 decode is ~1.5x slower than llama.cpp, and the gap widens with
  context (see above).
- qwen35 decode at ~9.4 tok/s is usable but unoptimised. The delta-net
  recurrence is sequential by construction: `delta_recur` loops the whole token
  range inside one dispatch of `n_v_heads` workgroups, so those layers get no
  token parallelism during prefill and cannot fill the GPU. A chunked
  formulation is the obvious next thing to attack.
- Neither architecture can rewind its cache, so prompt reuse is append-only:
  gemma4's ring buffers have overwritten the positions, and qwen35's recurrent
  state was never a history to begin with.
- The MTP/NextN draft head (`blk.64`) is skipped at load; no speculative
  decoding.
- Single request at a time, one GPU context; no batching across clients.
- Text only — both vocabularies carry image/audio/video tokens, but no
  multimodal path is implemented.
- The context is capped at `LLMOXIDE_CTX`, well below the models' 262144,
  since attention scratch scales with it.
