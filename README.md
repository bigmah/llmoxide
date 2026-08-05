# llmoxide

Hand-rolled inference for `gemma4-v2-Q4_K_M.gguf` — GGUF loader, k-quant
decoders, tokenizer, wgpu compute kernels, and an OpenAI-compatible server, in
Rust with no ML dependencies.

Built for one user on one machine (Apple M4 Pro, 48 GB), so it runs one request
at a time against one GPU context.

```
crates/gguf       GGUF v3 reader, mmap'd; Q4_K / Q6_K decoders
crates/tokenizer  gemma4 BPE (262144 tokens, 514906 merges)
crates/model      architecture config, CPU reference forward pass, sampling
crates/gpu        wgpu device, weight arena, WGSL kernels, GPU forward pass
crates/chat       prompt assembly + the model's tool-call DSL
crates/server     axum OpenAI-compatible API
```

## Running

```sh
cargo build --release
./target/release/llmoxide-serve gemma4-v2-Q4_K_M.gguf     # http://127.0.0.1:8080
```

`LLMOXIDE_CTX` (default 16384), `LLMOXIDE_PORT` (8080), `LLMOXIDE_BATCH` (256).

One-shot generation, either backend:

```sh
./target/release/llmoxide gemma4-v2-Q4_K_M.gguf "The capital of France is" 8
./target/release/llmoxide gemma4-v2-Q4_K_M.gguf "..." 8 --cpu
```

### opencode

`opencode.json` in this repo points opencode at the local server. It needs the
provider package once:

```sh
cd ~/.config/opencode && npm install @ai-sdk/openai-compatible
```

Then `opencode run --model llmoxide/gemma4-v2-Q4_K_M "..."`, or copy the
`provider` block into `~/.config/opencode/opencode.json` to use it anywhere.

## Correctness

The architecture is genuinely unusual — see [ARCHITECTURE.md](ARCHITECTURE.md)
for the four things that will silently produce garbage if you assume the
Gemma 2/3 shape. Everything is checked against llama.cpp rather than asserted:

| what | check |
|---|---|
| tokenizer | exact id-for-id match with `llama-tokenize` on 13 cases + a 3547-token file |
| CPU forward | **byte-identical** greedy output to `llama-completion --temp 0` |
| GPU kernels | every matvec within ~1e-7 of the CPU dequant-dot, on real weights |
| GPU forward | all 773 intermediate tensors match the CPU path across 48 layers (~1e-6) |

Two tools reproduce this:

```sh
./target/release/kernels gemma4-v2-Q4_K_M.gguf     # per-kernel vs CPU
./target/release/bisect  gemma4-v2-Q4_K_M.gguf     # per-layer GPU vs CPU
```

`bisect` reports the *first* diverging checkpoint, which is how the NaN in
GeGLU and the attention-scale error were both found.

## Performance

On an M4 Pro (20 GPU cores), measured, not projected. Decode is quoted with the
context it ran at, because attention cost grows with it:

| | before | now | llama.cpp (Metal) |
|---|---|---|---|
| decode @ 128 ctx | 12.7 tok/s | **22.3 tok/s** | 32.6 tok/s |
| decode @ 512 ctx | 8.6 tok/s | **18.4 tok/s** | |
| decode @ 2048 ctx | 5.0 tok/s | **12.9 tok/s** | |
| prefill, 374 tokens | 23.5 tok/s | **35.0 tok/s** | much higher |
| prefill, 1490 tokens | 20.7 tok/s | **32.7 tok/s** | |

Greedy output is unchanged token-for-token, and `bisect` still matches the CPU
path at every checkpoint.

Four things got it there, in rough order of how much they were worth:

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

### KV cache and prefix reuse

Sliding-window layers get a 1024-slot ring instead of a full-context
allocation — with 40 of 48 layers windowed, that is the difference between
~2 GB and ~90 GB at the model's full 262144-token context.

The cache is reused when a new prompt strictly extends what is resident.
Rewinding is deliberately not attempted: the ring has already overwritten the
positions a rewind would need. To make the common agentic case hit this path,
replayed assistant turns reproduce the empty thought channel the generation
prompt emits — otherwise every turn diverges from the cache at the first
assistant message and re-prefills the whole conversation.

## API

`GET /v1/models`, `POST /v1/chat/completions` (streaming and not), `GET /health`.

Supports `tools`, `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`,
`stop`, and a non-standard `enable_thinking` for the model's thought channel
(off by default).

Tool calls are the interesting part. The model does not emit JSON — it uses a
custom DSL where strings are delimited by the single token `<|"|>`:

```
<|tool_call>call:read_file{path:<|"|>src/main.rs<|"|>,limit:20}<tool_call|>
```

`crates/chat/src/dsl.rs` translates both directions. Two deliberate behaviours:

- Parsing works on **token ids**, not decoded text, because the quote marker and
  channel markers are control tokens a text decoder drops.
- Calls naming a tool the request never declared are **rejected**. The model
  will occasionally invent one, and a client that dispatched it would either
  error out or run something unintended.

## Known limitations

- Decode is ~1.5x slower than llama.cpp, and the gap widens with context (see
  above).
- Single request at a time; no batching across clients.
- Text only — the vocabulary has image/audio/video tokens and the checkpoint
  suppresses two of them, but no multimodal path is implemented.
- The context is capped at `LLMOXIDE_CTX`, well below the model's 262144, since
  the attention scores scratch buffer scales with it.
