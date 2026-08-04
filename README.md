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

On an M4 Pro (20 GPU cores), measured, not projected:

| | llmoxide | llama.cpp (Metal) |
|---|---|---|
| decode | 10.3 tok/s | 32.6 tok/s |
| prefill | 21 tok/s | much higher |

**This is the weak point.** It is fast enough to talk to and fast enough for
short agentic turns, but a cold 6000-token prompt takes minutes. Prefix caching
(below) makes subsequent turns cheap, so the cost is paid once per conversation
rather than once per turn.

What is already done: weights stay quantized on the GPU and are decoded inside
the matmul; the decode dispatch plan and its ~800 bind groups are built once and
reused; the matmul tiles 32 tokens per weight load; activations are read as
`vec4`. What profiling says is left: the kernels are **ALU-bound, not
bandwidth-bound** — raising the token tile from 8 to 32 bought almost nothing,
which means the dequantization arithmetic, not weight traffic, is the limit.
Closing the gap needs a real rewrite of the inner loop (llama.cpp uses Metal
simdgroup matrix ops here; WGSL has no equivalent, and Naga does not yet
implement the `subgroups` extension, so the tree reduction cannot be replaced
with a single `subgroupAdd`).

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

- Prefill and decode are ~3x slower than llama.cpp (see above).
- Single request at a time; no batching across clients.
- Text only — the vocabulary has image/audio/video tokens and the checkpoint
  suppresses two of them, but no multimodal path is implemented.
- The context is capped at `LLMOXIDE_CTX`, well below the model's 262144, since
  the attention scores scratch buffer scales with it.
