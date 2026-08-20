# qwen35 (Qwen3.5 27B) — architecture notes

Derived from `Qwen3.8-27B-Q6_K.gguf` metadata and llama.cpp b10090's `qwen35`
implementation (the reference the repo validates against). The interesting
part: this is a **hybrid stack** — three quarters of the layers are not
attention at all.

Verification status: on a complete 27B qwen35 checkpoint
(`Qwen3.8-27B-Uncensored-Cyber-Q6_K.gguf`, Q8_0/Q6_K tensors), **all 567
observation points match llama.cpp's Metal path within 4e-4** and greedy
decode is **byte-identical over 24 tokens**. An 8-layer random-weight
synthetic model additionally pins every intermediate on both prefill shapes —
see "Validation" below. The file this repo was pointed at
(`models/Qwen3.8-27B-Q6_K.gguf`) is a truncated download (17.8 of 20.9 GiB)
and cannot be loaded by anything until re-fetched; it differs from the
validated checkpoint only in carrying one extra NextN block.

## Shape

| | |
|---|---|
| blocks in file | 65 = 64 main + 1 NextN/MTP (`nextn_predict_layers = 1`) |
| main layers | 48 gated delta net : 16 full attention (`full_attention_interval = 4`, so indices 3, 7, … 63) |
| d_model | 5120 |
| FFN | 17408, SwiGLU |
| vocab | 248320, separate `output.weight` |
| RMS eps | 1e-6 |
| context | 262144 |

The MTP block (`blk.64`, with the `nextn.*` tensors) is a draft head for
speculative decoding. The main forward pass never executes it, and the loader
does not read it.

## Gated delta net layers (48 of 64)

Linear attention: state is a per-head 128×128 matrix updated by an outer
product per token, not a KV cache that grows. Fixed metadata names are
misleading — `ssm.time_step_rank` is the **v-head count** (48),
`ssm.group_count` the k/q-head count (16), `ssm.state_size` the head width
(128). `d_inner = 48 × 128 = 6144`.

Per token:

```
[q | k | v] = Wqkv · x                    # 2048 | 2048 | 6144 = 10240 channels
[q | k | v] = silu(causal_conv4([q|k|v])) # depthwise, kernel 4, over the fused stream
q, k        = l2norm per 128-dim head     # x / max(‖x‖, eps) — eps floors the norm
z           = Wgate · x                   # 6144, the output gate
per v-head h (its k/q head is h % 16 — llama.cpp repeats by tiling, not interleaving):
  g = exp(a_h · softplus(alpha_h(x) + dt_h))     # a = ssm_a, already -exp(A_log)
  b = sigmoid(beta_h(x))
  S ← g·S                                 # decay, whole-matrix scalar
  S ← S + b·(v − S k) ⊗ k                 # the delta rule
  o = (S q) / √128
out = Wout · ( rms(o)·ssm_norm ⊙ silu(z) )       # gated RMS norm, per head
```

Things that will bite:

1. **The k/q-head mapping is `h % 16`**, ggml's `repeat` tiling. HF-style
   `repeat_interleave` (`h / 3`) produces plausible-looking garbage.
2. **`ggml_l2_norm`'s eps is a floor on the norm** (`1/max(‖x‖, eps)`), not an
   addend under the root like RMS norm.
3. **`ssm_a` is stored negative** (`-exp(A_log)` baked in by the converter);
   multiplying by another sign flip makes the state explode.
4. The conv runs over the *fused* QKV stream before the split, and its state
   (last 3 columns) is what the recurrent cache actually stores alongside `S`.
5. State layout matters for validation: ggml stores each head's matrix
   transposed (rows are value dims). We adopt the same orientation so rows are
   contiguous in the recurrence.

Per-layer state: 3×10240 conv floats + 48×128×128 state floats ≈ 3.1 MB — the
whole 48-layer recurrent state is ~152 MB regardless of context length, which
is what makes 262k context tractable.

## Full attention layers (16 of 64)

| | |
|---|---|
| heads | 24 query, 4 KV (GQA group 6), head_dim 256 |
| Q projection | **fused per-head `[query | gate]`** — out dim 2·24·256 |
| QK norm | learned per-dim RMS gains (vectors, unlike gemma4's scalars) |
| RoPE | first 64 dims only, base 1e7, NeoX pairs `(i, i+32)`; dims 64–255 NoPE |
| scale | the standard `1/√256` |
| window | none — full context |

The metadata declares M-RoPE sections (`[11, 11, 10, 0]`, interleaved) for
vision inputs; with text-only input every section reads the same position, so
it reduces exactly to plain partial NeoX RoPE. Verified against llama.cpp.

After the softmax-weighted sum, each head's output is multiplied by
`sigmoid(gate)` — the second half of its fused Q row — before `Wo`. Forgetting
the gate leaves activations plausible and output subtly wrong.

V is used as projected: no norm, no rope.

## Block wiring

```
h = embed(tok)                       # NO sqrt(d_model) scale
x = rms(h) * attn_norm
h = h + attn_or_deltanet(x)
y = rms(h) * post_attention_norm     # "post" in name, but the FFN residual
h = h + Wd (silu(Wg y) * Wu y)       #  branches BEFORE it: it is a pre-FFN norm
logits = output · (rms(h) * output_norm)   # no softcap, no suppress list
```

## Tokenizer

Byte-level BPE (GPT-2 style), 248320 tokens, 247587 merges, pre-tokenizer
`qwen35` — the qwen2 split with `\p{M}` added so combining marks travel with
their letters, digits split individually. Our splitter is a direct port of
llama.cpp's hand-compiled matcher (`unicode_regex_split_custom_qwen35`), with
category tables generated from llama.cpp's own `unicode-data.cpp`.

Verified: exact id-for-id match with `llama-tokenize` on 18 varied cases plus
three full source files (~11k tokens), round-tripping through the decoder.

Control tokens: ChatML (`<|im_start|>` 248045, `<|im_end|>` 248046,
`<|endoftext|>` 248044 = bos/pad), plus *user-defined* single tokens
`<think>`/`</think>` (248068/9), `<tool_call>`/`</tool_call>` (248058/9),
`<tool_response>`/`</tool_response>` (248066/7).

**End-of-generation**: `<|im_end|>` and `<|endoftext|>`.

**BOS is never prepended.** When `tokenizer.ggml.add_bos_token` is absent,
llama.cpp defaults byte-level vocabs to *no* BOS (SentencePiece ones to yes),
and checkpoints exist with the key omitted. Getting this wrong is nearly
invisible — one stray `<|endoftext|>` up front — but the model then reads the
prompt as a document fragment: greedy decode of "The capital of France is
Paris.\n" ends the turn instead of continuing, more than a full logit away
from the correct next token. Found by diffing greedy output against
llama.cpp; the per-token margins made it look like a numeric near-tie until
llama-server's top-logprobs (fed the exact same ids) showed it wasn't.

## Chat format

ChatML with Qwen's XML-ish tool-call convention (see `chat::qwen`). Notable
template behaviors we reproduce: a default reasoning-effort preamble
(`xhigh`) in the system turn whenever thinking is enabled; assistant turns
replay their `<think>` block verbatim; tool *results* ride inside `user` turns
as `<tool_response>` blocks; with thinking disabled the generation prompt
closes an empty `<think>\n\n</think>` itself. Prompt assembly is validated
token-for-token against a jinja2 rendering of the checkpoint's embedded
template on conversations with and without tools.

Unlike llama.cpp's server we insert control tokens by id, so their spellings
in user text tokenize as plain text (the pre-tokenizer shatters them; merges
cannot cross pieces) and cannot forge a turn boundary.

## Validation

The full workflow, reproducible with what's in the repo:

```sh
# reference trace — use Metal (-ngl 99) for quantized checkpoints: llama.cpp's
# CPU backend quantizes activations to Q8_K, which adds multi-percent noise of
# its own; the Metal path computes on f32 activations and matches us to 4e-4
llama-eval-callback -m model.gguf -p "The capital of France is" -ngl 99 > eval.txt 2>&1
python3 scripts/parse-eval-callback.py eval.txt refs.json
./target/release/validate_qwen35 model.gguf refs.json 760,6511,314,9338,369

# greedy decode diff
llama-completion -m model.gguf -p "..." -n 24 --temp 0 -no-cnv --no-display-prompt
./target/release/llmoxide model.gguf "..." 24

# the all-f32 synthetic model for sharp (5e-3) whole-graph checks
python3 scripts/make-tiny-qwen35.py tiny-qwen35.gguf
```

(`scripts/gen-unicode-tables.py` documents where `tokenizer/src/unicode.rs`
came from: llama.cpp's own category tables, so classifications agree exactly.)

Results, in increasing order of integration:

* synthetic model (8 main layers + MTP block, real tokenizer metadata, F32):
  91 observation points match on 2- and 34-token prefills (worst 5e-4 /
  5e-3); greedy is token-identical until an argmax margin of 7e-5 — a genuine
  numerical coin-flip between llama.cpp's chunked prefill and our sequential
  recurrence.
* real 27B checkpoint: 567 observation points match the Metal trace within
  4e-4 across all 64 layers; greedy decode **byte-identical for 24 tokens**;
  the OpenAI endpoint serves it end to end (ChatML prompt, `<|im_end|>`
  stop).

## GPU backend

The wgpu path (`crates/gpu/src/qwen35.rs`) mirrors this graph step for step
and is the default at both entry points (`--cpu` / `LLMOXIDE_CPU=1` keep the
reference path). It reuses the gemma4 quant matvec and attention kernels, adds
Q8_0 + F32 matvec (`shaders/quant.wgsl`), the delta-net kernels
(`shaders/deltanet.wgsl` — the token loop lives inside `delta_recur`, one
workgroup per v-head, since the recurrence is sequential), and the qwen ops in
`shaders/ops.wgsl` (fused-Q split, partial RoPE, the SwiGLU/sigmoid gates).

Validated with `bisect_qwen35 <model> <ids>`, which diffs every CPU checkpoint
against the GPU `debug_stop`: **771/771 match** on the 27B checkpoint (logits
rel 1.3e-6, same argmax), and the tiny synthetic model matches at t=1/5/34
with greedy byte-identical. On an M4 Pro the 27B decodes at ~9.4 tok/s, ~9x
the CPU path.

Two failure modes here are *silent* — the GPU returns all zeros with no error
— and both are guarded now:

* **Upload past the working set.** Mapping every arena buffer at once doubles
  a 25 GB model with staging shadows; the writes are dropped. `Weights::upload`
  fills and flushes one buffer at a time and reads sampled spans back to
  confirm. `upload_check <model>` sweeps every tensor head.
* **Over-long command buffers.** Submitting a whole forward pass (~1500
  dispatches) as one command buffer trips Metal's GPU watchdog on cold
  pipelines and long prefills. `Qwen35Gpu::run` chunks at 64 dispatches per
  submit. A device-lost callback in `Gpu::new` turns any future watchdog kill
  into a printed message instead of silent zeros.
