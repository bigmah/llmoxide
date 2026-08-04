# gemma4 (12B) — architecture notes

Everything here was derived from `gemma4-v2-Q4_K_M.gguf` itself and verified
against `llama-eval-callback` tensor-by-tensor. Several details are *not* what
you would guess from the Gemma 2/3 lineage; those are called out.

Verification status: greedy generation is **byte-identical to llama.cpp** on
multiple prompts (`crates/model/src/bin/validate.rs`, and the `llmoxide` binary).

## Shape

| | |
|---|---|
| layers | 48 |
| d_model | 3840 |
| FFN | 15360 (GeGLU) |
| query heads | 16 |
| vocab | 262144, embeddings tied to the output projection |
| RMS eps | 1e-6 |
| context | 262144 |

Layers alternate **5 sliding-window : 1 global**, so indices 5, 11, … 47 are
global and the other 40 are windowed.

| | sliding-window (40) | global (8) |
|---|---|---|
| KV heads | 8 | **1** (MQA) |
| head_dim | 256 | **512** |
| window | 1024 | full |
| RoPE base | 10 000 | 1 000 000 |
| RoPE coverage | all 128 pairs | **first 64 pairs only** |
| V projection | `attn_v` | **none — reuses `attn_k`** |

## The four things that will bite you

### 1. Global layers ship no `attn_v` tensor

There are 667 tensors, not the 674 a uniform model would have: the eight global
layers are each missing `attn_v.weight`. They are not broken. On those layers
**K and V are the same projection**, differing only in what happens afterwards:

```
proj = attn_k · x
K    = rope(rms_norm(proj) * k_norm)
V    =       rms_norm(proj)              # no gain, no rope
```

### 2. V is always RMS-normalized, with no learned weight

On *every* layer, including the ordinary windowed ones. ggml shows this as a
bare `RMS_NORM(Vcur (reshaped))` with no following `MUL`. Easy to miss because
there is no `attn_v_norm` tensor to hint at it.

### 3. The attention softmax scale is 1.0, not 1/√head_dim

`attn_q_norm` and `attn_k_norm` are **uniform scalars**, not learned per-dim
vectors. `k_norm` follows an inverse-*dimension* law — 0.125 on the 256-wide
SWA layers, exactly 0.0625 on the 512-wide global ones (`32/head_dim`) — where a
plain gain would follow inverse-square-root. The attention temperature has been
folded into those weights.

Applying `1/sqrt(head_dim)` on top flattens the softmax so badly that every
position collapses onto the same hidden state within a few layers, and the model
emits the same filler token regardless of input. The failure is silent: per-layer
activations still look plausible, because the collapse only shows up when you
compare a token *other than the first*.

### 4. RoPE is partial, and only on global layers

`rope_freqs.weight` holds 256 per-pair frequency divisors: indices 0–63 are
`1.0`, 64–255 are the sentinel `1e30`, which drives those angles to zero.

It applies to **global layers only**. Confirmed empirically: after RoPE on a
global layer the highest head dimensions are bit-identical, while on a
sliding-window layer they still rotate by a small angle. So local layers keep
full positional resolution across their 1024-token window, and global layers
retain only the 64 lowest frequencies with the rest left as NoPE — which is what
makes the 262k context tractable.

Pairing is ggml NeoX: `(i, i + head_dim/2)`, *not* adjacent elements.

## Block graph

```
h = embed(tok) * sqrt(3840)              # 61.9677

per layer:
  x = rms(h) * attn_norm
  q = rope(rms_head(Wq x) * q_norm)      # rms_head is per-head, over head_dim
  k = rope(rms_head(Wk x) * k_norm)
  v =      rms_head(Wv x)                # no gain, no rope; Wv := Wk if global
  h = h + rms(Wo attn(q,k,v)) * post_attention_norm
  y = rms(h) * ffn_norm
  h = h + rms(Wd (gelu(Wg y) * Wu y)) * post_ffw_norm
  h = h * layer_output_scale             # per-layer scalar, whole residual stream

logits = tied_embed · (rms(h) * output_norm)
logits = 30 * tanh(logits / 30)          # final_logit_softcapping
logits[258882] = logits[258883] = -inf   # suppress_tokens: <image|>, <audio|>
```

`layer_output_scale` is a scalar per layer (0.0046 … 0.887) applied to the entire
residual stream after the FFN residual add. It does not shrink the stream to
nothing because `post_ffw_norm` is correspondingly large.

## Tokenizer

Rank-ordered **BPE**, 262144 tokens, 514906 merges.

`tokenizer.ggml.scores` is uniformly `-1000` — it carries no information, so this
is not a unigram/SentencePiece model despite the `▁` convention. Spaces become
`▁` (U+2581) before merging; `add_space_prefix` is false. There is no
pre-tokenizer regex — digits come out individually simply because no multi-digit
ASCII merges exist. Characters outside the vocab decompose to `<0xNN>` tokens.

Verified: exact match with `llama-tokenize` on 13 varied cases plus a 3547-token
source file.

## Control tokens

| id | token | role |
|---|---|---|
| 2 / 1 | `<bos>` / `<eos>` | |
| 105 / 106 | `<\|turn>` / `<turn\|>` | turn delimiters |
| 100 / 101 | `<\|channel>` / `<channel\|>` | thinking channel |
| 46 / 47 | `<\|tool>` / `<tool\|>` | tool declaration |
| 48 / 49 | `<\|tool_call>` / `<tool_call\|>` | tool invocation |
| 50 / 51 | `<\|tool_response>` / `<tool_response\|>` | tool result |
| 52 | `<\|"\|>` | string quote inside the tool DSL |

**End-of-generation**: `<eos>`, `<turn|>`, and `<|tool_response>`. That last one
matters for agentic use — after emitting a tool call the model opens a response
block and stops, waiting for the harness to supply the result.

## A note on validating against llama.cpp

llama.cpp's CPU kernels quantize activations to Q8_K and use integer dot
products. Our f32 path is the *more* accurate of the two: an independent f64
computation of the first V projection gives -0.3876, matching us exactly, versus
llama.cpp's -0.4198.

So do not validate on whole-tensor sums. Those are near-total cancellations of
thousands of terms and amplify that difference into apparent divergence even when
every element agrees closely. Compare element values against the tensor's own
magnitude, and check a token other than the first — position 0 attends only to
itself and will match even when attention is completely wrong.
