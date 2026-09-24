# Images

E4B ships a vision tower in a separate `mmproj` file. Point the session at one
and `image_url` content parts are encoded in place:

```bash
./target/release/llmoxide-fetch gemma4-e4b-q4
./target/release/llmoxide-fetch gemma4-e4b-mmproj

cargo run --release -p llmoxide --example image --features vision,gpu -- \
    models/gemma-4-E4B-it-Q4_K_M.gguf \
    models/mmproj-gemma-4-E4B-it-BF16.gguf \
    photo.jpg "What is in this image?"
```

For checking the tower by hand, llama.cpp's own `tools/mtmd/test-1.jpeg` is
the useful input: it is a newspaper front page, so a correct encoder reads the
headline and the date out of it rather than describing a plausible scene.

```rust
let mut s = Session::load(model, &LoadOptions::new().mmproj(mmproj))?;
s.complete(Request::new(vec![Message {
    role: "user".into(),
    content: Some(serde_json::json!([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}},
        {"type": "text", "text": "what is in this image?"},
    ])),
    ..Default::default()
}]))?;
```

The server takes the same shape (`llmoxide-serve model.gguf --mmproj
mmproj.gguf`), which is what makes an ordinary OpenAI client work unchanged.
`data:` URLs and local paths are read; **remote URLs are deliberately not
fetched**, because a server that dereferences a URL a client hands it is an
SSRF hole.

An image with no tower loaded is an error rather than a text-only answer —
quietly answering *about* an image the model never saw is the worst available
outcome.

## What the tower is

Not the SigLIP the name suggests. Every block is the text stack's block with
the sequence axis swapped for patches — RMSNorm, per-head Q/K norms, a gated
GELU feed-forward, post-norms on both residual branches — and the parts that
are genuinely its own are where the care went:

- **Two positional mechanisms, not one.** A learned `(x, y)` pair of lookup
  tables is added to the patch embedding, *and* a 2-D rotation runs inside
  attention, the low half of each head by column and the high half by row
  (`theta = 100`, against a text model's 1 000 000). Getting the axes backwards
  survives every content question and fails only on spatial ones, so the test
  for it is an image whose two coloured squares sit on the anti-diagonal —
  the diagonal is transpose-symmetric and would pass either way.
- **Attention is bidirectional**, and so is the text model's over the span the
  tower produced: llama.cpp clears causal attention for an encoded image and
  restores it afterwards. That span therefore has to prefill as a *single*
  batch, which is why an image is capped at 256 positions and why the prefill
  loop refuses to chunk it rather than silently splitting it in half.
- **The softmax scale is 1.0**, not `1/sqrt(head_dim)` — the same folded
  temperature the text stack uses.
- **The linears clamp.** Each weight may carry calibration ranges beside it
  (`.input_min`, `.output_max`, …); the input is clamped before the matmul and
  the result after. Ignoring them is fine on most images and wrong on the ones
  that saturate, which is the worst failure shape there is.

Two things about how an image enters the text stack are easy to get backwards,
and both are load-bearing:

- The rows are **not** scaled by `sqrt(d_model)`. Token embeddings are; these
  are already in the residual stream's space, and scaling them anyway
  multiplies the image by ~50.
- The per-layer embedding table has no token id to look up for an image
  position, so it falls back to **row 0** — the padding row — for every one of
  them, and only the projected half carries the image.

Resolution is native rather than square: the image is resampled so both sides
are a multiple of 48 and the area lands inside a token budget, which is why a
640×488 photo becomes a 13×10 grid of 130 tokens rather than a fixed 256.

## Where it runs

The tower follows the text model onto the GPU, and falls back to the CPU one
with a warning rather than failing the load if there is no adapter. Both are
kept: the CPU tower is written directly against llama.cpp's graph and is the
oracle the GPU one is checked against — the same relationship `model::cpu`
has to `crates/gpu`.

| 1170 patches (a 640×488 photo) | |
|---|---|
| CPU tower | 6.0 s |
| GPU tower | 0.87 s |
| llama.cpp (Metal) | ~0.2 s |

**Attention reuses `attn.wgsl` unchanged.** A tower has no KV cache, but a
cache with `window = 0` read at `base_pos = 0` *is* a flat
`[n_patches, kv_dim]` buffer, and the `bidi` flag that image spans already
needed in the text model opens the mask both ways — so K and V bind straight
into the cache slots and the mask comes out right. The patch convolution is
likewise a matmul once the image is lowered to `[n_patches, 16×16×3]`, which
is what the 4-D filter already looks like in memory; the kernel never consults
the GGUF's declared shape, only the `MatvecParams` handed to it.

Scratch is allocated per image rather than reserved for the largest one: the
scores buffer alone is `n² × n_heads` floats — 255 MB at the token ceiling and
a fifth of that for a typical photo — and this runs once per image, so the
allocation is not on any hot path.
