# qwen-image 2.1 (text-to-image) — architecture notes and plan

Scoped 2026-09-21. **Nothing here is implemented yet.** This is the spec and the
plan for generating images from text with `qwen-image-2.1-Q8_0.gguf`, written so
the work can be picked up cold. Every number below comes from the files on disk
or from the reference code listed under "Sources".

## Files

All four files are in `/Volumes/SD1/models/`. The three downloads were
SHA-256-checked by `llmoxide-fetch` against the hashes Hugging Face publishes,
which match the repo's `SHA256SUMS`.

| role | file | bytes | sha256 |
|---|---|---|---|
| diffusion transformer (DiT) | `qwen-image-2.1-Q8_0.gguf` | 7 591 557 792 | `9a7ec02f4c9d…` |
| text encoder | `qwen3vl_8b_bf16.safetensors` | 17 534 334 616 | `68bdc82bc1b6…` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 675 509 688 | `bb21f7473051…` |
| tokenizer | borrowed from `Qwen3-0.6B-Q8_0.gguf` (see below) | | |

They come from `abenzerps/Qwen-Image-2.1-GGUF`, which was earlier named
`…-Uncensored-GGUF` and serves the same files. According to its README:
* The GGUF was made with stable-diffusion.cpp (commit `1330ceb`) from the
  official `Qwen/Qwen-Image-2.1` at revision `b3179ad`.
* "Uncensored" means only that there is no safety checker.
* The text encoder and VAE are ComfyUI's repackaging (`Comfy-Org/Qwen-Image-2.1`).

To re-fetch:

```
cargo run --release -p llmoxide-hub --bin llmoxide-fetch -- --dir /Volumes/SD1/models \
  https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q8_0.gguf \
  https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors \
  https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/qwen3vl_8b_bf16.safetensors
```

What was established about these files:

* **The DiT GGUF has no metadata at all** (`n_kv = 0`): no architecture key and
  no config. `gguf::Gguf::open` loads it and its 297 tensors (checked). The
  config below therefore has to be hardcoded and asserted against tensor shapes.
* **The text encoder is stock Qwen3-VL-8B-Instruct**, not a version trained
  jointly with the image model. Six tensors are byte-identical between
  `Qwen/Qwen-Image-2.1/text_encoder` and `Qwen/Qwen3-VL-8B-Instruct`: the token
  embeddings, a q_proj, a down_proj, a k_norm, a layernorm and the final norm.
  So `Qwen/Qwen3-VL-8B-Instruct-GGUF` (`Qwen3VL-8B-Instruct-Q8_0.gguf`, 8.71 GB,
  arch `qwen3vl`) is a drop-in alternative that the existing GGUF loader is
  closer to reading.
* **The text-encoder safetensors layout:** 750 BF16 tensors named like plain
  Qwen3 (`model.layers.N.self_attn.q_proj.weight`, `model.embed_tokens.weight`,
  `model.norm.weight`). 351 of them are the vision tower (`visual.*`, needed only
  for image editing) and one is `lm_head` (unused). **It carries no tokenizer.**
* **The tokenizer matches `Qwen3-0.6B-Q8_0.gguf`, which is already on disk.**
  The tokens (151 936), merges (151 387), token types, `pre = qwen2`, eos 151645
  and `add_bos_token = false` all equal those in the `qwen3vl` GGUF, so load the
  tokenizer from the 0.6B file.
* **The VAE file** holds the encoder (102 tensors, needed only for editing) and
  the decoder. All tensors are BF16 with Wan-style names, and every conv kernel
  has temporal extent 1.
* **Not downloaded:** `qwen3vl_8b_int8_convrot.safetensors`. It uses a
  ComfyUI-specific rotated-int8 format that has no spec we can get at.

## Sources

The spec is diffusers `main` at `6256aa7666` ("Add Qwen-Image 2.1 (#14804)",
merged 2026-09-18):

* `src/diffusers/models/transformers/transformer_qwenimage21.py`
* `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`
* `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`
* `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
* configs: `Qwen/Qwen-Image-2.1/{transformer,vae,scheduler,text_encoder}/config.json`

**Runnable reference:** stable-diffusion.cpp (`docs/qwen_image_2.1.md`). It
produced this exact GGUF and runs on Metal, so it plays the role llama.cpp
played for the LLMs.

**Text-encoder half:** llama.cpp supports `qwen3vl`, so this half can be checked
with `llama-eval-callback` + `scripts/parse-eval-callback.py`, the same way
qwen3 was. That needs the Qwen GGUF, not the safetensors.

## Pipeline

```
prompt ─template─▶ tokens ─Qwen3-VL-8B, 36 layers─▶ last layer output, BEFORE the final norm
                                                    drop first 14 rows ─▶ ctx [L, 4096]
noise z ~ N(0, 1)  [64, H/16, W/16]
for each of 40 sigmas:   v = DiT(z, ctx, σ_i);   z += (σ_{i+1} − σ_i) · v
z · std + mean ─▶ VAE decoder ─▶ RGBA in [−1, 1] ─▶ PNG
```

**Sizes:**
* One latent token covers a 16×16 pixel tile: 512² → 1 024 image tokens,
  1024² → 4 096, 2048² → 16 384.
* Width and height must be multiples of 32, so the latent grid has even sides.
* 2048² is native. The listed aspect ratios are 2400×1792, 2528×1696 and
  2752×1536, plus their transposes.
* Smaller sizes run, but their quality hasn't been assessed.

## Text encoder (Qwen3-VL-8B-Instruct, text path only)

| | |
|---|---|
| layers | 36 |
| d_model | 4096 |
| attention | 32 q heads / 8 kv heads, head_dim 128, per-head q/k RMSNorm |
| FFN | 12288, SwiGLU |
| RoPE | θ = 5e6, MRoPE sections [24, 20, 20], interleaved |
| RMS eps | 1e-6 |

This is the dense qwen3 stack llmoxide already runs (`crates/model/src/qwen35/`,
`hybrid = false`).
* **MRoPE collapses to plain RoPE here.** MRoPE assigns each rotary frequency to
  the t, h or w position axis. A text token has the same position on all three,
  so for a text-only prompt MRoPE is ordinary NeoX RoPE.
* **DeepStack is inactive.** It injects vision features only at image positions,
  and a text-to-image prompt has none.

**The prompt, verbatim.** The pipeline builds this string itself rather than
calling the chat template, because the two tokenize differently. It ends with
`assistant` and a newline. An empty prompt becomes a single space.

```
<|im_start|>system
Comprehend and analyze the provided prompt.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

**Conditioning.** Take the output of the last decoder layer, **before
`model.norm`**, for every token, and drop the first 14 rows.
* The 14 rows are the system message `<|im_start|>system\n…<|im_end|>\n`:
  `[151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198]`
  (checked with `llama-tokenize`).
* Diffusers gets that count by tokenizing the system message, so recompute it
  rather than hardcoding it if the system prompt ever changes.
* Encode special tokens as ids. The `tok` binary treats `<|im_start|>` as
  literal text.
* In llmoxide terms the conditioning is the `l_out-35` trace point of
  `qwen35::cpu::forward_traced` (`crates/model/src/qwen35/cpu.rs:401`), not
  `result_norm`.
* The prompt is a few dozen tokens and runs once per image, so the CPU is fine.

**Work:**
* Accept `general.architecture = "qwen3vl"` in `qwen35/config.rs` and
  `model::Arch`, or load the safetensors under their HF names.
* Add a forward that returns every row of the final `h` instead of logits.

## Diffusion transformer

The config is the diffusers defaults, which equal the HF config. None of it is in
the GGUF.

| | |
|---|---|
| blocks | 32 single-stream: text and image share one sequence and the same weights |
| d_model | 4096 = 32 heads × 128 |
| MLP | 12288 (ratio 3), SwiGLU |
| latent in/out | 64 channels, patch size 1 (no patchify) |
| context in | 4096 (the Qwen3-VL hidden size) |
| RoPE | 3 axes (frame, height, width) with (16, 56, 56) dims, θ = 10 000 |
| eps | 1e-6 everywhere, `text_norm` included |
| biases | none anywhere |

Tensors (297 in total; GGUF dims are `[in, out]`):

| tensor | shape | type |
|---|---|---|
| `img_in.weight` | 64 → 4096 | BF16 |
| `txt_in.text_norm.weight` | 4096 | BF16 |
| `txt_in.in_layer.weight`, `txt_in.out_layer.weight` | 4096 → 4096 | BF16 |
| `time_text_embed.timestep_embedder.linear_1.weight` | 256 → 4096 | Q8_0 |
| `time_text_embed.timestep_embedder.linear_2.weight` | 4096 → 4096 | Q8_0 |
| `modulation.1.weight` | 4096 → 16384 | Q8_0 |
| `transformer_blocks.N.attn.to_{q,k,v}.weight`, `.to_out.0.weight` | 4096 → 4096 | Q8_0 |
| `transformer_blocks.N.attn.norm_{q,k}.weight` | 128 | BF16 |
| `transformer_blocks.N.img_mlp.{gate_layer,proj}.weight` | 4096 → 12288 | Q8_0 |
| `transformer_blocks.N.img_mlp.out.weight` | 12288 → 4096 | Q8_0 |
| `norm_out.linear.weight` | 4096 → 4096 | Q8_0 |
| `proj_out.weight` | 4096 → 64 | Q8_0 |

Forward pass for text-to-image, batch 1, with L text rows and
N = (H/16)·(W/16) image rows:

```
ctx  = out_layer(gelu_tanh(in_layer(rms(ctx) · (1 + text_norm))))  # [L, 4096]
img  = img_in(z)                                                  # [N, 4096], raster order
x    = [ctx ; img]                                                # text first, image appended

sinus(t)[i]       = cos(1000·t·f_i)                # i < 128, f_i = 10000^(−i/128)
sinus(t)[128 + i] = sin(1000·t·f_i)
temb(t) = linear_2(silu(linear_1(sinus(t))))
mod(t)  = modulation.1 · silu(temb(t))            # 4 × 4096: scale1 | gate1 | scale2 | gate2
          image rows use mod(σ), text rows use mod(0)            ("causal_condition")

every block, with the SAME mod for all 32 blocks:
  a      = layer_norm(x) · (1 + scale1)           # no affine, no shift
  q,k,v  = to_q·a, to_k·a, to_v·a
  q, k   = rms_norm per head, × norm_q / norm_k; then rope3
  x     += tanh(gate1) · to_out(softmax(q·kᵀ / √128 + mask) · v)
  m      = layer_norm(x) · (1 + scale2)
  x     += tanh(gate2) · out(silu(gate_layer·m) · proj·m)

v = proj_out(layer_norm(x) · (1 + norm_out.linear · silu(temb(σ))))   # image rows only; no shift
```

**Attention mask:** `(q ≥ k) or (q and k are in the same image block)`. For
text-to-image, text rows are causal over the text, and image rows see every text
row and every image row.

**Positions (rope3):**
* Text row j, counted from 0 after the 14 dropped rows, has position (j, j, j).
* Every image row has frame position L. Its (h, w) are centred on zero:
  h ∈ [−(H′ − ⌊H′/2⌋), ⌊H′/2⌋), which is −32 … 31 for H′ = 64. **Negative
  positions are real.**
* Frequencies per axis are `θ^(−2i/d)` for i < d/2. That gives 8 pairs for
  frame, 28 for height and 28 for width, concatenated in that order into 64
  pairs.
* **Pairs are interleaved:** (x[2j], x[2j+1]) is rotated as one complex number
  (diffusers' `use_real=False`, `view_as_complex`). This is not NeoX's pairing
  of (x[j], x[j+64]).

**Prefix cache:**
* Text rows never attend to image rows, and they are modulated at t = 0. So
  their K and V (after norm and RoPE) are the same at every step.
* Run the L text rows through the 32 blocks once, causally, and keep K and V
  per layer.
* Each denoising step then runs only the N image rows against
  [cached text K/V ; their own K/V], with no mask.
* This is the existing llmoxide pattern: a KV cache plus the `bidi` batch flag
  in `attn.wgsl`.
* With CFG, the negative prompt needs its own prefix cache.

**Things that will bite.** Each of these gives a plausible image or a blur
rather than an error:

1. **The text-encoder output must be taken before the final norm.** Diffusers
   had to hook around transformers 5.x, which returns the normed state. Its
   comment says the normed version is "a third of the signal … shows up first
   in rendered text".
2. **`txt_in.text_norm` is zero-centred.** The scale is `weight + 1`, computed
   in f32. Using the raw weight scales by roughly zero.
3. **Modulation is global.** A single `modulation.1` output is reused by every
   block, and there are no per-block modulation tensors. It has **two rows**,
   because text uses t = 0.
4. **The timestep sinusoid is cos first, then sin**, with argument `1000·σ`.
   The pipeline passes `t / 1000 = σ`, and the embedder multiplies by 1000 again.
5. **There is no shift anywhere.** Every adaptive norm is scale-only,
   `x · (1 + s)`, and the gates go through **tanh**.
6. **RoPE uses interleaved pairs**, three axes, and signed image positions.
7. **The GGUF has no metadata**, so a wrong hardcoded constant (eps, θ, the axis
   split) won't fail loudly.

## Sampler

Flow-matching Euler: `FlowMatchEulerDiscreteScheduler` with
`use_dynamic_shifting`, `time_shift_type = exponential` and
`shift_terminal = 0.02`. The default is 40 steps.

```
n    = image tokens = (H/16)·(W/16)
mu   = 0.5 + (0.9 − 0.5)·(n − 256)/(8192 − 256)     # not clamped: 2048² gives 1.313
s_i  = linspace(1, 1/steps, steps)
s_i  = e^mu / (e^mu + 1/s_i − 1)
s_i  = 1 − (1 − s_i)·(1 − 0.02)/(1 − s_last)        # stretched so the last one is 0.02
append 0
z   += (s_{i+1} − s_i) · DiT(z, σ = s_i)
```

Reference values, usable as unit-test vectors:

| n (size) | steps | mu | first sigmas | last sigmas |
|---|---|---|---|---|
| 256 (256²) | 4 | 0.500000 | 1.0, 0.744611, 0.426673 | 0.02, 0 |
| 4 096 (1024²) | 40 | 0.693548 | 1.0, 0.986964, 0.973593, 0.959875 | 0.113533, 0.067881, 0.02, 0 |
| 16 384 (2048²) | 40 | 1.312903 | 1.0, 0.992646, 0.985013, 0.977085 | 0.175179, 0.10223, 0.02, 0 |

**Guidance (CFG):**
* Diffusers defaults to **none** (`true_cfg_scale = 1`; its docs say the model is
  "meant to be sampled without guidance").
* stable-diffusion.cpp's example uses `--cfg-scale 6.0`.
* With CFG, `v = v_neg + s·(v_pos − v_neg)`. That is two DiT passes per step,
  so it doubles the cost. Start without it.

**Noise:** standard normal `[64, H/16, W/16]`, flattened to rows in raster
order: row = y·W′ + x, with 64 channels per row. No two random generators agree,
so matching a reference means injecting its noise tensor.

## VAE decoder

`AutoencoderKLQwenImage21` is Wan-2.2-style: 64 latent channels, 16× spatial
compression, **RGBA** output. It is named 3-D, but for a single image every op is
2-D, since all conv kernels in the file have temporal extent 1.

```
z   = z · latents_std + latents_mean        # per channel, 64 values each (below)
x   = conv2                                 # 1×1, 64 → 64, "post_quant_conv"
x   = decoder.conv1                         # 3×3, 64 → 1152
x   = decoder.middle                        # resblock → attention → resblock, 1152
x   = 5 up stages                           # table below
x   = decoder.head                          # rms → silu → 3×3 conv 144 → 4
out = clamp(x, −1, 1)                       # RGBA; (out + 1)/2 · 255 for 8-bit
```

| stage `decoder.upsamples.S` | channels | resblocks run at | then | shortcut from stage input |
|---|---|---|---|---|
| 0 | 1152 → 1152 | H/16 | ×2 nearest, 3×3 conv | DupUp, ft = 2 |
| 1 | 1152 → 1152 | H/8 | ×2 nearest, 3×3 conv | DupUp, ft = 2 |
| 2 | 1152 → 576 | H/4 | ×2 nearest, 3×3 conv | DupUp, ft = 2 |
| 3 | 576 → 288 | H/2 | ×2 nearest, 3×3 conv | DupUp, ft = 1 |
| 4 | 288 → 144 | H | none | none |

**Inside each stage:**
* `upsamples.{0,1,2}` are resblocks. The first one changes the channel count and
  has a 1×1 `shortcut` when in ≠ out.
* `upsamples.3.resample.1` is the 3×3 conv after the ×2 nearest upsample, which
  is plain pixel duplication.
* The stage output is resblocks → upsample, **plus** a parameter-free DupUp of
  the stage input:

```
DupUp(x)[o, 2h + b, 2w + d] = x[(((o·ft + ft − 1)·2 + b)·2 + d) / r, h, w]      # integer division
   r = out_ch · ft · 4 / in_ch      # channel repeat + pixel shuffle, keeping the last
                                    # temporal copy (diffusers' first_chunk slice)
```

**Layers:**
* **Resblock** (`residual.{0,2,3,6}`):
  `shortcut(x) + conv6(silu(rms3(conv2(silu(rms0(x))))))`.
* **Norms are RMS over channels, per pixel:** `x / ‖x‖₂ · √C · gamma` (this is
  `F.normalize`, with eps 1e-12 on the norm).
* **Attention (`middle.1`):**
  * the same channel RMS, then a 1×1 `to_qkv` with bias (q, k, v are channels
    [0, C), [C, 2C), [2C, 3C))
  * **one head** of 1152 over all (H/16)·(W/16) positions, softmax scale 1/√1152
  * a 1×1 `proj`, then the residual
* **Convs:** zero padding of 1 on every side. The `[out, in, 1, 3, 3]` weights
  drop their singleton dimension; the `resample.1` weights are already
  `[out, in, 3, 3]`.

**Things that will bite:**

1. **Skip `time_conv` for a single image.** The weights
   `upsamples.{0,1,2}.upsamples.3.time_conv` exist, but for a single frame the
   decoder's feature cache marks the first chunk "Rep" and never calls
   `time_conv`. Applying them doubles the channels and scrambles the image.
2. **`latents_mean` and `latents_std` are only in `vae/config.json`**, not in
   any weight file:

```
latents_mean =
     0.5126,  0.7721, -0.0631,  1.3506, -0.7855, -2.1025, -0.3458,  1.3722,
     1.8873, -1.7177, -0.6510,  0.2732,  0.7562, -0.6163, -1.0277,  3.8363,
     2.0210,  0.0472,  0.9320,  2.0087,  2.4954, -0.1391, -1.4249,  1.8464,
    -0.5236,  1.2826,  3.7046, -1.3035,  2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636,  3.0555,  0.0746, -3.0751, -0.1076,  1.7376,
    -1.0914, -1.9435, -0.2784, -1.3680,  0.4809, -0.4433,  0.3764,  0.5729,
    -2.0595,  1.0960, -1.3260, -2.0211, -5.0179,  0.5275,  4.0162,  1.8505,
     0.3026,  1.9373,  1.4937,  0.2632,  0.5547, -1.7121, -0.1562,  0.0304
latents_std =
     3.2001,  3.2936,  3.4321,  3.0091,  3.1061,  4.0379,  4.0705,  3.7910,
     3.0785,  3.6500,  3.9308,  3.0904,  2.8778,  3.7675,  3.7320,  5.0756,
     3.2864,  4.0397,  3.1317,  4.0443,  2.9249,  3.9454,  3.0988,  4.2489,
     3.4896,  3.8513,  3.9323,  3.4719,  3.7498,  4.2830,  3.5694,  4.2467,
     3.9037,  3.2947,  5.0770,  3.5075,  3.2700,  3.4767,  2.8063,  5.1125,
     3.5327,  4.7833,  3.1286,  4.1819,  3.8527,  3.8312,  3.5605,  4.3875,
     3.9624,  4.0168,  3.5643,  4.0550,  5.5614,  4.2963,  4.4080,  3.4959,
     3.8747,  3.7608,  3.5735,  3.1490,  3.7662,  3.6746,  3.4563,  3.8161
```

3. **Activations get large.** At 1024², the f32 tensors entering stage 4 and
   leaving stage 3's upsample are 1024² × 288 × 4 B ≈ 1.2 GB each; at 2048² they
   are four times that. Decode in tiles. Diffusers' tiled decode uses 256 px
   tiles with a 192 px stride and blends the overlaps.

## Compute budget

**Per DiT step:**
* Linear FLOPs = 2 · N · 218.1 M · 32, where 218.1 M is the weight count of one
  block.
* Attention ≈ 4 · N · (N + L) · 4096 · 32.
* The text prefix is cached, so it adds almost nothing per step.

| output | image tokens | DiT per step | × 40 steps | VAE decode |
|---|---|---|---|---|
| 512² | 1 024 | ~15 TFLOP | ~0.6 PFLOP | ~4 TFLOP |
| 1024² | 4 096 | ~66 TFLOP | ~2.6 PFLOP | ~17 TFLOP |
| 2048² | 16 384 | ~370 TFLOP | ~15 PFLOP | ~67 TFLOP |

**Time estimate:**
* The text encoder is about 0.5 TFLOP in total, which is negligible.
* Assume an effective 3–4 TFLOPS, a plausible but unmeasured target for a
  hand-tiled WGSL GEMM on the M4 Pro. Then 1024² takes about 11–15 min per
  image, 512² about 3 min and 2048² about an hour.
* CFG doubles the DiT part.
* The CPU path is only for validation at 256².

**Memory:**
* Weights: DiT 7.6 GB + text encoder 17.5 GB in BF16 (8.7 GB as the Q8_0 GGUF;
  free it after encoding) + VAE 0.7 GB. That fits in 48 GB.
* DiT activations at 1024²: about 4 100 rows × 12 288 × 4 B ≈ 200 MB per MLP
  buffer.

## What exists vs what to build

| need | in llmoxide today | to add |
|---|---|---|
| GGUF Q8_0 + BF16 | `gguf` crate, which opens the metadata-less DiT (checked) | hardcoded DiT config + shape asserts |
| safetensors | nothing | reader: u64 header length + JSON header + mmap (`serde_json` and `memmap2` are already deps); BF16 |
| tokenizer | `qwen2` pre-tokenizer; identical vocab in the Qwen3-0.6B GGUF | special-token-aware encode (the chat crate's path) |
| text encoder | dense qwen3 in `model::qwen35` | `qwen3vl` arch or safetensors names; return pre-norm rows |
| DiT | none | new module: loader, CPU forward, prefix cache |
| CPU ops | Q8_0/BF16 `matmul`, `rms_norm_mul`, `silu`, `gelu`, SwiGLU | layer norm without affine, modulate, 3×3 conv2d, upsample |
| GPU | `matvec_*_t` (matvec over a small token tile, no shared-memory reuse); `attn.wgsl` (n² scores, `bidi`); `rope`, `rope_2d`, `swiglu`, `rms_norm` | **tiled GEMM** (Q8_0, BF16); **flash-style attention**; layer_norm + modulate; tanh-gated residual; 3-axis interleaved RoPE (a per-token cos/sin table is simplest); conv2d; upsample / pixel shuffle |
| sampler | token sampling (`sample.rs`) | sigma schedule, Euler step, Gaussian RNG |
| output | `image` crate (png) in `crates/vision` | RGBA PNG writer |
| CLI | `llmoxide` binary | `image` subcommand: prompt, size, steps, seed, cfg |

**The GEMM is what makes or breaks this:**
* `matvec_q8_0_t` was built for decode and short prefills. It re-reads
  activations from global memory for every output row.
* At 4 096 rows × 40 steps, a real GEMM is the difference between minutes and
  hours.
* `attn.wgsl` materializes the score matrix. At 1024² that is
  4 096 × ~4 100 × 32 heads × 4 B ≈ 2.2 GB per layer.

## Build order

1. **safetensors reader.** Verify it on the VAE: 238 BF16 tensors, and the data
   ends exactly at EOF.
2. **Text encoder.** Dense qwen3 on the Comfy BF16 weights (or the `qwen3vl`
   GGUF), returning the `l_out-35` rows. Validate against llama.cpp's
   `llama-eval-callback` on the `qwen3vl` GGUF.
3. **DiT CPU reference.** One denoiser call at 256² (256 image tokens).
   Unit-test the sigma table and the timestep embedding.
4. **Sampler + VAE on CPU.** Produce a first PNG at 256². For the visual check,
   use a prompt with text in it ("a cat holding a sign that says 'llmoxide'"),
   since rendered text is the first thing to break.
5. **Compare with stable-diffusion.cpp.** Use the same prompt, the same noise
   (inject it; check what `sd-cli` allows) and the same steps.
6. **GPU.** The GEMM, flash attention and the small kernels, then the VAE convs.
   Diff against the CPU reference as was done for the vision tower (1.7e-5
   there).
7. **CLI + `llmoxide-fetch` aliases** for the three files.

**Out of scope for now:**
* Image editing. It needs the Qwen3-VL vision tower (`visual.*` in the same
  safetensors) and the VAE encoder.
* Transparency prompts ("This is an RGBA image with transparency. …").
* The prompt-rewriter models (`Qwen/Qwen-Image-2.1-PE-T2I`).
