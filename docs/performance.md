# Performance

On an M4 Pro (20 GPU cores), measured, not projected. Decode is quoted with the
context it ran at, because attention cost grows with it:

| gemma4 | before | now | llama.cpp (Metal) |
|---|---|---|---|
| decode @ 128 ctx | 12.7 tok/s | **22.3 tok/s** | 32.6 tok/s |
| decode @ 512 ctx | 8.6 tok/s | **18.4 tok/s** | |
| decode @ 2048 ctx | 5.0 tok/s | **12.9 tok/s** | |
| prefill, 374 tokens | 23.5 tok/s | **35.0 tok/s** | much higher |
| prefill, 1490 tokens | 20.7 tok/s | **32.7 tok/s** | |

| qwen35 27B (Q6_K) | before | now |
|---|---|---|
| prefill, 301 tokens | 14.0 tok/s | **78.7 tok/s** |
| decode | 8.9 tok/s | 9.4 tok/s |

The CPU reference path does ~1.1 tok/s decode and ~0.6 tok/s prefill.

Prefill gained 5.6x from a real GEMM. The old prefill path reused the decode
matvec across 2-token tiles, so a 300-token prompt read all 22 GB of weights 150
times. Now each workgroup dequantizes a weight tile once into threadgroup memory
and uses it for a whole tile of tokens. There are two implementations, picked in
`QuantKernels::new`:

- **`shaders/gemm_q6k.metal`**, on Apple GPUs. It is hand-written MSL, loaded
  through wgpu's experimental MSL passthrough, and runs the inner product on
  `simdgroup_float8x8` hardware matrix multiply-accumulates. WGSL cannot express
  these. It reaches ~4.4 TFLOP/s on an M4 Pro, all in f32. Staging in f16 was
  measured: it was no faster and 100x less accurate. Nothing checks the kernel
  against the pipeline layout, so its header spells out the buffer-slot mapping
  it relies on.
- **`shaders/gemm.wgsl`**, everywhere else, including the browser. It uses a
  register-blocked vec4-FMA tile and tops out near 2.5 TFLOP/s on the same GPU.
  `LLMOXIDE_NO_MSL=1` forces it on a Mac, so it stays tested.

Batches under 16 tokens stay on the matvec, as do the 48-row `ssm_alpha`/`ssm_beta`
projections. With the GEMM in place, prefill is almost entirely GEMM time:
16 TFLOP for 301 tokens at 4.4 TFLOP/s.

Decode is memory-bound and was already near the limit. The Q6_K matvec streams
~215 GB/s. A kernel that does nothing but read the same bytes reaches only
~225–242 GB/s (`GEMM_READBW=1` in the `gemm` bin). A decode kernel rewrite could
therefore buy at most ~10%.

`gemm` is the fast loop for this work. It loads only the named tensors, so it
runs in seconds against the 22 GB checkpoint where a full load takes minutes. It
times the matvec against the GEMM and checks both against the CPU reference:

```sh
cargo build --release -p llmoxide-gpu --bin gemm
./target/release/gemm models/Qwen3.8-27B-Q6_K.gguf 301        # n_tokens
./target/release/msl --gemm                                   # the WGSL GEMM as Metal
```

Three things about the WGSL GEMM's generated Metal are easy to trip on and are
written up at the top of `gemm.wgsl`. The main one: Naga silently drops
dynamic-component stores into workgroup vectors (`ws[i][c] = v`).

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
partials afterwards is the fix. gemma4 prefill still uses the matvec. The GEMM
only has a Q6_K path so far, and gemma4's checkpoints are mostly Q4_K, so
porting the dequant step is the missing piece.

Two dead ends are written up in `crates/gpu/src/shaders/quant.wgsl` so they
don't get retried: hoisting the dequantization above the token loop, and
widening the tile. Both look like obvious wins and both lose.

Most of that was settled by reading the generated Metal rather than guessing —
whether an accumulator reaches a register is not visible in the WGSL:

```sh
# `msl` pulls in naga, so it is behind a non-default feature rather than in
# every consumer's dependency tree.
cargo build --release -p llmoxide-gpu --features tools --bin msl
./target/release/msl              # quant kernels as MSL, as wgpu compiles them
./target/release/msl --barrier    # ...with the non-subgroup reduction
```

## Two silent failures at 27B scale

Both return all-zero buffers with no error, and both are guarded now — details
in [ARCHITECTURE-qwen35.md](../ARCHITECTURE-qwen35.md):

- **Uploads past the working set.** Mapping every arena buffer at once doubles a
  25 GB model with staging shadows and the writes are simply dropped.
  `Weights::upload` fills and flushes one buffer at a time and reads sampled
  spans back; `upload_check` sweeps every tensor head.
- **Over-long command buffers.** A whole forward pass is ~1500 dispatches, which
  trips Metal's GPU watchdog on cold pipelines and long prefills.
  `Qwen35Gpu::run` chunks at 64 dispatches per submit, and a device-lost
  callback in `Gpu::new` turns any future watchdog kill into a printed message.

## KV cache and prefix reuse

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

Reproducing the *text* is not enough, and this cost a silent regression: the
replay emitted the empty block as `text("\n") + text("") + text("\n")`, which
encodes to `[198, 198]`, while the generation prompt emits `text("\n\n")`,
which the BPE merges into the single id `271`. Identical strings, different
ids, and the cached prefix ended at that token — every qwen multi-turn
conversation re-prefilled from scratch. Prompt pieces have to be encoded in
the same *groupings*, not just the same order, because merges do not cross a
call boundary.
