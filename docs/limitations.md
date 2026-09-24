# Known limitations

- gemma4 decode is ~1.5x slower than llama.cpp, and the gap widens with
  context (see [Performance](performance.md)).
- qwen35 decode at ~9.4 tok/s is within ~10% of what this GPU's memory
  bandwidth allows for a 22 GB checkpoint (see [Performance](performance.md)). The delta-net
  recurrence is sequential by construction: `delta_recur` loops the whole token
  range inside one dispatch of `n_v_heads` workgroups, so those layers get no
  token parallelism during prefill and cannot fill the GPU. So far that has
  cost little: 301-token prefill takes about as long as its GEMM FLOPs alone
  predict. It will matter more for long prompts, where a chunked formulation is
  the fix.
- No architecture here can rewind its cache, so prompt reuse is append-only:
  gemma4's ring buffers have overwritten the positions, and qwen35's recurrent
  state was never a history to begin with.
- No speculative decoding, though two of the checkpoints ship a drafter for it:
  qwen35's MTP/NextN head (`blk.64`) is skipped at load, and Gemma 4 publishes a
  separate MTP drafter (`google/gemma-4-E4B-it-assistant`). Wiring either up
  needs a KV cache that can rewind on a rejected draft, which is the same gap as
  the entry above.
- Single request at a time, one GPU context; no batching across clients.
- The vision tower is ~3-4x off llama.cpp on Metal (0.87 s against ~0.2 s for
  a 1 170-patch image). Time grows faster than the patch count — 441 patches
  take 0.25 s, 2 304 take 2.33 s — because attention over patches is O(n^2)
  and `weighted_v` is shaped for the text model's 256- and 512-wide heads: at
  the tower's 64-wide ones it leaves three quarters of each workgroup idle.
  Fixing that means a second variant of a kernel the text path depends on,
  which is why it has not been done yet.
- Audio input is not implemented, though the same `mmproj` file carries a
  conformer encoder for it (`a.blk.*`, 12 blocks of 1024) and the vocabulary
  has the `<|audio>` pair to bracket it.
- One image per request is what has been exercised; several should work and
  are not tested. Each is capped at 256 positions so it prefills as one
  bidirectional batch.
- The context is capped at `LLMOXIDE_CTX`, well below the models' 262144,
  since attention scratch scales with it.
- Private mode covers this process, not the machine, and not the server: see
  [What this does not cover](privacy.md#what-this-does-not-cover) and [Serving](serving.md).
- The embedded (`--embed`) build is a 530 MB HTML file (854 MB at Q8_0) with no
  download progress and no separate caching of the model. It exists for offline
  distribution; hosting wants the two-file form.
- The browser build needs WebGPU and has no CPU fallback — wasm32's 4 GB
  address space cannot hold the weights at any useful quantization. It also
  cannot `mlock`, so it is the one entry point where "leaves nothing behind" is
  a weaker claim than elsewhere. See [In a browser](browser.md).
- Qwen3 0.6B is the only checkpoint here small enough to serve over the web,
  and it is a 0.6B model: fine for short exchanges, visibly limited past that,
  and prone to answering *about* your question rather than answering it. There
  is no larger option under a gigabyte in either family — see [Why not a
  smaller Gemma](browser.md#why-not-a-smaller-gemma).
- `qwen3` support is dense-attention only. The delta-net path it shares a module
  with is exercised by the 27B, not by any small checkpoint, so a regression
  there needs the 22 GB file to catch.
- The browser build's correctness rests on the native `bisect`, not on a check
  that runs in a browser: comparing per-checkpoint tensors there would mean
  shipping the CPU reference path, which is exactly what does not fit. The
  kernels are byte-identical WGSL and `Gpu::check_shaders` proves they compiled,
  but nothing verifies the browser's *numerics* against the CPU the way
  `bisect` does natively.
