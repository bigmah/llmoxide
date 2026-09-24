# In a browser

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

## One file, model included

```sh
scripts/build-web.sh --embed models/Qwen3-0.6B-Q4_K_M.gguf
```

Bakes the checkpoint into the page. Nothing else is needed at runtime: no
server, no network, no second file. Verified by running Chrome with DNS
blackholed (`--host-resolver-rules=MAP * 0.0.0.0`) against a `file://` URL, and
in Safari 26.6.

| embedded checkpoint | page | ready |
|---|---|---|
| `qwen3-0.6b-q4` (0.40 GB) | 530 MB | 1.0 s |
| `qwen3-0.6b` (0.64 GB) | 854 MB | 2.1 s |

The base64 has to arrive in pieces. V8 caps a single string at 536,870,888
characters and the Q8_0 model's base64 is 852,596,992 — so one blob is not
slow, it is unbuildable. `build-web.sh` emits 48 MB chunks either way and the
page drops each from the DOM as it decodes, since those strings are the largest
objects on it.

**Compressing the model buys nothing.** Quantized weights are close to random:
measured on this Q8_0, `gzip -9` gets 4.5% off and `zstd -19` 4.8%, which does
not pay for a decompressor in the page. What *is* worth doing is serving the
page with `Content-Encoding: gzip`, which takes the Q8_0 embedded build from
854 MB to **638 MB over the wire** — the base64 tax refunded almost exactly,
for one line of server config and no code.

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

## Why not a smaller Gemma

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

## Why the weights never enter wasm memory

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

## Two shaders that compile natively and not in a browser

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

## What does not survive the port

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
