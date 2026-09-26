# Models and downloads

Five checkpoints, each validated against tensor-by-tensor. All run on the GPU;
the CPU forward passes stay in the tree as the reference every kernel is checked
against, not as a fallback.

| alias | file | arch | size | sha256 |
|---|---|---|---|---|
| `gemma4` | `gemma4-v2-Q4_K_M.gguf` | gemma4, 12B, 48 layers | 7.38 GB | `0b9506ca…` |
| `gemma4-e4b` | `gemma-4-E4B-it-Q8_0.gguf` | gemma4, E4B, 42 layers | 8.03 GB | `34be82b1…` |
| `gemma4-e4b-q4` | `gemma-4-E4B-it-Q4_K_M.gguf` | gemma4, E4B, 42 layers | 5.34 GB | `d35a3aa7…` |
| `gemma4-e4b-mmproj` | `mmproj-gemma-4-E4B-it-BF16.gguf` | gemma4v vision tower | 0.99 GB | `bdfc4935…` |
| `qwen35` | `Qwen3.8-27B-OBLITERATED-Q6_K.gguf` | qwen35, 27B hybrid, 64 layers | 22.43 GB | `3535d4a1…` |
| `qwen3-0.6b` | `Qwen3-0.6B-Q8_0.gguf` | qwen3, 0.6B dense, 28 layers | 0.64 GB | `e150ed54…` |
| `qwen3-0.6b-q4` | `Qwen3-0.6B-Q4_K_M.gguf` | qwen3, 0.6B dense, 28 layers | 0.40 GB | `ac2d9771…` |

From [`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
and [`OBLITERATUS/Qwen3.8-27B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
respectively, and `gemma4-e4b` from
[`ggml-org/gemma-4-E4B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF).
`qwen3-0.6b` is the one small enough to *serve*: 0.64 GB, from
[`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF).
It is Q8_0 rather than a Q4 because at 0.6B the quantization error is the first
thing you notice, and 0.64 GB is already inside any sane download budget. There
is nothing smaller worth having in either family — see
[Why not a smaller Gemma](browser.md#why-not-a-smaller-gemma).

`gemma4-e4b-q4` is the same E4B at a size a browser tab will actually allocate,
from [`lmstudio-community`](https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF) —
that repo rather than the obvious ones because a GGUF is only loadable here if
*every* tensor is a type the decoders handle. ggml-org publishes E4B as Q8_0 or
Q4_0, and there is no Q4_0 decoder; bartowski's Q4_K_M mixes in 84 Q5_K
tensors, which there is also no decoder for. Both are rejected at load rather
than part-way through a forward pass, which is what the structure check in
`llmoxide-fetch` is for — it caught the second one after the download had
already passed its SHA-256.
`qwen35` is the interesting one architecturally: a hybrid stack where three
quarters of the layers are gated delta-net rather than attention.

Plain **qwen3** is not a separate implementation. Read as a config, it *is*
qwen35 with two things switched off: every layer is full attention rather than
delta-net, and `attn_q` carries no fused output gate, so it is `head_dim` wide
per head instead of `2 * head_dim`. Everything else — RMSNorm, GQA, per-dim
query and key norms, NeoX RoPE, SwiGLU — is the same code, so `qwen3` costs one
`Config::query_gate` flag and three branches rather than a second copy of an
attention block that already worked. The 27B is re-checked against the CPU path
at all 771 checkpoints after every change to it.

Two Qwen traps came with that. Its vocabulary uses the **classic `qwen2`
pre-tokenizer split**, which is the qwen35 one with every `\p{M}` term removed —
so combining marks no longer travel with the letters they modify. Picking the
wrong split does not fail, it just tokenizes subtly differently everywhere;
`pre::Marks` makes it one function with marks switched off rather than two
hand-ports that could drift. And Qwen3 ships **no `general.sampling.*`
metadata**, so a single hard-coded fallback would sample it with Gemma's
temperature of 1.0 against Qwen's published 0.7 — which on a 0.6B model reads
as the model being weak rather than the sampler being wrong.
`Sampling::family_default` picks per family.

One chat-format trap comes with it. The two checkpoints disagree on how to turn
*off* reasoning: the 12B's template suppresses it by making the generation prompt
open and immediately close an empty `thought` channel, and E4B's template has no
such suppressor. Emitting the 12B's form to E4B does not disable thinking — the
model finds the channel already closed, never opens another, and writes its
reasoning into the visible answer. `Special::closes_empty_thought` reads which
convention a checkpoint uses off its embedded template rather than assuming.

`gemma4-e4b` is the same architecture as the 12B only in name. The E-series
keeps a narrow 2560-wide residual stream and spends its parameters on a
*per-layer embedding* table instead — a 256-wide vector looked up per token per
layer, gated into the stream at the end of every block. That is what "E4B"
means: 4.5B effective parameters out of 8B on disk. It also shares KV across
the top of the stack, so layers 24–41 project Q only and attend into the cache
of layer 22 (sliding) or 23 (global), and it writes `head_count_kv` as a scalar
where the 12B writes a 48-entry array. Config reads all of this out of the GGUF;
see [`crates/model/src/config.rs`](../crates/model/src/config.rs).

```sh
./target/release/llmoxide-fetch                     # the table above, into models/
./target/release/llmoxide-fetch gemma4              # or one alias (--aliases lists all)
./target/release/llmoxide-fetch hf:owner/repo       # the best file in a repo that will run
./target/release/llmoxide-fetch hf:owner/repo:Q8_0  # ... at one quantization
./target/release/llmoxide-fetch hf:owner/repo/file.gguf
./target/release/llmoxide-fetch https://huggingface.co/owner/repo/blob/main/f.gguf
./target/release/llmoxide-fetch --check hf:owner/repo   # judge every file, download nothing
```

### Anything else on the Hub

The table is what this repo is validated around, not the limit of what it runs.
Every `qwen3`, `qwen35` and `gemma4` GGUF whose tensors are all F32, F16, BF16,
Q8_0, Q4_K or Q6_K loads, and most repos have one. The hard part is knowing
which file before downloading 20 GB of it, so the fetcher answers that first.

A GGUF's metadata and tensor table are at the front of the file. Before any
transfer, `llmoxide-fetch` range-reads that header (4 MB, doubling until it
parses) and puts it through the **real loaders**: `Arch::detect`, the config
parser for that architecture, and `Tokenizer::from_gguf`, over a
`Gguf::sparse` with no tensor bytes resident. There is no second list of what
is supported to drift from the first. A `qwen3moe`, a Q5_K tensor or an unknown
pre-tokenizer is refused up front with the reason (`--no-check` overrides it).

Given a repo, not a file, it lists the repo, drops split shards (the loader
opens one file) and mmproj files, and tries quantizations in the order Q4_K_M,
Q6_K, Q8_0, Q4_K_S, BF16, F16. It takes the first file that passes **and that
the GPU can run**. Two things make a file CPU-only: a `token_embd` that is not
Q6_K, Q8_0 or F32 (there is no gather kernel for other types), and F16 weights
(no F16 matvec). A CPU-only file is taken only if nothing in the repo does
better, and the fetcher says so when it does.

What a scan of the obvious repos found on 2026-09-26, which the extra aliases
below encode:

- **unsloth Q4_K_M is often not Q4_K_M.** The Qwen3.5 and Gemma 4 E-series
  builds mix in Q5_K tensors and are refused. From Qwen3-8B up, the Q4_K_M
  files load but carry a Q4_K `token_embd`, so they are CPU-only. lmstudio's
  Gemma 4 31B Q4_K_M uses a Q6_K embedding and runs on the GPU.
- **The small Qwen3.5s (0.8B to 9B) are the hybrid `qwen35` stack**, like the
  27B. The 0.8B was fetched this way and run: greedy output identical on the
  GPU, the CPU and llama.cpp. The others pass the loaders but have not been run.
- **Gemma 4 E2B does not load.** Its `feed_forward_length` is a per-layer
  array, which the gemma4 config reads as a scalar. That needs a port, not a
  download fix.
- **Nothing MoE runs**: `qwen3moe`, `qwen35moe`, Gemma 4 26B-A4B. Qwen3.5's
  mmproj files are Qwen's vision tower, not gemma4v, and are refused.

Aliases for those sizes. They are fetched by name only; a bare `llmoxide-fetch`
still takes just the table at the top:

| alias | file | size |
|---|---|---|
| `qwen3-1.7b` | unsloth `Qwen3-1.7B-Q4_K_M.gguf` | 1.11 GB |
| `qwen3-4b` | unsloth `Qwen3-4B-Q4_K_M.gguf` | 2.50 GB |
| `qwen3-8b` | unsloth `Qwen3-8B-Q6_K.gguf` | 6.73 GB |
| `qwen3-14b` | unsloth `Qwen3-14B-Q6_K.gguf` | 12.12 GB |
| `qwen3-32b` | unsloth `Qwen3-32B-Q6_K.gguf` | 26.88 GB |
| `qwen3.5-0.8b` | unsloth `Qwen3.5-0.8B-Q6_K.gguf` | 0.64 GB |
| `qwen3.5-2b` | unsloth `Qwen3.5-2B-Q8_0.gguf` | 2.01 GB |
| `qwen3.5-4b` | unsloth `Qwen3.5-4B-Q6_K.gguf` | 3.53 GB |
| `qwen3.5-9b` | unsloth `Qwen3.5-9B-Q6_K.gguf` | 7.46 GB |
| `qwen3.5-27b` | unsloth `Qwen3.5-27B-Q6_K.gguf` (stock, not the abliterated `qwen35`) | 22.45 GB |
| `gemma4-31b` | lmstudio-community `gemma-4-31B-it-Q4_K_M.gguf` | 18.69 GB |

Transfers **resume** — interrupt one, run the same command again, and it picks
up from the byte it stopped at. Everything reports progress, the SHA-256 passes
included, because at 21 GB a verification with no output is indistinguishable
from a hang.

Nothing is called finished until three things agree: the size matches
`x-linked-size`, the SHA-256 matches `x-linked-etag`, and `gguf::Gguf::open`
parses the result — which walks the tensor table and rejects any tensor running
past the end of the file. Only then does the `.part` file take its real name, so
an interrupted fetch can never be mistaken for a complete one. This repo already
lost time to a download that stopped at 17.8 of 20.9 GiB and wrote a
plausible-looking file; that is the check which would have caught it.

If the same checkpoint is already in the directory under a different name it is
hashed and hard-linked rather than downloaded again (`--no-adopt` to disable).
Re-uploads get renamed constantly, and re-fetching 21 GB you already have is the
most expensive mistake available here.

Two traps worth knowing if you touch `crates/hub`:

- A `/blob/` URL — what the website's copy button gives you — returns HTML. It
  is rewritten to `/resolve/`, otherwise you download a few kilobytes of markup
  that looks like a corrupt model.
- A resolve URL 302s to a CDN, and `x-linked-size` / `x-linked-etag` live on
  *that 302*, not on what it points at. Follow the redirect and you get the
  CDN's `etag` instead — the xet content hash, which is also 64 hex characters
  and is not the SHA-256 of the file. Verifying against it fails every honest
  download, at the end, after 21 GB. `probe` stops at the redirect for exactly
  this reason.
