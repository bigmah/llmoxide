# Correctness

Both architectures are genuinely unusual — see [ARCHITECTURE.md](../ARCHITECTURE.md)
for the four things that will silently produce garbage if you assume the
Gemma 2/3 shape, and [ARCHITECTURE-qwen35.md](../ARCHITECTURE-qwen35.md) for the
hybrid stack's own traps (head tiling, l2-norm eps, the fused query gate).
Everything is checked against llama.cpp rather than asserted:

| what | check |
|---|---|
| gemma4 tokenizer | exact id-for-id match with `llama-tokenize` on 13 cases + a 3547-token file |
| gemma4 CPU forward | **byte-identical** greedy output to `llama-completion --temp 0` |
| gemma4 GPU kernels | every matvec within ~1e-7 of the CPU dequant-dot, on real weights |
| gemma4 GPU forward | all 773 intermediate tensors match the CPU path across 48 layers (~1e-6) |
| gemma4v vision tower (CPU) | written against llama.cpp's `clip_graph_gemma4v`; a transpose-sensitive test image places both squares correctly, and the model reads the headline and date off a newspaper photo |
| gemma4v vision tower (GPU) | all 332 800 output values within 1.7e-5 of the CPU reference on a real image, at 441, 1 170 and 2 304 patches (`vision_check`) |
| E4B CPU forward | 674 tensors across all 42 layers traced against `llama-eval-callback`; KV-sharing boundary matches exactly (llama.cpp emits `Kcur` for layers 0–23 only) |
| E4B GPU forward | **471/471** checkpoints match the CPU path across 42 layers (~1e-6), per-layer embeddings included |
| E4B chat format | `reasoning_content` / `content` split matches `llama-server --jinja` on the same request, thinking on and off |
| qwen35 tokenizer | exact id-for-id match with `llama-tokenize` on 18 cases + 3 files (~11k tokens) |
| qwen35 CPU forward | 567 tensors across all 64 layers match `llama-eval-callback` within 4e-4 on a real 27B; **byte-identical** greedy output to `llama-completion --temp 0` |
| qwen35 GPU forward | **771/771** checkpoints match the CPU path on the 27B (logits rel 1.3e-6, same argmax) |
| qwen35 chat format | prompt ids match a jinja2 rendering of the embedded template, with and without tools |
| qwen3 tokenizer | exact id-for-id match with `llama-tokenize` on 21 cases and 6 files (30 610 tokens), with `--no-escape`; the only divergences are literal `<think>`/`<tool_call>` spellings in raw text, which are deliberately not matched as control tokens and which the validated qwen35 path treats the same way |
| qwen3 CPU forward | **byte-identical** greedy output to `llama-completion --temp 0` on 6 prompts / 288 tokens, code and prose |
| qwen3 GPU forward | **339/339** checkpoints match the CPU path across 28 layers, same argmax |
| qwen3 chat format | single-turn prompt ids match a jinja2 rendering of the embedded template exactly; multi-turn deliberately differs, see below |
| browser kernels | the WGSL rewrites for Tint leave `bisect` at **all checkpoints match** on E4B Q4_K_M, and `kernels` matching the CPU dequant-dot on *both* reduction paths (`LLMOXIDE_NO_SUBGROUP=1` is the one the browser takes) |

Tools that reproduce this:

```sh
./target/release/kernels        models/gemma4-v2-Q4_K_M.gguf   # per-kernel vs CPU
./target/release/bisect         models/gemma4-v2-Q4_K_M.gguf   # per-layer GPU vs CPU
./target/release/bisect_qwen35  <model> [ids]                  # per-checkpoint GPU vs CPU
./target/release/validate_qwen35 <model> <refs.json> <ids>     # vs llama-eval-callback
./target/release/upload_check   <model>                        # weight arena readback
cargo run --release -p llmoxide --example vision_check --features vision,gpu -- \
    models/mmproj-gemma-4-E4B-it-BF16.gguf photo.jpg           # vision tower GPU vs CPU
./target/release/wipe_check     <model> [prompt]               # wipe leaves no residue
./target/release/tok            <model> [text]                 # ids, vs llama-tokenize
./target/release/prompt         <model> < messages.json        # chat ids, vs a jinja render
```

`bisect` reports the *first* diverging checkpoint, which is how the NaN in
GeGLU and the attention-scale error were both found. The qwen35 workflow
(including a synthetic-checkpoint generator for fast whole-graph checks) is in
[ARCHITECTURE-qwen35.md](../ARCHITECTURE-qwen35.md) and `scripts/`.

One caveat on provenance: the tensor-level qwen35 numbers above were measured
against `Qwen3.8-27B-Uncensored-Cyber-Q6_K.gguf`, which carries no NextN block.
The checkpoint in `models/` is SHA-256 identical to the OBLITERATED build in the
table above — same architecture plus one NextN block, which the loader skips —
but has not been put back through `validate_qwen35` since.
