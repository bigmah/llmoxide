#!/usr/bin/env bash
# Build the blog post's browser demo: crates/lab for wasm32, plus its worker.
#
#   scripts/build-lab.sh [out-dir]      # default web/lab
#
# Writes lab.js (wasm-bindgen glue), lab_bg.wasm and lab-worker.js. The page
# that uses them starts the worker with
#   new Worker('<out-dir>/lab-worker.js', { type: 'module' })
# and points its `load` op at a checkpoint, e.g. SmolLM2-135M-Instruct-Q8_0.gguf.
#
# SIMD is switched on: every browser that runs module workers runs wasm
# SIMD128, and the engine's dot product is written so the compiler can use it.
#
# Panic messages carry source paths into the .wasm; the remaps keep the
# builder's home directory out of a file that gets published.
set -euo pipefail

cd "$(dirname "$0")/.."
out=${1:-web/lab}
mkdir -p "$out"

echo "==> cargo build (wasm32, simd128)"
( cd crates/lab
  RUSTFLAGS="-C target-feature=+simd128 --remap-path-prefix=$HOME=~ \
    --remap-path-prefix=$(rustc --print sysroot)=/rustc" \
    cargo build --release --lib --target wasm32-unknown-unknown )

echo "==> wasm-bindgen"
wasm-bindgen --target web --no-typescript --out-name lab --out-dir "$out" \
  crates/lab/target/wasm32-unknown-unknown/release/llmoxide_lab.wasm
cp crates/lab/web/lab-worker.js "$out/"

ls -la "$out"
