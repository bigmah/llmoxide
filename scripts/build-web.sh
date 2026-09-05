#!/usr/bin/env bash
# Build the browser chat REPL as one self-contained HTML file.
#
#   scripts/build-web.sh [--debug]
#
# Produces web/llmoxide.html: the shell in web/shell.html with the wasm-bindgen
# glue and a base64 copy of the module inlined. No other file is needed at
# runtime — the checkpoint is chosen from disk by whoever opens the page.
set -euo pipefail

cd "$(dirname "$0")/.."
root=$PWD
profile=release
[[ ${1:-} == --debug ]] && profile=debug

echo "==> cargo build ($profile)"
( cd crates/wasm
  if [[ $profile == release ]]; then
    cargo build --release --target wasm32-unknown-unknown
  else
    cargo build --target wasm32-unknown-unknown
  fi )

wasm=crates/wasm/target/wasm32-unknown-unknown/$profile/llmoxide_web.wasm
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

echo "==> wasm-bindgen"
if ! command -v wasm-bindgen >/dev/null; then
  want=$(grep -m1 '^wasm-bindgen ' crates/wasm/Cargo.lock -A1 | grep -m1 version | cut -d'"' -f2 || echo 0.2)
  echo "wasm-bindgen CLI not found. cargo install -f wasm-bindgen-cli --version $want" >&2
  exit 1
fi
wasm-bindgen --target web --no-typescript --out-dir "$out" "$wasm"

glue=$out/llmoxide_web.js
mod=$out/llmoxide_web_bg.wasm

# The page calls `__wbg_init` directly, because an inline module script cannot
# import its own default export. That name is wasm-bindgen's, not ours, so
# check it rather than discovering the rename as a blank page in a browser.
grep -Eq 'export (default __wbg_init|\{[^}]*__wbg_init as default)' "$glue" || {
  echo "wasm-bindgen no longer exports __wbg_init as its default." >&2
  echo "Update the initialiser call in web/shell.html to match:" >&2
  grep -n 'export.*default' "$glue" >&2
  exit 1
}

echo "==> inlining"
python3 - "$glue" "$mod" web/shell.html web/llmoxide.html <<'PY'
import base64, pathlib, sys

glue, mod, shell, dest = (pathlib.Path(p) for p in sys.argv[1:5])
js = glue.read_text()

# The glue goes inside a <script> element, so a literal "</script" anywhere in
# it would close the tag early and truncate the page. It has never contained
# one; fail loudly rather than emit something subtly broken if that changes.
if "</script" in js.lower():
    sys.exit("glue contains a literal '</script'; it needs escaping now")

html = shell.read_text()
b64 = base64.b64encode(mod.read_bytes()).decode()
# Substitute by hand rather than with str.replace's template handling: the glue
# is arbitrary JavaScript and backslash sequences in it must survive verbatim.
for marker, value in (("@GLUE@", js), ("@WASM_BASE64@", b64)):
    if marker not in html:
        sys.exit(f"{shell} has no {marker} marker")
    html = html.replace(marker, value, 1)
dest.write_text(html)

print(f"    wasm    {mod.stat().st_size / 1e6:6.2f} MB")
print(f"    glue    {len(js) / 1e3:6.1f} kB")
print(f"    page    {dest.stat().st_size / 1e6:6.2f} MB  -> {dest}")
PY

cat <<EOF

Open it:
  python3 -m http.server -d $root/web 8000   # then http://localhost:8000/llmoxide.html
or just open the file directly; WebGPU works from file:// in Chrome.

Then pick a checkpoint, e.g.
  $root/models/gemma-4-E4B-it-Q4_K_M.gguf
EOF
