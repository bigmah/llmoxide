# Serving (afterthought, and not private)

```sh
./target/release/llmoxide-serve models/gemma4-v2-Q4_K_M.gguf   # http://127.0.0.1:8080
```

`LLMOXIDE_CTX` (default 16384), `LLMOXIDE_PORT` (8080), `LLMOXIDE_BATCH` (256).
`LLMOXIDE_CPU=1` puts qwen35 on its CPU reference path; `LLMOXIDE_NO_SUBGROUP=1`
forces the barrier-tree row reduction.

**This is not a private session, and cannot be made into one from this side of
the socket.** The server keeps its own hands clean — locked and zeroed prompt
ids, the zeroing allocator (`LLMOXIDE_NO_ZEROIZE=1` to disable), no core dumps,
`POST /v1/wipe` to overwrite the resident conversation, and no request body in
the log. But whatever you point at it usually keeps a transcript, and that is
where the conversation actually persists. opencode writes every message in
plaintext to `~/.local/share/opencode/opencode.db`, has no option to turn that
off, and builds `export` / `import` / `stats` on top of it.

If you want the server anyway, put the client's store somewhere that does not
survive a reboot — opencode honours `XDG_DATA_HOME`, so a RAM disk works:

```sh
DISK=$(hdiutil attach -nomount ram://1048576)      # 512 MB
newfs_hfs -v ocram "$DISK" && mkdir -p /tmp/ocram
mount -t hfs "$DISK" /tmp/ocram
XDG_DATA_HOME=/tmp/ocram opencode
```

`hdiutil detach "$DISK"` ends it. This is a weaker guarantee than private mode
gives: those are ordinary pages, not `mlock`ed, so they can still reach
(encrypted) swap.

Two smaller notes. `PT_DENY_ATTACH` is opt-in here (`LLMOXIDE_PRIVATE=1`) rather
than automatic, because it blocks profilers and a server is the thing you
profile. And removing the request-body log was not enough on its own:
`serde_json` quotes the offending value inside its own error message, so
`ApiError` carries a detailed message for the client and a sanitized one —
category, line, column — for the log.

## opencode

`opencode.json` in this repo points opencode at the local server (port 8081 —
start the server with `LLMOXIDE_PORT=8081` or edit the `baseURL`). It needs the
provider package once:

```sh
cd ~/.config/opencode && npm install @ai-sdk/openai-compatible
```

Then `opencode run --model llmoxide/Qwen3.8-27B-Q6_K "..."`, or copy the
`provider` block into `~/.config/opencode/opencode.json` to use it anywhere.

## API

`GET /v1/models`, `POST /v1/chat/completions` (streaming and not), `POST /v1/wipe`,
`GET /health`.

Supports `tools`, `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`,
`stop`, and a non-standard `enable_thinking` for the model's thought channel
(off by default).

Tool calls are the interesting part, and each model has its own wire format.
gemma4 does not emit JSON — it uses a custom DSL where strings are delimited
by the single token `<|"|>`:

```
<|tool_call>call:read_file{path:<|"|>src/main.rs<|"|>,limit:20}<tool_call|>
```

qwen35 wraps an XML-ish block in `<tool_call>` control tokens
(`<function=read_file>` / `<parameter=path>`); `crates/chat/src/qwen.rs`
translates that, `crates/chat/src/dsl.rs` the gemma4 DSL. Two deliberate
behaviours shared by both:

- Parsing works on **token ids**, not decoded text, because the quote marker and
  channel markers are control tokens a text decoder drops.
- Calls naming a tool the request never declared are **rejected**. The model
  will occasionally invent one, and a client that dispatched it would either
  error out or run something unintended.
