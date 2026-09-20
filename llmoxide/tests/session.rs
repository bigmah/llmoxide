//! End-to-end tests against a real checkpoint.
//!
//! Skipped unless `LLMOXIDE_TEST_MODEL` points at a GGUF, because the smallest
//! checkpoint here is still 400 MB and cannot be committed:
//!
//!   LLMOXIDE_TEST_MODEL=web/weights.gguf \
//!     cargo test -p llmoxide --release --test session -- --test-threads=1
//!
//! Serially, because each test loads its own copy of the weights onto the GPU.

use llmoxide::chat::Message;
use llmoxide::{Flow, LoadOptions, Request, Session};

fn session() -> Option<Session> {
    let raw = std::env::var("LLMOXIDE_TEST_MODEL").ok()?;
    // Cargo runs a test with the package directory as the working directory,
    // not the workspace root, so resolve a relative path against the latter —
    // which is what anyone typing the command above will have meant.
    let path = std::path::Path::new(&raw);
    let path = if path.is_relative() {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join(path)
    } else {
        path.to_path_buf()
    };
    Some(
        Session::load(&path, &LoadOptions::new().n_ctx(4096))
            .unwrap_or_else(|e| panic!("load {}: {e}", path.display())),
    )
}

/// The regression this exists for: replaying an assistant turn used to encode
/// the empty thought channel as two `\n` ids where the generation prompt
/// emitted the merged `\n\n` id, so the cached prefix ended at the first
/// assistant message and every turn re-prefilled from scratch.
#[test]
fn prefix_reuse_hits_on_an_appended_turn() {
    let Some(mut s) = session() else { return };

    let mut history = vec![Message::user("what is the capital of France?")];
    let first = s
        .complete(Request::new(history.clone()).max_tokens(32))
        .unwrap();
    assert_eq!(first.cached_tokens, 0, "nothing is resident on turn one");
    assert!(!first.completion.content.is_empty());

    history.push(Message::assistant(first.completion.content));
    history.push(Message::user("and its population?"));
    let second = s
        .complete(Request::new(history).max_tokens(32))
        .unwrap();

    assert!(
        second.cached_tokens > first.prompt_tokens,
        "turn two should reuse turn one's prompt and reply, got {} of {}",
        second.cached_tokens,
        second.prompt_tokens
    );
    assert!(second.cached_tokens < second.prompt_tokens, "the new user turn is not cached");
}

#[test]
fn wipe_forfeits_the_cached_prefix() {
    let Some(mut s) = session() else { return };

    let history = vec![Message::user("count to three")];
    s.complete(Request::new(history.clone()).max_tokens(16)).unwrap();
    s.wipe();

    let after = s.complete(Request::new(history).max_tokens(16)).unwrap();
    assert_eq!(after.cached_tokens, 0, "a wipe leaves nothing to reuse");
}

#[test]
fn streaming_pieces_reassemble_into_the_completion() {
    let Some(mut s) = session() else { return };

    let mut streamed = String::new();
    let out = s
        .generate(Request::user("say hello").max_tokens(24), |piece| {
            streamed.push_str(piece);
            Flow::Continue
        })
        .unwrap();

    assert_eq!(streamed.trim(), out.completion.content.trim());
}

#[test]
fn a_callback_can_stop_generation() {
    let Some(mut s) = session() else { return };

    let mut seen = 0usize;
    let out = s
        .generate(Request::user("count from one to fifty").max_tokens(200), |_| {
            seen += 1;
            if seen >= 3 {
                Flow::Stop
            } else {
                Flow::Continue
            }
        })
        .unwrap();

    assert_eq!(seen, 3);
    assert_eq!(out.reason, llmoxide::FinishReason::Cancelled);
    assert!(out.generated <= 4, "stopped early, got {}", out.generated);
}
