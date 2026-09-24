//! A multi-turn conversation, streamed token by token.
//!
//!   cargo run --release -p llmoxide --example chat_stream -- models/Qwen3-0.6B-Q8_0.gguf
//!
//! Shows the two things a chat client needs beyond `complete`: a callback that
//! sees text as it is produced, and an append-only history that lets the
//! session reuse the prefix already in its cache instead of re-running the
//! whole conversation each turn.

use std::io::Write;

use llmoxide::chat::Message;
use llmoxide::{Flow, LoadOptions, Request, Session};

fn main() -> llmoxide::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: chat_stream <model.gguf>");

    let mut session = Session::load(&path, &LoadOptions::new().n_ctx(8192))?;
    let mut history = vec![Message::system("You are terse. Answer in one sentence.")];

    for turn in ["what is the capital of France?", "and its population?"] {
        println!("\n\x1b[1m» {turn}\x1b[0m");
        history.push(Message::user(turn));

        let out = session.generate(
            Request::new(history.clone()).max_tokens(128),
            |piece| {
                print!("{piece}");
                std::io::stdout().flush().ok();
                Flow::Continue
            },
        )?;
        println!();
        eprintln!(
            "[{}/{} prompt tokens served from cache]",
            out.cached_tokens, out.prompt_tokens
        );

        // Appending the reply is what makes the next turn's prefix reusable.
        history.push(Message::assistant(out.completion.content));
    }

    // Overwrite the conversation, the device buffers and the locked pages.
    session.wipe();
    Ok(())
}
