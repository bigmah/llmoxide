//! The smallest thing another program can do with this crate.
//!
//!   cargo run --release -p llmoxide --example generate -- models/Qwen3-0.6B-Q8_0.gguf "hello"

use llmoxide::{LoadOptions, Request, Session};

fn main() -> llmoxide::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: generate <model.gguf> [prompt]");
    let prompt = args.next().unwrap_or_else(|| "name three prime numbers".into());

    let mut session = Session::load(&path, &LoadOptions::default())?;
    println!("{:?} on {:?}", session.info().arch, session.info().device);

    let out = session.complete(Request::user(prompt).max_tokens(256))?;

    println!("{}", out.completion.content);
    eprintln!(
        "[{} prompt tokens, {} generated, finished: {}]",
        out.prompt_tokens,
        out.generated,
        out.reason.as_str()
    );
    Ok(())
}
