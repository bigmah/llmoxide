//! Print the prompt ids for a conversation, for diffing against a jinja
//! rendering of the checkpoint's own chat template.
//!
//!   prompt <model.gguf> [--thinking] < messages.json
//!
//! `messages.json` is an OpenAI-shaped array. The output is the token id list
//! `build_prompt` produces, plus the text it decodes to, so a divergence can
//! be read rather than just counted.

use chat::Message;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: prompt <model.gguf> [--thinking] < messages.json"))?;
    let thinking = args.any(|a| a == "--thinking");

    let g = gguf::Gguf::open(&model)?;
    let tok = tokenizer::Tokenizer::from_gguf(&g)?;
    let messages: Vec<Message> = serde_json::from_str(&std::io::read_to_string(std::io::stdin())?)?;

    let ids = match model::Arch::detect(&g)? {
        model::Arch::Gemma4 => {
            let sp = chat::Special::new(&tok, g.str("tokenizer.chat_template").ok())?;
            chat::build_prompt(&tok, &sp, &messages, &[], thinking)
        }
        model::Arch::Qwen35 => {
            let sp = chat::qwen::Special::new(&tok)?;
            chat::qwen::build_prompt(&tok, &sp, &messages, &[], thinking)
        }
    };

    println!("{ids:?}");
    eprintln!("--- decodes to ---\n{}", tok.decode(&ids));
    Ok(())
}
