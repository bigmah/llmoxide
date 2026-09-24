//! Ask a question about an image.
//!
//!   cargo run --release -p llmoxide --example image --features vision,gpu -- \
//!       models/gemma-4-E4B-it-Q4_K_M.gguf \
//!       models/mmproj-gemma-4-E4B-it-BF16.gguf \
//!       photo.jpg "what is in this image?" [--cpu]
//!
//! The vision tower always runs on the CPU; `--cpu` additionally puts the text
//! model there, which is the path validated against llama.cpp.

use llmoxide::{DevicePref, LoadOptions, Request, Session};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let think = args.iter().any(|a| a == "--think");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let (model, mmproj, image) = match pos.as_slice() {
        [m, p, i, ..] => (m.as_str(), p.as_str(), i.as_str()),
        _ => anyhow::bail!("usage: image <model.gguf> <mmproj.gguf> <image> [prompt] [--cpu]"),
    };
    let prompt = pos
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("Describe this image in one sentence.");

    let t0 = std::time::Instant::now();
    let mut s = Session::load(
        model,
        &LoadOptions::new()
            .n_ctx(4096)
            .mmproj(mmproj)
            .device(if cpu { DevicePref::Cpu } else { DevicePref::Gpu }),
    )?;
    eprintln!("loaded in {:?}", t0.elapsed());

    // The OpenAI content-part shape, which is also what the server accepts.
    let content = serde_json::json!([
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": prompt},
    ]);

    let t1 = std::time::Instant::now();
    let out = s.complete(
        Request::new(vec![llmoxide::prelude::Message {
            role: "user".into(),
            content: Some(content),
            ..Default::default()
        }])
        .thinking(think),
    )?;
    eprintln!("answered in {:?}", t1.elapsed());
    println!("{}", out.completion.content);
    Ok(())
}
