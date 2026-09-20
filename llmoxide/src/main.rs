//! Greedy generation on any supported architecture and backend.
//!
//! The CPU paths are validated tensor-by-tensor against llama.cpp, so running
//! the same prompt through both is how everything downstream is checked.
//!
//!   llmoxide <model.gguf> <prompt> [n] [--cpu]
//!
//! Both architectures run on the GPU by default; `--cpu` selects the
//! reference path. Deliberately greedy and chat-template-free: this is the
//! tool that answers "do the kernels still produce the right numbers", and a
//! sampler or a prompt template between you and the logits only gets in the
//! way. For actual use see `llmoxide-private`, or the `llmoxide` library.

use std::io::Write;

use llmoxide::backend::{self, DevicePref, LoadOptions};
use llmoxide::{gguf, tokenizer};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let use_cpu = args.iter().any(|a| a == "--cpu");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let model_path = pos
        .first()
        .cloned()
        .expect("usage: llmoxide <model.gguf> <prompt> [n] [--cpu]");
    let prompt = pos.get(1).map(|s| s.as_str()).unwrap_or_default();
    let n_gen: usize = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let raw = std::env::var("LLMOXIDE_RAW").is_ok();

    let t0 = std::time::Instant::now();
    let g = gguf::Gguf::open(model_path)?;
    let tok = tokenizer::Tokenizer::from_gguf(&g)?;
    let ids = tok.encode(prompt, true);
    let n_ctx = (ids.len() + n_gen + 8).next_power_of_two().max(2048);

    let opts = LoadOptions::new()
        .n_ctx(n_ctx)
        .max_batch(ids.len().max(1))
        .device(if use_cpu {
            DevicePref::Cpu
        } else {
            DevicePref::Gpu
        });
    let mut backend = backend::load(g, &opts)?;
    let info = backend.info().clone();
    eprintln!("{:?} on {:?}", info.arch, info.device);
    eprintln!("loaded in {:?}; prompt {} tokens", t0.elapsed(), ids.len());

    let t1 = std::time::Instant::now();
    let mut logits = backend.forward(&ids)?;
    let prefill = t1.elapsed();
    eprintln!(
        "prefill {:?} ({:.1} tok/s)",
        prefill,
        ids.len() as f64 / prefill.as_secs_f64()
    );

    let mut out = Vec::new();
    let mut dec = tokenizer::Decoder::new(&tok);
    let t2 = std::time::Instant::now();
    for _ in 0..n_gen {
        let next = argmax(&logits);
        if info.eog.contains(&next) {
            break;
        }
        out.push(next);
        if raw {
            print!("{}", tok.token_text(next).replace('\u{2581}', " "));
        } else {
            print!("{}", dec.push(next));
        }
        std::io::stdout().flush().ok();
        logits = backend.forward(&[next])?;
    }
    print!("{}", dec.flush());
    println!();

    let dt = t2.elapsed();
    eprintln!(
        "generated {} tokens in {:?} ({:.2} tok/s)",
        out.len(),
        dt,
        out.len() as f64 / dt.as_secs_f64()
    );
    eprintln!("ids: {out:?}");
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
