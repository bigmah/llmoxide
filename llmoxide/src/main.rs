//! Greedy generation on either backend.
//!
//! The CPU path is byte-exact against llama.cpp, so running the same prompt
//! through both is how the GPU kernels are checked end to end.
//!
//!   llmoxide <model.gguf> <prompt> [n] [--cpu]

use model::{cache::KvCache, config::Config, cpu::Cpu, weights::Weights};

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
    let cfg = Config::from_gguf(&g)?;
    let tok = tokenizer::Tokenizer::from_gguf(&g)?;
    let ids = tok.encode(prompt, true);
    let n_ctx = (ids.len() + n_gen + 8).next_power_of_two().max(2048);

    let mut backend: Box<dyn Backend> = if use_cpu {
        Box::new(CpuBackend::new(&g, &cfg, n_ctx, ids.len())?)
    } else {
        let gpu = gpu::Gpu::blocking_new()?;
        eprintln!("gpu: {}", gpu.adapter_name);
        Box::new(gpu::forward::GpuModel::load(
            gpu,
            &g,
            cfg.clone(),
            n_ctx,
            ids.len().max(1),
        )?)
    };
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
        if cfg.eog.contains(&next) {
            break;
        }
        out.push(next);
        if raw {
            print!("{}", tok.token_text(next).replace('\u{2581}', " "));
        } else {
            print!("{}", dec.push(next));
        }
        use std::io::Write;
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

trait Backend {
    fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>>;
}

impl Backend for gpu::forward::GpuModel {
    fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        gpu::forward::GpuModel::forward(self, tokens)
    }
}

/// Owns the mmap-backed weights alongside the executor so the borrow stays
/// valid for the lifetime of the run.
struct CpuBackend<'a> {
    cpu: Cpu<'a>,
    cache: KvCache,
}

impl<'a> CpuBackend<'a> {
    fn new(
        g: &'a gguf::Gguf,
        cfg: &'a Config,
        n_ctx: usize,
        batch: usize,
    ) -> anyhow::Result<Self> {
        let weights: &'a Weights<'a> = Box::leak(Box::new(Weights::load(g, cfg)?));
        Ok(Self {
            cpu: Cpu::new(cfg, weights, batch.max(1)),
            cache: KvCache::new(cfg, n_ctx),
        })
    }
}

impl Backend for CpuBackend<'_> {
    fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        Ok(self.cpu.forward(tokens, &mut self.cache))
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
