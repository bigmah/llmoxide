//! Bisect a GPU/CPU divergence by comparing the residual stream at each layer.
//!
//! The CPU path is byte-exact against llama.cpp, so the first checkpoint that
//! disagrees identifies the faulty kernel.

use model::{cache::KvCache, config::Config, cpu::Cpu, weights::Weights};

fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).expect("usage: bisect <model.gguf> [ids]");
    let tokens: Vec<u32> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "2,10979".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    let g = gguf::Gguf::open(&model)?;
    let cfg = Config::from_gguf(&g)?;
    let d = cfg.d_model;
    let t = tokens.len();

    // CPU reference: capture the residual stream at every checkpoint.
    let w = Weights::load(&g, &cfg)?;
    let mut cache = KvCache::new(&cfg, 2048);
    let mut cpu = Cpu::new(&cfg, &w, t);
    let mut want: Vec<(String, Vec<f32>)> = Vec::new();
    {
        let mut trace = |name: &str, data: &[f32]| {
            want.push((name.to_string(), data.to_vec()));
        };
        cpu.forward_traced(&tokens, &mut cache, &mut trace);
    }
    println!("cpu checkpoints: {}", want.len());

    let gpu = gpu::Gpu::blocking_new()?;
    let mut m = gpu::forward::GpuModel::load(gpu, &g, cfg.clone(), 2048, t)?;

    for (name, expect) in &want {
        m.reset();
        m.debug_stop = Some(name.clone());
        let Ok(got) = m.forward(&tokens) else { continue };
        let n = expect.len().min(got.len());

        let scale = expect.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        let mut worst = (0.0f32, 0usize);
        let mut nans = 0usize;
        for i in 0..n {
            if !got[i].is_finite() {
                nans += 1;
                if nans == 1 {
                    worst = (f32::INFINITY, i);
                }
                continue;
            }
            let dv = (expect[i] - got[i]).abs();
            if dv > worst.0 {
                worst = (dv, i);
            }
        }
        let rel = if nans > 0 { f32::INFINITY } else { worst.0 / scale };
        if nans > 0 {
            println!("{name:18} {nans}/{n} non-finite values");
        }
        println!(
            "{name:18} rel={rel:.3e}  worst@{} cpu={:+.5} gpu={:+.5}  {}",
            worst.1,
            expect[worst.1],
            got[worst.1],
            if rel < 1e-3 { "ok" } else { "MISMATCH" }
        );
        if rel >= 1e-3 {
            println!("\nfirst divergence at {name}; earlier checkpoints matched.");
            println!("cpu[0..6] = {:?}", &expect[..6.min(n)]);
            println!("gpu[0..6] = {:?}", &got[..6.min(n)]);
            // Also show the second token's row, where cross-position attention
            // first shows up.
            if t > 1 {
                println!("cpu[d..d+6] = {:?}", &expect[d..d + 6]);
                println!("gpu[d..d+6] = {:?}", &got[d..d + 6]);
            }
            return Ok(());
        }
    }
    println!("\nall checkpoints match");
    Ok(())
}
