//! Bisect a qwen35 GPU/CPU divergence checkpoint by checkpoint.
//!
//! The CPU path is validated tensor-by-tensor against llama.cpp, so the first
//! checkpoint that disagrees identifies the faulty kernel. Same idea as
//! `bisect` for gemma4.

use model::qwen35::{Config, Cpu, State, Weights};

fn main() -> anyhow::Result<()> {
    let model = std::env::args()
        .nth(1)
        .expect("usage: bisect_qwen35 <model.gguf> [ids]");
    let tokens: Vec<u32> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "785,6722,315,9625,374".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    let g = gguf::Gguf::open(&model)?;
    let cfg = Config::from_gguf(&g)?;
    let d = cfg.d_model;
    let t = tokens.len();

    // CPU reference: capture every checkpoint.
    let w = Weights::load(&g, &cfg)?;
    let mut state = State::new(&cfg, 2048);
    let mut cpu = Cpu::new(&cfg, &w, t);
    let mut want: Vec<(String, Vec<f32>)> = Vec::new();
    {
        let mut trace = |name: &str, data: &[f32]| {
            want.push((name.to_string(), data.to_vec()));
        };
        cpu.forward_traced(&tokens, &mut state, &mut trace);
    }
    println!("cpu checkpoints: {}", want.len());

    let gpu = gpu::Gpu::blocking_new()?;
    println!("adapter: {}", gpu.adapter_name);
    let mut m = gpu::qwen35::Qwen35Gpu::load(gpu, &g, cfg.clone(), 2048, t)?;

    let mut bad = 0usize;
    for (name, expect) in &want {
        if name == "result_output" {
            continue; // compared via the logits below
        }
        m.reset();
        m.debug_stop = Some(name.clone());
        let got = match m.forward(&tokens) {
            Ok(g) => g,
            Err(e) => {
                println!("{name:26} skipped ({e})");
                continue;
            }
        };
        // The CPU traces result_norm as the last row only; the GPU buffer
        // holds all t rows.
        let got: &[f32] = if name == "result_norm" && got.len() >= t * d {
            &got[(t - 1) * d..t * d]
        } else {
            &got
        };
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
            println!("{name:26} {nans}/{n} non-finite values");
        }
        let ok = rel < 1e-3;
        if !ok || std::env::var("LLMOXIDE_VERBOSE").is_ok() {
            println!(
                "{name:26} rel={rel:.3e}  worst@{} cpu={:+.5} gpu={:+.5}  {}",
                worst.1,
                expect[worst.1],
                got[worst.1],
                if ok { "ok" } else { "MISMATCH" }
            );
        }
        if !ok {
            bad += 1;
            if bad == 1 {
                println!("\nfirst divergence at {name}; earlier checkpoints matched.");
                println!("cpu[0..6] = {:?}", &expect[..6.min(n)]);
                println!("gpu[0..6] = {:?}", &got[..6.min(n)]);
            }
            if bad > 8 {
                println!("(stopping after 8 mismatches)");
                break;
            }
        }
    }

    // Final logits, full forward on both.
    m.reset();
    m.debug_stop = None;
    let got = m.forward(&tokens)?;
    let mut state = State::new(&cfg, 2048);
    let expect = cpu.forward(&tokens, &mut state);
    let scale = expect.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
    let worst = expect
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_top = argmax(&expect);
    let gpu_top = argmax(&got);
    println!(
        "logits: rel={:.3e}  argmax cpu={cpu_top} gpu={gpu_top}  {}",
        worst / scale,
        if cpu_top == gpu_top { "ok" } else { "MISMATCH" }
    );

    if bad == 0 && cpu_top == gpu_top {
        println!("\nall checkpoints match");
    }
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
