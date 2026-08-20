//! Diff the qwen35 CPU forward pass against llama.cpp's `eval-callback` trace.
//!
//! Same workflow as `validate` (the gemma4 tool): run
//! `llama-eval-callback -m model.gguf -p "..." -ngl 0`, parse the output into
//! a JSON list of `{name, op, dims, sum, head}` records, and hand it here with
//! the same token ids. The trace names in [`model::qwen35::cpu`] follow
//! llama.cpp's graph-callback names, and llama.cpp reuses some (`Kcur` is both
//! the raw projection and the post-RoPE tensor), so like the gemma tool we
//! compare against the *first* occurrence of each name.

use std::collections::HashMap;

use model::qwen35::{Config, Cpu, State, Weights};

#[derive(serde::Deserialize)]
struct Ref {
    name: String,
    #[allow(dead_code)]
    op: String,
    dims: Vec<usize>,
    #[allow(dead_code)]
    sum: f64,
    /// First three and last three elements of the tensor's first row.
    head: Option<Vec<f64>>,
}

/// Generate the reference with `-ngl 99` (Metal) for quantized checkpoints:
/// llama.cpp's *CPU* kernels quantize activations to Q8_K before their
/// integer dot products, which shows up as multi-percent per-element deltas
/// that are llama.cpp's noise, not ours — against the Metal trace the real
/// 27B matches within 4e-4, and the all-f32 synthetic model within 5e-3 on
/// either backend.
const TOLERANCE: f64 = 0.02;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(model_path), Some(ref_path), Some(ids)) = (args.next(), args.next(), args.next())
    else {
        eprintln!("usage: validate_qwen35 <model.gguf> <refs.json> <tok0,tok1,...>");
        std::process::exit(2)
    };
    let tokens: Vec<u32> = ids
        .split(',')
        .map(|s| s.trim().parse().expect("token id"))
        .collect();

    let refs: Vec<Ref> = serde_json::from_slice(&std::fs::read(&ref_path)?)?;
    let mut want: HashMap<&str, &Ref> = HashMap::new();
    for r in &refs {
        want.entry(r.name.as_str()).or_insert(r);
    }

    let g = gguf::Gguf::open(&model_path)?;
    let cfg = Config::from_gguf(&g)?;
    println!("{}\n", cfg.summary());

    let w = Weights::load(&g, &cfg)?;
    let mut state = State::new(&cfg, tokens.len().max(8));
    let mut cpu = Cpu::new(&cfg, &w, tokens.len());

    println!("tokens: {tokens:?}\n");

    let trajectory = std::env::var("LLMOXIDE_TRAJECTORY").is_ok();
    let mut checked = 0usize;
    let mut worst: Option<(String, f64, f64, f64)> = None;
    let mut failures: Vec<(String, f64, f64, f64)> = Vec::new();

    {
        let mut trace = |name: &str, data: &[f32]| {
            let Some(r) = want.get(name) else { return };
            let n_ref: usize = r.dims.iter().product();
            if n_ref != data.len() {
                failures.push((
                    format!("{name} [shape {n_ref} vs {}]", data.len()),
                    0.0,
                    0.0,
                    f64::INFINITY,
                ));
                return;
            }
            let Some(head) = r.head.as_ref().filter(|h| h.len() == 6) else {
                return;
            };
            let row = r.dims[0].min(data.len());
            if row < 3 {
                return;
            }
            let ours = [
                data[0], data[1], data[2],
                data[row - 3], data[row - 2], data[row - 1],
            ];

            checked += 1;
            // Judge each element against the tensor's own scale — see the
            // gemma4 tool for why per-element relative error misleads. The
            // reference prints with 4 decimals, so half a print ulp of the
            // difference is measurement noise, not model error; subtract it.
            const PRINT_ULP: f64 = 1e-4;
            let scale = head.iter().map(|v| v.abs()).fold(1e-4, f64::max);
            let mut worst_rel = 0.0f64;
            for (i, &want_v) in head.iter().enumerate() {
                let err = ((want_v - ours[i] as f64).abs() - PRINT_ULP / 2.0).max(0.0);
                worst_rel = worst_rel.max(err / scale);
            }
            if trajectory {
                println!(
                    "  {name:26} ref={:12.5} ours={:12.5} rel={worst_rel:.4}",
                    head[0], ours[0]
                );
            }
            if worst.as_ref().is_none_or(|w| worst_rel > w.1) {
                worst = Some((name.to_string(), worst_rel, head[0], ours[0] as f64));
            }
            if worst_rel > TOLERANCE {
                failures.push((name.to_string(), head[0], ours[0] as f64, worst_rel));
            }
        };

        let logits = cpu.forward_traced(&tokens, &mut state, &mut trace);
        drop(trace);

        let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("top-5 logits: {:?}\n", &top[..5]);
    }

    println!("compared {checked} tensors against llama.cpp");
    if let Some((n, rel, a, b)) = &worst {
        println!("worst relative delta: {rel:.3e} on {n} (ref {a:.4} vs ours {b:.4})");
    }

    if failures.is_empty() {
        println!("\nALL MATCH");
        Ok(())
    } else {
        println!("\n{} MISMATCHES (showing first 25):", failures.len());
        for (n, a, b, rel) in failures.iter().take(25) {
            println!("  {n:28} ref={a:14.5} ours={b:14.5} rel={rel:.4}");
        }
        anyhow::bail!("{} tensors diverged", failures.len())
    }
}
