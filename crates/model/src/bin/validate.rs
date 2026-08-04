//! Diff our CPU forward pass against llama.cpp's `eval-callback` trace.
//!
//! Usage:
//!   validate <model.gguf> <ref_sums.json> <tok0,tok1,...>
//!
//! The reference file is produced by parsing `llama-eval-callback` output; each
//! record is a tensor name and the sum of its elements. Sums are a coarse but
//! surprisingly sharp check: any error in shape, ordering, RoPE pairing, or
//! normalization moves them well outside float tolerance.

use std::collections::HashMap;

use model::{cache::KvCache, config::Config, cpu::Cpu, weights::Weights};

#[derive(serde::Deserialize)]
struct Ref {
    name: String,
    #[allow(dead_code)]
    op: String,
    dims: Vec<usize>,
    #[allow(dead_code)]
    sum: f64,
    /// First three and last three elements of the tensor's first row.
    ///
    /// Element values are the useful signal. Whole-tensor sums are near-total
    /// cancellations of thousands of terms, so they amplify llama.cpp's Q8_K
    /// activation quantization into apparent divergence even when every element
    /// agrees to a fraction of a percent.
    head: Option<Vec<f64>>,
}

/// Per-element relative tolerance. Generous because llama.cpp's CPU kernels
/// quantize activations to Q8_K before the dot product; our f32 path is the
/// more accurate of the two, and the gap shows up as a fraction of a percent
/// on well-conditioned values.
const TOLERANCE: f64 = 0.05;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        eprintln!("usage: validate <model.gguf> <ref_sums.json> <ids>");
        std::process::exit(2)
    });
    let ref_path = args.next().unwrap();
    let tokens: Vec<u32> = args
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().parse().expect("token id"))
        .collect();

    let refs: Vec<Ref> = serde_json::from_slice(&std::fs::read(&ref_path)?)?;
    // llama.cpp reuses scratch names like "norm-0"; keep the first occurrence of
    // each distinct name, which is the one our trace emits.
    let mut want: HashMap<&str, &Ref> = HashMap::new();
    for r in &refs {
        want.entry(r.name.as_str()).or_insert(r);
    }

    let g = gguf::Gguf::open(&model_path)?;
    let cfg = Config::from_gguf(&g)?;
    println!("{}\n", cfg.summary());

    let w = Weights::load(&g, &cfg)?;
    let mut cache = KvCache::new(&cfg, tokens.len().max(8));
    let mut cpu = Cpu::new(&cfg, &w, tokens.len());

    println!("tokens: {tokens:?}\n");

    // Print a named tensor in ggml's layout so it can be eyeballed against the
    // eval-callback output row for row.
    let dump_name = std::env::var("LLMOXIDE_DUMP").ok();
    let dump_rows: usize = std::env::var("LLMOXIDE_DUMP_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let trajectory = std::env::var("LLMOXIDE_TRAJECTORY").is_ok();
    let mut checked = 0usize;
    let mut worst: Option<(String, f64, f64, f64)> = None;
    let mut failures: Vec<(String, f64, f64, f64)> = Vec::new();

    {
        let mut trace = |name: &str, data: &[f32]| {
            if dump_name.as_deref() == Some(name) {
                let r = want.get(name).map(|r| r.dims.clone()).unwrap_or_default();
                // ggml dims are innermost-first; rows are dim0, grouped by dim1.
                let (row, per_block) = match r.len() {
                    0 => (data.len(), 1),
                    1 => (r[0], 1),
                    _ => (r[0], r[1]),
                };
                println!("--- {name} dims={r:?} row={row} rows_per_block={per_block}");
                for (bi, block) in data.chunks(row * per_block).enumerate() {
                    println!("  block{bi}:");
                    for (ri, rw) in block.chunks(row).enumerate().take(dump_rows) {
                        let n = rw.len();
                        println!(
                            "    row{ri}: {:>10.4},{:>10.4},{:>10.4}, ..., {:>10.4},{:>10.4},{:>10.4}",
                            rw[0], rw[1], rw[2], rw[n - 3], rw[n - 2], rw[n - 1]
                        );
                    }
                }
            }
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
            // ggml prints the first three and last three of the first row.
            let row = r.dims[0].min(data.len());
            let ours = [
                data[0], data[1], data[2],
                data[row - 3], data[row - 2], data[row - 1],
            ];

            checked += 1;
            // Judge each element against the tensor's own scale rather than its
            // own magnitude. An element that happens to sit near zero is the
            // result of cancellation, where llama.cpp's Q8_K activation
            // quantization shows up as a huge *relative* error but a negligible
            // absolute one — scoring those per-element would drown out real bugs.
            let scale = head
                .iter()
                .map(|v| v.abs())
                .fold(1e-4, f64::max);
            let mut worst_rel = 0.0f64;
            for (i, &want_v) in head.iter().enumerate() {
                worst_rel = worst_rel.max((want_v - ours[i] as f64).abs() / scale);
            }
            if trajectory {
                println!(
                    "  {name:24} ref={:12.5} ours={:12.5} rel={worst_rel:.4}",
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

        let logits = cpu.forward_traced(&tokens, &mut cache, &mut trace);
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
