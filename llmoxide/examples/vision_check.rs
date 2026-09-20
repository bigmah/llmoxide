//! Diff the GPU vision tower against the CPU reference, on a real image.
//!
//!   cargo run --release --example vision_check --features vision,gpu -- \
//!       models/mmproj-gemma-4-E4B-it-BF16.gguf photo.jpg
//!
//! The CPU tower is the oracle — it is the one written directly against
//! llama.cpp's graph — so this is the same relationship `bisect` has to
//! `model::cpu`. Exits non-zero if the two disagree beyond tolerance.

use llmoxide::{gguf, model, vision};

/// Per-element relative tolerance. The two paths differ in summation order
/// (the CPU tower accumulates in four chains, the GPU one reduces across a
/// subgroup), so exact equality is not the bar; agreement well inside f32
/// round-off is.
const TOLERANCE: f32 = 2e-3;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mmproj = args
        .next()
        .unwrap_or_else(|| "models/mmproj-gemma-4-E4B-it-BF16.gguf".into());
    let image = args.next().unwrap_or_else(|| "test-image.jpg".into());

    let cfg = model::vision::Config::from_gguf(&gguf::Gguf::open(&mmproj)?)?;
    println!("{}", cfg.summary());
    let planar = vision::prepare(&std::fs::read(&image)?, &cfg)?;
    let (nx, ny) = planar.grid(cfg.patch_size);
    println!("{image}: {}x{} px, {nx}x{ny} patches", planar.w, planar.h);

    let t0 = std::time::Instant::now();
    let cpu = vision::Vision::new(gguf::Gguf::open(&mmproj)?)?;
    let want = cpu.encode(&planar)?;
    println!("cpu  {:?} -> {} rows of {}", t0.elapsed(), want.n, want.proj_dim);

    let t1 = std::time::Instant::now();
    let dev = llmoxide::gpu::Gpu::blocking_new()?;
    println!("gpu  adapter: {}", dev.adapter_name);
    let tower = llmoxide::gpu::vision::VisionGpu::load(dev, gguf::Gguf::open(&mmproj)?, cfg)?;
    let t2 = std::time::Instant::now();
    let mut got = tower.encode(&planar.data, planar.w, planar.h)?;
    println!("gpu  {:?} (load {:?})", t2.elapsed(), t1.elapsed() - t2.elapsed());
    // Repeat: the first encode pays for shader specialization and a cold
    // allocator, neither of which a second image pays again.
    for i in 1..3 {
        let t = std::time::Instant::now();
        got = tower.encode(&planar.data, planar.w, planar.h)?;
        println!("gpu  encode {i}: {:?}", t.elapsed());
    }

    anyhow::ensure!(
        got.len() == want.rows.len(),
        "length mismatch: gpu {} vs cpu {}",
        got.len(),
        want.rows.len()
    );

    // Scale the tolerance by the row's own magnitude: these are activations,
    // not probabilities, and an absolute epsilon would be meaningless.
    let scale = want
        .rows
        .iter()
        .fold(0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    let mut bad = 0usize;
    for (i, (&a, &b)) in want.rows.iter().zip(&got).enumerate() {
        let rel = (a - b).abs() / scale;
        if rel > worst {
            worst = rel;
            worst_at = i;
        }
        if rel > TOLERANCE {
            bad += 1;
        }
    }
    println!(
        "compared {} values, peak |cpu| {scale:.4}\n  worst {worst:.2e} at {worst_at} \
         (cpu {:.5}, gpu {:.5})\n  over tolerance ({TOLERANCE:.0e}): {bad}",
        got.len(),
        want.rows[worst_at],
        got[worst_at],
    );

    anyhow::ensure!(bad == 0, "{bad} values disagree beyond {TOLERANCE:.0e}");
    println!("ok: GPU tower matches the CPU reference");
    Ok(())
}
