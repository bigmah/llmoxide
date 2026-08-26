//! Verify that a wipe actually removes the conversation from device memory.
//!
//! The claim `wipe` makes is not one to take on trust: `reset` looks like it
//! clears the cache and does not, and a `clear_buffer` that was queued but never
//! submitted looks identical from the host side. So this runs a prompt, confirms
//! the device buffers are full of it, wipes, and reads every one of them back
//! again.
//!
//!   wipe_check <model.gguf> [prompt]
//!
//! Exits non-zero if a single non-zero word survives, which is what makes it
//! usable from a test script.

fn main() -> anyhow::Result<()> {
    let model = std::env::args()
        .nth(1)
        .expect("usage: wipe_check <model.gguf> [prompt]");
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is Paris and the year is 1789".into());

    let g = gguf::Gguf::open(&model)?;
    let arch = model::Arch::detect(&g)?;
    let tok = tokenizer::Tokenizer::from_gguf(&g)?;
    let ids = tok.encode(&prompt, true);
    let n_ctx = 1024;

    let gpu = gpu::Gpu::blocking_new()?;
    println!("adapter: {}\narch: {arch:?}\nprompt: {} tokens", gpu.adapter_name, ids.len());

    let (before, after) = match arch {
        model::Arch::Gemma4 => {
            let cfg = model::config::Config::from_gguf(&g)?;
            let mut m = gpu::forward::GpuModel::load(gpu, &g, cfg, n_ctx, ids.len().max(1))?;
            m.forward(&ids)?;
            let before = m.residue();
            m.wipe();
            (before, m.residue())
        }
        model::Arch::Qwen35 => {
            let cfg = model::qwen35::Config::from_gguf(&g)?;
            let mut m = gpu::qwen35::Qwen35Gpu::load(gpu, &g, cfg, n_ctx, ids.len().max(1))?;
            m.forward(&ids)?;
            let before = m.residue();
            m.wipe();
            (before, m.residue())
        }
    };

    let live: usize = before.iter().map(|(_, nz, _)| nz).sum();
    let words: usize = before.iter().map(|(_, _, n)| n).sum();
    println!(
        "\nafter prefill : {live} non-zero words across {} buffers ({:.1} MB)",
        before.len(),
        words as f64 * 4.0 / 1e6
    );

    // A wipe that "passes" because the buffers were empty to begin with would
    // prove nothing.
    anyhow::ensure!(
        live > 0,
        "nothing was resident before the wipe; the check would be vacuous"
    );

    let mut leaked = 0usize;
    for ((name, nz, n), (_, nz_after, _)) in before.iter().zip(&after) {
        if *nz_after > 0 {
            println!("  LEAK {name:20} {nz_after}/{n} words survived (was {nz})");
            leaked += nz_after;
        }
    }

    if leaked == 0 {
        println!("after wipe    : 0 non-zero words. every buffer clean.");
        Ok(())
    } else {
        println!("after wipe    : {leaked} non-zero words survived.");
        std::process::exit(1);
    }
}
