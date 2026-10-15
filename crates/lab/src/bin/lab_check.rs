//! Check the lab engine natively before it goes anywhere near a browser.
//!
//!   lab_check <model.gguf> <prompt> [n]
//!
//! Greedy and template-free, so the output can be diffed against
//!   llama-completion -m <model.gguf> -p <prompt> -n <n> --temp 0 -no-cnv
//! Then one chat turn through the session, to see the template and sampler.

use std::time::Instant;

use llmoxide_lab::engine::{Cache, Model};
use llmoxide_lab::session::Lab;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: lab_check <model.gguf> <prompt> [n]");
    let prompt = args.get(1).map(String::as_str).unwrap_or("The capital of France is");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);

    let g = gguf::Gguf::open(path)?;
    let tok = tokenizer::Tokenizer::from_gguf(&g)?;
    let mut m = Model::load(&g)?;
    eprintln!("{:?}", m.cfg);

    let ids = tok.encode(prompt, true);
    let mut cache = Cache::new(&m.cfg, ids.len() + n + 1);
    let t0 = Instant::now();
    let mut logits = m.forward(&mut cache, &ids, None);
    let prefill = t0.elapsed();
    let mut out = Vec::new();
    let t1 = Instant::now();
    for _ in 0..n {
        let id = model::sample::argmax(&logits);
        if m.cfg.eog.contains(&id) {
            break;
        }
        out.push(id);
        logits = m.forward(&mut cache, &[id], None);
    }
    let decode = t1.elapsed();
    println!("{}{}", prompt, tok.decode(&out));
    eprintln!(
        "prefill {} tok in {:?} ({:.1} tok/s), decode {} tok in {:?} ({:.1} tok/s)",
        ids.len(),
        prefill,
        ids.len() as f64 / prefill.as_secs_f64(),
        out.len(),
        decode,
        out.len() as f64 / decode.as_secs_f64()
    );
    eprintln!("ids: {out:?}");

    let mut lab = Lab::new(&g, 1024, 256)?;
    for q in ["What is the capital of France?", "And what language do they speak there?"] {
        let t = Instant::now();
        let (ran, reused) = lab.chat_begin(q)?;
        print!("\n> {q}\n[prefill {ran} tokens, reused {reused}, {:?}]\n", t.elapsed());
        loop {
            let (text, done) = lab.chat_next();
            print!("{text}");
            if done {
                break;
            }
        }
        println!();
    }
    lab.inspect("The capital of France is", false)?;
    for (i, row) in lab.lens(3).iter().enumerate() {
        let s: Vec<String> = row.iter().map(|(id, p)| format!("{:?} {:.2}", lab.piece(*id), p)).collect();
        eprintln!("lens {i:2}: {}", s.join("  "));
    }
    let nn = lab.neighbors(tok.encode(" king", false)[0], 8);
    eprintln!("near ' king': {:?}", nn.iter().map(|(i, s)| (lab.piece(*i), *s)).collect::<Vec<_>>());
    let before = lab.chat_cache.nonzero();
    lab.wipe();
    eprintln!("chat cache nonzero floats: {before} -> {}", lab.chat_cache.nonzero());
    Ok(())
}
