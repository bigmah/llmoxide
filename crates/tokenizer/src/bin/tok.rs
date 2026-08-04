//! Tokenize stdin (or an argument) and print ids, for diffing against
//! `llama-tokenize --ids`.

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: tok <model.gguf> [text]");
    let g = gguf::Gguf::open(&model)?;
    let t = tokenizer::Tokenizer::from_gguf(&g)?;

    let text = match args.next() {
        Some(s) => s,
        None => std::io::read_to_string(std::io::stdin())?,
    };
    let ids = t.encode(&text, true);
    println!("{ids:?}");
    // Round-trip check: decoding must reproduce the input exactly.
    let back = t.decode(&ids);
    if back != text {
        eprintln!("ROUNDTRIP MISMATCH\n  in : {text:?}\n  out: {back:?}");
        std::process::exit(1);
    }
    Ok(())
}
