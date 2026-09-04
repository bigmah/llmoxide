//! Download a checkpoint from Hugging Face, resumably and verified.
//!
//!   llmoxide-fetch                      # every model this repo is built around
//!   llmoxide-fetch gemma4               # one of them
//!   llmoxide-fetch hf:owner/repo/f.gguf
//!   llmoxide-fetch https://huggingface.co/owner/repo/blob/main/f.gguf
//!
//!   --dir <path>   where to put it (default: models)
//!   --force        re-download even if the file is already there
//!   --no-verify    skip the hash and GGUF checks (they are the point; don't)
//!   --no-adopt     download even if the identical file is already there under
//!                  a different name

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", HELP);
        return Ok(());
    }

    let force = args.iter().any(|a| a == "--force");
    let verify = !args.iter().any(|a| a == "--no-verify");
    let adopt = !args.iter().any(|a| a == "--no-adopt");
    let dir = flag(&args, "--dir").unwrap_or_else(|| "models".to_string());
    let dir = std::path::PathBuf::from(dir);

    let mut specs: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--dir" {
            it.next();
        } else if !a.starts_with("--") {
            specs.push(a.clone());
        }
    }
    // No argument means every model the repo is about.
    if specs.is_empty() {
        specs = hub::ALIASES.iter().map(|(n, _)| n.to_string()).collect();
    }

    for (i, spec) in specs.iter().enumerate() {
        if i > 0 {
            println!();
        }
        let url = hub::resolve_url(spec)?;
        println!("resolving {spec}");
        let remote = hub::probe(&url)?;
        println!(
            "  {}  {}{}",
            remote.filename,
            hub::progress::bytes(remote.size),
            match &remote.sha256 {
                Some(s) => format!("  sha256 {}…", &s[..12]),
                None => String::new(),
            }
        );
        let report = hub::fetch(&remote, &dir, force, verify, adopt)?;
        if !report.already_had_it {
            println!("saved to {}", report.path.display());
        }
    }
    Ok(())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}

const HELP: &str = "\
llmoxide-fetch — download GGUF checkpoints from Hugging Face

  llmoxide-fetch                       every model this repo is built around
  llmoxide-fetch gemma4                one alias
  llmoxide-fetch hf:owner/repo/f.gguf  shorthand
  llmoxide-fetch <https url>           a blob/blame/resolve URL

  --dir <path>   destination directory (default: models)
  --force        re-download even if already present
  --no-verify    skip sha256 and GGUF structure checks
  --no-adopt     always download, even if the same file is already in --dir
                 under another name

Transfers resume: interrupt one and run the same command again.
";
