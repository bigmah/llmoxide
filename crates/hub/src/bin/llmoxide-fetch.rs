//! Download a checkpoint from Hugging Face, resumably and verified.
//!
//!   llmoxide-fetch                      # the checkpoints this repo is built around
//!   llmoxide-fetch gemma4               # one alias (`--aliases` lists them)
//!   llmoxide-fetch hf:owner/repo        # best file in the repo that llmoxide can run
//!   llmoxide-fetch hf:owner/repo:Q8_0   # ... of one quantization
//!   llmoxide-fetch hf:owner/repo/f.gguf
//!   llmoxide-fetch https://huggingface.co/owner/repo/blob/main/f.gguf
//!
//!   --check        judge the file (or every file in the repo) and download nothing
//!   --dir <path>   where to put it (default: models)
//!   --force        re-download even if the file is already there
//!   --no-check     download without first reading the header to see if it will load
//!   --no-verify    skip the hash and GGUF checks (they are the point; don't)
//!   --no-adopt     download even if the identical file is already there under
//!                  a different name

// The library of this package is `llmoxide_hub`; keep the short name in the code.
use llmoxide_hub as hub;

use hub::check::Verdict;
use hub::Target;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", HELP);
        return Ok(());
    }
    if args.iter().any(|a| a == "--aliases") {
        for a in hub::ALIASES {
            println!("{:<22} {}{}", a.name, a.url, if a.core { "  [default]" } else { "" });
        }
        return Ok(());
    }

    let force = args.iter().any(|a| a == "--force");
    let verify = !args.iter().any(|a| a == "--no-verify");
    let adopt = !args.iter().any(|a| a == "--no-adopt");
    let precheck = !args.iter().any(|a| a == "--no-check");
    let check_only = args.iter().any(|a| a == "--check");
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
    // No argument means the models the repo is about — not every alias, which
    // would be a few hundred gigabytes.
    if specs.is_empty() {
        specs = hub::ALIASES.iter().filter(|a| a.core).map(|a| a.name.to_string()).collect();
    }

    let mut failed = 0;
    for (i, spec) in specs.iter().enumerate() {
        if i > 0 {
            println!();
        }
        println!("resolving {spec}");
        let target = hub::parse_spec(spec)?;
        if check_only {
            failed += check(&target)?;
            continue;
        }

        let (remote, verdict) = match &target {
            Target::File(url) => {
                let remote = hub::probe(url)?;
                let verdict = if precheck {
                    let v = hub::check::check_remote(url, remote.size).map_err(|e| {
                        anyhow::anyhow!("{spec}: llmoxide cannot run this file: {e:#}\n(--no-check downloads it anyway)")
                    })?;
                    Some(v)
                } else {
                    None
                };
                (remote, verdict)
            }
            Target::Repo { repo, quant } => {
                let (remote, v) = hub::repo::pick(repo, quant.as_deref())?;
                (remote, Some(v))
            }
        };
        println!(
            "  {}  {}{}",
            remote.filename,
            hub::progress::bytes(remote.size),
            match &remote.sha256 {
                Some(s) => format!("  sha256 {}…", &s[..12]),
                None => String::new(),
            }
        );
        if let Some(v) = &verdict {
            show(v);
        }
        let report = hub::fetch(&remote, &dir, force, verify, adopt)?;
        if !report.already_had_it {
            println!("saved to {}", report.path.display());
        }
        if let (Target::Repo { repo, .. }, Some(v)) = (&target, &verdict) {
            if v.arch == "gemma4" {
                hint_mmproj(repo);
            }
        }
    }
    if failed > 0 {
        anyhow::bail!("{failed} file(s) llmoxide cannot run");
    }
    Ok(())
}

/// `--check`: judge without downloading. Returns how many files failed.
fn check(target: &Target) -> anyhow::Result<usize> {
    let files: Vec<(String, String, u64)> = match target {
        Target::File(url) => {
            let r = hub::probe(url)?;
            vec![(r.filename.clone(), url.clone(), r.size)]
        }
        Target::Repo { repo, quant } => {
            let entries = hub::repo::list(repo)?;
            let mut files: Vec<_> = hub::repo::candidates(&entries, quant.as_deref());
            files.extend(hub::repo::mmprojs(&entries, quant.as_deref()).into_iter().cloned());
            let shards = entries.iter().filter(|e| e.is_shard()).count();
            if shards > 0 {
                println!("  ({shards} split-GGUF shards not checked: llmoxide loads a single file)");
            }
            files
                .into_iter()
                .map(|e| (e.path.clone(), hub::repo::resolve_url(repo, &e.path), e.size))
                .collect()
        }
    };
    let mut failed = 0;
    for (name, url, size) in files {
        match hub::check::check_remote(&url, size) {
            Ok(v) => {
                println!("  ok    {name}  {}", hub::progress::bytes(size));
                show(&v);
            }
            Err(e) => {
                failed += 1;
                println!("  no    {name}  {}: {e:#}", hub::progress::bytes(size));
            }
        }
    }
    Ok(failed)
}

fn show(v: &Verdict) {
    println!("        {}", v.summary);
    for b in &v.gpu_blockers {
        println!("        CPU only (--cpu): {b}");
    }
}

/// Gemma 4 can take images; point at the projector if the repo ships one.
fn hint_mmproj(repo: &str) {
    let Ok(entries) = hub::repo::list(repo) else { return };
    for e in hub::repo::mmprojs(&entries, None) {
        println!("image input: llmoxide-fetch hf:{repo}/{}", e.path);
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}

const HELP: &str = "\
llmoxide-fetch — download GGUF checkpoints from Hugging Face

  llmoxide-fetch                       the checkpoints this repo is built around
  llmoxide-fetch gemma4                one alias (--aliases lists them all)
  llmoxide-fetch hf:owner/repo         the best file in the repo llmoxide can run
  llmoxide-fetch hf:owner/repo:Q8_0    ... of that quantization
  llmoxide-fetch hf:owner/repo/f.gguf  one file
  llmoxide-fetch <https url>           a repo, or a blob/blame/resolve file URL

Before downloading, the file's header is read with a range request and run
through the real loaders: an architecture, tokenizer or quantization llmoxide
cannot run is refused up front instead of after the transfer.

  --check        judge the file, or every file in the repo, and download nothing
  --aliases      list the aliases
  --dir <path>   destination directory (default: models)
  --force        re-download even if already present
  --no-check     skip the header check and download anyway
  --no-verify    skip sha256 and GGUF structure checks
  --no-adopt     always download, even if the same file is already in --dir
                 under another name

Transfers resume: interrupt one and run the same command again.
";
