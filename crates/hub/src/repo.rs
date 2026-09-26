//! Picking a file out of a Hugging Face repo.
//!
//! A GGUF repo is usually a dozen quantizations of one model, and most of them
//! are types llmoxide has no decoder for. Given `hf:owner/repo` this lists the
//! repo, orders its `.gguf` files by a preferred quantization, and takes the
//! first one whose header passes [`crate::check`] — so what gets downloaded is
//! the best file that will actually load, not the first name that looked right.

use anyhow::{bail, Context};

use crate::check::{self, Verdict};
use crate::Remote;

/// Quantizations in the order they are tried when none is asked for.
///
/// Q4_K_M first, as llama.cpp and Ollama default to it: the body is Q4_K and
/// the sensitive tensors Q6_K, both of which the kernels handle. Then the
/// higher-fidelity types. Tags are matched against the upper-cased file name,
/// and `Q4_K_M` does not match unsloth's `UD-Q4_K_XL` (which is Q5_K inside).
pub const PREFERENCE: &[&str] = &["Q4_K_M", "Q6_K", "Q8_0", "Q4_K_S", "BF16", "F16", "F32"];

/// One `.gguf` in a repo listing.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Path inside the repo — files can sit in per-quant subdirectories.
    pub path: String,
    pub size: u64,
}

impl Entry {
    fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }

    /// Vision projector, not a language model.
    pub fn is_mmproj(&self) -> bool {
        self.name().to_ascii_lowercase().contains("mmproj")
    }

    /// One shard of a split GGUF (`-00001-of-00003.gguf`). The loader opens a
    /// single file, so shards are never candidates.
    pub fn is_shard(&self) -> bool {
        let n = self.name();
        let stem = n.strip_suffix(".gguf").unwrap_or(n);
        let b = stem.as_bytes();
        b.len() >= 15
            && &b[b.len() - 15..b.len() - 14] == b"-"
            && &b[b.len() - 9..b.len() - 5] == b"-of-"
            && b[b.len() - 14..b.len() - 9].iter().all(u8::is_ascii_digit)
            && b[b.len() - 5..].iter().all(u8::is_ascii_digit)
    }

    /// Rank in [`PREFERENCE`], or past the end if the name carries no known
    /// tag.
    fn rank(&self) -> usize {
        let up = self.name().to_ascii_uppercase();
        PREFERENCE
            .iter()
            .position(|tag| has_tag(&up, tag))
            .unwrap_or(PREFERENCE.len())
    }
}

/// `tag` appears in `name` as a whole token: `Q6_K` matches `x-Q6_K.gguf`
/// but not `x-Q6_K_L.gguf`, and `F16` does not match `BF16`.
fn has_tag(name: &str, tag: &str) -> bool {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
    name.match_indices(tag).any(|(i, _)| {
        boundary(name[..i].chars().next_back()) && boundary(name[i + tag.len()..].chars().next())
    })
}

/// Every `.gguf` in `repo` (`owner/name`), from the Hub's tree API.
pub fn list(repo: &str) -> anyhow::Result<Vec<Entry>> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let body = crate::agent()
        .get(&url)
        .call()
        .with_context(|| format!("listing {repo} (is the name right, and is the repo public?)"))?
        .into_body()
        .read_to_string()?;
    let json: serde_json::Value = serde_json::from_str(&body).context("tree API returned non-JSON")?;
    let items = json.as_array().context("tree API returned no file list")?;
    Ok(items
        .iter()
        .filter(|i| i["type"] == "file")
        .filter_map(|i| {
            let path = i["path"].as_str()?;
            path.ends_with(".gguf").then(|| Entry {
                path: path.to_string(),
                size: i["size"].as_u64().unwrap_or(0),
            })
        })
        .collect())
}

pub fn resolve_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{path}")
}

/// Candidate model files in the order they should be tried: `quant` narrows to
/// files carrying that tag, and within that [`PREFERENCE`] decides, smallest
/// first on a tie.
pub fn candidates(entries: &[Entry], quant: Option<&str>) -> Vec<Entry> {
    let quant = quant.map(str::to_ascii_uppercase);
    let mut out: Vec<Entry> = entries
        .iter()
        .filter(|e| !e.is_mmproj() && !e.is_shard())
        .filter(|e| {
            quant
                .as_deref()
                .is_none_or(|q| has_tag(&e.name().to_ascii_uppercase(), q))
        })
        .cloned()
        .collect();
    out.sort_by_key(|e| (e.rank(), e.size));
    out
}

/// The first candidate whose header passes and that the GPU can run, with its
/// verdict — or, if every passing file is CPU-only, the first of those. A
/// CPU-only pick is said out loud; a 27B on the CPU path is not a usable chat.
/// Rejections are printed as they happen, since each one is a network round
/// trip.
pub fn pick(repo: &str, quant: Option<&str>) -> anyhow::Result<(Remote, Verdict)> {
    let entries = list(repo)?;
    let cands = candidates(&entries, quant);
    if cands.is_empty() {
        let shards = entries.iter().filter(|e| e.is_shard()).count();
        match quant {
            Some(q) => bail!("{repo} has no single-file .gguf tagged {q}"),
            None if shards > 0 => bail!(
                "{repo} only has split GGUFs ({shards} shards); llmoxide loads a single file"
            ),
            None => bail!("{repo} has no .gguf files"),
        }
    }
    let mut cpu_only: Option<(Remote, Verdict)> = None;
    for e in &cands {
        let url = resolve_url(repo, &e.path);
        let remote = crate::probe(&url)?;
        match check::check_remote(&url, remote.size) {
            Ok(v) if v.gpu_blockers.is_empty() => return Ok((remote, v)),
            Ok(v) => {
                println!("  skip {}: CPU only ({})", e.path, v.gpu_blockers.join("; "));
                cpu_only.get_or_insert((remote, v));
            }
            Err(err) => println!("  skip {}: {err:#}", e.path),
        }
    }
    match cpu_only {
        Some((remote, v)) => {
            println!("  nothing here runs on the GPU; taking {} for the CPU path", remote.filename);
            Ok((remote, v))
        }
        None => bail!("nothing in {repo} is a file llmoxide can run"),
    }
}

/// The vision projectors in a listing, narrowed to `quant` if given.
pub fn mmprojs<'a>(entries: &'a [Entry], quant: Option<&str>) -> Vec<&'a Entry> {
    let quant = quant.map(str::to_ascii_uppercase);
    entries
        .iter()
        .filter(|e| e.is_mmproj())
        .filter(|e| {
            quant
                .as_deref()
                .is_none_or(|q| has_tag(&e.name().to_ascii_uppercase(), q))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(path: &str) -> Entry {
        Entry { path: path.into(), size: 0 }
    }

    #[test]
    fn tags_match_whole_tokens_only() {
        assert!(has_tag("QWEN3-8B-Q6_K.GGUF", "Q6_K"));
        assert!(!has_tag("QWEN3-8B-Q6_K_L.GGUF", "Q6_K"));
        assert!(!has_tag("QWEN3-8B-BF16.GGUF", "F16"));
        assert!(has_tag("QWEN3-8B-F16.GGUF", "F16"));
        assert!(!has_tag("QWEN3-8B-UD-Q4_K_XL.GGUF", "Q4_K_M"));
    }

    #[test]
    fn shards_and_mmproj_are_not_candidates() {
        assert!(e("Q4_K_M/m-Q4_K_M-00001-of-00002.gguf").is_shard());
        assert!(!e("m-Q4_K_M.gguf").is_shard());
        let got = candidates(
            &[
                e("mmproj-BF16.gguf"),
                e("m-Q8_0.gguf"),
                e("m-Q4_K_M.gguf"),
                e("m-UD-Q4_K_XL.gguf"),
                e("m-Q6_K-00001-of-00002.gguf"),
            ],
            None,
        );
        let names: Vec<_> = got.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(names, ["m-Q4_K_M.gguf", "m-Q8_0.gguf", "m-UD-Q4_K_XL.gguf"]);
    }

    #[test]
    fn explicit_quant_filters() {
        let got = candidates(&[e("m-Q8_0.gguf"), e("m-Q4_K_M.gguf")], Some("q8_0"));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "m-Q8_0.gguf");
    }
}
