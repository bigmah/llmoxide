//! Can llmoxide run this file? Answered from the header alone, before the
//! transfer.
//!
//! A GGUF's metadata and tensor table sit at the front of the file, so a few
//! megabytes fetched with a `Range` request say everything a loader would
//! refuse over: the architecture, the tokenizer's pre-tokenizer, and the type
//! of every tensor. Finding out after 20 GB that a file is `qwen3moe`, or
//! quantized to Q5_K, is the failure this module exists to prevent.
//!
//! The verdict runs the real loaders — [`model::Arch::detect`], the config
//! parsers and [`tokenizer::Tokenizer::from_gguf`] — over a
//! [`gguf::Gguf::sparse`] with no tensor bytes resident, rather than keeping a
//! second list of what is supported that would drift from the first.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;

use anyhow::{bail, Context};
use gguf::{GgmlType, Header};

/// First read. Big enough for a Qwen vocab in one round trip; a Gemma one
/// (262k tokens) takes a second.
const FIRST_READ: usize = 4 << 20;
/// No real header is near this. A file that has not finished its tensor table
/// by here is not something to keep downloading in the hope that it will.
const MAX_HEADER: usize = 256 << 20;

/// What the header says, once the loaders have accepted it.
pub struct Verdict {
    /// `general.architecture`, e.g. `qwen3`, `gemma4`, `clip`.
    pub arch: String,
    /// One line for a person: layer count, width, quantization mix.
    pub summary: String,
    /// Reasons the GPU path would refuse a file the CPU path runs.
    pub gpu_blockers: Vec<String>,
}

/// Fetch just enough of `url` to parse its header, and judge it.
///
/// `size` is the file's full length when known (0 if not); a header whose
/// tensors run past it means the *remote* copy is truncated.
pub fn check_remote(url: &str, size: u64) -> anyhow::Result<Verdict> {
    let header = fetch_header(url)?;
    if size > 0 {
        header
            .check_bounds(size)
            .context("the file on the server is itself truncated")?;
    }
    judge(header)
}

/// Judge a header already in hand.
pub fn judge(header: Header) -> anyhow::Result<Verdict> {
    // Bytes per tensor type, for the summary and the GPU check.
    let mut mix: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut types: Vec<GgmlType> = Vec::new();
    for t in &header.tensors {
        *mix.entry(t.ty.name()).or_default() += t.byte_len() as u64;
        if !types.contains(&t.ty) {
            types.push(t.ty);
        }
    }
    let embd = header.info("token_embd.weight").ok().map(|t| t.ty);

    let g = gguf::Gguf::sparse(header, HashMap::new())?;
    let arch = g.str("general.architecture")?.to_string();

    let mut gpu_blockers = Vec::new();
    let what = if arch == "clip" {
        model::vision::Config::from_gguf(&g)?;
        "vision tower (mmproj)".to_string()
    } else {
        match model::Arch::detect(&g)? {
            model::Arch::Gemma4 => {
                model::Config::from_gguf(&g)?;
            }
            model::Arch::Qwen35 => {
                model::qwen35::Config::from_gguf(&g)?;
            }
        }
        tokenizer::Tokenizer::from_gguf(&g)?;

        // Mirrors `Pipelines::embed_for` in the gpu crate.
        if let Some(ty) = embd {
            if !matches!(ty, GgmlType::Q6K | GgmlType::Q8_0 | GgmlType::F32) {
                gpu_blockers.push(format!("no GPU embedding kernel for a {} token_embd", ty.name()));
            }
        }

        let k = |s: &str| format!("{arch}.{s}");
        let layers = g.usize(&k("block_count")).unwrap_or(0);
        let width = g.usize(&k("embedding_length")).unwrap_or(0);
        format!("{layers} layers, width {width}")
    };

    // Both the text stack and the vision tower go through `pipeline_for`.
    if types.contains(&GgmlType::F16) {
        gpu_blockers.push("no GPU matvec kernel for F16 weights".to_string());
    }

    let total: u64 = mix.values().sum::<u64>().max(1);
    let mut mix: Vec<_> = mix.into_iter().collect();
    mix.sort_by(|a, b| b.1.cmp(&a.1));
    let mix = mix
        .iter()
        .filter(|(_, n)| n * 100 / total >= 1)
        .map(|(name, n)| format!("{name} {}%", n * 100 / total))
        .collect::<Vec<_>>()
        .join(" ");

    Ok(Verdict {
        summary: format!("{arch} · {what} · {mix}"),
        arch,
        gpu_blockers,
    })
}

/// Read a growing prefix of `url` until the header parses.
fn fetch_header(url: &str) -> anyhow::Result<Header> {
    let mut buf: Vec<u8> = Vec::new();
    let mut want = FIRST_READ;
    loop {
        let have = buf.len();
        let resp = crate::agent()
            .get(url)
            .header("Range", &format!("bytes={have}-{}", want - 1))
            .call()
            .with_context(|| format!("GET {url} (header)"))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            bail!("GET {url} returned {status}");
        }
        // A server that ignores the range sends the whole file from byte 0.
        // Take what we asked for from the top and drop the connection.
        if status != 206 {
            buf.clear();
        }
        let limit = (want - buf.len()) as u64;
        resp.into_body()
            .into_reader()
            .take(limit)
            .read_to_end(&mut buf)?;
        let short = buf.len() < want;

        match Header::parse(&buf) {
            Ok(h) => return Ok(h),
            Err(gguf::Error::Eof(_)) if !short && want < MAX_HEADER => want *= 2,
            Err(gguf::Error::Eof(_)) if short => bail!("file ends inside its own header"),
            Err(gguf::Error::BadTensorType(n)) => bail!(
                "has {} tensors, which llmoxide has no decoder for \
                 (supported: F32 F16 BF16 Q8_0 Q4_K Q6_K)",
                ggml_type_name(n)
            ),
            Err(e) => return Err(e.into()),
        }
    }
}

/// Names for the ggml types [`GgmlType`] rejects, so the refusal says `Q5_K`
/// rather than `13`.
fn ggml_type_name(n: u32) -> String {
    let name = match n {
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        13 => "Q5_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => return format!("ggml type {n}"),
    };
    name.to_string()
}
