//! Fetching checkpoints from Hugging Face.
//!
//! The requirement that shapes this module is not bandwidth, it is that a
//! 21 GB transfer *will* be interrupted, and that a checkpoint which arrives
//! short is worse than one that does not arrive at all: this repo already lost
//! time to a download that stopped at 17.8 of 20.9 GiB, wrote a plausible-
//! looking file, and was rejected by everything that tried to open it.
//!
//! So the download is resumable, and nothing is called finished until three
//! things agree:
//!
//! * the byte count matches `x-linked-size`,
//! * the SHA-256 matches `x-linked-etag` — Hugging Face publishes both on the
//!   `HEAD` of a resolve URL, so verification costs one extra request, and
//! * [`gguf::Gguf::open`] parses it, which walks the tensor table and refuses
//!   any tensor whose data runs past the end of the file. That is the check the
//!   truncated download would have failed.
//!
//! Only then does the `.part` file take its real name, so an interrupted fetch
//! can never be mistaken for a complete one.

pub mod progress;

use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use sha2::{Digest, Sha256};

use progress::Progress;

/// Read size for the transfer and for hashing. Large enough that per-chunk
/// overhead is irrelevant at gigabyte scale, small enough that the meter still
/// moves several times a second on a fast link.
const CHUNK: usize = 8 << 20;

/// The checkpoints this repo is built around, so neither has to be a URL.
pub const ALIASES: &[(&str, &str)] = &[
    (
        "gemma4",
        "https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF/resolve/main/gemma4-v2-Q4_K_M.gguf",
    ),
    (
        "qwen35",
        "https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/resolve/main/Qwen3.8-27B-OBLITERATED-Q6_K.gguf",
    ),
];

/// Turn an alias, a `hf:owner/repo/file` shorthand, or any Hugging Face file URL
/// into something that serves bytes.
///
/// The URL you get from the website's "copy link" is a `/blob/` (or `/blame/`)
/// page, which returns HTML — downloading one yields a few kilobytes that look
/// like a corrupt model. Rewriting to `/resolve/` is the whole difference.
pub fn resolve_url(spec: &str) -> anyhow::Result<String> {
    if let Some((_, url)) = ALIASES.iter().find(|(name, _)| *name == spec) {
        return Ok((*url).to_string());
    }
    if let Some(rest) = spec.strip_prefix("hf:") {
        let parts: Vec<&str> = rest.splitn(3, '/').collect();
        if parts.len() != 3 {
            bail!("expected hf:<owner>/<repo>/<file>, got hf:{rest}");
        }
        return Ok(format!(
            "https://huggingface.co/{}/{}/resolve/main/{}",
            parts[0], parts[1], parts[2]
        ));
    }
    if !spec.starts_with("https://") {
        bail!(
            "not a known alias, hf: spec, or https URL: {spec}\naliases: {}",
            ALIASES.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
        );
    }
    Ok(spec.replace("/blob/", "/resolve/").replace("/blame/", "/resolve/"))
}

pub struct Remote {
    pub url: String,
    pub size: u64,
    /// SHA-256 as published by Hugging Face, when it publishes one.
    pub sha256: Option<String>,
    pub filename: String,
}

fn agent() -> ureq::Agent {
    // No global timeout: the deadline for a 21 GB body is not knowable up
    // front, and a transfer that is still moving must not be killed. Stalls are
    // caught by the read timeout instead, and shown by the meter.
    ureq::Agent::config_builder()
        .timeout_global(None)
        .build()
        .into()
}

/// An agent that stops at the redirect instead of following it.
///
/// This matters more than it looks. A resolve URL 302s to a CDN, and the
/// headers that describe the *file* — `x-linked-size`, `x-linked-etag` — are on
/// that 302, put there by Hugging Face. Follow it and you get the CDN's own
/// headers instead, whose `etag` is the xet content hash: also 64 hex
/// characters, also plausible, and not the SHA-256 of the file. Verifying
/// against it fails every honest download, at the very end, after 21 GB.
fn probe_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(None)
        .max_redirects(0)
        .max_redirects_will_error(false)
        .build()
        .into()
}

/// Ask what is at the other end before committing to a transfer.
pub fn probe(url: &str) -> anyhow::Result<Remote> {
    let resp = probe_agent()
        .head(url)
        .call()
        .with_context(|| format!("HEAD {url}"))?;
    let status = resp.status();
    let redirected = status.is_redirection();
    if !status.is_success() && !redirected {
        bail!("HEAD {url} returned {status}");
    }
    let header = |name: &str| {
        resp.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };

    // `x-linked-size` is the real object size; `content-length` on a redirect
    // describes the redirect body, not the file.
    let size = header("x-linked-size")
        .or_else(|| if redirected { None } else { header("content-length") })
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // Only ever `x-linked-etag`, and only from a response Hugging Face itself
    // produced. A plain `etag` here is the git blob SHA-1 for a small file
    // (harmless, wrong length, filtered) — but a CDN's would be the xet hash,
    // which is indistinguishable by shape and wrong.
    let sha256 = header("x-linked-etag")
        .or_else(|| if redirected { None } else { header("etag") })
        .map(|e| e.trim_matches('"').trim_start_matches("W/").to_string())
        .filter(|e| e.len() == 64 && e.chars().all(|c| c.is_ascii_hexdigit()));

    let filename = url
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("model.gguf")
        .split('?')
        .next()
        .unwrap_or("model.gguf")
        .to_string();

    Ok(Remote {
        url: url.to_string(),
        size,
        sha256,
        filename,
    })
}

/// SHA-256 a file, reporting progress — at 21 GB this is a minute of apparent
/// silence otherwise.
pub fn hash_file(path: &Path, label: &str) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let total = f.metadata()?.len();
    let mut meter = Progress::new(label, total);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        meter.add(n as u64);
    }
    meter.finish();
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn name_of(p: &Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().into_owned()
}

/// A file in `dir`, other than `dest`, whose length is exactly `size`.
fn find_twin(dir: &Path, dest: &Path, size: u64) -> anyhow::Result<Option<PathBuf>> {
    if size == 0 {
        return Ok(None);
    }
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if p == dest || !p.is_file() {
            continue;
        }
        if p.extension().is_some_and(|e| e == "part") {
            continue;
        }
        if std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0) == size {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

pub struct Report {
    pub path: PathBuf,
    /// True when the file was already present and verified, so nothing moved.
    pub already_had_it: bool,
}

/// Download `remote` into `dir`, resuming a previous attempt if one is there.
pub fn fetch(
    remote: &Remote,
    dir: &Path,
    force: bool,
    verify: bool,
    adopt: bool,
) -> anyhow::Result<Report> {
    std::fs::create_dir_all(dir)?;
    let dest = dir.join(&remote.filename);
    let part = dir.join(format!("{}.part", remote.filename));

    if dest.exists() && !force {
        let have = std::fs::metadata(&dest)?.len();
        if have == remote.size {
            if verify {
                if let Some(want) = &remote.sha256 {
                    let got = hash_file(&dest, &format!("verify {}", remote.filename))?;
                    if &got == want {
                        println!("already present and verified: {}", dest.display());
                        return Ok(Report { path: dest, already_had_it: true });
                    }
                    bail!(
                        "{} exists but its hash does not match the remote\n  local  {got}\n  remote {want}\nre-run with --force to replace it",
                        dest.display()
                    );
                }
            }
            println!("already present: {} ({})", dest.display(), progress::bytes(have));
            return Ok(Report { path: dest, already_had_it: true });
        }
        bail!(
            "{} exists but is {}, expected {} — this is what a truncated download looks like.\nre-run with --force to replace it",
            dest.display(),
            progress::bytes(have),
            progress::bytes(remote.size)
        );
    }

    // The same checkpoint is often already on disk under a different name —
    // re-uploads get renamed, and a repo's file name rarely matches what you
    // called it locally. Downloading 21 GB you already have is the most
    // expensive possible mistake here, so look for a twin first. Size is the
    // cheap filter; the hash is what decides.
    if adopt && !force {
        if let Some(twin) = find_twin(dir, &dest, remote.size)? {
            if let Some(want) = &remote.sha256 {
                println!(
                    "{} is exactly {} — checking whether it is this file",
                    twin.display(),
                    progress::bytes(remote.size)
                );
                let got = hash_file(&twin, &format!("hash {}", name_of(&twin)))?;
                if &got == want {
                    // A hard link, not a copy or a rename: costs no disk, and
                    // leaves the name it already had working.
                    std::fs::hard_link(&twin, &dest)?;
                    println!(
                        "identical to the remote file — linked {} -> {} instead of downloading {}",
                        dest.display(),
                        twin.display(),
                        progress::bytes(remote.size)
                    );
                    return Ok(Report { path: dest, already_had_it: true });
                }
                println!("different file, downloading");
            }
        }
    }

    // Resume: whatever is already in `.part` is bytes we do not have to fetch
    // again, but they still have to go through the hasher.
    let mut hasher = Sha256::new();
    let mut have = if force { 0 } else { part.metadata().map(|m| m.len()).unwrap_or(0) };
    if have > remote.size {
        println!("partial file is larger than the remote; starting over");
        have = 0;
    }
    if have > 0 {
        println!(
            "resuming {} at {} of {}",
            remote.filename,
            progress::bytes(have),
            progress::bytes(remote.size)
        );
        let mut f = std::fs::File::open(&part)?;
        let mut meter = Progress::new(format!("rehash {}", remote.filename), have);
        let mut buf = vec![0u8; CHUNK];
        let mut read = 0u64;
        while read < have {
            let want = CHUNK.min((have - read) as usize);
            let n = f.read(&mut buf[..want])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            read += n as u64;
            meter.add(n as u64);
        }
        meter.finish();
        have = read;
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&part)?;
    file.set_len(have)?;
    file.seek(std::io::SeekFrom::End(0))?;

    if have < remote.size || remote.size == 0 {
        let mut req = agent().get(&remote.url);
        if have > 0 {
            req = req.header("Range", &format!("bytes={have}-"));
        }
        let resp = req.call().with_context(|| format!("GET {}", remote.url))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            bail!("GET returned {status}");
        }
        // A server that ignores the range answers 200 with the whole file. Take
        // it from the top rather than appending it to what we already had.
        if have > 0 && status != 206 {
            println!("server ignored the range request; restarting from zero");
            file.set_len(0)?;
            file.seek(std::io::SeekFrom::Start(0))?;
            hasher = Sha256::new();
            have = 0;
        }

        let mut meter =
            Progress::new(remote.filename.clone(), remote.size.max(have)).at(have);
        let mut reader = resp.into_body().into_reader();
        let mut buf = vec![0u8; CHUNK];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            meter.add(n as u64);
        }
        meter.finish();
    }

    file.flush()?;
    file.sync_all()?;
    let got = file.metadata()?.len();
    drop(file);

    // ---- three independent checks before this file gets to keep its name ----

    if remote.size > 0 && got != remote.size {
        bail!(
            "short transfer: got {} of {}. the partial file is kept at {}, re-run to resume",
            progress::bytes(got),
            progress::bytes(remote.size),
            part.display()
        );
    }

    if verify {
        if let Some(want) = &remote.sha256 {
            let digest = hex(&hasher.finalize());
            if &digest != want {
                bail!(
                    "hash mismatch — the bytes are not what the server published\n  got    {digest}\n  want   {want}\nthe partial file is kept at {}",
                    part.display()
                );
            }
            println!("sha256 ok  {digest}");
        } else {
            println!("note: the server published no hash, so only size was checked");
        }

        if remote.filename.ends_with(".gguf") {
            // Walks the tensor table and rejects any tensor whose data runs
            // past the end of the file — the exact failure a short download
            // produces, and one a size check alone can miss if the server also
            // lied about the length.
            gguf::Gguf::open(&part)
                .with_context(|| format!("{} downloaded but does not parse as GGUF", part.display()))?;
            println!("gguf ok    tensor table fits the file");
        }
    }

    std::fs::rename(&part, &dest)?;
    Ok(Report { path: dest, already_had_it: false })
}
