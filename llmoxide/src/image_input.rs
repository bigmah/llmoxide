//! Turning an OpenAI-shaped content part into residual-stream rows.
//!
//! This is the adapter between what a client sends and what the vision tower
//! takes. It accepts the spellings that actually turn up in the wild:
//!
//! ```json
//! {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR…"}}
//! {"type": "image_url", "image_url": {"url": "file:///abs/path.jpg"}}
//! {"type": "image_url", "image_url": {"url": "/abs/path.jpg"}}
//! {"type": "image",     "path": "/abs/path.jpg"}
//! {"type": "image",     "data": "iVBOR…"}
//! ```
//!
//! Remote `http(s)` URLs are deliberately *not* fetched. A server that
//! dereferences URLs on a client's say-so is an SSRF hole, and this one is
//! meant to be run against local files and pasted data.

use serde_json::Value;

use crate::{Error, Result};

/// Decode standard base64, ignoring whitespace. Padding is optional.
fn base64(s: &str) -> Option<Vec<u8>> {
    const INVALID: u8 = 0xFF;
    let mut table = [INVALID; 256];
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut i = 0;
    while i < alphabet.len() {
        table[alphabet[i] as usize] = i as u8;
        i += 1;
    }

    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &b in s.as_bytes() {
        if b.is_ascii_whitespace() || b == b'=' {
            continue;
        }
        let v = table[b as usize];
        if v == INVALID {
            return None;
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// Pull the encoded image bytes out of a content part.
pub fn part_bytes(part: &Value) -> Result<Vec<u8>> {
    let url = part
        .get("image_url")
        .and_then(|u| match u {
            // Both `{"image_url": {"url": …}}` and the shorthand
            // `{"image_url": "…"}` are seen from real clients.
            Value::Object(_) => u.get("url").and_then(Value::as_str),
            Value::String(s) => Some(s.as_str()),
            _ => None,
        })
        .or_else(|| part.get("url").and_then(Value::as_str));

    if let Some(raw) = part.get("data").and_then(Value::as_str) {
        return base64(raw).ok_or_else(|| Error::Image("image data is not valid base64".into()));
    }
    if let Some(path) = part.get("path").and_then(Value::as_str) {
        return std::fs::read(path).map_err(|e| Error::Image(format!("{path}: {e}")));
    }

    let url = url.ok_or_else(|| Error::Image("image part carries no url, path or data".into()))?;

    if let Some(rest) = url.strip_prefix("data:") {
        let payload = rest
            .split_once(";base64,")
            .map(|(_, b)| b)
            .or_else(|| rest.split_once(',').map(|(_, b)| b))
            .ok_or_else(|| Error::Image("malformed data: URL".into()))?;
        return base64(payload).ok_or_else(|| Error::Image("data: URL is not valid base64".into()));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(Error::Image(
            "remote image URLs are not fetched; send the bytes as a data: URL".into(),
        ));
    }
    let path = url.strip_prefix("file://").unwrap_or(url);
    std::fs::read(path).map_err(|e| Error::Image(format!("{path}: {e}")))
}

/// Adapts the vision tower to the prompt builder's `ImageEncoder`.
pub struct Encoder<'a> {
    pub vision: &'a vision::Vision,
    /// Positions one image may occupy. The span attends to itself in both
    /// directions, so it has to prefill in a single batch.
    pub max_tokens: usize,
}

impl chat::ImageEncoder for Encoder<'_> {
    fn encode(&self, part: &Value) -> anyhow::Result<(Vec<f32>, usize)> {
        let bytes = part_bytes(part)?;
        let prepared = vision::prepare(&bytes, &self.vision.cfg)?;
        let out = self.vision.encode(&prepared)?;
        anyhow::ensure!(
            out.n <= self.max_tokens,
            "image encodes to {} positions but max_batch is {}; \
             raise LoadOptions::max_batch or send a smaller image",
            out.n,
            self.max_tokens
        );
        tracing::info!(
            tokens = out.n,
            grid = format!("{}x{}", out.grid.0, out.grid.1),
            "encoded image"
        );
        Ok((out.rows, out.n))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_round_trips_a_known_vector() {
        assert_eq!(super::base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(super::base64("aGVsbG8").unwrap(), b"hello");
        assert_eq!(super::base64("aGVs\nbG8=").unwrap(), b"hello");
        assert!(super::base64("not base64!").is_none());
    }
}
