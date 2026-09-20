//! Bytes on disk to the tensor the tower consumes.
//!
//! Three steps, each of which has to match llama.cpp exactly or the encoder
//! sees a different image than the reference does:
//!
//! 1. **Size.** The aspect ratio is preserved and both sides are snapped to a
//!    multiple of `patch * n_merge`, then the area is pulled inside the token
//!    budget. This is `smart_resize` in the transformers code.
//! 2. **Resample.** Bilinear, *align-corners* (the ratio divides by
//!    `target - 1`, not `target`), with a truncating cast back to `u8`.
//!    Rounding instead of truncating moves roughly a third of the pixels by
//!    one level, which is small but not nothing.
//! 3. **Scale.** `mean = 0`, `std = 1` in this checkpoint, so normalization is
//!    just `v / 255`; the graph then maps that to `[-1, 1]`, which is folded
//!    in here.

use crate::config::VisionConfig;

/// A resized image in the layout `ggml_conv_2d` wants: channel-planar,
/// `[c][y][x]`, already scaled to `[-1, 1]`.
pub struct Planar {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Planar {
    /// Patch grid this image produces.
    pub fn grid(&self, patch: usize) -> (usize, usize) {
        (self.w / patch, self.h / patch)
    }
}

/// Pick the resized dimensions: aspect preserved, both sides a multiple of
/// `align`, area within `[min_px, max_px]`.
pub fn smart_resize(
    w: usize,
    h: usize,
    align: usize,
    min_px: usize,
    max_px: usize,
) -> (usize, usize) {
    let f = align as f32;
    let round_by = |x: f32| ((x / f).round() * f) as usize;
    let ceil_by = |x: f32| ((x / f).ceil() * f) as usize;
    let floor_by = |x: f32| ((x / f).floor() * f) as usize;

    // Snap first, then correct the area — the order matters, because snapping
    // can push a just-inside image over the ceiling.
    let mut w_bar = align.max(round_by(w as f32));
    let mut h_bar = align.max(round_by(h as f32));

    let area = (w * h) as f32;
    if w_bar * h_bar > max_px {
        let beta = (area / max_px as f32).sqrt();
        w_bar = align.max(floor_by(w as f32 / beta));
        h_bar = align.max(floor_by(h as f32 / beta));
    } else if w_bar * h_bar < min_px {
        let beta = (min_px as f32 / area).sqrt();
        w_bar = ceil_by(w as f32 * beta);
        h_bar = ceil_by(h as f32 * beta);
    }
    (w_bar, h_bar)
}

/// Align-corners bilinear resample of an interleaved RGB8 buffer.
fn resize_bilinear(src: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    let mut out = vec![0u8; tw * th * 3];
    let x_ratio = if tw > 1 {
        (sw - 1) as f32 / (tw - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if th > 1 {
        (sh - 1) as f32 / (th - 1) as f32
    } else {
        0.0
    };

    for y in 0..th {
        let py = y as f32 * y_ratio;
        let y0 = (py as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let yf = py - y0 as f32;
        for x in 0..tw {
            let px = x as f32 * x_ratio;
            let x0 = (px as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let xf = px - x0 as f32;

            let p = |xx: usize, yy: usize, c: usize| src[(yy * sw + xx) * 3 + c] as f32;
            for c in 0..3 {
                let top = p(x0, y0, c) + (p(x1, y0, c) - p(x0, y0, c)) * xf;
                let bot = p(x0, y1, c) + (p(x1, y1, c) - p(x0, y1, c)) * xf;
                // Truncating, matching llama.cpp's cast.
                out[(y * tw + x) * 3 + c] = (top + (bot - top) * yf) as u8;
            }
        }
    }
    out
}

/// Decode any supported container and prepare it for the tower.
pub fn prepare(bytes: &[u8], cfg: &VisionConfig) -> anyhow::Result<Planar> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("could not decode image: {e}"))?
        .to_rgb8();
    let (sw, sh) = (img.width() as usize, img.height() as usize);
    anyhow::ensure!(sw > 0 && sh > 0, "image has a zero dimension");
    Ok(prepare_rgb8(img.as_raw(), sw, sh, cfg))
}

/// The same, for pixels that are already decoded and interleaved RGB8.
pub fn prepare_rgb8(src: &[u8], sw: usize, sh: usize, cfg: &VisionConfig) -> Planar {
    let (tw, th) = smart_resize(
        sw,
        sh,
        cfg.align(),
        cfg.image_min_pixels,
        cfg.image_max_pixels,
    );
    let resized = resize_bilinear(src, sw, sh, tw, th);

    let n = tw * th;
    let mut data = vec![0f32; n * 3];
    for y in 0..th {
        for x in 0..tw {
            let s = (y * tw + x) * 3;
            let d = y * tw + x;
            for c in 0..3 {
                // v/255 to normalize, then the graph's `2v - 1`.
                data[c * n + d] = (resized[s + c] as f32 / 255.0) * 2.0 - 1.0;
            }
        }
    }
    Planar { w: tw, h: th, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_snaps_and_respects_bounds() {
        let (align, min_px, max_px) = (48, 40 * 2304, 280 * 2304);
        // A large photo is pulled under the ceiling, still a multiple of 48.
        let (w, h) = smart_resize(4032, 3024, align, min_px, max_px);
        assert_eq!(w % align, 0);
        assert_eq!(h % align, 0);
        assert!(w * h <= max_px, "{w}x{h} over the ceiling");
        // Aspect is preserved to within one alignment step.
        let want = 4032.0 / 3024.0;
        assert!((w as f32 / h as f32 - want).abs() < 0.1);

        // A tiny icon is pushed up to the floor rather than left alone.
        let (w, h) = smart_resize(32, 32, align, min_px, max_px);
        assert!(w * h >= min_px, "{w}x{h} under the floor");
        assert_eq!(w % align, 0);
    }

    #[test]
    fn resize_is_identity_at_matching_size() {
        let src: Vec<u8> = (0..4 * 4 * 3).map(|i| i as u8).collect();
        assert_eq!(resize_bilinear(&src, 4, 4, 4, 4), src);
    }
}
