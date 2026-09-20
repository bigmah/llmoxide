//! The gemma4v vision tower, on the CPU.
//!
//! Written against llama.cpp's `clip_graph_gemma4v::build` and validated
//! against it, in the same spirit as the text reference path next door: clear
//! enough to diff tensor by tensor, not fast.
//!
//! The block is the text stack's block with the sequence axis swapped for
//! patches, so the shapes below should look familiar:
//!
//! ```text
//! x  = conv16x16(image) + pos_x[col] + pos_y[row]
//! h  = rms(x) * ln1
//! q  = rope2d(rms_head(Wq h) * q_norm)     // low half by column, high by row
//! k  = rope2d(rms_head(Wk h) * k_norm)
//! v  =        rms_head(Wv h)               // no weight, no rotation
//! x  = x + rms(Wo attn(q,k,v)) * attn_post_norm
//! y  = rms(x) * ln2
//! x  = x + rms(Wd (gelu(Wg y) * Wu y)) * ffn_post_norm
//! ```
//!
//! Three details are easy to get wrong and silently produce plausible
//! garbage:
//!
//! - **Attention is bidirectional.** Patches are not a causal sequence.
//! - **The softmax scale is 1.0**, not `1/sqrt(head_dim)` — the same folded
//!   temperature the text stack uses, and for the same reason.
//! - **Linears clamp.** Each weight may carry calibration ranges beside it
//!   (`.input_min`/`.output_max`/…); llama.cpp clamps the input before the
//!   matmul and the result after. Ignoring them is fine on most images and
//!   wrong on the ones that saturate, which is the worst failure shape.

use std::collections::HashMap;

use anyhow::Context;
use gguf::{Gguf, TensorView};
use model::ops;
use rayon::prelude::*;

use crate::config::VisionConfig;
use crate::preprocess::Planar;

#[inline]
fn bf16(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// `Y = W X` over `n` rows, with a BF16 fast path.
///
/// Every weight in the tower is BF16, and `ops::matmat` reaches them through
/// `dot_row`, whose single accumulator makes each add wait on the one before
/// it — the loop runs at floating-point *latency* rather than throughput.
/// Four independent chains is most of a 3x here, which matters because one
/// image is ~180 G multiply-adds and this runs on the CPU even when the text
/// model is on the GPU.
///
/// The partial-sum order differs from `dot_row`'s, so the last few bits do
/// too; nothing downstream is compared against `dot_row` for these tensors.
fn matmat(y: &mut [f32], w: &TensorView<'_>, x: &[f32], n: usize) {
    let (in_dim, out_dim) = (w.in_dim(), w.out_dim());
    let raw: Option<&[u16]> = (w.ty() == gguf::GgmlType::BF16)
        .then(|| bytemuck::try_cast_slice(w.data).ok())
        .flatten();
    let Some(raw) = raw else {
        // Anything else (or a misaligned mapping) takes the shared path.
        return ops::matmat(y, w, x, n);
    };

    // A task per token, all output rows inside. The transposed order — a task
    // per output row, tokens inside — looks better for weight reuse and
    // measures ~1.7x *worse*: it streams the whole activation block once per
    // row and needs a transpose afterwards, while this way the token's own
    // 3 KB of `x` stays in L1 and the weight reads stay sequential enough for
    // the prefetcher. Measured, not reasoned.
    y.par_chunks_mut(out_dim).enumerate().for_each(|(t, col)| {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for (o, out) in col.iter_mut().enumerate() {
            let row = &raw[o * in_dim..(o + 1) * in_dim];
            let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
            let mut i = 0;
            while i + 4 <= in_dim {
                a0 += bf16(row[i]) * xt[i];
                a1 += bf16(row[i + 1]) * xt[i + 1];
                a2 += bf16(row[i + 2]) * xt[i + 2];
                a3 += bf16(row[i + 3]) * xt[i + 3];
                i += 4;
            }
            let mut acc = (a0 + a1) + (a2 + a3);
            while i < in_dim {
                acc += bf16(row[i]) * xt[i];
                i += 1;
            }
            *out = acc;
        }
    });
}

/// Calibration range carried beside a weight, for `Gemma4ClippableLinear`.
#[derive(Clone, Copy, Debug)]
struct Clamp {
    in_lo: f32,
    in_hi: f32,
    out_lo: f32,
    out_hi: f32,
}

/// Rows an encoded image contributes to the residual stream.
pub struct ImageEmbeds {
    /// `n * proj_dim` floats, row-major.
    pub rows: Vec<f32>,
    pub n: usize,
    pub proj_dim: usize,
    /// Pooled token grid, for reporting.
    pub grid: (usize, usize),
}

pub struct Vision {
    g: Gguf,
    pub cfg: VisionConfig,
    clamp: HashMap<String, Clamp>,
}

impl Vision {
    pub fn open(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::new(Gguf::open(path)?)
    }

    pub fn new(g: Gguf) -> anyhow::Result<Self> {
        let cfg = VisionConfig::from_gguf(&g)?;
        // The tower is read once per image, not once per token, but the
        // mapping is still faulted in up front: the alternative is a page
        // fault storm in the middle of the first encode.
        g.prefault();
        let clamp = Self::clamp_table(&g);
        tracing::info!("{}", cfg.summary());
        Ok(Self { g, cfg, clamp })
    }

    /// Collect the per-weight calibration ranges. A weight with no scalars
    /// beside it simply does not appear.
    fn clamp_table(g: &Gguf) -> HashMap<String, Clamp> {
        let scalar = |name: &str| -> Option<f32> {
            let t = g.tensor_opt(name)?;
            t.as_f32().and_then(|v| v.first().copied())
        };
        let mut out = HashMap::new();
        for i in 0..g.header().tensors.len() {
            let name = g.tensor_at(i).info.name.clone();
            let Some(stem) = name.strip_suffix(".weight") else {
                continue;
            };
            let c = Clamp {
                in_lo: scalar(&format!("{stem}.input_min")).unwrap_or(f32::NEG_INFINITY),
                in_hi: scalar(&format!("{stem}.input_max")).unwrap_or(f32::INFINITY),
                out_lo: scalar(&format!("{stem}.output_min")).unwrap_or(f32::NEG_INFINITY),
                out_hi: scalar(&format!("{stem}.output_max")).unwrap_or(f32::INFINITY),
            };
            if c.in_lo.is_finite() || c.in_hi.is_finite() || c.out_lo.is_finite() || c.out_hi.is_finite()
            {
                out.insert(name, c);
            }
        }
        out
    }

    fn tensor(&self, name: &str) -> anyhow::Result<TensorView<'_>> {
        self.g
            .tensor(name)
            .with_context(|| format!("mmproj is missing {name}"))
    }

    /// An F32 norm gain, borrowed in place.
    fn gain(&self, name: &str) -> anyhow::Result<&[f32]> {
        self.tensor(name)?
            .as_f32()
            .with_context(|| format!("{name} is not F32"))
    }

    /// `y = W x` over `n` rows, honouring the weight's calibration ranges.
    fn linear(&self, name: &str, y: &mut [f32], x: &[f32], n: usize) -> anyhow::Result<()> {
        let w = self.tensor(name)?;
        match self.clamp.get(name) {
            None => matmat(y, &w, x, n),
            Some(c) => {
                let mut xc = x.to_vec();
                for v in xc.iter_mut() {
                    *v = v.clamp(c.in_lo, c.in_hi);
                }
                matmat(y, &w, &xc, n);
                for v in y.iter_mut() {
                    *v = v.clamp(c.out_lo, c.out_hi);
                }
            }
        }
        Ok(())
    }

    /// Encode one prepared image into residual-stream rows.
    pub fn encode(&self, img: &Planar) -> anyhow::Result<ImageEmbeds> {
        let cfg = &self.cfg;
        let (nx, ny) = img.grid(cfg.patch_size);
        anyhow::ensure!(
            nx > 0 && ny > 0,
            "image is smaller than one patch after resizing"
        );
        anyhow::ensure!(
            nx % cfg.n_merge == 0 && ny % cfg.n_merge == 0,
            "patch grid {nx}x{ny} is not a multiple of the {} pooling kernel",
            cfg.n_merge
        );
        anyhow::ensure!(
            nx.max(ny) <= cfg.pos_table_len,
            "patch grid {nx}x{ny} exceeds the {}-entry position table",
            cfg.pos_table_len
        );

        let n = nx * ny;
        let mut x = self.patch_embed(img, nx, ny)?;
        self.add_position(&mut x, nx, ny)?;
        for il in 0..cfg.n_layers {
            self.block(il, &mut x, n, nx)?;
        }
        self.pool_and_project(&x, nx, ny)
    }

    /// 16x16 stride-16 convolution over the image, one output row per patch.
    /// No bias tensor exists for this checkpoint.
    fn patch_embed(&self, img: &Planar, nx: usize, ny: usize) -> anyhow::Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (p, d) = (cfg.patch_size, cfg.d_model);
        let filt = self.tensor("v.patch_embd.weight")?;
        let f = filt
            .as_f32()
            .context("v.patch_embd.weight is not F32")?;
        anyhow::ensure!(
            f.len() == p * p * 3 * d,
            "patch filter is {} elements, expected {}",
            f.len(),
            p * p * 3 * d
        );

        let plane = img.w * img.h;
        let mut out = vec![0f32; nx * ny * d];
        out.par_chunks_mut(d).enumerate().for_each(|(pi, row)| {
            let (px, py) = (pi % nx, pi / nx);
            for (o, dst) in row.iter_mut().enumerate() {
                let fo = o * p * p * 3;
                let mut acc = 0f32;
                for c in 0..3 {
                    for ky in 0..p {
                        let src = c * plane + (py * p + ky) * img.w + px * p;
                        let fk = fo + c * p * p + ky * p;
                        for kx in 0..p {
                            acc += img.data[src + kx] * f[fk + kx];
                        }
                    }
                }
                *dst = acc;
            }
        });
        Ok(out)
    }

    /// Add the learned positional vectors. One table is indexed by column and
    /// one by row; they live in a single tensor, `x` first.
    fn add_position(&self, x: &mut [f32], nx: usize, ny: usize) -> anyhow::Result<()> {
        let d = self.cfg.d_model;
        let len = self.cfg.pos_table_len;
        let pos = self.tensor("v.position_embd.weight")?;
        let p = pos.as_f32().context("v.position_embd.weight is not F32")?;

        x.par_chunks_mut(d).enumerate().for_each(|(pi, row)| {
            let (col, r) = (pi % nx, pi / nx);
            let tx = col * d;
            let ty = (len + r) * d;
            for (k, v) in row.iter_mut().enumerate() {
                *v += p[tx + k] + p[ty + k];
            }
        });
        let _ = ny;
        Ok(())
    }

    fn block(&self, il: usize, x: &mut [f32], n: usize, nx: usize) -> anyhow::Result<()> {
        let cfg = &self.cfg;
        let (d, hd, nh) = (cfg.d_model, cfg.head_dim, cfg.n_heads);
        let eps = cfg.eps;
        let t = |s: &str| format!("v.blk.{il}.{s}");

        // --- pre-attention norm -------------------------------------------
        let ln1 = self.gain(&t("ln1.weight"))?;
        let mut h = x.to_vec();
        h.par_chunks_mut(d).for_each(|row| ops::rms_norm_mul(row, ln1, eps));

        // --- q / k / v -----------------------------------------------------
        let mut q = vec![0f32; n * d];
        let mut k = vec![0f32; n * d];
        let mut v = vec![0f32; n * d];
        self.linear(&t("attn_q.weight"), &mut q, &h, n)?;
        self.linear(&t("attn_k.weight"), &mut k, &h, n)?;
        self.linear(&t("attn_v.weight"), &mut v, &h, n)?;

        let q_norm = self.gain(&t("attn_q_norm.weight"))?;
        let k_norm = self.gain(&t("attn_k_norm.weight"))?;
        let theta = cfg.rope_theta;
        let half = hd / 2;

        // Q and K: per-head norm, then the 2-D rotation. The low half of each
        // head rotates by the patch's column, the high half by its row.
        let rope_heads = |buf: &mut [f32], gain: &[f32]| {
            buf.par_chunks_mut(hd).enumerate().for_each(|(i, head)| {
                ops::rms_norm_mul(head, gain, eps);
                let tok = i / nh;
                let (col, row) = (tok % nx, tok / nx);
                ops::rope_neox(&mut head[..half], col, theta, &[]);
                ops::rope_neox(&mut head[half..], row, theta, &[]);
            });
        };
        rope_heads(&mut q, q_norm);
        rope_heads(&mut k, k_norm);

        // V is normalized per head with no learned gain and never rotated.
        v.par_chunks_mut(hd).for_each(|head| ops::rms_norm(head, eps));

        // --- attention (bidirectional, scale 1.0) --------------------------
        let mut attn = vec![0f32; n * d];
        attn.par_chunks_mut(d).enumerate().for_each(|(i, out)| {
            let mut scores = vec![0f32; n];
            for hq in 0..nh {
                let qh = &q[i * d + hq * hd..][..hd];
                for (j, s) in scores.iter_mut().enumerate() {
                    let kh = &k[j * d + hq * hd..][..hd];
                    *s = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                }
                ops::softmax(&mut scores);

                let o = &mut out[hq * hd..][..hd];
                o.fill(0.0);
                for (j, &w) in scores.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let vh = &v[j * d + hq * hd..][..hd];
                    for (acc, &vv) in o.iter_mut().zip(vh) {
                        *acc += w * vv;
                    }
                }
            }
        });

        // --- output projection, post-norm, residual ------------------------
        let mut proj = vec![0f32; n * d];
        self.linear(&t("attn_out.weight"), &mut proj, &attn, n)?;
        let post = self.gain(&t("attn_post_norm.weight"))?;
        proj.par_chunks_mut(d).for_each(|row| ops::rms_norm_mul(row, post, eps));
        for (xv, &pv) in x.iter_mut().zip(proj.iter()) {
            *xv += pv;
        }

        // --- feed-forward ---------------------------------------------------
        let ln2 = self.gain(&t("ln2.weight"))?;
        let mut y = x.to_vec();
        y.par_chunks_mut(d).for_each(|row| ops::rms_norm_mul(row, ln2, eps));

        let mut gate = vec![0f32; n * cfg.ffn_dim];
        let mut up = vec![0f32; n * cfg.ffn_dim];
        self.linear(&t("ffn_gate.weight"), &mut gate, &y, n)?;
        self.linear(&t("ffn_up.weight"), &mut up, &y, n)?;
        gate.par_iter_mut().zip(up.par_iter()).for_each(|(g, &u)| {
            *g = ops::gelu(*g) * u;
        });

        let mut down = vec![0f32; n * d];
        self.linear(&t("ffn_down.weight"), &mut down, &gate, n)?;
        let fpost = self.gain(&t("ffn_post_norm.weight"))?;
        down.par_chunks_mut(d).for_each(|row| ops::rms_norm_mul(row, fpost, eps));
        for (xv, &pv) in x.iter_mut().zip(down.iter()) {
            *xv += pv;
        }
        Ok(())
    }

    /// Average-pool the patch grid, scale, normalize, project into the text
    /// model's residual stream.
    fn pool_and_project(&self, x: &[f32], nx: usize, ny: usize) -> anyhow::Result<ImageEmbeds> {
        let cfg = &self.cfg;
        let (d, m) = (cfg.d_model, cfg.n_merge);
        let (ox, oy) = (nx / m, ny / m);
        let n_out = ox * oy;

        let mut pooled = vec![0f32; n_out * d];
        let inv = 1.0 / (m * m) as f32;
        pooled.par_chunks_mut(d).enumerate().for_each(|(pi, row)| {
            let (bx, by) = (pi % ox, pi / ox);
            for ky in 0..m {
                for kx in 0..m {
                    let src = ((by * m + ky) * nx + bx * m + kx) * d;
                    for (acc, &sv) in row.iter_mut().zip(&x[src..src + d]) {
                        *acc += sv;
                    }
                }
            }
            for v in row.iter_mut() {
                *v *= inv;
            }
        });

        // The pooler scales by sqrt(d) before the embedder's norm.
        let s = (d as f32).sqrt();
        pooled.par_iter_mut().for_each(|v| *v *= s);
        // Plain RMS norm: the embedder has no gain tensor.
        pooled.par_chunks_mut(d).for_each(|row| ops::rms_norm(row, cfg.eps));

        let mut rows = vec![0f32; n_out * cfg.proj_dim];
        self.linear("mm.input_projection.weight", &mut rows, &pooled, n_out)?;

        Ok(ImageEmbeds {
            rows,
            n: n_out,
            proj_dim: cfg.proj_dim,
            grid: (ox, oy),
        })
    }
}
