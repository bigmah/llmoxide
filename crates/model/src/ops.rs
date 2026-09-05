//! Scalar reference kernels.
//!
//! These define the semantics the WGSL kernels must reproduce. Correctness
//! first: the only concession to speed is rayon over independent output rows,
//! which does not change results.

use gguf::{quant, TensorView};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(target_arch = "wasm32")]
use serial::ParIterMut as _;

/// `par_iter_mut` with no threads behind it.
///
/// rayon compiles for `wasm32-unknown-unknown` but cannot run there: building
/// its pool calls `std::thread::spawn`, which panics. Real wasm threads would
/// need `SharedArrayBuffer`, which needs COOP/COEP response headers — and the
/// browser build is a single HTML file that may be opened straight off disk,
/// where there are no headers to set. So the reference kernels run one row at
/// a time. Results are identical; only the wall clock differs, and nothing in
/// the browser build calls these on the hot path (the GPU does that work).
#[cfg(target_arch = "wasm32")]
mod serial {
    pub trait ParIterMut {
        type Item;
        fn par_iter_mut(&mut self) -> std::slice::IterMut<'_, Self::Item>;
    }

    impl<T> ParIterMut for [T] {
        type Item = T;
        fn par_iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
            self.iter_mut()
        }
    }
}

/// Root-mean-square norm over `x`, in place, without a learned weight.
///
/// ggml computes the mean in f32 and does not subtract the mean, matching this.
pub fn rms_norm(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// RMS norm followed by an elementwise weight.
///
/// Note the weights in this checkpoint already include Gemma's `+1` offset —
/// the converter baked it in, so this is a plain multiply.
pub fn rms_norm_mul(x: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), w.len());
    rms_norm(x, eps);
    for (v, &g) in x.iter_mut().zip(w) {
        *v *= g;
    }
}

/// `y = W x`, where `W` is a GGUF weight with `ne = [in_dim, out_dim]`.
///
/// Each output element is an independent dot product over a quantized row, so
/// dequantization is fused into the accumulation rather than materialized.
pub fn matvec(y: &mut [f32], w: &TensorView<'_>, x: &[f32]) {
    let in_dim = w.in_dim();
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), w.out_dim());

    let row_bytes = w.ty().bytes_for(in_dim);
    let ty = w.ty();
    let data = w.data;

    y.par_iter_mut().enumerate().for_each(|(o, out)| {
        *out = quant::dot_row(ty, &data[o * row_bytes..(o + 1) * row_bytes], x);
    });
}

/// Batched `Y = W X` over `n` column vectors laid out contiguously.
///
/// Parallelising over output rows (not tokens) keeps every thread streaming a
/// distinct weight row, which is the bandwidth-bound part of prefill.
pub fn matmat(y: &mut [f32], w: &TensorView<'_>, x: &[f32], n: usize) {
    let (in_dim, out_dim) = (w.in_dim(), w.out_dim());
    debug_assert_eq!(x.len(), in_dim * n);
    debug_assert_eq!(y.len(), out_dim * n);

    let row_bytes = w.ty().bytes_for(in_dim);
    let ty = w.ty();
    let data = w.data;

    // Transposed accumulation would need a scratch buffer; instead each task
    // owns one output row across all tokens and writes with a stride.
    let mut cols: Vec<&mut [f32]> = y.chunks_exact_mut(out_dim).collect();
    cols.par_iter_mut().enumerate().for_each(|(t, col)| {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for (o, out) in col.iter_mut().enumerate() {
            *out = quant::dot_row(ty, &data[o * row_bytes..(o + 1) * row_bytes], xt);
        }
    });
}

/// ggml's `GELU`, the tanh approximation used by `GEGLU`.
#[inline]
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEFF: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEFF * x * x)).tanh())
}

/// `out = gelu(gate) * up`, the gated FFN activation.
pub fn geglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = gelu(g) * u;
    }
}

/// In-place softmax, numerically stabilized by subtracting the max.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Every position masked out; leave zeros rather than emitting NaN.
        x.fill(0.0);
        return;
    }
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Apply rotary embeddings to one head, ggml NeoX-style.
///
/// Pairs are `(i, i + head_dim/2)` — *not* adjacent elements. `factors` divides
/// each pair's angle; the `1e30` entries in this checkpoint therefore leave
/// their dimensions untouched, which is how partial RoPE is expressed.
pub fn rope_neox(head: &mut [f32], pos: usize, base: f32, factors: &[f32]) {
    let d = head.len();
    let half = d / 2;
    let theta_scale = (base as f64).powf(-2.0 / d as f64);

    let mut theta = pos as f64;
    for i in 0..half {
        let factor = factors.get(i).copied().unwrap_or(1.0) as f64;
        let angle = theta / factor;
        let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);

        let (x0, x1) = (head[i], head[i + half]);
        head[i] = x0 * cos - x1 * sin;
        head[i + half] = x0 * sin + x1 * cos;

        theta *= theta_scale;
    }
}

/// Batched `Y = W X` that stays parallel for a single column.
///
/// [`matmat`] parallelizes over tokens, which is right for prefill and
/// catastrophic for decode: a one-token batch runs on one core. Decode is the
/// qwen35 CPU path's steady state, so it matters there.
pub fn matmul(y: &mut [f32], w: &TensorView<'_>, x: &[f32], n: usize) {
    if n == 1 {
        matvec(y, w, x);
    } else {
        matmat(y, w, x, n);
    }
}

/// `x * sigmoid(x)`, the SwiGLU activation (ggml `SILU`).
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`, with ggml's large-`x` shortcut to avoid overflow.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Scale `x` to unit L2 norm, in place. Matches ggml's `l2_norm`: `eps` floors
/// the *norm* (`1/max(‖x‖, eps)`), it is not added under the square root.
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let scale = 1.0 / norm.max(eps);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// `cap * tanh(x / cap)`, applied elementwise to the final logits.
pub fn soft_cap(x: &mut [f32], cap: f32) {
    let inv = 1.0 / cap;
    for v in x.iter_mut() {
        *v = cap * (*v * inv).tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_gives_unit_rms() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        rms_norm(&mut x, 0.0);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6, "rms was {rms}");
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut h = vec![0.5, -1.5, 2.0, 0.25];
        let before = h.clone();
        rope_neox(&mut h, 0, 10_000.0, &[1.0, 1.0]);
        for (a, b) in h.iter().zip(&before) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_sentinel_factor_disables_rotation() {
        let mut h = vec![0.5, -1.5, 2.0, 0.25];
        let before = h.clone();
        // Huge factors drive every angle to zero regardless of position.
        rope_neox(&mut h, 137, 10_000.0, &[1e30, 1e30]);
        for (a, b) in h.iter().zip(&before) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn rope_pairs_across_the_half_boundary() {
        // A pure rotation must preserve the norm of each (i, i+half) pair.
        let mut h: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
        let before = h.clone();
        rope_neox(&mut h, 3, 10_000.0, &[1.0; 4]);
        for i in 0..4 {
            let n0 = before[i].hypot(before[i + 4]);
            let n1 = h[i].hypot(h[i + 4]);
            assert!((n0 - n1).abs() < 1e-5, "pair {i}: {n0} vs {n1}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn softmax_all_masked_is_zero_not_nan() {
        let mut x = vec![f32::NEG_INFINITY; 4];
        softmax(&mut x);
        assert!(x.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn soft_cap_saturates() {
        let mut x = vec![0.0, 1e9, -1e9];
        soft_cap(&mut x, 30.0);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 30.0).abs() < 1e-3);
        assert!((x[2] + 30.0).abs() < 1e-3);
    }
}
