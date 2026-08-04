//! k-quant block decoders.
//!
//! Layouts mirror ggml's `block_q4_K` / `block_q6_K` exactly — these are on-disk
//! formats, so the bit twiddling is not ours to redesign. The GPU path decodes
//! the same bits in WGSL; this module is the reference the shaders are checked
//! against.

use half::f16;

use crate::GgmlType;

/// Elements per k-quant super-block.
pub const QK_K: usize = 256;

/// `d` (f16) + `dmin` (f16) + 12 packed 6-bit scale/min pairs + 128 nibble bytes.
pub const Q4K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;
/// 128 low-nibble bytes + 64 high-bit bytes + 16 i8 scales + `d` (f16).
pub const Q6K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

const _: () = assert!(Q4K_BLOCK_BYTES == 144);
const _: () = assert!(Q6K_BLOCK_BYTES == 210);

#[inline(always)]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Unpack sub-block `j`'s 6-bit scale and min from Q4_K's 12 packed bytes.
///
/// Sub-blocks 0..4 store both fields plainly in the low 6 bits of bytes 0..8;
/// sub-blocks 4..8 borrow their top 2 bits from the high bits of bytes 0..8.
#[inline(always)]
fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Decode one 256-element Q4_K super-block.
///
/// Values are `d * scale * nibble - dmin * min`, with an independent 6-bit
/// `(scale, min)` pair per 32-element sub-block.
#[inline]
pub fn dequant_q4k_block(blk: &[u8], out: &mut [f32]) {
    debug_assert_eq!(blk.len(), Q4K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);

    let d = f16_at(blk, 0);
    let dmin = f16_at(blk, 2);
    let scales = &blk[4..16];
    let qs = &blk[16..144];

    // Each iteration covers 64 output elements: 32 from low nibbles, 32 from high.
    for pair in 0..4 {
        let (sc1, m1) = q4k_scale_min(pair * 2, scales);
        let (sc2, m2) = q4k_scale_min(pair * 2 + 1, scales);
        let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);

        let q = &qs[pair * 32..pair * 32 + 32];
        let lo = &mut out[pair * 64..];
        for l in 0..32 {
            lo[l] = d1 * (q[l] & 0xF) as f32 - min1;
            lo[l + 32] = d2 * (q[l] >> 4) as f32 - min2;
        }
    }
}

/// Decode one 256-element Q6_K super-block.
///
/// Each value is a signed 6-bit quant (4 low bits from `ql`, 2 high from `qh`,
/// biased by -32) times an i8 sub-block scale times the f16 super-block scale.
#[inline]
pub fn dequant_q6k_block(blk: &[u8], out: &mut [f32]) {
    debug_assert_eq!(blk.len(), Q6K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);

    let d = f16_at(blk, 208);
    let ql = &blk[0..128];
    let qh = &blk[128..192];
    let sc: &[i8] = bytemuck::cast_slice(&blk[192..208]);

    // Two halves of 128 elements; within each, four interleaved 32-element groups.
    for n in 0..2 {
        let ql = &ql[n * 64..];
        let qh = &qh[n * 32..];
        let sc = &sc[n * 8..];
        let y = &mut out[n * 128..];

        for l in 0..32 {
            let is = l / 16;
            let h = qh[l];
            let q1 = ((ql[l] & 0xF) | ((h & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;

            y[l] = d * sc[is] as f32 * q1 as f32;
            y[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
            y[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
            y[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
        }
    }
}

/// Dequantize a contiguous run of `out.len()` elements of the given type.
pub fn dequant(ty: GgmlType, data: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            let src: &[f32] = bytemuck::cast_slice(&data[..out.len() * 4]);
            out.copy_from_slice(src);
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        GgmlType::BF16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        GgmlType::Q4K => {
            debug_assert_eq!(out.len() % QK_K, 0);
            for (blk, o) in data
                .chunks_exact(Q4K_BLOCK_BYTES)
                .zip(out.chunks_exact_mut(QK_K))
            {
                dequant_q4k_block(blk, o);
            }
        }
        GgmlType::Q6K => {
            debug_assert_eq!(out.len() % QK_K, 0);
            for (blk, o) in data
                .chunks_exact(Q6K_BLOCK_BYTES)
                .zip(out.chunks_exact_mut(QK_K))
            {
                dequant_q6k_block(blk, o);
            }
        }
    }
}

/// Dot product of a quantized weight row with an f32 activation vector.
///
/// Fuses dequantization into the accumulation so the CPU reference path never
/// materializes a dequantized row. Mirrors what the WGSL matvec kernels do.
pub fn dot_row(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    match ty {
        GgmlType::Q4K => {
            let mut acc = 0f32;
            for (bi, blk) in row.chunks_exact(Q4K_BLOCK_BYTES).enumerate() {
                let d = f16_at(blk, 0);
                let dmin = f16_at(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let xb = &x[bi * QK_K..];

                for pair in 0..4 {
                    let (sc1, m1) = q4k_scale_min(pair * 2, scales);
                    let (sc2, m2) = q4k_scale_min(pair * 2 + 1, scales);
                    let q = &qs[pair * 32..pair * 32 + 32];
                    let xl = &xb[pair * 64..];

                    // Accumulate quant*x and sum(x) separately per sub-block so the
                    // scale and min each multiply in once instead of 32 times.
                    let (mut s1, mut x1, mut s2, mut x2) = (0f32, 0f32, 0f32, 0f32);
                    for l in 0..32 {
                        let a = xl[l];
                        let b = xl[l + 32];
                        s1 += (q[l] & 0xF) as f32 * a;
                        x1 += a;
                        s2 += (q[l] >> 4) as f32 * b;
                        x2 += b;
                    }
                    acc += d * sc1 as f32 * s1 - dmin * m1 as f32 * x1;
                    acc += d * sc2 as f32 * s2 - dmin * m2 as f32 * x2;
                }
            }
            acc
        }
        GgmlType::Q6K => {
            let mut acc = 0f32;
            for (bi, blk) in row.chunks_exact(Q6K_BLOCK_BYTES).enumerate() {
                let d = f16_at(blk, 208);
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc: &[i8] = bytemuck::cast_slice(&blk[192..208]);
                let xb = &x[bi * QK_K..];

                let mut sub = 0f32;
                for n in 0..2 {
                    let ql = &ql[n * 64..];
                    let qh = &qh[n * 32..];
                    let sc = &sc[n * 8..];
                    let y = &xb[n * 128..];
                    for l in 0..32 {
                        let is = l / 16;
                        let h = qh[l];
                        let q1 = ((ql[l] & 0xF) | ((h & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                        sub += sc[is] as f32 * q1 as f32 * y[l]
                            + sc[is + 2] as f32 * q2 as f32 * y[l + 32]
                            + sc[is + 4] as f32 * q3 as f32 * y[l + 64]
                            + sc[is + 6] as f32 * q4 as f32 * y[l + 96];
                    }
                }
                acc += d * sub;
            }
            acc
        }
        GgmlType::F32 => {
            let w: &[f32] = bytemuck::cast_slice(row);
            w.iter().zip(x).map(|(a, b)| a * b).sum()
        }
        GgmlType::F16 => row
            .chunks_exact(2)
            .zip(x)
            .map(|(c, b)| f16::from_le_bytes([c[0], c[1]]).to_f32() * b)
            .sum(),
        GgmlType::BF16 => row
            .chunks_exact(2)
            .zip(x)
            .map(|(c, b)| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) * b)
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dot_row` must agree with dequantize-then-dot, since the GPU kernels are
    /// validated against `dot_row` but the shapes are validated against `dequant`.
    fn check_dot_matches_dequant(ty: GgmlType, block_bytes: usize) {
        let mut raw = vec![0u8; block_bytes * 3];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = ((i * 37 + 11) % 251) as u8;
        }
        // Keep the f16 scales in a sane range so the comparison is meaningful.
        for b in 0..3 {
            let off = b * block_bytes;
            let (d_at, dmin_at) = match ty {
                GgmlType::Q4K => (off, Some(off + 2)),
                _ => (off + 208, None),
            };
            raw[d_at..d_at + 2].copy_from_slice(&f16::from_f32(0.0123).to_le_bytes());
            if let Some(m) = dmin_at {
                raw[m..m + 2].copy_from_slice(&f16::from_f32(0.0071).to_le_bytes());
            }
        }

        let n = QK_K * 3;
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

        let mut deq = vec![0f32; n];
        dequant(ty, &raw, &mut deq);
        let expect: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = dot_row(ty, &raw, &x);

        let tol = expect.abs() * 1e-4 + 1e-3;
        assert!(
            (got - expect).abs() < tol,
            "{}: dot_row={got} dequant-dot={expect}",
            ty.name()
        );
    }

    #[test]
    fn q4k_dot_matches_dequant() {
        check_dot_matches_dequant(GgmlType::Q4K, Q4K_BLOCK_BYTES);
    }

    #[test]
    fn q6k_dot_matches_dequant() {
        check_dot_matches_dequant(GgmlType::Q6K, Q6K_BLOCK_BYTES);
    }

    #[test]
    fn q4k_scale_min_unpacks_six_bits() {
        // All-ones input: every 6-bit field should read back as 63.
        let q = [0xFFu8; 12];
        for j in 0..8 {
            let (sc, m) = q4k_scale_min(j, &q);
            assert_eq!((sc, m), (63, 63), "sub-block {j}");
        }
    }

    #[test]
    fn q4k_zero_quants_give_negative_min() {
        // d=1, dmin=1, scales byte 0 => sc=0,m=0 for j<4; nibbles zero.
        let mut blk = vec![0u8; Q4K_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        blk[2..4].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        blk[4] = 5; // sub-block 0 scale = 5
        blk[8] = 7; // sub-block 0 min = 7
        let mut out = vec![0f32; QK_K];
        dequant_q4k_block(&blk, &mut out);
        // quant nibble is 0, so value = 0*5 - 1*7
        assert_eq!(out[0], -7.0);
    }

    #[test]
    fn q6k_zero_quants_center_at_minus_32() {
        let mut blk = vec![0u8; Q6K_BLOCK_BYTES];
        blk[208..210].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        blk[192] = 2; // scales[0] = 2
        let mut out = vec![0f32; QK_K];
        dequant_q6k_block(&blk, &mut out);
        // (0 | 0) - 32 = -32, times scale 2, times d 1
        assert_eq!(out[0], -64.0);
    }
}
