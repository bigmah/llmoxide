//! Weight upload.
//!
//! All 667 tensors are packed into a small number of large storage buffers.
//! Two constraints shape the layout:
//!
//! * A single binding tops out at ~4 GB on this adapter while the model is
//!   7.4 GB, so tensors are packed into several buffers and no tensor may
//!   straddle a boundary.
//! * Shaders address weights as `array<u32>`, so every tensor starts on a
//!   4-byte boundary — and Q6_K blocks, which are 210 bytes on disk, are
//!   repacked to a 224-byte stride so each *block* is u32-aligned too.

use gguf::{GgmlType, Gguf, TensorView};

/// Bytes per Q6_K block once repacked for the GPU (210 on disk, padded).
pub const Q6K_GPU_BLOCK_BYTES: usize = 224;
/// Q4_K blocks are already 36 u32 and need no repacking.
pub const Q4K_GPU_BLOCK_BYTES: usize = 144;
/// Q8_0 blocks are 34 bytes on disk (f16 scale + 32 i8); repacked to 36 so the
/// scale word and every quant word are u32-aligned.
pub const Q8_0_GPU_BLOCK_BYTES: usize = 36;

/// Largest buffer the packer will build when the caller does not say.
///
/// Native adapters here report a ~4 GB binding limit, so this is a margin
/// under it rather than the limit itself; the packer is greedy and only checks
/// before appending. A browser is a different story — WebGPU's *default*
/// `maxStorageBufferBindingSize` is 128 MB and even the adapter maximum is
/// commonly 2 GB — so [`Plan::build_with`] takes the real number and
/// [`Gpu::arena_buffer_bytes`](crate::Gpu::arena_buffer_bytes) derives it from
/// the device.
pub const TARGET_BUFFER_BYTES: u64 = 3 << 30;
/// Tensor start alignment. 256 keeps every block type aligned and matches the
/// usual `min_storage_buffer_offset_alignment`.
const TENSOR_ALIGN: u64 = 256;

/// Where a tensor ended up.
#[derive(Debug, Clone, Copy)]
pub struct Handle {
    pub buffer: usize,
    /// Offset in u32 units, which is what the shaders index by.
    pub base_u32: u32,
    pub in_dim: u32,
    pub out_dim: u32,
    pub ty: GgmlType,
}

/// GPU-side size of `n` elements of `ty`.
pub fn gpu_bytes(ty: GgmlType, n: usize) -> usize {
    match ty {
        GgmlType::Q4K => n / gguf::quant::QK_K * Q4K_GPU_BLOCK_BYTES,
        GgmlType::Q6K => n / gguf::quant::QK_K * Q6K_GPU_BLOCK_BYTES,
        GgmlType::F32 => n * 4,
        GgmlType::F16 | GgmlType::BF16 => n * 2,
        GgmlType::Q8_0 => n / gguf::quant::QK8_0 * Q8_0_GPU_BLOCK_BYTES,
    }
}

/// On-disk and GPU-side bytes per block, for a type's repack.
///
/// Both are 1 for the dense types, whose "blocks" are single elements — which
/// makes any byte count a whole number of them, so a dense tensor can be split
/// anywhere.
pub fn block_strides(ty: GgmlType) -> (usize, usize) {
    match ty {
        GgmlType::Q6K => (gguf::quant::Q6K_BLOCK_BYTES, Q6K_GPU_BLOCK_BYTES),
        GgmlType::Q8_0 => (gguf::quant::Q8_0_BLOCK_BYTES, Q8_0_GPU_BLOCK_BYTES),
        GgmlType::Q4K => (gguf::quant::Q4K_BLOCK_BYTES, Q4K_GPU_BLOCK_BYTES),
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => (1, 1),
    }
}

/// Copy a whole number of blocks into `dst`, applying the type's repack.
///
/// Split out from [`encode_tensor`] so a tensor too large to hold twice — the
/// E4B token embedding is ~590 MB once repacked — can be moved a piece at a
/// time. Both slices must cover the *same* block count, which is what
/// [`block_strides`] is for; a dense type has a stride of one byte, so any
/// split is legal there.
pub fn encode_blocks(ty: GgmlType, src: &[u8], dst: &mut [u8]) {
    match ty {
        GgmlType::Q6K => {
            // 210 -> 224 bytes; the trailing 14 bytes are padding the shader
            // never reads, but zeroing keeps the upload deterministic.
            for (src, out) in src
                .chunks_exact(gguf::quant::Q6K_BLOCK_BYTES)
                .zip(dst.chunks_exact_mut(Q6K_GPU_BLOCK_BYTES))
            {
                out[..gguf::quant::Q6K_BLOCK_BYTES].copy_from_slice(src);
                out[gguf::quant::Q6K_BLOCK_BYTES..].fill(0);
            }
        }
        GgmlType::Q8_0 => {
            // 34 -> 36 bytes: the f16 scale keeps its word to itself so the 32
            // quants that follow start on a u32 boundary.
            for (src, out) in src
                .chunks_exact(gguf::quant::Q8_0_BLOCK_BYTES)
                .zip(dst.chunks_exact_mut(Q8_0_GPU_BLOCK_BYTES))
            {
                out[..2].copy_from_slice(&src[..2]);
                out[2..4].fill(0);
                out[4..].copy_from_slice(&src[2..]);
            }
        }
        _ => dst[..src.len()].copy_from_slice(src),
    }
}

/// Copy one tensor into `dst`, applying the Q6_K repack.
pub fn encode_tensor(t: &TensorView<'_>, dst: &mut [u8]) {
    encode_blocks(t.ty(), t.data, dst)
}

/// Plans the packing before any GPU memory is touched, so allocation failures
/// surface as a clear error rather than a driver abort mid-upload.
pub struct Plan {
    pub handles: Vec<(String, Handle)>,
    pub buffer_sizes: Vec<u64>,
}

impl Plan {
    pub fn build(g: &Gguf, names: impl IntoIterator<Item = String>) -> anyhow::Result<Self> {
        Self::build_with(g, names, TARGET_BUFFER_BYTES)
    }

    /// Plan against a specific maximum buffer size.
    ///
    /// Only the tensor *table* is read, never the payload, so this also works
    /// on a [`gguf::Header`] parsed from a prefix — which is how the browser
    /// build knows what to allocate before it has fetched a single weight.
    pub fn build_with(
        header: &gguf::Header,
        names: impl IntoIterator<Item = String>,
        target: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(target >= TENSOR_ALIGN, "buffer target {target} is too small");
        let mut handles = Vec::new();
        let mut buffer_sizes: Vec<u64> = vec![0];

        for name in names {
            let t = header.info(&name)?;
            let size = gpu_bytes(t.ty, t.elem_count()) as u64;
            anyhow::ensure!(
                size <= target,
                "tensor {name} is {size} bytes, larger than the {target}-byte \
                 buffer limit this device allows"
            );

            let cur = buffer_sizes.last_mut().expect("at least one buffer");
            let start = cur.next_multiple_of(TENSOR_ALIGN);
            let (buffer, base) = if start + size > target {
                buffer_sizes.push(size);
                (buffer_sizes.len() - 1, 0)
            } else {
                *cur = start + size;
                (buffer_sizes.len() - 1, start)
            };

            handles.push((
                name,
                Handle {
                    buffer,
                    base_u32: (base / 4) as u32,
                    in_dim: t.in_dim() as u32,
                    out_dim: t.out_dim() as u32,
                    ty: t.ty,
                },
            ));
        }

        Ok(Self {
            handles,
            buffer_sizes,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        self.buffer_sizes.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6k_repack_preserves_payload_and_pads() {
        let src = vec![7u8; gguf::quant::Q6K_BLOCK_BYTES * 2];
        let info = gguf::TensorInfo {
            name: "t".into(),
            dims: vec![512],
            ty: GgmlType::Q6K,
            offset: 0,
        };
        let view = TensorView {
            info: &info,
            data: &src,
        };
        let mut dst = vec![0xAAu8; gpu_bytes(GgmlType::Q6K, 512)];
        encode_tensor(&view, &mut dst);

        assert_eq!(dst.len(), 448);
        for b in 0..2 {
            let blk = &dst[b * Q6K_GPU_BLOCK_BYTES..(b + 1) * Q6K_GPU_BLOCK_BYTES];
            assert!(blk[..210].iter().all(|&x| x == 7), "payload block {b}");
            assert!(blk[210..].iter().all(|&x| x == 0), "padding block {b}");
        }
    }

    #[test]
    fn gpu_block_strides_are_u32_aligned() {
        assert_eq!(Q4K_GPU_BLOCK_BYTES % 4, 0);
        assert_eq!(Q6K_GPU_BLOCK_BYTES % 4, 0);
    }
}
