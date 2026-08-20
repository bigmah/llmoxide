//! Minimal, zero-copy GGUF v2/v3 reader.
//!
//! The file is mmap'd and never copied; [`TensorView`] hands out borrowed byte
//! slices that point straight into the mapping. Dequantization lives in
//! [`quant`], and the shapes follow ggml's `ne` convention: `ne[0]` is the
//! fastest-varying axis, which for a weight matrix is the *input* dimension.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

pub mod quant;

pub const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic {0:#x})")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0}")]
    BadVersion(u32),
    #[error("unexpected end of file at offset {0}")]
    Eof(usize),
    #[error("unknown metadata value type {0}")]
    BadValueType(u32),
    #[error("unknown ggml tensor type {0}")]
    BadTensorType(u32),
    #[error("invalid utf-8 in {0}")]
    Utf8(&'static str),
    #[error("tensor {0:?} not found")]
    NoTensor(String),
    #[error("metadata key {0:?} not found")]
    NoKey(String),
    #[error("metadata key {key:?} is {actual}, wanted {wanted}")]
    WrongType {
        key: String,
        wanted: &'static str,
        actual: &'static str,
    },
    #[error("tensor {name:?} runs past end of file ({end} > {len})")]
    TensorOutOfBounds { name: String, end: u64, len: u64 },
}

pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// ggml tensor types
// ---------------------------------------------------------------------------

/// The subset of ggml types we can actually read. Anything else is rejected at
/// load time rather than producing garbage later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q4K = 12,
    Q6K = 14,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            30 => Self::BF16,
            other => return Err(Error::BadTensorType(other)),
        })
    }

    /// Elements per quantization block (1 for dense types).
    pub const fn block_elems(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 => quant::QK8_0,
            Self::Q4K | Self::Q6K => quant::QK_K,
        }
    }

    /// Bytes per quantization block.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => quant::Q8_0_BLOCK_BYTES,
            Self::Q4K => quant::Q4K_BLOCK_BYTES,
            Self::Q6K => quant::Q6K_BLOCK_BYTES,
        }
    }

    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4K | Self::Q6K)
    }

    /// Storage size of `n` elements. `n` must be a multiple of
    /// [`block_elems`](Self::block_elems) for quantized types.
    pub const fn bytes_for(self, n: usize) -> usize {
        n / self.block_elems() * self.block_bytes()
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
            Self::BF16 => "BF16",
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Array),
}

/// Arrays are kept in typed form; the tokenizer vocab alone is 262 144 strings,
/// so boxing every element as a `Value` would be gratuitous.
#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Empty,
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Empty => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "[u8]",
            Self::I8(_) => "[i8]",
            Self::U16(_) => "[u16]",
            Self::I16(_) => "[i16]",
            Self::U32(_) => "[u32]",
            Self::I32(_) => "[i32]",
            Self::U64(_) => "[u64]",
            Self::I64(_) => "[i64]",
            Self::F32(_) => "[f32]",
            Self::F64(_) => "[f64]",
            Self::Bool(_) => "[bool]",
            Self::String(_) => "[string]",
            Self::Empty => "[]",
        }
    }

    /// Widen any integer array to `u32`. GGUF writers are inconsistent about
    /// integer widths, so callers that want counts should go through this.
    pub fn as_u32_vec(&self) -> Option<Vec<u32>> {
        Some(match self {
            Self::U8(v) => v.iter().map(|&x| x as u32).collect(),
            Self::I8(v) => v.iter().map(|&x| x as u32).collect(),
            Self::U16(v) => v.iter().map(|&x| x as u32).collect(),
            Self::I16(v) => v.iter().map(|&x| x as u32).collect(),
            Self::U32(v) => v.clone(),
            Self::I32(v) => v.iter().map(|&x| x as u32).collect(),
            Self::U64(v) => v.iter().map(|&x| x as u32).collect(),
            Self::I64(v) => v.iter().map(|&x| x as u32).collect(),
            Self::Empty => Vec::new(),
            _ => return None,
        })
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(a) => a.type_name(),
        }
    }

    /// Any integer type widened to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Self::U8(v) => v as u64,
            Self::I8(v) => v as u64,
            Self::U16(v) => v as u64,
            Self::I16(v) => v as u64,
            Self::U32(v) => v as u64,
            Self::I32(v) => v as u64,
            Self::U64(v) => v,
            Self::I64(v) => v as u64,
            Self::Bool(v) => v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        Some(match *self {
            Self::F32(v) => v,
            Self::F64(v) => v as f32,
            _ => return None,
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            Self::U8(v) => Some(v != 0),
            Self::U32(v) => Some(v != 0),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor descriptor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml `ne` order: `dims[0]` varies fastest.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    pub fn byte_len(&self) -> usize {
        self.ty.bytes_for(self.elem_count())
    }

    /// Input dimension for a weight matrix (`ne[0]`).
    pub fn in_dim(&self) -> usize {
        self.dims.first().copied().unwrap_or(1) as usize
    }

    /// Output dimension for a weight matrix (`ne[1]`, or 1 for a vector).
    pub fn out_dim(&self) -> usize {
        self.dims.get(1).copied().unwrap_or(1) as usize
    }
}

/// A tensor's raw bytes plus enough metadata to interpret them.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn ty(&self) -> GgmlType {
        self.info.ty
    }

    pub fn dims(&self) -> &[u64] {
        &self.info.dims
    }

    pub fn elem_count(&self) -> usize {
        self.info.elem_count()
    }

    pub fn in_dim(&self) -> usize {
        self.info.in_dim()
    }

    pub fn out_dim(&self) -> usize {
        self.info.out_dim()
    }

    /// Borrow as `f32`, only valid for `F32` tensors.
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        if self.info.ty != GgmlType::F32 {
            return None;
        }
        // GGUF tensor offsets are aligned to `general.alignment` (>= 4), and
        // the mapping itself is page-aligned, so this cast is sound in practice;
        // `try_cast_slice` keeps it sound in principle too.
        bytemuck::try_cast_slice(self.data).ok()
    }

    /// Dequantize the whole tensor to `f32`.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.elem_count()];
        self.dequant_into(&mut out);
        out
    }

    /// Dequantize the whole tensor into a caller-provided buffer.
    pub fn dequant_into(&self, out: &mut [f32]) {
        quant::dequant(self.info.ty, self.data, out);
    }

    /// Dequantize a single row (`ne[0]` elements starting at row `row`).
    pub fn dequant_row_into(&self, row: usize, out: &mut [f32]) {
        let n = self.in_dim();
        debug_assert_eq!(out.len(), n);
        let bytes = self.info.ty.bytes_for(n);
        let start = row * bytes;
        quant::dequant(self.info.ty, &self.data[start..start + bytes], out);
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

pub struct Gguf {
    _file: File,
    mmap: Mmap,
    /// Absolute file offset of the tensor data section.
    data_offset: u64,
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: we require the model file not be mutated while mapped, which
        // is the same contract every mmap-based loader operates under.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(file, mmap)
    }

    fn from_mmap(file: File, mmap: Mmap) -> Result<Self> {
        let mut c = Cursor::new(&mmap);

        let magic = c.u32()?;
        if magic != MAGIC {
            return Err(Error::BadMagic(magic));
        }
        let version = c.u32()?;
        if version != 2 && version != 3 {
            return Err(Error::BadVersion(version));
        }

        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;

        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let val = c.value(ty)?;
            metadata.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut index = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ty = GgmlType::from_u32(c.u32()?)?;
            let offset = c.u64()?;
            index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);
        let data_offset = align_up(c.pos as u64, alignment);

        let file_len = mmap.len() as u64;
        for t in &tensors {
            let end = data_offset + t.offset + t.byte_len() as u64;
            if end > file_len {
                return Err(Error::TensorOutOfBounds {
                    name: t.name.clone(),
                    end,
                    len: file_len,
                });
            }
        }

        Ok(Self {
            _file: file,
            mmap,
            data_offset,
            version,
            metadata,
            tensors,
            index,
        })
    }

    /// Hint the OS to fault the whole mapping in. Worth doing once up front so
    /// the first token doesn't pay for 7 GB of page faults.
    pub fn prefault(&self) {
        // `Mmap::advise` is best-effort; failure just means slower first token.
        let _ = self.mmap.advise(memmap2::Advice::WillNeed);
    }

    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn tensor_data_offset(&self) -> u64 {
        self.data_offset
    }

    // -- metadata accessors -------------------------------------------------

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        let v = self.metadata.get(key).ok_or_else(|| Error::NoKey(key.into()))?;
        v.as_u64().ok_or_else(|| Error::WrongType {
            key: key.into(),
            wanted: "integer",
            actual: v.type_name(),
        })
    }

    pub fn usize(&self, key: &str) -> Result<usize> {
        Ok(self.u64(key)? as usize)
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        let v = self.metadata.get(key).ok_or_else(|| Error::NoKey(key.into()))?;
        v.as_f32().ok_or_else(|| Error::WrongType {
            key: key.into(),
            wanted: "f32",
            actual: v.type_name(),
        })
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        let v = self.metadata.get(key).ok_or_else(|| Error::NoKey(key.into()))?;
        v.as_str().ok_or_else(|| Error::WrongType {
            key: key.into(),
            wanted: "string",
            actual: v.type_name(),
        })
    }

    pub fn array(&self, key: &str) -> Result<&Array> {
        let v = self.metadata.get(key).ok_or_else(|| Error::NoKey(key.into()))?;
        v.as_array().ok_or_else(|| Error::WrongType {
            key: key.into(),
            wanted: "array",
            actual: v.type_name(),
        })
    }

    /// Integer array widened to `u32`, e.g. per-layer KV head counts.
    pub fn u32_array(&self, key: &str) -> Result<Vec<u32>> {
        let a = self.array(key)?;
        a.as_u32_vec().ok_or_else(|| Error::WrongType {
            key: key.into(),
            wanted: "integer array",
            actual: a.type_name(),
        })
    }

    pub fn bool_array(&self, key: &str) -> Result<Vec<bool>> {
        let a = self.array(key)?;
        match a {
            Array::Bool(v) => Ok(v.clone()),
            Array::U8(v) => Ok(v.iter().map(|&x| x != 0).collect()),
            Array::Empty => Ok(Vec::new()),
            other => Err(Error::WrongType {
                key: key.into(),
                wanted: "bool array",
                actual: other.type_name(),
            }),
        }
    }

    pub fn f32_array(&self, key: &str) -> Result<Vec<f32>> {
        let a = self.array(key)?;
        match a {
            Array::F32(v) => Ok(v.clone()),
            Array::F64(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Array::Empty => Ok(Vec::new()),
            other => Err(Error::WrongType {
                key: key.into(),
                wanted: "f32 array",
                actual: other.type_name(),
            }),
        }
    }

    pub fn string_array(&self, key: &str) -> Result<&[String]> {
        let a = self.array(key)?;
        match a {
            Array::String(v) => Ok(v),
            other => Err(Error::WrongType {
                key: key.into(),
                wanted: "string array",
                actual: other.type_name(),
            }),
        }
    }

    // -- tensor accessors ---------------------------------------------------

    pub fn has_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| Error::NoTensor(name.into()))?;
        Ok(self.tensor_at(i))
    }

    pub fn tensor_opt(&self, name: &str) -> Option<TensorView<'_>> {
        self.index.get(name).map(|&i| self.tensor_at(i))
    }

    pub fn tensor_at(&self, i: usize) -> TensorView<'_> {
        let info = &self.tensors[i];
        let start = (self.data_offset + info.offset) as usize;
        TensorView {
            info,
            data: &self.mmap[start..start + info.byte_len()],
        }
    }

    /// Total bytes occupied by tensor data.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.byte_len() as u64).sum()
    }
}

const fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

// ---------------------------------------------------------------------------
// Byte cursor
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Eof(self.pos))?;
        if end > self.buf.len() {
            return Err(Error::Eof(self.pos));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let b = self.take(n)?;
        // Tokenizer vocabs contain byte-fallback tokens that are not valid
        // UTF-8 on their own; those live in tensor-adjacent arrays we read
        // lossily rather than failing the whole load.
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => Value::Array(self.array()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(Error::BadValueType(other)),
        })
    }

    fn array(&mut self) -> Result<Array> {
        let elem_ty = self.u32()?;
        let n = self.u64()? as usize;
        Ok(match elem_ty {
            0 => Array::U8(self.take(n)?.to_vec()),
            1 => Array::I8(bytemuck::cast_slice(self.take(n)?).to_vec()),
            2 => Array::U16(self.le_vec(n, 2, |b| u16::from_le_bytes(b.try_into().unwrap()))?),
            3 => Array::I16(self.le_vec(n, 2, |b| i16::from_le_bytes(b.try_into().unwrap()))?),
            4 => Array::U32(self.le_vec(n, 4, |b| u32::from_le_bytes(b.try_into().unwrap()))?),
            5 => Array::I32(self.le_vec(n, 4, |b| i32::from_le_bytes(b.try_into().unwrap()))?),
            6 => Array::F32(self.le_vec(n, 4, |b| f32::from_le_bytes(b.try_into().unwrap()))?),
            7 => Array::Bool(self.take(n)?.iter().map(|&b| b != 0).collect()),
            8 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.string()?);
                }
                Array::String(v)
            }
            10 => Array::U64(self.le_vec(n, 8, |b| u64::from_le_bytes(b.try_into().unwrap()))?),
            11 => Array::I64(self.le_vec(n, 8, |b| i64::from_le_bytes(b.try_into().unwrap()))?),
            12 => Array::F64(self.le_vec(n, 8, |b| f64::from_le_bytes(b.try_into().unwrap()))?),
            // An empty array carries a placeholder element type we can ignore.
            _ if n == 0 => Array::Empty,
            other => return Err(Error::BadValueType(other)),
        })
    }

    fn le_vec<T>(&mut self, n: usize, width: usize, f: impl Fn(&[u8]) -> T) -> Result<Vec<T>> {
        let bytes = self.take(n * width)?;
        Ok(bytes.chunks_exact(width).map(f).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_multiple() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(15_821_832, 32), 15_821_856);
    }

    #[test]
    fn block_sizes_match_ggml() {
        assert_eq!(GgmlType::Q4K.bytes_for(256), 144);
        assert_eq!(GgmlType::Q6K.bytes_for(256), 210);
        assert_eq!(GgmlType::F32.bytes_for(10), 40);
    }
}
