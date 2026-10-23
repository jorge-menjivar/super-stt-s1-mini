// SPDX-License-Identifier: GPL-3.0-only
//! The GGUF file S1-mini ships in: its tensor directory, and dequantizing its
//! tensors.
//!
//! The model is published as llama.cpp's Q4_K_M build, the one the model card
//! recommends and measured its accuracy on: most matrices in Q4_K, the output
//! head and half the value and down projections in Q6_K, and the norms in f32.
//! Burn has no kernels for llama.cpp's block quantizations, so every tensor is
//! dequantized once, as it is loaded, into the dense type the model computes
//! in. The weights are then the Q4_K_M build's values exactly, rounded to f16
//! on a GPU.
//!
//! Only what that file needs is read: GGUF versions 2 and 3, and tensors in
//! f32, f16, Q4_K and Q6_K. Of the metadata only the architecture and the data
//! alignment are kept. The shapes come from `config.json` instead, which the
//! kernel build reads before this file has downloaded, and the load holds every
//! tensor here to them.
//!
//! The block layouts and their arithmetic are ggml's (`dequantize_row_q4_K`
//! and `dequantize_row_q6_K` in `ggml-quants.c`), which candle's `k_quants.rs`,
//! what this backend ran on before, follows step for step.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::f16;

/// Elements in a k-quant block.
const QK_K: usize = 256;

/// Data offsets are multiples of this unless the file says otherwise.
const DEFAULT_ALIGNMENT: u64 = 32;

/// Below this many blocks a tensor is converted on the calling thread.
const PARALLEL_MIN_BLOCKS: usize = 1 << 10;

/// How a tensor's elements are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    F32,
    F16,
    /// 256 elements in 144 bytes: eight sub-blocks of 32 4-bit values, each
    /// with a 6-bit scale and a 6-bit minimum, under an f16 scale for each.
    Q4K,
    /// 256 elements in 210 bytes: sixteen sub-blocks of 16 6-bit values, each
    /// with an 8-bit scale, under one f16 scale.
    Q6K,
}

impl Kind {
    /// The kind of ggml type `id`, if it is one this reader dequantizes.
    fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            12 => Some(Self::Q4K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }

    /// Elements per block, and the bytes they take.
    fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4K => (QK_K, 144),
            Self::Q6K => (QK_K, 210),
        }
    }
}

/// One tensor of the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Outermost dimension first, as PyTorch writes it. GGUF lists them the
    /// other way round, innermost first, over the same row-major bytes.
    pub shape: Vec<usize>,
    pub kind: Kind,
    /// From the start of the file.
    offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// The bytes the tensor takes in the file.
    pub fn byte_len(&self) -> usize {
        let (elements, bytes) = self.kind.block();
        self.elements() / elements * bytes
    }
}

/// An open GGUF file: its tensor directory, and the file to read them from.
#[derive(Debug)]
pub struct Gguf {
    file: File,
    /// `general.architecture`.
    pub architecture: Option<String>,
    pub tensors: Vec<TensorInfo>,
}

impl Gguf {
    /// Read the header and the tensor directory of the file at `path`. No
    /// tensor is read until [`Self::read`] asks for it.
    ///
    /// # Errors
    /// If the file is not a GGUF this reader understands, holds a tensor of a
    /// type it does not dequantize, or is shorter than its directory says.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let len = file.metadata()?.len();
        let mut reader = Reader(BufReader::new(file));
        let (architecture, tensors) = read_directory(&mut reader, len)
            .with_context(|| format!("reading {}", path.display()))?;
        Ok(Self {
            file: reader.0.into_inner(),
            architecture,
            tensors,
        })
    }

    /// The raw bytes of `tensor`.
    ///
    /// # Errors
    /// If the file cannot be read.
    pub fn read(&self, tensor: &TensorInfo) -> std::io::Result<Vec<u8>> {
        let mut bytes = vec![0; tensor.byte_len()];
        self.file.read_exact_at(&mut bytes, tensor.offset)?;
        Ok(bytes)
    }
}

/// The header, the metadata and the tensor directory, with every tensor's
/// offset made absolute and checked against the file's length.
fn read_directory(r: &mut Reader, file_len: u64) -> Result<(Option<String>, Vec<TensorInfo>)> {
    let mut magic = [0; 4];
    r.0.read_exact(&mut magic)?;
    ensure!(&magic == b"GGUF", "not a GGUF file");
    let version = r.u32()?;
    ensure!(
        matches!(version, 2 | 3),
        "GGUF version {version} is not supported, only 2 and 3"
    );
    let tensor_count = r.u64()?;
    let metadata_count = r.u64()?;

    let mut architecture = None;
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..metadata_count {
        let key = r.string()?;
        let kind = r.u32()?;
        match (key.as_str(), kind) {
            ("general.architecture", STRING) => architecture = Some(r.string()?),
            ("general.alignment", UINT32) => alignment = u64::from(r.u32()?),
            _ => r
                .skip_value(kind)
                .with_context(|| format!("skipping {key}"))?,
        }
    }
    ensure!(
        alignment.is_power_of_two(),
        "an alignment of {alignment} is not a power of two"
    );

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = r.string()?;
        let dims = r.u32()?;
        let mut shape = (0..dims)
            .map(|_| r.u64().map(usize::try_from))
            .collect::<std::io::Result<Result<Vec<_>, _>>>()??;
        let id = r.u32()?;
        let offset = r.u64()?;
        let kind = Kind::from_id(id).with_context(|| {
            format!("{name} is of ggml type {id}, which this backend does not dequantize")
        })?;
        let (block, _) = kind.block();
        ensure!(
            shape.first().is_some_and(|inner| inner % block == 0),
            "{name}'s rows of {shape:?} do not split into blocks of {block}"
        );
        shape.reverse();
        tensors.push(TensorInfo {
            name,
            shape,
            kind,
            offset,
        });
    }

    let directory_end = r.0.stream_position()?;
    let data = directory_end.next_multiple_of(alignment);
    for tensor in &mut tensors {
        ensure!(
            tensor.offset.is_multiple_of(alignment),
            "{} starts at {}, off the alignment of {alignment}",
            tensor.name,
            tensor.offset
        );
        tensor.offset += data;
        let end = tensor.offset + tensor.byte_len() as u64;
        ensure!(
            end <= file_len,
            "{} runs to byte {end}, past the end of the file at {file_len}: is it complete?",
            tensor.name
        );
    }
    Ok((architecture, tensors))
}

// The metadata value types this reader tells apart.
const UINT8: u32 = 0;
const INT8: u32 = 1;
const UINT16: u32 = 2;
const INT16: u32 = 3;
const UINT32: u32 = 4;
const INT32: u32 = 5;
const FLOAT32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const UINT64: u32 = 10;
const INT64: u32 = 11;
const FLOAT64: u32 = 12;

/// Little-endian reads over the header.
struct Reader(BufReader<File>);

impl Reader {
    fn u32(&mut self) -> std::io::Result<u32> {
        let mut bytes = [0; 4];
        self.0.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> std::io::Result<u64> {
        let mut bytes = [0; 8];
        self.0.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn string(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?)?;
        let mut bytes = vec![0; len];
        self.0.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }

    fn skip(&mut self, bytes: u64) -> Result<()> {
        self.0.seek(SeekFrom::Current(i64::try_from(bytes)?))?;
        Ok(())
    }

    /// Skips a value of type `kind`: the tokenizer's 150 thousand strings are
    /// in the metadata, and none of them is needed here.
    fn skip_value(&mut self, kind: u32) -> Result<()> {
        match kind {
            UINT8 | INT8 | BOOL => self.skip(1),
            UINT16 | INT16 => self.skip(2),
            UINT32 | INT32 | FLOAT32 => self.skip(4),
            UINT64 | INT64 | FLOAT64 => self.skip(8),
            STRING => {
                let len = self.u64()?;
                self.skip(len)
            }
            ARRAY => {
                let element = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count {
                    self.skip_value(element)?;
                }
                Ok(())
            }
            _ => bail!("unknown metadata value type {kind}"),
        }
    }
}

/// What a tensor can be dequantized into.
pub trait Dense: Copy + Default + Send + Sync {
    fn from_f32(value: f32) -> Self;
}

impl Dense for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl Dense for f16 {
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
}

/// Dequantizes `bytes`, a whole tensor stored as `kind`, into `out`, splitting
/// the work across every core.
///
/// # Panics
/// If `bytes` and `out` disagree on the number of elements.
pub fn dequantize<E: Dense>(kind: Kind, bytes: &[u8], out: &mut [E]) {
    let (block_elements, block_bytes) = kind.block();
    let blocks = bytes.len() / block_bytes;
    assert!(
        bytes.len() == blocks * block_bytes && out.len() == blocks * block_elements,
        "{} bytes of {kind:?} are not {} elements",
        bytes.len(),
        out.len()
    );
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(blocks.div_ceil(PARALLEL_MIN_BLOCKS))
        .max(1);
    let per_thread = blocks.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (from, to) in bytes
            .chunks(per_thread * block_bytes)
            .zip(out.chunks_mut(per_thread * block_elements))
        {
            scope.spawn(move || dequantize_blocks(kind, from, to));
        }
    });
}

fn dequantize_blocks<E: Dense>(kind: Kind, bytes: &[u8], out: &mut [E]) {
    match kind {
        Kind::F32 => {
            for (from, to) in bytes.as_chunks::<4>().0.iter().zip(out) {
                *to = E::from_f32(f32::from_le_bytes(*from));
            }
        }
        Kind::F16 => {
            for (from, to) in bytes.as_chunks::<2>().0.iter().zip(out) {
                *to = E::from_f32(f16::from_le_bytes(*from).to_f32());
            }
        }
        Kind::Q4K => blockwise(bytes.as_chunks::<144>().0, out, q4k_block),
        Kind::Q6K => blockwise(bytes.as_chunks::<210>().0, out, q6k_block),
    }
}

/// Dequantizes k-quant `blocks` with `block` into `out`, through f32.
fn blockwise<const N: usize, E: Dense>(
    blocks: &[[u8; N]],
    out: &mut [E],
    block: fn(&[u8; N], &mut [f32; QK_K]),
) {
    let mut values = [0f32; QK_K];
    for (from, to) in blocks.iter().zip(out.as_chunks_mut::<QK_K>().0) {
        block(from, &mut values);
        for (value, to) in values.iter().zip(to) {
            *to = E::from_f32(*value);
        }
    }
}

fn half_at(bytes: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
}

/// One Q4_K block: `d` and `dmin` in f16, twelve bytes packing eight 6-bit
/// scales and eight 6-bit minimums, then 128 bytes of 4-bit values. Each run
/// of 64 values is two sub-blocks: the low nibbles of 32 bytes, then their
/// high nibbles.
fn q4k_block(block: &[u8; 144], out: &mut [f32; QK_K]) {
    let d = half_at(block, 0);
    let min = half_at(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];
    for (run, (quants, out)) in qs
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<64>().0)
        .enumerate()
    {
        let (sc, m) = scale_min_k4(2 * run, scales);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(2 * run + 1, scales);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        let (low, high) = out.split_at_mut(32);
        for ((q, low), high) in quants.iter().zip(low).zip(high) {
            *low = d1 * f32::from(q & 0xF) - m1;
            *high = d2 * f32::from(q >> 4) - m2;
        }
    }
}

/// The scale and minimum of sub-block `j` of a Q4_K block: the first four in
/// the low six bits of bytes 0–3 and 4–7, the last four split between the
/// nibbles of bytes 8–11 and the top two bits of bytes 0–7.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// One Q6_K block: 128 bytes of low nibbles, 64 bytes of high bit pairs,
/// sixteen signed 8-bit scales, then `d` in f16. Each half of 128 values
/// takes 64 low-nibble bytes, 32 high-pair bytes and eight scales.
fn q6k_block(block: &[u8; 210], out: &mut [f32; QK_K]) {
    let d = half_at(block, 208);
    for (half, out) in out.as_chunks_mut::<128>().0.iter_mut().enumerate() {
        let ql = &block[64 * half..64 * half + 64];
        let qh = &block[128 + 32 * half..128 + 32 * half + 32];
        let sc = &block[192 + 8 * half..192 + 8 * half + 8];
        let scale = |i: usize| f32::from(i8::from_le_bytes([sc[i]]));
        for l in 0..32 {
            let is = l / 16;
            let six = |low: u8, shift: u8| {
                let q = (low | (((qh[l] >> shift) & 3) << 4)).cast_signed() - 32;
                f32::from(q)
            };
            out[l] = d * scale(is) * six(ql[l] & 0xF, 0);
            out[l + 32] = d * scale(is + 2) * six(ql[l + 32] & 0xF, 2);
            out[l + 64] = d * scale(is + 4) * six(ql[l] >> 4, 4);
            out[l + 96] = d * scale(is + 6) * six(ql[l + 32] >> 4, 6);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A seeded xorshift, for bytes that are the same every run.
    pub(crate) struct Rng(pub u64);

    impl Rng {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        pub(crate) fn byte(&mut self) -> u8 {
            self.next().to_le_bytes()[3]
        }
    }

    fn half_bytes(value: f32) -> [u8; 2] {
        f16::from_f32(value).to_le_bytes()
    }

    /// A Q4_K block written field by field: `scales[i]` and `mins[i]` are the
    /// six-bit scale and minimum of sub-block `i`, `quants[k]` the four-bit
    /// value of element `k`.
    pub(crate) fn q4k_bytes(
        d: f32,
        dmin: f32,
        scales: [u8; 8],
        mins: [u8; 8],
        quants: &[u8],
    ) -> Vec<u8> {
        let mut packed = [0u8; 12];
        for j in 0..8 {
            let (sc, m) = (scales[j], mins[j]);
            if j < 4 {
                packed[j] |= sc & 63;
                packed[j + 4] |= m & 63;
            } else {
                packed[j + 4] |= (sc & 0xF) | ((m & 0xF) << 4);
                packed[j - 4] |= (sc >> 4) << 6;
                packed[j] |= (m >> 4) << 6;
            }
        }
        let mut qs = [0u8; 128];
        for (k, &q) in quants.iter().enumerate() {
            let (run, within) = (k / 64, k % 64);
            let byte = &mut qs[32 * run + within % 32];
            *byte |= if within < 32 { q & 0xF } else { (q & 0xF) << 4 };
        }
        let mut block = Vec::with_capacity(144);
        block.extend(half_bytes(d));
        block.extend(half_bytes(dmin));
        block.extend(packed);
        block.extend(qs);
        block
    }

    /// A Q6_K block written field by field: `scales[i]` is the scale of
    /// sub-block `i` of sixteen, `quants[k]` the six-bit value of element `k`
    /// before the offset of 32 is taken off.
    pub(crate) fn q6k_bytes(d: f32, scales: [i8; 16], quants: &[u8]) -> Vec<u8> {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for (k, &q) in quants.iter().enumerate() {
            let (half, within) = (k / 128, k % 128);
            let (quarter, l) = (within / 32, within % 32);
            let low = &mut ql[64 * half + l + if quarter % 2 == 1 { 32 } else { 0 }];
            *low |= if quarter < 2 { q & 0xF } else { (q & 0xF) << 4 };
            qh[32 * half + l] |= ((q >> 4) & 3) << (2 * quarter);
        }
        let mut block = Vec::with_capacity(210);
        block.extend(ql);
        block.extend(qh);
        block.extend(scales.map(i8::to_le_bytes).concat());
        block.extend(half_bytes(d));
        block
    }

    /// Every element of a Q4_K block is its sub-block's scale times its value,
    /// less its sub-block's minimum, in the order ggml lays them out.
    #[test]
    fn a_q4k_block_is_scale_times_value_less_min() {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        let scales: [u8; 8] = std::array::from_fn(|_| rng.byte() & 63);
        let mins: [u8; 8] = std::array::from_fn(|_| rng.byte() & 63);
        let quants: Vec<u8> = (0..QK_K).map(|_| rng.byte() & 0xF).collect();
        // Both stored in f16, which 0.0125 is not.
        let d = f16::from_f32(0.012_5).to_f32();
        let dmin = 0.003_906_25;
        let block = q4k_bytes(d, dmin, scales, mins, &quants);
        let mut out = vec![0f32; QK_K];
        dequantize(Kind::Q4K, &block, &mut out);
        for (k, value) in out.iter().enumerate() {
            let sub = k / 32;
            let expected =
                d * f32::from(scales[sub]) * f32::from(quants[k]) - dmin * f32::from(mins[sub]);
            assert!(
                (value - expected).abs() < 1e-6,
                "{k}: {value} against {expected}"
            );
        }
    }

    /// Every element of a Q6_K block is its sub-block's signed scale times its
    /// value less 32.
    #[test]
    fn a_q6k_block_is_signed_scale_times_value() {
        let mut rng = Rng(0xfedc_ba98_7654_3210);
        let scales: [i8; 16] = std::array::from_fn(|_| rng.byte().cast_signed());
        let quants: Vec<u8> = (0..QK_K).map(|_| rng.byte() & 63).collect();
        let d = 0.000_976_562_5;
        let block = q6k_bytes(d, scales, &quants);
        let mut out = vec![0f32; QK_K];
        dequantize(Kind::Q6K, &block, &mut out);
        for (k, value) in out.iter().enumerate() {
            let expected = d * f32::from(scales[k / 16]) * f32::from(i16::from(quants[k]) - 32);
            assert!(
                (value - expected).abs() < 1e-6,
                "{k}: {value} against {expected}"
            );
        }
    }

    /// A tensor of many blocks, split across threads, comes out as each block
    /// does on its own, and f16 output is the f32 output rounded.
    #[test]
    fn a_tensor_dequantizes_block_by_block_across_threads() {
        let mut rng = Rng(42);
        let blocks = 3 * PARALLEL_MIN_BLOCKS + 7;
        let bytes: Vec<u8> = (0..blocks)
            .flat_map(|_| {
                let scales = std::array::from_fn(|_| rng.byte().cast_signed());
                let quants: Vec<u8> = (0..QK_K).map(|_| rng.byte() & 63).collect();
                q6k_bytes(f32::from(rng.byte()) / 4096.0, scales, &quants)
            })
            .collect();
        let mut whole = vec![0f32; blocks * QK_K];
        dequantize(Kind::Q6K, &bytes, &mut whole);
        let blocks_and_values = bytes
            .as_chunks::<210>()
            .0
            .iter()
            .zip(whole.as_chunks::<QK_K>().0);
        for (block, expected) in blocks_and_values {
            let mut one = [0f32; QK_K];
            q6k_block(block, &mut one);
            assert_eq!(one.map(f32::to_bits), expected.map(f32::to_bits));
        }
        let mut halves = vec![f16::ZERO; blocks * QK_K];
        dequantize(Kind::Q6K, &bytes, &mut halves);
        assert!(
            halves
                .iter()
                .zip(&whole)
                .all(|(h, f)| *h == f16::from_f32(*f))
        );
    }

    #[test]
    fn plain_floats_read_as_themselves() {
        let values = [1.5f32, -0.25, 3.0e-3, 65504.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = [0f32; 4];
        dequantize(Kind::F32, &bytes, &mut out);
        assert_eq!(out.map(f32::to_bits), values.map(f32::to_bits));
        let bytes: Vec<u8> = values.iter().flat_map(|v| half_bytes(*v)).collect();
        dequantize(Kind::F16, &bytes, &mut out);
        let halves = values.map(|v| f16::from_f32(v).to_f32().to_bits());
        assert_eq!(out.map(f32::to_bits), halves);
    }

    /// A GGUF file with `tensors`, each `(name, ggml type, dims innermost
    /// first, bytes)`, and `metadata` written before them.
    pub(crate) fn write_gguf(
        path: &Path,
        metadata: &[(&str, MetadataValue)],
        tensors: &[(String, u32, Vec<u64>, Vec<u8>)],
    ) {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((metadata.len() as u64).to_le_bytes());
        let string = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        };
        for (key, value) in metadata {
            string(&mut out, key);
            match value {
                MetadataValue::String(s) => {
                    out.extend(STRING.to_le_bytes());
                    string(&mut out, s);
                }
                MetadataValue::U32(v) => {
                    out.extend(UINT32.to_le_bytes());
                    out.extend(v.to_le_bytes());
                }
                MetadataValue::Strings(items) => {
                    out.extend(ARRAY.to_le_bytes());
                    out.extend(STRING.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    for item in items {
                        string(&mut out, item);
                    }
                }
                MetadataValue::F32s(items) => {
                    out.extend(ARRAY.to_le_bytes());
                    out.extend(FLOAT32.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    for item in items {
                        out.extend(item.to_le_bytes());
                    }
                }
            }
        }
        let mut offset = 0u64;
        let mut offsets = Vec::new();
        for (name, kind, dims, bytes) in tensors {
            string(&mut out, name);
            out.extend(u32::try_from(dims.len()).unwrap().to_le_bytes());
            for dim in dims {
                out.extend(dim.to_le_bytes());
            }
            out.extend(kind.to_le_bytes());
            out.extend(offset.to_le_bytes());
            offsets.push(offset);
            offset = (offset + bytes.len() as u64).next_multiple_of(DEFAULT_ALIGNMENT);
        }
        let alignment = usize::try_from(DEFAULT_ALIGNMENT).unwrap();
        out.resize(out.len().next_multiple_of(alignment), 0);
        let data = out.len();
        for ((_, _, _, bytes), offset) in tensors.iter().zip(offsets) {
            out.resize(data + usize::try_from(offset).unwrap(), 0);
            out.extend(bytes);
        }
        std::fs::write(path, out).unwrap();
    }

    /// The metadata values [`write_gguf`] writes.
    pub(crate) enum MetadataValue {
        String(&'static str),
        U32(u32),
        Strings(Vec<&'static str>),
        F32s(Vec<f32>),
    }

    #[test]
    fn the_directory_reads_back_with_shapes_outermost_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.gguf");
        let norm: Vec<u8> = (0..4u8).flat_map(|v| f32::from(v).to_le_bytes()).collect();
        let quants: Vec<u8> = (0..QK_K).map(|k| u8::try_from(k % 64).unwrap()).collect();
        let matrix: Vec<u8> = (0..2)
            .flat_map(|_| q6k_bytes(0.5, [1; 16], &quants))
            .collect();
        write_gguf(
            &path,
            &[
                ("general.architecture", MetadataValue::String("qwen3")),
                (
                    "tokenizer.ggml.tokens",
                    MetadataValue::Strings(vec!["a", "bc"]),
                ),
                ("tokenizer.ggml.scores", MetadataValue::F32s(vec![0.5, 1.0])),
                ("qwen3.block_count", MetadataValue::U32(28)),
            ],
            &[
                ("norm".into(), 0, vec![4], norm),
                ("matrix".into(), 14, vec![256, 2], matrix.clone()),
            ],
        );
        let gguf = Gguf::open(&path).unwrap();
        assert_eq!(gguf.architecture.as_deref(), Some("qwen3"));
        let [norm, m] = &gguf.tensors[..] else {
            panic!("{:?}", gguf.tensors)
        };
        assert_eq!(
            (norm.name.as_str(), &norm.shape[..], norm.kind),
            ("norm", &[4][..], Kind::F32)
        );
        assert_eq!(
            (m.shape.as_slice(), m.kind, m.byte_len()),
            (&[2, 256][..], Kind::Q6K, 420)
        );
        assert_eq!(gguf.read(m).unwrap(), matrix);
        let mut values = [0f32; 4];
        dequantize(norm.kind, &gguf.read(norm).unwrap(), &mut values);
        assert_eq!(
            values.map(f32::to_bits),
            [0f32, 1.0, 2.0, 3.0].map(f32::to_bits)
        );
    }

    #[test]
    fn files_it_cannot_read_are_refused_by_what_is_wrong() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.gguf");
        let err = |path: &Path| format!("{:#}", Gguf::open(path).unwrap_err());

        std::fs::write(&path, b"GGML\x03\0\0\0").unwrap();
        assert!(err(&path).contains("not a GGUF"), "{}", err(&path));

        let q5k = vec![0u8; 176];
        write_gguf(&path, &[], &[("w".into(), 13, vec![256], q5k)]);
        assert!(err(&path).contains("ggml type 13"), "{}", err(&path));

        write_gguf(&path, &[], &[("w".into(), 12, vec![100], vec![0; 144])]);
        assert!(err(&path).contains("blocks of 256"), "{}", err(&path));

        write_gguf(&path, &[], &[("w".into(), 12, vec![256], vec![0; 144])]);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(err(&path).contains("past the end"), "{}", err(&path));

        let missing = dir.path().join("missing.gguf");
        assert!(err(&missing).contains("missing.gguf"), "{}", err(&missing));
    }
}
