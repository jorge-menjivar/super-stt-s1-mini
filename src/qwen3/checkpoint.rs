// SPDX-License-Identifier: GPL-3.0-only
//! The GGUF's tensors, as the parameters of [`crate::qwen3::model`]'s modules.
//!
//! llama.cpp names a Qwen3 checkpoint's tensors its own way —
//! `blk.3.attn_q.weight` for what `transformers` calls
//! `model.layers.3.self_attn.q_proj.weight` — and [`param_path`] maps them onto
//! the module tree. Every tensor reaches the store unread: its bytes are read
//! and dequantized only as the store applies it, straight into the type the
//! model computes in, so a load holds about one tensor's worth of host memory
//! at a time.
//!
//! Every weight is stored `[d_output, d_input]`, as PyTorch writes it, which
//! is the column-major layout the linear layers keep on a GPU. On flex they
//! are row-major (see [`crate::qwen3::linear_config`]), and the linear weights
//! are transposed as they are dequantized.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::nn::LinearLayout;
use burn::tensor::{DType, Shape, TensorData, f16};
use burn_store::ApplyResult;
use burn_store::bridge;
use burn_store::burn_pack::{Error as PackError, Tensor as PackTensor};

use crate::gguf::{self, Dense, Gguf, TensorInfo};

/// What a tensor is to the module that takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// A linear layer's weight, `[d_output, d_input]` in the file.
    Linear,
    /// The token embedding, `[vocab, hidden]` in the file and in the module.
    Table,
    /// A norm's scale.
    Vector,
}

/// The module parameter the GGUF tensor `name` fills, and what it is to it.
/// `None` for a tensor this port has no parameter for.
fn param_path(name: &str) -> Option<(String, Role)> {
    let top = match name {
        "token_embd.weight" => Some(("embed_tokens.weight", Role::Table)),
        "output.weight" => Some(("lm_head.weight", Role::Linear)),
        "output_norm.weight" => Some(("transformer.norm.gamma", Role::Vector)),
        _ => None,
    };
    if let Some((path, role)) = top {
        return Some((path.to_string(), role));
    }
    let (layer, tensor) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (param, role) = match tensor {
        "attn_norm.weight" => ("input_layernorm.gamma", Role::Vector),
        "ffn_norm.weight" => ("post_attention_layernorm.gamma", Role::Vector),
        "attn_q.weight" => ("self_attn.q_proj.weight", Role::Linear),
        "attn_k.weight" => ("self_attn.k_proj.weight", Role::Linear),
        "attn_v.weight" => ("self_attn.v_proj.weight", Role::Linear),
        "attn_output.weight" => ("self_attn.o_proj.weight", Role::Linear),
        "attn_q_norm.weight" => ("self_attn.q_norm.gamma", Role::Vector),
        "attn_k_norm.weight" => ("self_attn.k_norm.gamma", Role::Vector),
        "ffn_gate.weight" => ("mlp.gate_proj.weight", Role::Linear),
        "ffn_up.weight" => ("mlp.up_proj.weight", Role::Linear),
        "ffn_down.weight" => ("mlp.down_proj.weight", Role::Linear),
        _ => return None,
    };
    Some((format!("transformer.layers.{layer}.{param}"), role))
}

/// Every tensor of `gguf`, named by the parameter it fills, to be dequantized
/// into `dtype` when it is applied. `read` counts the file's bytes as each
/// tensor's are read.
///
/// A tensor this port has no parameter for keeps its GGUF name, which no
/// parameter has, so the store reports it unused and [`check`] refuses it.
///
/// A checkpoint without an output head of its own ties it to the embedding,
/// as Qwen3-0.6B's configuration says; S1-mini's GGUF carries one, a copy of
/// the embedding quantized to Q6_K rather than Q4_K.
pub fn tensors(
    gguf: &Arc<Gguf>,
    layout: LinearLayout,
    dtype: DType,
    read: &Arc<AtomicU64>,
) -> Vec<PackTensor> {
    let mut tensors: Vec<PackTensor> = gguf
        .tensors
        .iter()
        .map(|info| {
            let (path, role) = param_path(&info.name).unwrap_or((info.name.clone(), Role::Vector));
            deferred(gguf, info, path, role, layout, dtype, read)
        })
        .collect();
    let has_head = gguf.tensors.iter().any(|t| t.name == "output.weight");
    if !has_head
        && let Some(embedding) = gguf.tensors.iter().find(|t| t.name == "token_embd.weight")
    {
        tensors.push(deferred(
            gguf,
            embedding,
            "lm_head.weight".to_string(),
            Role::Linear,
            layout,
            dtype,
            read,
        ));
    }
    tensors
}

fn deferred(
    gguf: &Arc<Gguf>,
    info: &TensorInfo,
    path: String,
    role: Role,
    layout: LinearLayout,
    dtype: DType,
    read: &Arc<AtomicU64>,
) -> PackTensor {
    let transpose = role == Role::Linear && matches!(layout, LinearLayout::Row);
    let mut shape = info.shape.clone();
    if transpose {
        shape.reverse();
    }
    let (gguf, info, read) = (Arc::clone(gguf), info.clone(), Arc::clone(read));
    bridge::deferred(path, dtype, Shape::from(shape), None, move || {
        let bytes = gguf
            .read(&info)
            .map_err(|e| PackError::IoError(format!("reading {}: {e}", info.name)))?;
        read.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(if dtype == DType::F16 {
            dense::<f16>(&info, &bytes, transpose)
        } else {
            dense::<f32>(&info, &bytes, transpose)
        })
    })
}

/// `bytes`, the whole of the tensor `info`, dequantized, and transposed when
/// it is a matrix to be stored the other way round.
fn dense<E: Dense + burn::tensor::Element>(
    info: &TensorInfo,
    bytes: &[u8],
    transpose: bool,
) -> TensorData {
    let mut values = vec![E::default(); info.elements()];
    gguf::dequantize(info.kind, bytes, &mut values);
    match (transpose, info.shape.as_slice()) {
        (true, &[rows, cols]) => TensorData::new(transposed(&values, rows, cols), [cols, rows]),
        _ => TensorData::new(values, info.shape.clone()),
    }
}

/// The `cols × rows` transpose of the row-major `rows × cols` matrix
/// `values`, written a band of output rows per core.
fn transposed<E: Dense>(values: &[E], rows: usize, cols: usize) -> Vec<E> {
    /// The input rows read together, so that each pass over a band touches a
    /// few cache lines of every output row rather than one element.
    const TILE: usize = 64;
    let mut out = vec![E::default(); values.len()];
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(cols.div_ceil(TILE))
        .max(1);
    let band = cols.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (i, chunk) in out.chunks_mut(band * rows).enumerate() {
            scope.spawn(move || {
                let first = i * band;
                for r0 in (0..rows).step_by(TILE) {
                    for (j, out_row) in chunk.chunks_mut(rows).enumerate() {
                        let c = first + j;
                        for r in r0..(r0 + TILE).min(rows) {
                            out_row[r] = values[r * cols + c];
                        }
                    }
                }
            });
        }
    });
    out
}

/// Refuse a load that left any parameter unfilled, or any tensor of the file
/// unused. A checkpoint carrying weights this port never reads is one it does
/// not implement, and running it anyway would be quietly wrong.
///
/// # Errors
/// Names what went wrong, with a few of the parameters or tensors at fault.
pub fn check(result: &ApplyResult) -> Result<(), String> {
    if !result.errors.is_empty() {
        let errors: Vec<String> = result.errors.iter().map(ToString::to_string).collect();
        return Err(errors.join(", "));
    }
    if !result.missing.is_empty() {
        let missing: Vec<&str> = result.missing.iter().map(|(p, _)| p.as_str()).collect();
        return Err(format!(
            "{} parameters were not found in the checkpoint, e.g. {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        ));
    }
    if !result.unused.is_empty() {
        return Err(format!(
            "{} tensors of the checkpoint belong to no parameter, e.g. {:?}",
            result.unused.len(),
            &result.unused[..result.unused.len().min(5)]
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tensor_of_a_block_has_its_parameter() {
        let names = [
            ("attn_norm", "input_layernorm.gamma"),
            ("ffn_norm", "post_attention_layernorm.gamma"),
            ("attn_q", "self_attn.q_proj.weight"),
            ("attn_k", "self_attn.k_proj.weight"),
            ("attn_v", "self_attn.v_proj.weight"),
            ("attn_output", "self_attn.o_proj.weight"),
            ("attn_q_norm", "self_attn.q_norm.gamma"),
            ("attn_k_norm", "self_attn.k_norm.gamma"),
            ("ffn_gate", "mlp.gate_proj.weight"),
            ("ffn_up", "mlp.up_proj.weight"),
            ("ffn_down", "mlp.down_proj.weight"),
        ];
        for (gguf, param) in names {
            let (path, _) = param_path(&format!("blk.27.{gguf}.weight")).unwrap();
            assert_eq!(path, format!("transformer.layers.27.{param}"));
        }
        assert_eq!(
            param_path("output.weight"),
            Some(("lm_head.weight".to_string(), Role::Linear))
        );
        assert_eq!(
            param_path("token_embd.weight"),
            Some(("embed_tokens.weight".to_string(), Role::Table))
        );
        for unknown in [
            "blk.x.attn_q.weight",
            "blk.0.attn_q.bias",
            "rope_freqs.weight",
            "blk",
        ] {
            assert_eq!(param_path(unknown), None, "{unknown}");
        }
    }

    #[test]
    fn a_transpose_swaps_rows_and_columns() {
        let (rows, cols) = (130, 70);
        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let out = transposed(&values, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let (got, want) = (out[c * rows + r], values[r * cols + c]);
                assert_eq!(got.to_bits(), want.to_bits(), "({r}, {c})");
            }
        }
    }
}
