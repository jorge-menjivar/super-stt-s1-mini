// SPDX-License-Identifier: GPL-3.0-only
//! Qwen3, the architecture S1-mini is fine-tuned from, on Burn.
//!
//! - [`transformer`]: the decoder stack (RMS norm, per-head query and key
//!   norms, rotary positions, grouped-query attention, `SwiGLU`);
//! - [`model`]: the stack between its token embedding and its output head,
//!   and greedy decoding over it;
//! - [`checkpoint`]: the GGUF's tensors as the parameters of those modules;
//! - [`config`]: the shape of it all, from `config.json`.
//!
//! # Provenance
//!
//! The candle backend this replaces ran
//! `candle_transformers::models::quantized_qwen3` at
//! `jorge-menjivar/candle@eef47d5`. The decoder here is the Qwen ASR
//! backend's, whose own decoder is Qwen3-0.6B's as well, and the decoding
//! around it follows that backend's: a captured decode step per attention
//! window, and prompts padded to a few lengths, so that CubeCL compiles and
//! tunes a handful of shapes rather than one per request.

pub mod checkpoint;
pub mod config;
pub mod model;
pub mod transformer;

use burn::nn::{LinearConfig, LinearLayout};
use burn::prelude::*;

/// The configuration of every linear layer here, on `device`.
///
/// On a GPU the weight keeps the column-major, `[d_output, d_input]` layout
/// the GGUF stores it in. That is not only about loading without a transpose.
/// Decoding a token runs every linear layer of the decoder on a single row,
/// and the matmul kernels for that product are very sensitive to the layout
/// of the matrix: the Burn port of Qwen3-TTS measured ~55 µs with a row-major
/// weight against ~12 µs, the memory-bandwidth limit, with a column-major one
/// on an RTX 3090.
///
/// Burn's flex CPU backend is the other way round. It multiplies by a
/// column-major matrix through a strided path, and on one thread below 192³
/// multiply-adds, which is every projection of a decode step. On Qwen3-0.6B's
/// shapes the Qwen ASR backend measured 1.48 s a token that way against 0.04
/// s row-major, on a Ryzen 9 5900X. So on flex the weights are row-major,
/// `[d_input, d_output]`, and transposed once as they are loaded (see
/// [`checkpoint`]).
pub(crate) fn linear_config(d_input: usize, d_output: usize, device: &Device) -> LinearConfig {
    LinearConfig::new(d_input, d_output).with_layout(linear_layout(device))
}

/// The layout [`linear_config`] gives a weight on `device`.
pub(crate) fn linear_layout(device: &Device) -> LinearLayout {
    if is_flex(device) {
        LinearLayout::Row
    } else {
        LinearLayout::Col
    }
}

/// Whether `device` is Burn's flex CPU backend.
#[cfg(feature = "flex")]
fn is_flex(device: &Device) -> bool {
    *device == Device::flex()
}

/// Whether `device` is Burn's flex CPU backend, which a build without the
/// `flex` feature does not have.
#[cfg(not(feature = "flex"))]
fn is_flex(_: &Device) -> bool {
    false
}

/// The device the unit tests run on.
#[cfg(test)]
pub(crate) fn test_device() -> Device {
    crate::engine::select_device(None).0
}
