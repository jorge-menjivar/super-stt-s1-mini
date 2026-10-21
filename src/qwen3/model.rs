// SPDX-License-Identifier: GPL-3.0-only
//! The language model: the token embedding, the decoder stack and the output
//! head, and greedy decoding over them.
//!
//! A request is one prompt, prefilled up to [`MAX_PREFILL_PIECE`] positions at
//! a time, then decoded a token at a time until an end-of-turn token.
//!
//! Decoding a token is one pass of the decoder on one row, a few hundred small
//! kernels, and launching them one at a time is most of what it costs on a
//! GPU. So the step is captured as a graph and replayed: the token and its
//! position are read from device buffers the host rewrites in place, the
//! key/value caches are preallocated and written at the position the buffer
//! holds, and the step ends by writing its greedy choice back into the token
//! buffer — so the host's only round trip per token is reading that choice to
//! see whether it ended the turn. On a device without graph support the same
//! closure simply runs again.
//!
//! Every shape here is bucketed, because CubeCL compiles and tunes per shape:
//! the prompt is padded at its end to a power of two for the prefill, and a
//! step attends over a power-of-two window of the caches — one captured step
//! per window, all sharing the same caches. See [`Decoder`].
//!
//! Taken from the Qwen ASR backend's `src/qwen3/model.rs`, with the audio
//! tower and the splice of its output into the prompt taken out.

// `slice_assign` takes one range per dimension, in an array: one for a one-dimensional tensor.
#![allow(clippy::single_range_in_vec_init)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use burn::module::{ModuleMapper, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear};
use burn::prelude::*;
use burn::tensor::{DType, Graph, capture, f16};
use burn_store::ModuleSnapshot;

use crate::gguf::Gguf;
use crate::qwen3::config::TextConfig;
use crate::qwen3::transformer::{Transformer, TransformerState};
use crate::qwen3::{checkpoint, linear_config, linear_layout};

/// The module tree the checkpoint's tensors are applied to, by the paths
/// [`checkpoint`] names them with.
#[derive(Module, Debug)]
struct Model {
    embed_tokens: Embedding,
    transformer: Transformer,
    lm_head: Linear,
}

impl Model {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
            transformer: Transformer::init(cfg, device),
            lm_head: linear_config(cfg.hidden_size, cfg.vocab_size, device)
                .with_bias(false)
                .init(device),
        }
    }
}

/// Replaces every float parameter with zeros of its shape in `dtype`, made the
/// way the store makes a loaded one, and without initializing it first.
struct Zeros {
    dtype: DType,
}

impl ModuleMapper for Zeros {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let shape = param.lazy_shape();
        let device = param.lazy_device();
        let data = if self.dtype == DType::F16 {
            TensorData::zeros::<f16, _>(shape)
        } else {
            TensorData::zeros::<f32, _>(shape)
        };
        let id = param.id;
        param.transform_for_load(Tensor::from_data(data, (&device, self.dtype)), id)
    }
}

/// What the captured decode steps read and write besides the model, shared by
/// all of them.
struct StepState {
    transformer: TransformerState,
    /// The token a step feeds, (1, 1), overwritten by the step with the token
    /// it chooses.
    token: Tensor<2, Int>,
    /// The position of that token, (1).
    pos: Tensor<1, Int>,
}

/// A captured decode step.
type Step = Graph<Tensor<1, Int>, Box<dyn FnMut() -> Tensor<1, Int>>>;

/// The decode steps: one captured per attention window, all over the same
/// caches.
///
/// A step's shape is fixed by the window it attends over, not by the
/// position it writes, so a handful of windows — powers of two from
/// [`MIN_WINDOW`] — serve every position, and the warm-up compiles and tunes
/// each of them before the first request. The caches are sized for the
/// largest window used so far and grow, rarely, by doubling; a grown cache has
/// moved, and every step captured over the old one is captured again, lazily.
struct Decoder {
    state: Rc<RefCell<StepState>>,
    steps: HashMap<usize, Step>,
}

/// The smallest attention window a decode step uses.
const MIN_WINDOW: usize = 256;

/// The window a step at `pos` attends over: the smallest power of two that
/// holds its position, and at least [`MIN_WINDOW`].
pub fn window_for(pos: usize) -> usize {
    (pos + 1).next_power_of_two().max(MIN_WINDOW)
}

/// The shortest prompt the prefill runs on.
const MIN_PREFILL: usize = 64;

/// The length a prompt of `len` tokens is padded to for the prefill: a power
/// of two, so that the prefill's shapes are a handful the warm-up covers
/// rather than one per request. The padding goes at the end, where causal
/// attention never sees it and the decode steps overwrite or mask it.
pub fn prefill_len(len: usize) -> usize {
    len.next_power_of_two().max(MIN_PREFILL)
}

/// The most positions the prefill runs at once; a longer prompt goes through
/// in pieces at increasing offsets, which computes the same. Only memory rides
/// on it, and no prompt the chunker makes is this long.
const MAX_PREFILL_PIECE: usize = 1024;

/// A loaded Qwen3 checkpoint.
pub struct Qwen3 {
    model: Model,
    /// Built on the first request and kept for the next, whose prefill simply
    /// overwrites the caches from position 0.
    decoder: Option<Decoder>,
    config: TextConfig,
    device: Device,
    dtype: DType,
}

// The graph and the modules have nothing printable; the configuration and the
// placement are what a log wants.
impl std::fmt::Debug for Qwen3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3")
            .field("config", &self.config)
            .field(
                "decode_capacity",
                &self
                    .decoder
                    .as_ref()
                    .map(|d| d.state.borrow().transformer.capacity()),
            )
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl Qwen3 {
    /// Builds the model `config` describes from the tensors of `gguf`,
    /// dequantized into `dtype`. `read` counts the file's bytes as they are
    /// read.
    ///
    /// # Errors
    /// If the configuration describes a model this port does not implement,
    /// the file is not a Qwen3 checkpoint, a tensor cannot be read, or the
    /// tensors do not fill every parameter exactly.
    pub fn load(
        config: &TextConfig,
        gguf: &Arc<Gguf>,
        dtype: DType,
        device: &Device,
        read: &Arc<AtomicU64>,
    ) -> Result<Self, String> {
        config.validate()?;
        if let Some(architecture) = &gguf.architecture
            && architecture != "qwen3"
        {
            return Err(format!("the checkpoint is a {architecture}, not a qwen3"));
        }
        // The weights go to the persistent pool: exact-fit slices that live
        // for the process. In the dynamic pools they would share pages with
        // the activations, and those pools keep whatever the worst moment of
        // a workload asked for.
        let model = device.memory_persistent_allocations((), |()| {
            let mut model = Model::init(config, device);
            let tensors = checkpoint::tensors(gguf, linear_layout(device), dtype, read);
            let result = model.apply(tensors, None, None, false);
            checkpoint::check(&result).map(|()| model)
        })?;
        Ok(Self::assemble(model, config, dtype, device))
    }

    /// The model `config` describes with every weight zero, in `dtype`: the
    /// shapes, dtypes and layouts of a loaded one, so it runs the same kernels
    /// without the checkpoint. Unlike [`Self::load`]'s, its weights live in
    /// the pools a cleanup hands back to the driver.
    ///
    /// # Errors
    /// If the configuration describes a model this port does not implement.
    pub fn zeroed(config: &TextConfig, dtype: DType, device: &Device) -> Result<Self, String> {
        config.validate()?;
        let model = Model::init(config, device).map(&mut Zeros { dtype });
        Ok(Self::assemble(model, config, dtype, device))
    }

    fn assemble(model: Model, config: &TextConfig, dtype: DType, device: &Device) -> Self {
        Self {
            model,
            decoder: None,
            config: config.clone(),
            device: device.clone(),
            dtype,
        }
    }

    /// The device the model runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn embed(&self, ids: &[u32]) -> Tensor<3> {
        let len = ids.len();
        let ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
        let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), &self.device);
        self.model.embed_tokens.forward(ids)
    }

    /// Decodes greedily after `prompt` until one of `eos` or `max_new_tokens`
    /// tokens.
    ///
    /// `on_token` sees every token as it is chosen, the ending one excluded,
    /// and [`ControlFlow::Break`] ends the decoding after it — how a cancel
    /// stops paying for the rest of a request. Returns the tokens chosen,
    /// without the ending one.
    ///
    /// # Panics
    /// If `prompt` is empty, or the device fails a read.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos: &[u32],
        mut on_token: impl FnMut(u32) -> ControlFlow<()>,
    ) -> Vec<u32> {
        let len = prompt.len();
        assert!(len > 0, "an empty prompt has nothing to decode after");
        let padded = prefill_len(len);
        let embeds = self.embed(prompt);
        let embeds = if padded > len {
            let hidden = self.config.hidden_size;
            let padding = Tensor::zeros([1, padded - len, hidden], (&self.device, self.dtype));
            Tensor::cat(vec![embeds, padding], 1)
        } else {
            embeds
        };
        // The prefill writes every padded position, and the first step
        // attends over its window.
        self.ensure_capacity(padded.max(window_for(len)));

        let first = {
            let decoder = self.decoder.as_ref().expect("just ensured");
            let state = &mut *decoder.state.borrow_mut();
            let mut last = None;
            for start in (0..padded).step_by(MAX_PREFILL_PIECE) {
                let n = MAX_PREFILL_PIECE.min(padded - start);
                let hidden = self.model.transformer.forward(
                    embeds.clone().narrow(1, start, n),
                    start,
                    &mut state.transformer,
                );
                if (start..start + n).contains(&(len - 1)) {
                    last = Some(hidden.narrow(1, len - 1 - start, 1));
                }
            }
            let last = last.expect("the last position lies in some piece");
            let token = self.model.lm_head.forward(last).argmax(2).reshape([1, 1]);
            state
                .token
                .inplace(|buffer| buffer.slice_assign([0..1, 0..1], token));
            read_token(&state.token.clone().reshape([1]))
        };

        let mut tokens = Vec::new();
        let mut token = first;
        while !eos.contains(&token) {
            tokens.push(token);
            if on_token(token).is_break() || tokens.len() >= max_new_tokens {
                break;
            }
            // The token just chosen is fed at the position after the prompt
            // and every token before it.
            token = self.step(len + tokens.len() - 1, token);
        }
        tokens
    }

    /// [`Self::generate`] one operation at a time: no padding, no capture,
    /// and a cache of exactly the positions it needs. What the captured path
    /// must agree with.
    #[cfg(test)]
    pub fn generate_plain(&self, prompt: &[u32], max_new_tokens: usize, eos: &[u32]) -> Vec<u32> {
        let len = prompt.len();
        let mut state =
            TransformerState::new(&self.config, len + max_new_tokens, self.dtype, &self.device);
        let hidden = self
            .model
            .transformer
            .forward(self.embed(prompt), 0, &mut state);
        let mut logits = self.model.lm_head.forward(hidden.narrow(1, len - 1, 1));
        let mut chosen = Vec::new();
        loop {
            let token = read_token(&logits.argmax(2).reshape([1]));
            if eos.contains(&token) {
                break;
            }
            chosen.push(token);
            if chosen.len() >= max_new_tokens {
                break;
            }
            let hidden = self.model.transformer.forward(
                self.embed(&[token]),
                len + chosen.len() - 1,
                &mut state,
            );
            logits = self.model.lm_head.forward(hidden);
        }
        chosen
    }

    /// Makes sure the caches hold at least `needed` positions, growing them —
    /// and dropping every captured step, which replays against the old ones —
    /// when they do not. Returns whether they grew.
    fn ensure_capacity(&mut self, needed: usize) -> bool {
        let capacity = needed.next_power_of_two();
        match &mut self.decoder {
            Some(decoder) if decoder.state.borrow().transformer.capacity() >= needed => false,
            Some(decoder) => {
                // The steps replay against the old caches, so they go before
                // the caches move.
                decoder.steps.clear();
                decoder.state.borrow_mut().transformer.grow(capacity);
                true
            }
            None => {
                self.decoder = Some(Decoder {
                    state: Rc::new(RefCell::new(StepState {
                        transformer: TransformerState::new(
                            &self.config,
                            capacity,
                            self.dtype,
                            &self.device,
                        ),
                        token: Tensor::zeros([1, 1], &self.device),
                        pos: Tensor::zeros([1], &self.device),
                    })),
                    steps: HashMap::new(),
                });
                true
            }
        }
    }

    /// Captures the decode step attending over `window` positions.
    fn capture_step(&self, state: &Rc<RefCell<StepState>>, window: usize) -> Step {
        // The capture's warm-up runs write the cache at the position the
        // buffer holds: the last one, which lies outside every smaller window,
        // stays masked in its own until a request reaches it, and is
        // overwritten before it is read.
        {
            let state = &mut *state.borrow_mut();
            let last = state.transformer.capacity() - 1;
            write_position(&mut state.pos, last, &self.device);
        }
        let model = self.model.clone();
        let step_state = Rc::clone(state);
        let mut run: Box<dyn FnMut() -> Tensor<1, Int>> = Box::new(move || {
            let state = &mut *step_state.borrow_mut();
            let xs = model.embed_tokens.forward(state.token.clone());
            let hidden =
                model
                    .transformer
                    .forward_at(xs, &state.pos, &mut state.transformer, window);
            let next = model.lm_head.forward(hidden).argmax(2).reshape([1, 1]);
            state
                .token
                .inplace(|buffer| buffer.slice_assign([0..1, 0..1], next.clone()));
            next.reshape([1])
        });
        // Compiled and autotuned outside the capture: tuning allocates the
        // buffers of every candidate it benchmarks, which a capture would keep.
        let _ = run().into_data();
        capture(&self.device, run)
    }

    /// Feeds `token`, which the step buffer holds, at `pos` and returns the
    /// next one.
    fn step(&mut self, pos: usize, token: u32) -> u32 {
        let window = window_for(pos);
        let mut disturbed = self.ensure_capacity(window);
        let decoder = self.decoder.as_ref().expect("just ensured");
        if !decoder.steps.contains_key(&window) {
            let state = Rc::clone(&decoder.state);
            let step = self.capture_step(&state, window);
            let decoder = self.decoder.as_mut().expect("just ensured");
            decoder.steps.insert(window, step);
            disturbed = true;
        }
        let decoder = self.decoder.as_mut().expect("just ensured");
        if disturbed {
            // A capture's warm-up runs overwrote the token buffer with tokens
            // of their own. The value comes from the host, where the caller
            // already has it.
            let token = Tensor::<2, Int>::from_data([[i64::from(token)]], &self.device);
            decoder
                .state
                .borrow_mut()
                .token
                .inplace(|buffer| buffer.slice_assign([0..1, 0..1], token));
        }
        write_position(&mut decoder.state.borrow_mut().pos, pos, &self.device);
        let step = decoder.steps.get_mut(&window).expect("just captured");
        // Safety: every buffer the step reads or writes is kept alive by the
        // shared state or by the modules its closure owns, the writes above
        // and the read below go through the same device client and stream as
        // the replay, and nothing else touches those buffers meanwhile.
        let next = unsafe { step.replay() }.clone();
        read_token(&next)
    }
}

/// Writes `pos` into the one-element `buffer` in place: the buffer is what the
/// captured step reads its position from, so it must stay where it is.
fn write_position(buffer: &mut Tensor<1, Int>, pos: usize, device: &Device) {
    let pos = i64::try_from(pos).expect("a position fits in an i64");
    let at = Tensor::<1, Int>::from_data([pos], device);
    buffer.inplace(|buffer| buffer.slice_assign([0..1], at));
}

/// The one round trip of a decode step.
fn read_token(token: &Tensor<1, Int>) -> u32 {
    let value = token
        .clone()
        .into_data()
        .iter::<i64>()
        .next()
        .expect("a token tensor holds one value");
    u32::try_from(value).expect("a token id is a vocabulary index")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    /// The fixture, loaded straight from its files in `dtype`.
    fn load(dtype: DType) -> Qwen3 {
        let backend = fixture::backend();
        let config: TextConfig = serde_json::from_value(fixture::config()).unwrap();
        let gguf = Arc::new(Gguf::open(&backend.model_dir().join(fixture::GGUF)).unwrap());
        let read = Arc::default();
        let model =
            Qwen3::load(&config, &gguf, dtype, &crate::qwen3::test_device(), &read).unwrap();
        let total: usize = gguf
            .tensors
            .iter()
            .map(crate::gguf::TensorInfo::byte_len)
            .sum();
        assert_eq!(
            read.load(std::sync::atomic::Ordering::Relaxed),
            total as u64
        );
        model
    }

    const PROMPT: &[u32] = &[257, 115, 104, 105, 258, 97, 98];

    /// The captured decode, and a decode far enough to grow its caches past
    /// the first window, choose the tokens a plain step-by-step decode does.
    #[test]
    fn the_captured_decode_matches_the_step_by_step_one() {
        let mut model = load(DType::F32);
        let steps = 300;
        let mut seen = Vec::new();
        let captured = model.generate(PROMPT, steps, &[], |t| {
            seen.push(t);
            ControlFlow::Continue(())
        });
        assert_eq!(captured.len(), steps);
        assert_eq!(seen, captured);
        // Noise, but not one token over and over: the comparisons below
        // would hold of that trivially.
        let distinct: std::collections::HashSet<u32> = captured.iter().copied().collect();
        assert!(distinct.len() > 10, "{captured:?}");
        let plain = model.generate_plain(PROMPT, steps, &[]);
        assert_eq!(plain, captured);
        assert!(format!("{model:?}").contains("decode_capacity"));

        // An ending token ends it, and is not returned.
        let eos = captured[5];
        let first = captured.iter().position(|&t| t == eos).unwrap();
        let ended = model.generate(PROMPT, steps, &[eos], |_| ControlFlow::Continue(()));
        assert_eq!(ended[..], captured[..first]);
        assert_eq!(model.generate_plain(PROMPT, steps, &[eos]), ended);
    }

    /// A prompt longer than the prefill runs at once goes through in pieces,
    /// and decodes as the plain path does.
    #[test]
    fn a_long_prompt_prefills_in_pieces() {
        let mut model = load(DType::F32);
        let prompt: Vec<u32> = (0..MAX_PREFILL_PIECE + 40)
            .map(|i| u32::try_from(i % 250).unwrap())
            .collect();
        let captured = model.generate(&prompt, 6, &[], |_| ControlFlow::Continue(()));
        assert_eq!(model.generate_plain(&prompt, 6, &[]), captured);
    }

    #[test]
    fn a_break_ends_the_decode_after_that_token() {
        let mut model = load(DType::F32);
        let mut count = 0;
        let tokens = model.generate(PROMPT, 50, &[], |_| {
            count += 1;
            if count == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn a_checkpoint_loads_and_decodes_in_f16() {
        let mut model = load(DType::F16);
        let tokens = model.generate(PROMPT, 4, &[], |_| ControlFlow::Continue(()));
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn a_zeroed_model_decodes_at_the_same_shapes() {
        let config: TextConfig = serde_json::from_value(fixture::config()).unwrap();
        let mut model = Qwen3::zeroed(&config, DType::F32, &crate::qwen3::test_device()).unwrap();
        let tokens = model.generate(PROMPT, 4, &[], |_| ControlFlow::Continue(()));
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn windows_and_prefills_come_in_powers_of_two() {
        assert_eq!(window_for(0), MIN_WINDOW);
        assert_eq!(window_for(MIN_WINDOW), 2 * MIN_WINDOW);
        assert_eq!(prefill_len(1), MIN_PREFILL);
        assert_eq!(prefill_len(MIN_PREFILL + 1), 2 * MIN_PREFILL);
    }
}
