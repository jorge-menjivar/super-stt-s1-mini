// SPDX-License-Identifier: GPL-3.0-only
//! S1-mini inference on candle: a GGUF Qwen3 decoder driven greedily.
//!
//! S1-mini is a fine-tune of Qwen3-0.6B, so it loads through candle's
//! `quantized_qwen3` with no model code of its own. The engine owns the
//! weights, the tokenizer and the KV cache; [`Engine::normalize`] builds the
//! trained prompt around a transcript, decodes until the end-of-turn token,
//! and returns the text — chunking a transcript that overflows the model's
//! input budget first (see [`crate::chunk`]).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context as _, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{D, Device, Tensor, utils};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use tokenizers::Tokenizer;

use crate::chunk;
use crate::prompt::{Control, format_prompt};

/// Transcript tokens per piece. The model card recommends inputs under about a
/// thousand tokens in total; the prompt scaffolding takes ~70 of those, and
/// the rest is headroom so a piece never sits right at the edge.
pub const INPUT_TOKEN_BUDGET: usize = 768;

/// Hard ceiling on generated tokens per piece, from the model card's
/// `max_new_tokens=1024`. A cleanup never legitimately runs that long for a
/// piece under [`INPUT_TOKEN_BUDGET`]; the cap is a backstop against a decode
/// that fails to stop, so a runaway costs seconds rather than the daemon's
/// whole timeout.
pub const MAX_NEW_TOKENS: usize = 1024;

/// A loaded S1-mini ready to normalize transcripts.
pub struct Engine {
    model: Qwen3,
    tokenizer: Tokenizer,
    device: Device,
    /// Token ids that end the assistant turn. Qwen3 marks it with
    /// `<|im_end|>`; `<|endoftext|>` is the base model's generic stop and is
    /// honored too, so a decode that reaches either halts.
    stop_tokens: Vec<u32>,
}

impl Engine {
    /// Load the model from a directory holding one `*.gguf` and a
    /// `tokenizer.json`, on CUDA when built with it and the device is up,
    /// otherwise on the CPU. `force_cpu` skips the accelerator even when
    /// available.
    pub fn load(model_dir: &Path, force_cpu: bool) -> Result<Self> {
        let files = resolve_files(model_dir)?;

        let device = if !force_cpu && utils::cuda_is_available() {
            log::info!("S1-mini: using CUDA device");
            Device::new_cuda(0).context("Failed to create CUDA device")?
        } else {
            log::info!("S1-mini: using CPU");
            Device::Cpu
        };

        let start = std::time::Instant::now();
        let mut file = std::fs::File::open(&files.weights)
            .with_context(|| format!("open {}", files.weights.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(&files.weights))
            .context("read GGUF")?;
        let tensor_bytes: usize = content
            .tensor_infos
            .values()
            .map(|t| t.shape.elem_count() * t.ggml_dtype.type_size() / t.ggml_dtype.block_size())
            .sum();
        let model =
            Qwen3::from_gguf(content, &mut file, &device).context("build Qwen3 from GGUF")?;
        log::info!(
            "S1-mini: loaded {} ({} MB) in {:.1}s",
            files.weights.display(),
            tensor_bytes / 1_000_000,
            start.elapsed().as_secs_f64()
        );

        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", files.tokenizer.display()))?;
        let stop_tokens: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        anyhow::ensure!(
            stop_tokens.contains(&tokenizer.token_to_id("<|im_end|>").unwrap_or(u32::MAX)),
            "tokenizer has no <|im_end|> token; this is not a Qwen3 tokenizer"
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            stop_tokens,
        })
    }

    /// Device label for `GET /v1/status`.
    pub fn device_label(&self) -> &'static str {
        match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
        }
    }

    /// Normalize a raw transcript under `control`.
    ///
    /// A transcript over [`INPUT_TOKEN_BUDGET`] is cut into pieces, each
    /// normalized in turn and the results joined. `cancel` is polled once per
    /// generated token; when it is raised the decode stops and this returns an
    /// error, so a request the daemon gave up on does not keep the GPU busy.
    ///
    /// An empty result is a valid answer — a transcript of nothing but filler
    /// normalizes to nothing — and is returned as such.
    pub fn normalize(
        &mut self,
        text: &str,
        control: Control,
        cancel: &AtomicBool,
    ) -> Result<String> {
        let pieces = chunk::chunk(text, INPUT_TOKEN_BUDGET, |s| self.count_tokens(s));
        if pieces.len() > 1 {
            log::info!(
                "S1-mini: transcript over {INPUT_TOKEN_BUDGET} tokens; normalizing in {} pieces",
                pieces.len()
            );
        }
        let mut outputs = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            let prompt = format_prompt(piece, control);
            outputs.push(self.generate(&prompt, cancel)?);
        }
        Ok(chunk::join(&outputs))
    }

    /// Tokens in `text` as the model will see them (no special tokens).
    fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map_or(0, |enc| enc.get_ids().len())
    }

    /// Greedily decode one prompt to its end-of-turn token.
    ///
    /// S1-mini is trained for greedy decoding; sampling only adds variance.
    /// The KV cache is cleared first, since every request is a fresh
    /// conversation, and the prompt is fed in one forward pass.
    fn generate(&mut self, prompt: &str, cancel: &AtomicBool) -> Result<String> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?
            .get_ids()
            .to_vec();

        // Generation is bounded by the input: a cleanup is never much longer
        // than what it cleans, so a piece's ceiling follows its own length
        // while the model card's absolute cap still applies.
        let max_new = (2 * prompt_tokens.len() + 64).min(MAX_NEW_TOKENS);

        self.model.clear_kv_cache();
        let start = std::time::Instant::now();
        let generated = self.device.clone().with_context(|| -> Result<Vec<u32>> {
            let mut generated = Vec::new();
            let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let mut next = argmax(&self.model.forward(&input, 0)?)?;

            while !self.stop_tokens.contains(&next) && generated.len() < max_new {
                if cancel.load(Ordering::SeqCst) {
                    bail!("cancelled");
                }
                generated.push(next);
                let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                let offset = prompt_tokens.len() + generated.len() - 1;
                next = argmax(&self.model.forward(&input, offset)?)?;
            }
            Ok(generated)
        })?;

        if generated.len() >= max_new {
            log::warn!(
                "S1-mini: decode hit the {max_new}-token cap without an end-of-turn token; \
                 returning what it produced"
            );
        }
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decode output: {e}"))?;
        let elapsed = start.elapsed().as_secs_f64();
        let generated_count = u32::try_from(generated.len()).unwrap_or(u32::MAX);
        log::debug!(
            "S1-mini: {} prompt tokens, {generated_count} generated in {elapsed:.2}s ({:.1} tok/s)",
            prompt_tokens.len(),
            f64::from(generated_count) / elapsed.max(f64::EPSILON)
        );
        Ok(text.trim().to_string())
    }
}

/// The highest-scoring token of a `[1, vocab]` logits row.
fn argmax(logits: &Tensor) -> Result<u32> {
    Ok(logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?)
}

#[derive(Debug)]
struct ModelFiles {
    weights: PathBuf,
    tokenizer: PathBuf,
}

/// Find the one `*.gguf` and the `tokenizer.json` the manifest provisions into
/// a model directory. Named by extension rather than by exact filename so the
/// same engine serves every quantization the manifest lists.
fn resolve_files(dir: &Path) -> Result<ModelFiles> {
    let tokenizer = dir.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer.exists(),
        "tokenizer.json not found in {}",
        dir.display()
    );
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read dir {}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    ggufs.sort();
    let weights = match ggufs.as_slice() {
        [one] => one.clone(),
        [] => bail!("no .gguf in {}", dir.display()),
        many => bail!(
            "expected one .gguf in {}, found {}: {many:?}",
            dir.display(),
            many.len()
        ),
    };
    Ok(ModelFiles { weights, tokenizer })
}

#[cfg(test)]
mod tests {
    use super::resolve_files;

    #[test]
    fn a_model_dir_needs_a_tokenizer_and_exactly_one_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_files(dir.path()).unwrap_err().to_string();
        assert!(err.contains("tokenizer.json"), "{err}");

        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let err = resolve_files(dir.path()).unwrap_err().to_string();
        assert!(err.contains("no .gguf"), "{err}");

        std::fs::write(dir.path().join("s1-mini-q4_k_m.gguf"), b"").unwrap();
        let files = resolve_files(dir.path()).unwrap();
        assert!(files.weights.ends_with("s1-mini-q4_k_m.gguf"));

        std::fs::write(dir.path().join("another.gguf"), b"").unwrap();
        let err = resolve_files(dir.path()).unwrap_err().to_string();
        assert!(err.contains("expected one .gguf"), "{err}");
    }
}
