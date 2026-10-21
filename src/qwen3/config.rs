// SPDX-License-Identifier: GPL-3.0-only
//! The shape of the decoder, from the `config.json` S1-mini's own repository
//! publishes beside its weights.
//!
//! Only what the port reads is declared; the file carries every field of the
//! `transformers` config class it was saved from, and serde ignores the rest.
//! It is read rather than the GGUF's metadata because the kernel build runs
//! before the GGUF has downloaded, and both must build the same model.

use burn::prelude::*;
use burn::tensor::activation::silu;
use serde::Deserialize;

/// The activation of the MLP. Qwen3 uses SiLU, and so does every checkpoint
/// this port runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Silu,
}

impl Activation {
    pub fn forward<const D: usize>(self, xs: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Silu => silu(xs),
        }
    }
}

/// How the rotary embedding is scaled, `rope_scaling`. S1-mini ships `null`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_rope_type() -> String {
    "default".to_string()
}

/// A Qwen3 causal language model's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub use_sliding_window: bool,
}

impl TextConfig {
    /// Refuse a checkpoint whose shape this port does not implement, rather
    /// than run it and write nonsense.
    ///
    /// # Errors
    /// Names the first property that is out of reach.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(scaling) = &self.rope_scaling
            && scaling.rope_type != "default"
        {
            return Err(format!(
                "rope_type {:?} is not implemented, only \"default\"",
                scaling.rope_type
            ));
        }
        if self.attention_bias {
            return Err("attention_bias is not implemented".to_string());
        }
        if self.use_sliding_window {
            return Err("use_sliding_window is not implemented".to_string());
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "{} key/value heads do not divide {} query heads",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "a head of {} channels cannot be rotated in halves",
                self.head_dim
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S1-mini's `config.json`, as its repository publishes it.
    pub const S1_MINI: &str = r#"{
      "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
      "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
      "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 3072,
      "max_position_embeddings": 40960, "max_window_layers": 28, "model_type": "qwen3",
      "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
      "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
      "use_cache": true, "use_sliding_window": false, "vocab_size": 151936
    }"#;

    #[test]
    fn the_shipped_config_parses_and_validates() {
        let config: TextConfig = serde_json::from_str(S1_MINI).unwrap();
        config.validate().unwrap();
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads / config.num_key_value_heads, 2);
        assert!(config.rope_scaling.is_none());
    }

    #[test]
    fn a_rope_scaling_without_a_type_is_the_default() {
        let config = S1_MINI.replace(r#""rope_scaling": null"#, r#""rope_scaling": {"x": 1}"#);
        let config: TextConfig = serde_json::from_str(&config).unwrap();
        assert_eq!(config.rope_scaling.as_ref().unwrap().rope_type, "default");
        config.validate().unwrap();
    }

    /// Every shape this port does not implement is refused, by name.
    #[test]
    fn shapes_out_of_reach_are_refused() {
        for (from, to, named) in [
            (
                r#""rope_scaling": null"#,
                r#""rope_scaling": {"rope_type": "yarn"}"#,
                "yarn",
            ),
            (
                r#""attention_bias": false"#,
                r#""attention_bias": true"#,
                "attention_bias",
            ),
            (
                r#""use_sliding_window": false"#,
                r#""use_sliding_window": true"#,
                "use_sliding_window",
            ),
            (
                r#""num_attention_heads": 16"#,
                r#""num_attention_heads": 12"#,
                "do not divide 12",
            ),
            (r#""head_dim": 128"#, r#""head_dim": 127"#, "127 channels"),
        ] {
            assert!(S1_MINI.contains(from), "{from}");
            let config: TextConfig = serde_json::from_str(&S1_MINI.replace(from, to)).unwrap();
            let err = config.validate().unwrap_err();
            assert!(err.contains(named), "{to}: {err}");
        }
    }

    #[test]
    fn another_activation_does_not_parse() {
        let config = S1_MINI.replace(r#""hidden_act": "silu""#, r#""hidden_act": "gelu""#);
        assert!(serde_json::from_str::<TextConfig>(&config).is_err());
    }
}
