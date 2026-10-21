// SPDX-License-Identifier: GPL-3.0-only
//! A tiny S1-mini, written on the fly for the tests that load, warm up and
//! normalize: CI never has the real weights.
//!
//! The checkpoint has the real one's layout — llama.cpp's tensor names, Q4_K
//! for most matrices, Q6_K for the value and down projections and the output
//! head, f32 norms, f16 query and key norms — at a size that loads in
//! milliseconds. Its weights are seeded random quantized values, so what it
//! writes is noise, but the same noise every run. Its tokenizer is byte-level
//! BPE with no merges: the 256 byte symbols, then the special tokens the
//! prompt names, and the system prompt and the default control line as a
//! token each, so that a prompt is about as short as the real tokenizer makes
//! it and a decode bounded by its length stays short.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::gguf::tests::{MetadataValue, Rng, q4k_bytes, q6k_bytes, write_gguf};

/// The model the fixture is installed as: the manifest's.
pub const MODEL: &str = "s1-mini-q4_k_m";

/// The checkpoint's file name, as the manifest provisions it.
pub const GGUF: &str = "s1-mini-q4_k_m.gguf";

/// The tokens after the 256 bytes, with whether each is special.
fn added() -> [(String, bool); 7] {
    [
        ("<|endoftext|>".to_string(), true),
        ("<|im_start|>".to_string(), true),
        ("<|im_end|>".to_string(), true),
        ("<think>".to_string(), false),
        ("</think>".to_string(), false),
        (crate::prompt::SYSTEM_PROMPT.to_string(), true),
        (crate::prompt::Control::default().line(), true),
    ]
}
const VOCAB: usize = 256 + 7;

// Every matrix's rows hold whole k-quant blocks of 256.
const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 512;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const LAYERS: usize = 2;

/// A backend directory holding the fixture as [`MODEL`], removed when dropped.
pub struct Backend {
    pub dir: PathBuf,
}

impl Backend {
    /// The model's own directory.
    pub fn model_dir(&self) -> PathBuf {
        self.dir.join("models").join(MODEL)
    }

    /// Replace the model's `config.json` with what `edit` makes of it.
    pub fn edit_config(&self, edit: impl FnOnce(&mut Value)) {
        let path = self.model_dir().join("config.json");
        let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        edit(&mut value);
        std::fs::write(path, value.to_string()).unwrap();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A fresh backend directory with the fixture in it.
pub fn backend() -> Backend {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "s1-mini-fixture-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let backend = Backend { dir };
    let model = backend.model_dir();
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), config().to_string()).unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer().to_string()).unwrap();
    write_checkpoint(&model.join(GGUF), &tensor_names());
    backend
}

/// Writes the fixture's checkpoint to `path` with the tensors `names`, every
/// one the fixture knows how to make.
pub fn write_checkpoint(path: &Path, names: &[String]) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let tensors: Vec<_> = names
        .iter()
        .map(|name| {
            let (kind, dims, bytes) = tensor(name, &mut rng);
            (name.clone(), kind, dims, bytes)
        })
        .collect();
    write_gguf(
        path,
        &[("general.architecture", MetadataValue::String("qwen3"))],
        &tensors,
    );
}

/// The fixture's `config.json`, shaped like S1-mini's.
pub fn config() -> Value {
    json!({
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": false,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": true,
        "use_sliding_window": false,
        "vocab_size": VOCAB
    })
}

/// Every tensor name of the checkpoint, in the order llama.cpp writes them.
pub fn tensor_names() -> Vec<String> {
    let mut names: Vec<String> = ["output.weight", "output_norm.weight", "token_embd.weight"]
        .map(String::from)
        .into();
    for i in 0..LAYERS {
        for t in [
            "attn_k",
            "attn_k_norm",
            "attn_norm",
            "attn_output",
            "attn_q",
            "attn_q_norm",
            "attn_v",
            "ffn_down",
            "ffn_gate",
            "ffn_norm",
            "ffn_up",
        ] {
            names.push(format!("blk.{i}.{t}.weight"));
        }
    }
    names
}

/// The ggml type, dims (innermost first) and bytes of the tensor `name`.
fn tensor(name: &str, rng: &mut Rng) -> (u32, Vec<u64>, Vec<u8>) {
    let short = name.rsplit_once(".weight").map_or(name, |(n, _)| n);
    let short = short.split('.').next_back().unwrap_or(short);
    let dims = |inner: usize, outer: usize| vec![inner as u64, outer as u64];
    match short {
        "output" => (14, dims(HIDDEN, VOCAB), q6k(rng, HIDDEN * VOCAB, HIDDEN)),
        // Small next to what the layers add, so the next token turns on the
        // context attended to rather than on the last token alone, which
        // would send the decode round a short cycle.
        "token_embd" => (12, dims(HIDDEN, VOCAB), q4k(rng, HIDDEN * VOCAB, HIDDEN)),
        "output_norm" | "attn_norm" | "ffn_norm" => (
            0,
            vec![HIDDEN as u64],
            [1f32; HIDDEN]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        "attn_q_norm" | "attn_k_norm" => {
            let one = burn::tensor::f16::ONE.to_le_bytes();
            (1, vec![HEAD_DIM as u64], one.repeat(HEAD_DIM))
        }
        "attn_q" => (
            12,
            dims(HIDDEN, HEADS * HEAD_DIM),
            q4k(rng, HIDDEN * HEADS * HEAD_DIM, HIDDEN),
        ),
        "attn_k" => (
            12,
            dims(HIDDEN, KV_HEADS * HEAD_DIM),
            q4k(rng, HIDDEN * KV_HEADS * HEAD_DIM, HIDDEN),
        ),
        "attn_v" => (
            14,
            dims(HIDDEN, KV_HEADS * HEAD_DIM),
            q6k(rng, HIDDEN * KV_HEADS * HEAD_DIM, HIDDEN),
        ),
        "attn_output" => (
            12,
            dims(HEADS * HEAD_DIM, HIDDEN),
            q4k(rng, HEADS * HEAD_DIM * HIDDEN, HEADS * HEAD_DIM),
        ),
        "ffn_gate" | "ffn_up" => (
            12,
            dims(HIDDEN, INTERMEDIATE),
            q4k(rng, HIDDEN * INTERMEDIATE, HIDDEN),
        ),
        "ffn_down" => (
            14,
            dims(INTERMEDIATE, HIDDEN),
            q6k(rng, INTERMEDIATE * HIDDEN, INTERMEDIATE),
        ),
        _ => panic!("the fixture has no tensor {name}"),
    }
}

/// The bound of values scaled by their fan-in, as an initializer would, so
/// the activations stay in range through the layers.
#[allow(clippy::cast_precision_loss)]
fn bound(fan_in: usize) -> f32 {
    (3.0 / fan_in as f32).sqrt()
}

/// `elements` Q4_K values about zero, within [`bound`] of `fan_in`: each
/// sub-block's minimum is about half its range, so `q` of 0–15 centres near
/// 7.5.
fn q4k(rng: &mut Rng, elements: usize, fan_in: usize) -> Vec<u8> {
    let d = 2.0 * bound(fan_in) / (63.0 * 15.0);
    (0..elements / 256)
        .flat_map(|_| {
            let scales: [u8; 8] = std::array::from_fn(|_| rng.byte() & 63);
            let quants: Vec<u8> = (0..256).map(|_| rng.byte() & 0xF).collect();
            q4k_bytes(d, 15.0 * d, scales, scales.map(|s| s / 2), &quants)
        })
        .collect()
}

/// `elements` Q6_K values about zero, within [`bound`] of `fan_in`.
fn q6k(rng: &mut Rng, elements: usize, fan_in: usize) -> Vec<u8> {
    let d = bound(fan_in) / (127.0 * 32.0);
    (0..elements / 256)
        .flat_map(|_| {
            let scales: [i8; 16] = std::array::from_fn(|_| rng.byte().cast_signed());
            let quants: Vec<u8> = (0..256).map(|_| rng.byte() & 63).collect();
            q6k_bytes(d, scales, &quants)
        })
        .collect()
}

/// A byte-level BPE with no merges: every byte its own token.
fn tokenizer() -> Value {
    let vocab: serde_json::Map<String, Value> = byte_symbols()
        .into_iter()
        .enumerate()
        .map(|(id, symbol)| (symbol.to_string(), json!(id)))
        .collect();
    let added: Vec<Value> = added()
        .iter()
        .enumerate()
        .map(|(i, (content, special))| {
            json!({
                "id": 256 + i, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": special
            })
        })
        .collect();
    let byte_level = json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true
    });
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": byte_level,
        "post_processor": null,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": []
        }
    })
}

/// GPT-2's byte-to-symbol table, which byte-level BPE vocabularies are
/// written in: the printable bytes stand for themselves, the rest are moved
/// up past 255.
fn byte_symbols() -> Vec<char> {
    let printable = |b: u32| {
        (u32::from(b'!')..=u32::from(b'~')).contains(&b)
            || (0xa1..=0xac).contains(&b)
            || (0xae..=0xff).contains(&b)
    };
    let mut shifted = 0;
    (0..256u32)
        .map(|b| {
            let c = if printable(b) {
                b
            } else {
                shifted += 1;
                255 + shifted
            };
            char::from_u32(c).unwrap()
        })
        .collect()
}
