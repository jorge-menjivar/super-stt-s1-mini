// SPDX-License-Identifier: GPL-3.0-only
//! Gated real-model E2E. Provision `models/s1-mini-q4_k_m` under
//! SUPER_STT_BACKEND_DIR (the two files `backend.toml` lists for it), then:
//!   just test-e2e <backend-dir> [--features cuda]
//! or, spelled out:
//!   SUPER_STT_TEST_S1_MINI=1 SUPER_STT_BACKEND_DIR=<dir> \
//!   cargo test [--features cuda] --test process_e2e -- --nocapture
//!
//! Pass `SUPER_STT_TEST_DEVICE=cpu` to load on the CPU even in a CUDA build.

#![allow(clippy::doc_markdown)] // env var names in shell-command doc comment

mod common;
use common::{Backend, MODEL};
use std::path::PathBuf;
use std::time::Duration;

/// The model card's own examples, measured on the BF16 weights under the
/// default control line. The Q4_K_M build is what ships, so exact matches are
/// not asserted — the checks are for the transformation the model is for.
const RAW: &str =
    "so um i need to like send the the report by uh friday no wait make that thursday";

fn spawn_ready() -> Option<impl std::future::Future<Output = Backend>> {
    if std::env::var("SUPER_STT_TEST_S1_MINI").is_err() {
        return None; // self-skip unless explicitly enabled
    }
    let backend_dir = PathBuf::from(
        std::env::var("SUPER_STT_BACKEND_DIR")
            .unwrap_or_else(|_| panic!("SUPER_STT_BACKEND_DIR (must contain models/{MODEL})")),
    );
    assert!(
        backend_dir.join("models").join(MODEL).exists(),
        "provision models/{MODEL} under SUPER_STT_BACKEND_DIR first"
    );
    let device = std::env::var("SUPER_STT_TEST_DEVICE").unwrap_or_default();
    Some(async move {
        let backend = Backend::spawn(&backend_dir).await;
        assert_eq!(backend.load(MODEL, &device).await, 202);
        let s = backend
            .wait_for_state(&["ready", "error"], Duration::from_secs(600))
            .await;
        assert_eq!(s["state"], "ready", "load failed: {s}");
        println!("=== loaded on {} ===", s["device"]);
        backend
    })
}

#[tokio::test]
async fn normalizes_the_model_cards_example() {
    let Some(backend) = spawn_ready() else { return };
    let backend = backend.await;

    let (code, body) = backend.process(RAW, &[]).await;
    assert_eq!(code, 200, "process failed: {body}");
    let text = body["text"].as_str().unwrap_or("");
    println!("=== S1-MINI ===\n{RAW}\n→ {text}\n===============");
    assert!(!text.trim().is_empty(), "expected a non-empty result");
    let lower = text.to_lowercase();
    assert!(lower.contains("thursday"), "kept the correction: {text:?}");
    assert!(
        !lower.contains("friday"),
        "dropped the false start: {text:?}"
    );
    assert!(!lower.contains(" um "), "dropped the filler: {text:?}");
    assert!(
        text.chars().next().is_some_and(char::is_uppercase),
        "semi-formal capitalizes: {text:?}"
    );
}

#[tokio::test]
async fn the_control_line_steers_the_register() {
    let Some(backend) = spawn_ready() else { return };
    let backend = backend.await;

    let raw = "hmm im gonna be late theres a cute dog outside i cant just walk past him";
    let (code, casual) = backend.process(raw, &[("styling", "casual")]).await;
    assert_eq!(code, 200, "{casual}");
    let (code, formal) = backend.process(raw, &[("styling", "formal")]).await;
    assert_eq!(code, 200, "{formal}");
    let casual = casual["text"].as_str().unwrap();
    let formal = formal["text"].as_str().unwrap();
    println!("casual → {casual}\nformal → {formal}");
    assert_eq!(casual, casual.to_lowercase(), "casual is all lowercase");
    assert!(
        formal.contains("I am"),
        "formal expands contractions: {formal:?}"
    );
}

#[tokio::test]
async fn filler_only_input_normalizes_to_nothing() {
    let Some(backend) = spawn_ready() else { return };
    let backend = backend.await;

    let (code, body) = backend.process("um", &[]).await;
    assert_eq!(code, 200, "{body}");
    assert_eq!(
        body["text"], "",
        "the model card says this is the correct answer"
    );
}

/// A transcript far over the input budget is chunked, normalized piecewise
/// and rejoined — every piece cleaned, nothing dropped.
#[tokio::test]
async fn a_long_transcript_is_normalized_in_pieces() {
    let Some(backend) = spawn_ready() else { return };
    let backend = backend.await;

    let sentence =
        "um so the next item on the list is number {n} which we should uh discuss on monday. ";
    let long: String = (1..=120)
        .map(|n| sentence.replace("{n}", &n.to_string()))
        .collect();
    let (code, body) = backend.process(&long, &[]).await;
    assert_eq!(code, 200, "{body}");
    let text = body["text"].as_str().unwrap();
    println!(
        "=== {} chars in → {} chars out ===\n{}…\n…{}",
        long.len(),
        text.len(),
        &text[..text.len().min(200)],
        &text[text.len().saturating_sub(200)..]
    );
    assert!(text.contains("120"), "the tail survived chunking: {text:?}");
    assert!(
        !text.to_lowercase().contains(" um "),
        "every piece was cleaned"
    );
}
