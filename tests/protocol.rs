// SPDX-License-Identifier: GPL-3.0-only
//! Mock-daemon protocol test against the real binary with NO model provisioned.
//! Verifies the `/v1` HTTP surface, the pre-load `starting` state, the role
//! refusals, and a graceful load-error path. CPU-only — runs in CI.

mod common;
use common::{Backend, MODEL};
use std::time::Duration;

#[tokio::test]
async fn ping_and_starting_state() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Backend::spawn(dir.path()).await;

    let (code, body) = backend
        .request("GET", "/v1/ping", &[], Vec::new())
        .await
        .unwrap();
    assert_eq!(code, 200);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["message"], "pong");

    let s = backend.status().await;
    assert_eq!(s["state"], "starting", "no load yet → starting");
}

#[tokio::test]
async fn process_before_load_is_not_ready() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Backend::spawn(dir.path()).await;

    let (code, body) = backend.process("so um hello there", &[]).await;
    assert_eq!(code, 409, "{body}");
    assert_eq!(body["message"], "not_ready");
}

#[tokio::test]
async fn transcribe_is_the_wrong_role() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Backend::spawn(dir.path()).await;

    let body = serde_json::to_vec(&serde_json::json!({ "audio_data": [0.0f32] })).unwrap();
    let (code, body) = backend
        .request("POST", "/v1/transcribe", &[], body)
        .await
        .unwrap();
    assert_eq!(code, 400);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["message"], "unsupported_role");
}

#[tokio::test]
async fn cancel_with_nothing_in_flight_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Backend::spawn(dir.path()).await;

    let (code, body) = backend
        .request("POST", "/v1/cancel", &[], Vec::new())
        .await
        .unwrap();
    assert_eq!(code, 409);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["message"], "nothing_in_progress");
}

#[tokio::test]
async fn load_without_weights_reports_error() {
    let dir = tempfile::tempdir().unwrap(); // no models/ subdir
    let backend = Backend::spawn(dir.path()).await;

    assert_eq!(backend.load(MODEL, "cpu").await, 202);

    // Weights dir is missing → the engine load fails → state becomes `error`.
    let s = backend
        .wait_for_state(&["error"], Duration::from_secs(30))
        .await;
    assert_eq!(s["state"], "error");
    assert_eq!(s["model"]["name"], MODEL);
    assert!(
        s.get("reason").and_then(|r| r.as_str()).is_some(),
        "error state must carry a reason: {s}"
    );
}
