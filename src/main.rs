// SPDX-License-Identifier: GPL-3.0-only
//! S1-mini subprocess backend: serves the Super STT `/v1` contract over a
//! pathname Unix socket (`SUPER_STT_BACKEND_SOCKET`) for the **post-processor**
//! role, loading the model from `SUPER_STT_BACKEND_DIR/models/<name>`.
//! Self-contained — no super-stt deps.
//!
//! The daemon hands a finished transcript to `POST /v1/process`; this backend
//! runs it through S1-mini by Superwhisper and returns the cleaned text. It
//! never sees audio.

// doc lint trips on prose like "candle"/"super-stt".
#![allow(clippy::doc_markdown)]

mod chunk;
mod engine;
mod prompt;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::UnixListener;

use engine::Engine;
use prompt::Control;

#[derive(Clone, Copy)]
enum LoadState {
    Starting,
    Loading,
    Ready,
    Error,
}

impl LoadState {
    fn as_str(self) -> &'static str {
        match self {
            LoadState::Starting => "starting",
            LoadState::Loading => "loading",
            LoadState::Ready => "ready",
            LoadState::Error => "error",
        }
    }
}

struct Status {
    state: LoadState,
    model: Option<String>,
    device: Option<String>,
    reason: Option<String>,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            state: LoadState::Starting,
            model: None,
            device: None,
            reason: None,
        }
    }
}

struct AppState {
    backend_dir: PathBuf,
    status: Mutex<Status>,
    engine: Mutex<Option<Engine>>,
    /// Raised by `/v1/cancel` while a request is decoding; the decode loop
    /// polls it per token. Cleared when the next request starts.
    cancel: AtomicBool,
    /// Whether a `/v1/process` is currently decoding, so `/v1/cancel` can tell
    /// "cancelled it" from "nothing to cancel".
    in_flight: AtomicBool,
}

impl AppState {
    fn new(backend_dir: PathBuf) -> Self {
        Self {
            backend_dir,
            status: Mutex::new(Status::default()),
            engine: Mutex::new(None),
            cancel: AtomicBool::new(false),
            in_flight: AtomicBool::new(false),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let socket = std::env::var("SUPER_STT_BACKEND_SOCKET")
        .context("SUPER_STT_BACKEND_SOCKET must be set")?;
    let backend_dir =
        std::env::var("SUPER_STT_BACKEND_DIR").context("SUPER_STT_BACKEND_DIR must be set")?;

    let state = Arc::new(AppState::new(PathBuf::from(backend_dir)));
    let app = router(state);

    if let Some(parent) = std::path::Path::new(&socket).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).with_context(|| format!("bind {socket}"))?;
    log::info!("S1-mini backend serving /v1 on {socket}");

    loop {
        let (stream, _) = listener.accept().await?;
        let app = app.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = TowerToHyperService::new(app);
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                log::debug!("connection ended: {e}");
            }
        });
    }
}

/// Build the `/v1` router. Extracted from `main` so handlers can be exercised
/// in-process by the tests below without spawning the binary.
fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/ping", get(ping))
        .route("/v1/status", get(get_status))
        .route("/v1/load", post(load))
        .route("/v1/process", post(process))
        .route("/v1/transcribe", post(transcribe))
        .route("/v1/cancel", post(cancel))
        .with_state(state)
}

async fn ping() -> Json<Value> {
    Json(json!({ "status": "success", "message": "pong" }))
}

async fn get_status(State(s): State<Arc<AppState>>) -> Json<Value> {
    let st = s.status.lock().unwrap();
    let mut out = json!({ "status": "success", "state": st.state.as_str() });
    if let Some(m) = &st.model {
        out["model"] = json!({ "name": m });
    }
    if let Some(d) = &st.device {
        out["device"] = json!(d);
    }
    if let Some(r) = &st.reason {
        out["reason"] = json!(r);
    }
    Json(out)
}

#[derive(Deserialize)]
struct LoadReq {
    name: String,
    #[serde(default)]
    device: Option<String>,
}

async fn load(State(s): State<Arc<AppState>>, Json(req): Json<LoadReq>) -> impl IntoResponse {
    {
        let mut st = s.status.lock().unwrap();
        // Contract: reject a concurrent load. A model switch is a fresh load
        // after the daemon tears this backend down, so only an in-flight load on
        // this instance trips `already_loading`.
        if matches!(st.state, LoadState::Loading) {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "status": "error", "message": "already_loading" })),
            );
        }
        st.state = LoadState::Loading;
        st.model = Some(req.name.clone());
        st.device = None;
        st.reason = None;
    }
    let dir = s.backend_dir.join("models").join(&req.name);
    let force_cpu = req.device.as_deref() == Some("cpu");
    let s2 = Arc::clone(&s);
    tokio::spawn(async move {
        let res = tokio::task::spawn_blocking(move || Engine::load(&dir, force_cpu)).await;
        match res {
            Ok(Ok(engine)) => {
                let label = engine.device_label().to_string();
                *s2.engine.lock().unwrap() = Some(engine);
                let mut st = s2.status.lock().unwrap();
                st.device = Some(label);
                st.state = LoadState::Ready;
                log::info!("model loaded; ready");
            }
            Ok(Err(e)) => {
                let mut st = s2.status.lock().unwrap();
                st.state = LoadState::Error;
                st.reason = Some(format!("{e:#}"));
                log::error!("model load failed: {e:#}");
            }
            Err(e) => {
                let mut st = s2.status.lock().unwrap();
                st.state = LoadState::Error;
                st.reason = Some(format!("load task panicked: {e}"));
            }
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Loading started" })),
    )
}

#[derive(Deserialize)]
struct ProcessReq {
    #[serde(default)]
    text: Option<String>,
}

/// The control line for a request, from the `x-stt-option-*` headers the
/// daemon injects for each declared `[[options]]` value the user has set.
fn control_from_headers(headers: &HeaderMap) -> Control {
    Control::from_options(|name| {
        headers
            .get(format!("x-stt-option-{name}"))
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    })
}

async fn process(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ProcessReq>,
) -> (StatusCode, Json<Value>) {
    if !matches!(s.status.lock().unwrap().state, LoadState::Ready) {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "status": "error", "message": "not_ready" })),
        );
    }
    // Contract: a missing or blank `text` is a client error, not a processing
    // failure — the daemon never sends an empty transcript.
    let text = req.text.unwrap_or_default();
    if text.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "message": "invalid_text" })),
        );
    }
    let control = control_from_headers(&headers);

    s.cancel.store(false, Ordering::SeqCst);
    s.in_flight.store(true, Ordering::SeqCst);
    let s2 = Arc::clone(&s);
    let result = tokio::task::spawn_blocking(move || {
        let mut guard = s2.engine.lock().unwrap();
        let engine = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("engine not loaded"))?;
        engine.normalize(&text, control, &s2.cancel)
    })
    .await;
    s.in_flight.store(false, Ordering::SeqCst);

    match result {
        Ok(Ok(text)) => (
            StatusCode::OK,
            Json(json!({ "status": "success", "text": text })),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                json!({ "status": "error", "message": "processing_failed", "detail": format!("{e:#}") }),
            ),
        ),
        // A task panic is still a processing failure; the contract documents
        // `processing_failed` for 500, so report that (the panic is in `detail`).
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                json!({ "status": "error", "message": "processing_failed", "detail": format!("panicked: {e}") }),
            ),
        ),
    }
}

/// This backend serves the post-processor role only. Saying so is more useful
/// than a bare 404 if a daemon ever routes audio here.
async fn transcribe() -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "status": "error",
            "message": "unsupported_role",
            "detail": "This backend post-processes transcripts; it does not transcribe audio.",
        })),
    )
}

/// Stop the decode in flight, if there is one. The `/v1/process` it belonged
/// to answers `500 processing_failed`, and the daemon keeps the raw transcript.
async fn cancel(State(s): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    if !s.in_flight.load(Ordering::SeqCst) {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "status": "error", "message": "nothing_in_progress" })),
        );
    }
    s.cancel.store(true, Ordering::SeqCst);
    (
        StatusCode::OK,
        Json(json!({ "status": "success", "message": "Cancelled" })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    use prompt::{Context, Structure, Styling};

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState::new(std::env::temp_dir()))
    }

    async fn json_body(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn post_json(path: &str, body: &Value) -> Request<Body> {
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    #[tokio::test]
    async fn ping_returns_pong() {
        let resp = router(test_state())
            .oneshot(Request::get("/v1/ping").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["message"], "pong");
    }

    #[tokio::test]
    async fn status_is_starting_before_load() {
        let resp = router(test_state())
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(json_body(resp).await["state"], "starting");
    }

    #[test]
    fn load_state_wire_strings() {
        assert_eq!(LoadState::Starting.as_str(), "starting");
        assert_eq!(LoadState::Loading.as_str(), "loading");
        assert_eq!(LoadState::Ready.as_str(), "ready");
        assert_eq!(LoadState::Error.as_str(), "error");
    }

    #[tokio::test]
    async fn status_includes_populated_fields() {
        let state = test_state();
        {
            let mut st = state.status.lock().unwrap();
            st.state = LoadState::Ready;
            st.model = Some("s1-mini-q4_k_m".to_string());
            st.device = Some("cuda".to_string());
            st.reason = Some("recovered".to_string());
        }
        let resp = router(state)
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let v = json_body(resp).await;
        assert_eq!(v["state"], "ready");
        assert_eq!(v["model"]["name"], "s1-mini-q4_k_m");
        assert_eq!(v["device"], "cuda");
        assert_eq!(v["reason"], "recovered");
    }

    #[tokio::test]
    async fn load_rejects_concurrent_load() {
        let state = test_state();
        state.status.lock().unwrap().state = LoadState::Loading;
        let resp = router(state)
            .oneshot(post_json("/v1/load", &json!({ "name": "s1-mini-q4_k_m" })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "already_loading");
    }

    #[tokio::test]
    async fn load_sets_model_name_synchronously() {
        // The handler sets the model name before spawning the load task, and the
        // error path leaves it intact — so it's readable right after the 202.
        let state = test_state();
        let resp = router(Arc::clone(&state))
            .oneshot(post_json(
                "/v1/load",
                &json!({ "name": "s1-mini-q4_k_m", "device": "cpu" }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert_eq!(
            state.status.lock().unwrap().model.as_deref(),
            Some("s1-mini-q4_k_m")
        );
    }

    #[tokio::test]
    async fn load_missing_weights_transitions_to_error() {
        // backend_dir is a temp dir with no models/, so the load fails fast.
        let state = test_state();
        let resp = router(Arc::clone(&state))
            .oneshot(post_json(
                "/v1/load",
                &json!({ "name": "s1-mini-q4_k_m", "device": "cpu" }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        // The load runs in a spawned task; with no weights it must reach `error`.
        for _ in 0..250 {
            if matches!(state.status.lock().unwrap().state, LoadState::Error) {
                let reason = state.status.lock().unwrap().reason.clone().unwrap();
                assert!(reason.contains("tokenizer.json"), "{reason}");
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("load did not reach error state");
    }

    #[tokio::test]
    async fn process_before_ready_conflicts() {
        let resp = router(test_state())
            .oneshot(post_json("/v1/process", &json!({ "text": "hello" })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "not_ready");
    }

    /// Ready state, but no usable `text` → 400 invalid_text (contract), before
    /// the engine is ever touched (so no model is needed to exercise it).
    #[tokio::test]
    async fn process_blank_text_is_invalid() {
        for body in [json!({}), json!({ "text": "" }), json!({ "text": "   " })] {
            let state = test_state();
            state.status.lock().unwrap().state = LoadState::Ready;
            let resp = router(state)
                .oneshot(post_json("/v1/process", &body))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(json_body(resp).await["message"], "invalid_text");
        }
    }

    /// `ready` with no engine behind it cannot happen in the binary, but it is
    /// the cheapest way to drive the failure arm of `process` without weights.
    #[tokio::test]
    async fn process_without_an_engine_is_a_processing_failure() {
        let state = test_state();
        state.status.lock().unwrap().state = LoadState::Ready;
        let resp = router(state)
            .oneshot(post_json("/v1/process", &json!({ "text": "hello" })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let v = json_body(resp).await;
        assert_eq!(v["message"], "processing_failed");
        assert_eq!(v["detail"], "engine not loaded");
    }

    #[tokio::test]
    async fn transcribe_is_refused_as_the_wrong_role() {
        let resp = router(test_state())
            .oneshot(post_json("/v1/transcribe", &json!({ "audio_data": [0.0] })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["message"], "unsupported_role");
    }

    #[tokio::test]
    async fn cancel_with_nothing_in_flight_conflicts() {
        let resp = router(test_state())
            .oneshot(Request::post("/v1/cancel").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "nothing_in_progress");
    }

    #[tokio::test]
    async fn cancel_raises_the_flag_for_a_decode_in_flight() {
        let state = test_state();
        state.in_flight.store(true, Ordering::SeqCst);
        let resp = router(Arc::clone(&state))
            .oneshot(Request::post("/v1/cancel").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["message"], "Cancelled");
        assert!(state.cancel.load(Ordering::SeqCst));
    }

    /// The daemon injects one `x-stt-option-<name>` header per option the user
    /// set; each steers exactly its own axis and an unset one keeps its default.
    #[test]
    fn control_is_read_from_the_option_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-stt-option-styling", "casual".parse().unwrap());
        headers.insert("x-stt-option-structure", "lists".parse().unwrap());
        assert_eq!(
            control_from_headers(&headers),
            Control {
                styling: Styling::Casual,
                structure: Structure::Lists,
                context: Context::General,
            }
        );
        assert_eq!(control_from_headers(&HeaderMap::new()), Control::default());
    }
}
