// SPDX-License-Identifier: GPL-3.0-only
//! S1-mini subprocess backend: serves the Super STT `/v1` contract over a
//! pathname Unix socket (`SUPER_STT_BACKEND_SOCKET`) for the **post-processor**
//! role, loading the model from `SUPER_STT_BACKEND_DIR/models/<name>`.
//! Self-contained — no super-stt deps.
//!
//! The daemon hands a finished transcript to `POST /v1/process`; this backend
//! runs it through S1-mini by Superwhisper and returns the cleaned text. It
//! never sees audio.
//!
//! The model runs on Burn, whose GPU kernels CubeCL compiles at runtime and
//! keeps in `SUPER_STT_BACKEND_CACHE_DIR` — the one writable directory a
//! daemon new enough to grant it provides.

// doc lint trips on prose like "candle"/"super-stt".
#![allow(clippy::doc_markdown)]

mod chunk;
mod engine;
#[cfg(test)]
mod fixture;
mod gguf;
#[cfg(test)]
mod manifest_probe;
mod model_thread;
mod progress;
mod prompt;
mod qwen3;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

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

use engine::{Engine, MODELS};
use model_thread::ModelThread;
use progress::{Phase, Report};
use prompt::Control;

#[derive(Clone, Copy)]
enum LoadState {
    Starting,
    Preparing,
    Prepared,
    Loading,
    Ready,
    Error,
}

impl LoadState {
    fn as_str(self) -> &'static str {
        match self {
            LoadState::Starting => "starting",
            LoadState::Preparing => "preparing",
            LoadState::Prepared => "prepared",
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
    /// How far a prepare or a load has got; reported only while `state` is
    /// `preparing` or `loading`.
    load: Report,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            state: LoadState::Starting,
            model: None,
            device: None,
            reason: None,
            load: Report::default(),
        }
    }
}

struct AppState {
    backend_dir: PathBuf,
    status: Mutex<Status>,
    /// The model, which lives on a thread of its own because it is not `Send`.
    /// Every use of it is a job sent there, and jobs run one at a time, in
    /// order: a load sent during a prepare waits for it, and a request sent
    /// during a load waits for that.
    model: ModelThread,
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
            model: ModelThread::spawn(),
            cancel: AtomicBool::new(false),
            in_flight: AtomicBool::new(false),
        }
    }

    /// The status, whatever a panicking holder left it as: it is plain data
    /// with no invariant to repair.
    fn status(&self) -> std::sync::MutexGuard<'_, Status> {
        self.status.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stop any request and free the model on its thread, so the device has
    /// nothing left to do when the process exits and the driver tears itself
    /// down. Gives up after [`SHUTDOWN_WAIT`], on a prepare or a load still
    /// running.
    async fn shut_down(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        let released = self.model.run(|slot| {
            if let Some(engine) = slot.take() {
                engine.release();
            }
        });
        if tokio::time::timeout(SHUTDOWN_WAIT, released).await.is_err() {
            log::warn!("the model thread is busy; exiting without freeing the model");
        }
    }
}

/// How long a shutdown waits for the model thread.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(5);

/// Names the one writable directory the sandbox grants for keeping things
/// across runs, where the compiled kernels go.
const ENV_CACHE_DIR: &str = "SUPER_STT_BACKEND_CACHE_DIR";

/// Where the compiled kernels go.
///
/// The granted directory when there is one. Otherwise a directory under the
/// unit's private `/tmp`, which is writable but dies with the process, so a
/// GPU build compiles and tunes its kernels again on every load, and every
/// load reports itself as an initial setup.
fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ENV_CACHE_DIR) {
        log::info!("keeping compiled kernels in {}", Path::new(&dir).display());
        return PathBuf::from(dir);
    }
    let dir = std::env::temp_dir().join("s1-mini-cache");
    log::warn!(
        "{ENV_CACHE_DIR} is not set, so compiled GPU kernels live in {} and are rebuilt on \
         every start; a daemon that grants a cache directory makes them persist",
        dir.display()
    );
    dir
}

fn main() -> anyhow::Result<()> {
    // CubeCL's ROCm compiler logs its whole IR after every pass at `info`,
    // and the daemon runs backends at `info`. So it starts at `warn`, which
    // `RUST_LOG` can still raise by naming it.
    env_logger::Builder::new()
        .filter_module("pliron", log::LevelFilter::Warn)
        .parse_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    // Before anything touches a device: the kernel cache's location is frozen
    // the first time CubeCL reads its configuration, and the driver's own
    // cache the first time CUDA initializes.
    let cache = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache) {
        log::warn!("creating {}: {e}", cache.display());
    }
    // The NVIDIA driver keeps its PTX-to-SASS translations under
    // `$HOME/.nv`, read-only in the sandbox. Only when nothing chose a place:
    // the daemon's own choice wins.
    if std::env::var_os("CUDA_CACHE_PATH").is_none() {
        // Safety: no other thread exists yet — the runtime is built below, and
        // nothing before this line spawns one.
        unsafe { std::env::set_var("CUDA_CACHE_PATH", cache.join("nv")) };
    }
    engine::configure_kernel_cache(&cache);

    let backend_dir =
        std::env::var("SUPER_STT_BACKEND_DIR").context("SUPER_STT_BACKEND_DIR must be set")?;

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the async runtime")?
        .block_on(serve(PathBuf::from(backend_dir)))
}

async fn serve(backend_dir: PathBuf) -> anyhow::Result<()> {
    let socket = std::env::var("SUPER_STT_BACKEND_SOCKET")
        .context("SUPER_STT_BACKEND_SOCKET must be set")?;
    // Before the socket exists: the daemon may send SIGTERM as soon as it
    // sees it, and one that lands before the handler kills the process
    // outright, leaving the socket behind.
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("installing the SIGTERM handler")?;

    let state = Arc::new(AppState::new(backend_dir));
    let app = router(Arc::clone(&state));

    if let Some(parent) = Path::new(&socket).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).with_context(|| format!("bind {socket}"))?;
    log::info!("S1-mini backend serving /v1 on {socket}");

    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(pair) => pair,
                Err(e) => {
                    log::error!("accept failed: {e}");
                    continue;
                }
            },
            // The daemon stops a backend with SIGTERM: close the socket and go.
            _ = terminate.recv() => {
                log::info!("SIGTERM; shutting down");
                let _ = std::fs::remove_file(&socket);
                state.shut_down().await;
                return Ok(());
            }
        };
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
        .route("/v1/prepare", post(prepare))
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
    let st = s.status();
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
    if matches!(st.state, LoadState::Preparing | LoadState::Loading) {
        if let Some(phase) = st.load.phase {
            out["phase"] = json!(phase.as_str());
        }
        if let Some(step) = st.load.step {
            out["step"] = json!(step.as_str());
        }
        if let Some(progress) = st.load.progress {
            out["progress"] = json!(progress);
        }
    }
    Json(out)
}

#[derive(Deserialize)]
struct LoadReq {
    name: String,
    #[serde(default)]
    device: Option<String>,
}

fn invalid_model() -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "status": "error", "message": "invalid_model" })),
    )
}

fn thread_gone() -> (StatusCode, Json<Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(
            json!({ "status": "error", "message": "load_failed", "detail": "the model thread is gone" }),
        ),
    )
}

/// Build the kernels a load of `name` will need from the model's small files,
/// while its weights are still downloading.
///
/// It runs on the model thread, so a load sent meanwhile waits for it. It
/// never fails that load: a prepare that errors or panics is logged and goes
/// back to `starting`, and the load then builds the kernels itself.
async fn prepare(State(s): State<Arc<AppState>>, Json(req): Json<LoadReq>) -> impl IntoResponse {
    // Contract: a model this backend does not declare is `invalid_model`.
    if !MODELS.contains(&req.name.as_str()) {
        return invalid_model();
    }
    let accepted = (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Preparing started" })),
    );
    {
        let mut st = s.status();
        match st.state {
            LoadState::Loading | LoadState::Ready => {
                return (
                    StatusCode::CONFLICT,
                    Json(json!({ "status": "error", "message": "already_loading" })),
                );
            }
            LoadState::Preparing | LoadState::Prepared => return accepted,
            LoadState::Starting | LoadState::Error => {}
        }
        st.model = Some(req.name.clone());
        st.device = None;
        st.reason = None;
        st.load = Report::default();
        if Engine::kernels_built(&req.name) {
            st.state = LoadState::Prepared;
            return accepted;
        }
        st.state = LoadState::Preparing;
    }
    let s2 = Arc::clone(&s);
    let queued = s
        .model
        .submit(move |_| run_prepare(&s2, &req.name, req.device.as_deref()));
    if !queued {
        s.status().state = LoadState::Starting;
        return thread_gone();
    }
    accepted
}

/// Runs on the model thread: a prepare, and the state it ends in.
fn run_prepare(s: &AppState, name: &str, device: Option<&str>) {
    let report = |load: Report| {
        let mut st = s.status();
        if matches!(st.state, LoadState::Preparing | LoadState::Loading) {
            st.load = load;
        }
    };
    let built = engine::catching(|| Engine::build_kernels(&s.backend_dir, name, device, &report))
        .unwrap_or_else(|panic| Err(anyhow::anyhow!("panicked: {panic}")));
    let built = match built {
        Ok(()) => {
            log::info!("kernels built; prepared");
            true
        }
        Err(e) => {
            log::warn!("building the kernels ahead of the load failed: {e:#}");
            false
        }
    };
    let mut st = s.status();
    // A load that arrived meanwhile owns the status now.
    if matches!(st.state, LoadState::Preparing) {
        st.state = if built {
            LoadState::Prepared
        } else {
            LoadState::Starting
        };
        st.load = Report::default();
    }
}

async fn load(State(s): State<Arc<AppState>>, Json(req): Json<LoadReq>) -> impl IntoResponse {
    // Contract: a model this backend does not declare is `invalid_model`.
    if !MODELS.contains(&req.name.as_str()) {
        return invalid_model();
    }
    let preparing = {
        let mut st = s.status();
        // Contract: reject a concurrent load. A model switch is a fresh load
        // after the daemon tears this backend down, so only an in-flight load on
        // this instance trips `already_loading`.
        if matches!(st.state, LoadState::Loading) {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "status": "error", "message": "already_loading" })),
            );
        }
        // A prepare still building goes on reporting its progress, as this
        // load's, and the whole load then counts as the initial setup it is.
        let preparing = matches!(st.state, LoadState::Preparing);
        st.state = LoadState::Loading;
        st.model = Some(req.name.clone());
        st.device = None;
        st.reason = None;
        if !preparing {
            st.load = Report::default();
        }
        preparing
    };
    let s2 = Arc::clone(&s);
    let queued = s.model.submit(move |slot| {
        let report = |mut load: Report| {
            if preparing {
                load.phase = Some(Phase::InitialSetup);
            }
            let mut st = s2.status();
            if matches!(st.state, LoadState::Loading) {
                st.load = load;
            }
        };
        // A panic here — an allocation the device refused while the weights
        // were mapped, or a lost device while the old model's memory was
        // freed, say — would otherwise be caught by the model thread and
        // leave the status saying `loading` for good.
        let loaded = engine::catching(|| {
            // The previous model's memory goes first, so it is back before
            // the new one asks for any.
            *slot = None;
            Engine::load(&s2.backend_dir, &req.name, req.device.as_deref(), &report)
        })
        .map_err(|panic| format!("load task panicked: {panic}"))
        .and_then(|loaded| loaded.map_err(|e| e.to_string()));
        let mut st = s2.status();
        match loaded {
            Ok(engine) => {
                st.device = Some(engine.device_name().to_string());
                st.state = LoadState::Ready;
                *slot = Some(engine);
                log::info!("model loaded; ready");
            }
            Err(reason) => {
                log::error!("model load failed: {reason}");
                st.state = LoadState::Error;
                st.reason = Some(reason);
            }
        }
    });
    if !queued {
        let mut st = s.status();
        st.state = LoadState::Error;
        st.reason = Some("the model thread is gone".to_string());
        return thread_gone();
    }
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

/// Clears the in-flight flag on every way out of a request, a panic included.
struct Idle<'a>(&'a AtomicBool);

impl Drop for Idle<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

async fn process(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ProcessReq>,
) -> (StatusCode, Json<Value>) {
    if !matches!(s.status().state, LoadState::Ready) {
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

    let s2 = Arc::clone(&s);
    let result = s
        .model
        .run(move |slot| {
            let Some(engine) = slot.as_mut() else {
                anyhow::bail!("engine not loaded");
            };
            s2.cancel.store(false, Ordering::SeqCst);
            s2.in_flight.store(true, Ordering::SeqCst);
            let _idle = Idle(&s2.in_flight);
            engine.normalize(&text, control, &s2.cancel)
        })
        .await;

    match result {
        Some(Ok(text)) => (
            StatusCode::OK,
            Json(json!({ "status": "success", "text": text })),
        ),
        Some(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                json!({ "status": "error", "message": "processing_failed", "detail": format!("{e:#}") }),
            ),
        ),
        // A panic is still a processing failure; the contract documents
        // `processing_failed` for 500. The model thread drops the engine after
        // one, and the next load builds a fresh one.
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "status": "error",
                "message": "processing_failed",
                "detail": "the request panicked or the model thread is gone",
            })),
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

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    use progress::Step;
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

    fn get(path: &str) -> Request<Body> {
        Request::get(path).body(Body::empty()).unwrap()
    }

    /// Waits up to ten minutes: a GPU build's warm-up builds its kernels
    /// during the load, which took over a minute for the fixture with the
    /// other tests compiling theirs on the same card.
    async fn wait_for(state: &AppState, done: impl Fn(LoadState) -> bool) -> LoadState {
        for _ in 0..60_000 {
            let now = state.status().state;
            if done(now) {
                return now;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the state never got there");
    }

    #[tokio::test]
    async fn ping_returns_pong() {
        let resp = router(test_state()).oneshot(get("/v1/ping")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["message"], "pong");
    }

    #[tokio::test]
    async fn status_is_starting_before_load() {
        let resp = router(test_state())
            .oneshot(get("/v1/status"))
            .await
            .unwrap();
        assert_eq!(json_body(resp).await["state"], "starting");
    }

    #[test]
    fn load_state_wire_strings() {
        assert_eq!(LoadState::Starting.as_str(), "starting");
        assert_eq!(LoadState::Preparing.as_str(), "preparing");
        assert_eq!(LoadState::Prepared.as_str(), "prepared");
        assert_eq!(LoadState::Loading.as_str(), "loading");
        assert_eq!(LoadState::Ready.as_str(), "ready");
        assert_eq!(LoadState::Error.as_str(), "error");
    }

    #[tokio::test]
    async fn status_includes_populated_fields() {
        let state = test_state();
        {
            let mut st = state.status();
            st.state = LoadState::Ready;
            st.model = Some("s1-mini-q4_k_m".to_string());
            st.device = Some("cuda".to_string());
            st.reason = Some("recovered".to_string());
        }
        let resp = router(state).oneshot(get("/v1/status")).await.unwrap();
        let v = json_body(resp).await;
        assert_eq!(v["state"], "ready");
        assert_eq!(v["model"]["name"], "s1-mini-q4_k_m");
        assert_eq!(v["device"], "cuda");
        assert_eq!(v["reason"], "recovered");
    }

    #[tokio::test]
    async fn status_reports_load_progress_only_while_preparing_or_loading() {
        let state = test_state();
        {
            let mut st = state.status();
            st.state = LoadState::Loading;
            st.load = Report {
                phase: Some(Phase::InitialSetup),
                step: Some(Step::BuildingKernels),
                progress: Some(0.5),
            };
        }
        let v = json_body(
            router(Arc::clone(&state))
                .oneshot(get("/v1/status"))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(v["state"], "loading");
        assert_eq!(v["phase"], "initial_setup");
        assert_eq!(v["step"], "building_kernels");
        assert_eq!(v["progress"], 0.5);

        state.status().state = LoadState::Preparing;
        let v = json_body(
            router(Arc::clone(&state))
                .oneshot(get("/v1/status"))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(v["state"], "preparing");
        assert_eq!(v["step"], "building_kernels");
        assert_eq!(v["progress"], 0.5);

        for done in [LoadState::Prepared, LoadState::Ready] {
            state.status().state = done;
            let v = json_body(
                router(Arc::clone(&state))
                    .oneshot(get("/v1/status"))
                    .await
                    .unwrap(),
            )
            .await;
            for field in ["phase", "step", "progress"] {
                assert!(v.get(field).is_none(), "{field} outside a load: {v}");
            }
        }
    }

    #[tokio::test]
    async fn an_undeclared_model_is_invalid() {
        for route in ["/v1/load", "/v1/prepare"] {
            for name in ["whisper-tiny", "../s1-mini-q4_k_m"] {
                let state = test_state();
                let resp = router(Arc::clone(&state))
                    .oneshot(post_json(route, &json!({ "name": name })))
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{route} {name}");
                assert_eq!(json_body(resp).await["message"], "invalid_model");
                assert!(matches!(state.status().state, LoadState::Starting));
            }
        }
    }

    #[tokio::test]
    async fn load_rejects_concurrent_load() {
        let state = test_state();
        state.status().state = LoadState::Loading;
        let resp = router(state)
            .oneshot(post_json("/v1/load", &json!({ "name": "s1-mini-q4_k_m" })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "already_loading");
    }

    #[tokio::test]
    async fn load_sets_model_name_synchronously() {
        // The handler sets the model name before queueing the load, and the
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
        assert_eq!(state.status().model.as_deref(), Some("s1-mini-q4_k_m"));
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
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Loading)).await;
        assert!(matches!(ended, LoadState::Error));
        let reason = state.status().reason.clone().unwrap();
        assert!(reason.contains("tokenizer.json"), "{reason}");
    }

    #[tokio::test]
    async fn prepare_is_refused_once_a_load_has_begun() {
        for begun in [LoadState::Loading, LoadState::Ready] {
            let state = test_state();
            state.status().state = begun;
            let resp = router(state)
                .oneshot(post_json(
                    "/v1/prepare",
                    &json!({ "name": "s1-mini-q4_k_m" }),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::CONFLICT);
            assert_eq!(json_body(resp).await["message"], "already_loading");
        }
    }

    #[tokio::test]
    async fn a_second_prepare_is_left_alone() {
        for begun in [LoadState::Preparing, LoadState::Prepared] {
            let state = test_state();
            state.status().state = begun;
            let resp = router(Arc::clone(&state))
                .oneshot(post_json(
                    "/v1/prepare",
                    &json!({ "name": "s1-mini-q4_k_m" }),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::ACCEPTED);
            assert_eq!(json_body(resp).await["message"], "Preparing started");
            assert_eq!(state.status().state.as_str(), begun.as_str());
        }
    }

    /// Without its files the build fails, and the state goes back to
    /// `starting` with no reason: the load will build the kernels itself. A
    /// CPU build has no kernels to build and is prepared at once.
    #[tokio::test]
    async fn a_prepare_without_its_files_goes_back_to_starting() {
        let state = test_state();
        let resp = router(Arc::clone(&state))
            .oneshot(post_json(
                "/v1/prepare",
                &json!({ "name": "s1-mini-q4_k_m" }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Preparing)).await;
        if Engine::kernels_built("s1-mini-q4_k_m") {
            assert!(matches!(ended, LoadState::Prepared));
        } else {
            assert!(matches!(ended, LoadState::Starting));
            assert!(state.status().reason.is_none());
        }
    }

    #[tokio::test]
    async fn a_load_during_a_prepare_waits_for_it_and_keeps_its_progress() {
        let state = test_state();
        {
            let mut st = state.status();
            st.state = LoadState::Preparing;
            st.load = Report {
                phase: Some(Phase::InitialSetup),
                step: Some(Step::BuildingKernels),
                progress: Some(0.5),
            };
        }
        // Holds the model thread, as a prepare still building does, then ends
        // the way a prepare without its files does.
        let (release, held) = std::sync::mpsc::channel::<()>();
        let preparing = Arc::clone(&state);
        assert!(state.model.submit(move |_| {
            held.recv().unwrap();
            run_prepare(&preparing, "s1-mini-q4_k_m", None);
        }));
        let app = router(Arc::clone(&state));
        let resp = app
            .clone()
            .oneshot(post_json("/v1/load", &json!({ "name": "s1-mini-q4_k_m" })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let v = json_body(app.oneshot(get("/v1/status")).await.unwrap()).await;
        assert_eq!(v["state"], "loading");
        assert_eq!(v["phase"], "initial_setup");
        assert_eq!(v["step"], "building_kernels");
        assert_eq!(v["progress"], 0.5);
        release.send(()).unwrap();
        // The load fails on the missing weights, and the prepare's ending
        // does not overwrite that.
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Loading)).await;
        assert!(matches!(ended, LoadState::Error));
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
            state.status().state = LoadState::Ready;
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
        state.status().state = LoadState::Ready;
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

    /// A router over a backend directory holding the fixture, loaded.
    async fn loaded(backend: &fixture::Backend) -> (Router, Arc<AppState>) {
        let state = Arc::new(AppState::new(backend.dir.clone()));
        let app = router(Arc::clone(&state));
        let resp = app
            .clone()
            .oneshot(post_json("/v1/load", &json!({ "name": fixture::MODEL })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Loading)).await;
        assert!(
            matches!(ended, LoadState::Ready),
            "{:?}",
            state.status().reason
        );
        (app, state)
    }

    #[tokio::test]
    async fn a_loaded_model_processes_a_transcript() {
        let backend = fixture::backend();
        let (app, state) = loaded(&backend).await;
        let v = json_body(app.clone().oneshot(get("/v1/status")).await.unwrap()).await;
        assert_eq!(v["state"], "ready");
        assert_eq!(v["device"], engine::BUILT_FOR);
        assert_eq!(v["model"]["name"], fixture::MODEL);
        let resp = app
            .oneshot(
                Request::post("/v1/process")
                    .header("content-type", "application/json")
                    .header("x-stt-option-styling", "casual")
                    .body(Body::from(r#"{"text": "so um hello there"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = json_body(resp).await;
        assert_eq!(v["status"], "success");
        assert!(v["text"].is_string(), "{v}");
        assert!(!state.in_flight.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_shutdown_frees_the_model() {
        let backend = fixture::backend();
        let (_, state) = loaded(&backend).await;
        state.shut_down().await;
        assert_eq!(state.model.run(|slot| slot.is_none()).await, Some(true));
    }

    /// Prepared at once on the CPU, which builds no kernels; on a GPU, built
    /// from the fixture's small files.
    #[tokio::test]
    async fn a_prepared_model_loads_and_processes() {
        let backend = fixture::backend();
        let state = Arc::new(AppState::new(backend.dir.clone()));
        let app = router(Arc::clone(&state));
        let req = json!({ "name": fixture::MODEL });
        let resp = app
            .clone()
            .oneshot(post_json("/v1/prepare", &req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Preparing)).await;
        assert!(matches!(ended, LoadState::Prepared));
        let resp = app
            .clone()
            .oneshot(post_json("/v1/load", &req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Loading)).await;
        assert!(
            matches!(ended, LoadState::Ready),
            "{:?}",
            state.status().reason
        );
        let resp = app
            .oneshot(post_json("/v1/process", &json!({ "text": "hello" })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
