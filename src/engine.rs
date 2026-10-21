// SPDX-License-Identifier: GPL-3.0-only
//! The S1-mini session the `/v1` handlers drive: which device to run on,
//! where the files are, how a transcript becomes prompts, and how the tokens
//! that come back become text. The model itself is in [`crate::qwen3`].
//!
//! S1-mini is a fine-tune of Qwen3-0.6B. [`Engine::normalize`] builds the
//! trained prompt around a transcript, decodes greedily until the end-of-turn
//! token, and returns the text — chunking a transcript that overflows the
//! model's input budget first (see [`crate::chunk`]).
//!
//! Everything here is synchronous and compute-bound, and the model is not
//! `Send`, so it lives on [`crate::model_thread`] and the handlers send it
//! jobs.

use std::fmt::Write as _;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result, anyhow, bail};
use burn::prelude::Device;
use burn::tensor::DType;
use tokenizers::Tokenizer;

use crate::chunk;
use crate::gguf::Gguf;
use crate::progress::{Finish, Measure, Phase, Report, Step, Tracker};
use crate::prompt::{Control, format_prompt};
use crate::qwen3::config::TextConfig;
use crate::qwen3::model::Qwen3;

/// The models `backend.toml` declares, the only names a load or a prepare
/// accepts. A test holds this list to the manifest's.
pub const MODELS: &[&str] = &["s1-mini-q4_k_m"];

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

/// Prompt lengths, in tokens, the warm-up decodes after on a GPU.
///
/// CubeCL compiles and tunes a kernel per shape, and the model buckets its
/// shapes so that few exist: the prefill runs a power of two of prompt
/// positions, and each decode step a power of two of cache positions from 256
/// (see [`crate::qwen3::model`]). A piece's prompt is at most about 840
/// tokens, [`INPUT_TOKEN_BUDGET`] and the scaffolding, and it decodes at most
/// [`MAX_NEW_TOKENS`] after it, so every request lands in prefills of 128 to
/// 1024 and windows of 256 to 2048, which these rungs cover:
///
/// | prompt | prefill | decode windows |
/// | -----: | ------: | -------------: |
/// |   1022 |    1024 |    1024, 2048  |
/// |    600 |    1024 |          1024  |
/// |    400 |     512 |           512  |
/// |    200 |     256 |           256  |
/// |    100 |     128 |           256  |
///
/// The longest goes first: its last step grows the caches to 2048 positions,
/// which drops the steps captured over the smaller ones, and the rungs after
/// it capture theirs over the caches every request then uses.
const WARM_UP_PROMPTS: [usize; 5] = [1022, 600, 400, 200, 100];

/// The one rung a CPU build warms, which compiles nothing: it only proves the
/// whole path runs before `ready` says so.
const CPU_WARM_UP_PROMPTS: [usize; 1] = [100];

/// Tokens each warm-up rung decodes: enough for three steps, which reaches
/// the next window from 1022. Only the shapes matter, not the tokens.
const WARM_UP_TOKENS: usize = 4;

/// Decoder layers [`Engine::build_kernels`] keeps: two, so that what runs
/// between one layer and the next is built too.
const BUILD_LAYERS: usize = 2;

/// The prompt lengths this build's warm-up decodes after.
fn warm_up_prompts() -> &'static [usize] {
    if ON_GPU {
        &WARM_UP_PROMPTS
    } else {
        &CPU_WARM_UP_PROMPTS
    }
}

/// The accelerator this build was compiled for, as the manifest names it.
///
/// One build serves one accelerator: Burn's backends are cargo features, and
/// the asset that carries this binary declares the matching `accel`.
pub const BUILT_FOR: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "rocm") {
    "rocm"
} else if cfg!(feature = "vulkan") {
    "vulkan"
} else if cfg!(feature = "metal") {
    "metal"
} else {
    "cpu"
};

/// Whether [`BUILT_FOR`] is a GPU, which decides the dtype and the warm-up.
const ON_GPU: bool = cfg!(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
));

/// The device this build runs on, and the name `GET /v1/status` reports.
///
/// The daemon sends the accelerator it resolved for the installed asset, or
/// nothing. A build has exactly one backend compiled in, so the request is a
/// cross-check rather than a choice, and a mismatch is worth a line in the log
/// because it means the wrong asset was installed for the host.
#[must_use]
pub fn select_device(requested: Option<&str>) -> (Device, &'static str) {
    if let Some(d) = requested.map(str::trim).filter(|s| !s.is_empty())
        && !d.eq_ignore_ascii_case(BUILT_FOR)
    {
        log::warn!(
            "the daemon asked for {d:?} but this build only has {BUILT_FOR}; using {BUILT_FOR}"
        );
    }
    // One arm per backend, in the order a build that somehow enabled several
    // would prefer them: cargo features are additive, so the arms have to
    // exclude each other by hand.
    #[cfg(feature = "cuda")]
    return (Device::cuda(0), BUILT_FOR);
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    return (Device::rocm(0), BUILT_FOR);
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
    return (
        Device::vulkan(burn::prelude::DeviceKind::DefaultDevice),
        BUILT_FOR,
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        feature = "metal"
    ))]
    return (
        Device::metal(burn::prelude::DeviceKind::DefaultDevice),
        BUILT_FOR,
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        feature = "flex"
    ))]
    return (Device::flex(), BUILT_FOR);
    #[cfg(not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "metal",
        feature = "flex"
    )))]
    (Device::default(), BUILT_FOR)
}

/// The type the model computes in on `device`: f16 on every GPU, and f32 on
/// the CPU or a GPU without f16.
///
/// The weights are Q4_K and Q6_K dequantized, a few significant bits each, so
/// f16 holds them closely; it halves the weights and their bandwidth against
/// f32. bf16 would give up three more mantissa bits for range the
/// dequantized weights do not need, and it is not a type every CubeCL backend
/// compiles, whatever the device reports: the Voxtral backend's notes have
/// the details. On the CPU a half-width type is slower than f32 rather than
/// faster.
fn compute_dtype(device: &Device) -> DType {
    if ON_GPU && device.supports_dtype(DType::F16) {
        DType::F16
    } else {
        DType::F32
    }
}

// What a cold warm-up wrote to the kernel cache, CUDA and Vulkan on an RTX
// 3090 (see `docs/port-notes.md`). ROCm and Metal have not been measured and
// use Vulkan's count.
const CUDA_CACHE_ENTRIES: u64 = 1131;
const VULKAN_CACHE_ENTRIES: u64 = 584;
const ROCM_CACHE_ENTRIES: u64 = VULKAN_CACHE_ENTRIES;
const METAL_CACHE_ENTRIES: u64 = VULKAN_CACHE_ENTRIES;

/// The entries a first load's warm-up writes to the kernel cache, which is
/// what its `building_kernels` progress is measured against: tuning results
/// and compiled kernels both. Only the pace of the bar rides on it; past the
/// estimate it slows down short of the end rather than stopping (see
/// [`crate::progress`]).
const WARM_UP_CACHE_ENTRIES: u64 = if cfg!(feature = "cuda") {
    CUDA_CACHE_ENTRIES
} else if cfg!(feature = "rocm") {
    ROCM_CACHE_ENTRIES
} else if cfg!(feature = "metal") {
    METAL_CACHE_ENTRIES
} else {
    VULKAN_CACHE_ENTRIES
};

/// Where CubeCL keeps this backend's kernels, once [`configure_kernel_cache`]
/// has said.
static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Point CubeCL's kernel cache at `cache_dir`.
///
/// CubeCL compiles every kernel it meets at runtime and keeps them on disk so
/// only the first run of a build pays. Left to itself it would write under
/// `$HOME`, which the sandbox mounts read-only, so it would recompile on every
/// load.
///
/// Must run before the first device is created, because the configuration is
/// frozen the first time anything reads it.
pub fn configure_kernel_cache(cache_dir: &Path) {
    use burn::cubecl::config::cache::CacheConfig;
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    config.environment.path = CacheConfig::Directory(cache_dir.to_path_buf());
    let _ = CACHE_DIR.set(cache_dir.to_path_buf());
    // `false` means something already read the configuration and this call is
    // too late to matter. Nothing in this backend touches a device before
    // `main` calls this, so it is a guard rather than a case to handle.
    if !CubeClRuntimeConfig::try_set(config) {
        log::warn!("the CubeCL configuration was already read; the kernel cache keeps its default");
    }
}

/// Where a load records that `model` finished a warm-up with this build, so
/// that the next load of it is not an initial setup. It sits beside the
/// kernels it vouches for, so clearing the cache clears it too.
///
/// Named by the binary's build ID, which is what CubeCL keys its compiled
/// kernels on: any rebuild, a release that keeps its version included,
/// recompiles all of them, and should say so. Without a build ID the crate
/// version stands in; every release asset has one, the GNU build ID on Linux
/// and `LC_UUID` on macOS.
fn warm_marker(model: &str) -> Option<PathBuf> {
    let build = buildid::build_id().map_or_else(
        || env!("CARGO_PKG_VERSION").to_string(),
        |id| {
            id.iter().fold(String::new(), |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            })
        },
    );
    CACHE_DIR.get().map(|dir| {
        dir.join("s1-mini-warm")
            .join(format!("{model}-{BUILT_FOR}-{build}"))
    })
}

/// Writes `marker`, if there is one, recording a finished warm-up.
fn mark_warm(marker: Option<&Path>) {
    let Some(marker) = marker else {
        return;
    };
    let written = marker
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(marker, b""));
    if let Err(e) = written {
        log::warn!("could not mark {} warm: {e}", marker.display());
    }
}

/// Why a load failed.
#[derive(Debug)]
pub enum LoadError {
    /// The build's accelerator could not be initialized: no driver, no device.
    DeviceUnavailable(String),
    /// Anything else: a missing file, a checkpoint this port cannot read.
    Failed(anyhow::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceUnavailable(m) => write!(f, "the {BUILT_FOR} device is unavailable: {m}"),
            Self::Failed(e) => write!(f, "{e:#}"),
        }
    }
}

impl From<anyhow::Error> for LoadError {
    fn from(e: anyhow::Error) -> Self {
        Self::Failed(e)
    }
}

/// Run `f` and turn a panic into an error: CubeCL panics rather than returns
/// when a runtime cannot find its driver or device, or a kernel will not
/// compile.
pub(crate) fn catching<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|panic| {
        panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "panicked".to_string())
    })
}

/// A loaded S1-mini ready to normalize transcripts.
pub struct Engine {
    model: Qwen3,
    tokenizer: Tokenizer,
    /// Token ids that end the assistant turn. Qwen3 marks it with
    /// `<|im_end|>`; `<|endoftext|>` is the base model's generic stop and is
    /// honored too, so a decode that reaches either halts.
    stop_tokens: Vec<u32>,
    /// The most tokens a piece decodes, [`MAX_NEW_TOKENS`] but in the tests,
    /// whose model decodes noise and never ends a turn of its own.
    max_new_tokens: usize,
    device_name: &'static str,
}

// The tokenizer has no `Debug` and the model little worth printing.
impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("model", &self.model)
            .field("stop_tokens", &self.stop_tokens)
            .field("device_name", &self.device_name)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Load `model_name` from `backend_dir/models/<name>`, then warm it up,
    /// reporting how far it has got into `report` as it goes.
    ///
    /// # Errors
    /// [`LoadError::DeviceUnavailable`] if the accelerator cannot be
    /// initialized, [`LoadError::Failed`] for anything else.
    pub fn load(
        backend_dir: &Path,
        model_name: &str,
        device: Option<&str>,
        report: &(dyn Fn(Report) + Sync),
    ) -> Result<Self, LoadError> {
        // A CPU build compiles no kernels, so it has no first load to set
        // apart. A GPU build without a cache directory compiles on every load,
        // and every load is an initial setup.
        let marker = warm_marker(model_name).filter(|_| ON_GPU);
        let phase = if !ON_GPU || marker.as_ref().is_some_and(|m| m.exists()) {
            Phase::Loading
        } else {
            Phase::InitialSetup
        };
        log::info!("loading {model_name} ({phase:?})");
        let tracker = Tracker::new(phase, report);
        let (engine, warmed) = std::thread::scope(|scope| {
            scope.spawn(|| tracker.run());
            // Stops the sampler however this ends, a panic included: the
            // scope waits for it before it lets a panic through.
            let _finish = Finish(&tracker);
            Self::load_tracked(backend_dir, model_name, device, phase, &tracker)
        })?;
        if warmed {
            mark_warm(marker.as_deref());
        }
        Ok(engine)
    }

    /// Compile and tune the kernels a load of `model_name` needs from its
    /// `config.json` and `tokenizer.json` alone, while its weights are still
    /// on their way: the load's warm-up, on a model of [`BUILD_LAYERS`]
    /// layers whose weights are zeros. The kernels depend on the shapes,
    /// dtypes and layouts the model runs at, never on its weights, and every
    /// decoder layer runs the same ones. Its memory is handed back before this
    /// returns, so the load that follows has the device to itself.
    ///
    /// # Errors
    /// If a file is missing or unreadable, or the warm-up cannot run.
    pub fn build_kernels(
        backend_dir: &Path,
        model_name: &str,
        device: Option<&str>,
        report: &(dyn Fn(Report) + Sync),
    ) -> Result<()> {
        let dir = backend_dir.join("models").join(model_name);
        let tokenizer = load_tokenizer(&model_file(&dir, "tokenizer.json")?)?;
        let mut config = read_config(&model_file(&dir, "config.json")?)?;
        config.num_hidden_layers = config.num_hidden_layers.min(BUILD_LAYERS);
        let stop_tokens = stop_tokens(&tokenizer)?;

        let (device, device_name) = select_device(device);
        let dtype = compute_dtype(&device);
        log::info!(
            "building {model_name}'s kernels on {device_name} ({dtype:?}) from {}",
            dir.display()
        );
        let entries = crate::progress::cache_entries();
        let tracker = Tracker::new(Phase::InitialSetup, report);
        let warmed = std::thread::scope(|scope| {
            scope.spawn(|| tracker.run());
            let _finish = Finish(&tracker);
            tracker.enter(
                Step::BuildingKernels,
                Measure::cache_entries(WARM_UP_CACHE_ENTRIES),
            );
            let mut engine = Self {
                model: Qwen3::zeroed(&config, dtype, &device).map_err(|e| anyhow!(e))?,
                tokenizer,
                stop_tokens,
                max_new_tokens: MAX_NEW_TOKENS,
                device_name,
            };
            engine.warm_up(&AtomicU64::new(0))
        });
        device.memory_cleanup();
        if let Err(e) = device.sync() {
            log::warn!("syncing the device after building the kernels failed: {e:?}");
        }
        log::info!(
            "the kernel build wrote {} kernel-cache entries",
            crate::progress::cache_entries().saturating_sub(entries)
        );
        if warmed? {
            mark_warm(warm_marker(model_name).as_deref());
        }
        Ok(())
    }

    /// Whether a load of `model_name` would find its kernels built by this
    /// build. Always, on the CPU, which builds none.
    #[must_use]
    pub fn kernels_built(model_name: &str) -> bool {
        !ON_GPU || warm_marker(model_name).is_some_and(|marker| marker.exists())
    }

    /// [`Self::load`] under its tracker. Returns the engine and whether its
    /// warm-up ran to the end.
    fn load_tracked(
        backend_dir: &Path,
        model_name: &str,
        device: Option<&str>,
        phase: Phase,
        tracker: &Tracker<'_>,
    ) -> Result<(Self, bool), LoadError> {
        let dir = backend_dir.join("models").join(model_name);
        let files = resolve_files(&dir)?;
        let config = read_config(&files.config)?;
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let stop_tokens = stop_tokens(&tokenizer)?;
        let gguf = Arc::new(Gguf::open(&files.weights)?);

        let (device, device_name) = select_device(device);
        // One tensor through the device before half a gigabyte of weights:
        // the runtime initializes on first use, and a host without the driver
        // or the device fails here, where it can be named for what it is.
        catching(|| {
            let probe = burn::prelude::Tensor::<1>::zeros([1], &device);
            let _ = probe.into_data();
        })
        .map_err(LoadError::DeviceUnavailable)?;

        let dtype = compute_dtype(&device);
        log::info!(
            "loading {} on {device_name} ({dtype:?})",
            files.weights.display()
        );
        let started = std::time::Instant::now();
        let total: u64 = gguf.tensors.iter().map(|t| t.byte_len() as u64).sum();
        let read = Arc::new(AtomicU64::new(0));
        tracker.enter(
            Step::LoadingWeights,
            Measure::Count {
                done: Arc::clone(&read),
                total,
            },
        );
        let model = catching(|| Qwen3::load(&config, &gguf, dtype, &device, &read))
            .unwrap_or_else(|panic| Err(format!("panicked: {panic}")))
            .map_err(|e| anyhow!("building the model from {}: {e}", files.weights.display()))?;
        log::info!(
            "dequantized {} MB of weights in {:.1?}",
            total / 1_000_000,
            started.elapsed()
        );

        let mut engine = Self {
            model,
            tokenizer,
            stop_tokens,
            max_new_tokens: MAX_NEW_TOKENS,
            device_name,
        };
        let runs = Arc::new(AtomicU64::new(0));
        let (step, measure) = match phase {
            Phase::InitialSetup => (
                Step::BuildingKernels,
                Measure::cache_entries(WARM_UP_CACHE_ENTRIES),
            ),
            Phase::Loading => (
                Step::WarmingUp,
                Measure::Count {
                    done: Arc::clone(&runs),
                    total: warm_up_prompts().len() as u64,
                },
            ),
        };
        let entries = crate::progress::cache_entries();
        tracker.enter(step, measure);
        let warmed = engine.warm_up(&runs).map_err(LoadError::Failed);
        // What `WARM_UP_CACHE_ENTRIES` is calibrated against, on a cold load.
        log::info!(
            "the warm-up wrote {} kernel-cache entries",
            crate::progress::cache_entries().saturating_sub(entries)
        );
        Ok((engine, warmed?))
    }

    /// Device label for `GET /v1/status`.
    #[must_use]
    pub fn device_name(&self) -> &'static str {
        self.device_name
    }

    /// Drop the model and wait until the device has freed it.
    pub fn release(self) {
        let device = self.model.device().clone();
        drop(self);
        if let Err(e) = device.sync() {
            log::warn!("syncing the device after freeing the model failed: {e:?}");
        }
    }

    /// Compile and tune the kernels, so the first request does not.
    ///
    /// CubeCL compiles every GPU kernel the first time it meets one and tunes
    /// each operation against its candidates, and on a cold cache that would
    /// otherwise land on whoever dictates first, after `GET /v1/status` has
    /// already said `ready`. See [`WARM_UP_PROMPTS`] for why the prompts are
    /// several lengths rather than one.
    ///
    /// A failure after the first run is logged and swallowed: the model
    /// normalizes, and refusing the load over a longer prompt would turn a
    /// slow first request into no service at all. A failure on the first run
    /// fails the load, because then nothing normalizes — a device that says it
    /// computes in a type its compiler cannot lower fails every kernel, and a
    /// load that reported `ready` would fail every request after it.
    ///
    /// Counts each run into `runs` as it ends, and returns whether all of
    /// them did.
    fn warm_up(&mut self, runs: &AtomicU64) -> Result<bool> {
        let started = std::time::Instant::now();
        let prompts = warm_up_prompts();
        for &len in prompts {
            let prompt = vec![0; len];
            let run = catching(|| {
                self.model
                    .generate(&prompt, WARM_UP_TOKENS, &[], |_| ControlFlow::Continue(()))
            });
            if let Err(panic) = run {
                if runs.load(Ordering::Relaxed) == 0 {
                    bail!("the model cannot run: {panic}");
                }
                log::warn!("the warm-up failed after a {len}-token prompt: {panic}");
                return Ok(false);
            }
            runs.fetch_add(1, Ordering::Relaxed);
        }
        log::info!(
            "warmed {} prompt lengths up in {:.1?}",
            prompts.len(),
            started.elapsed()
        );
        Ok(true)
    }

    /// Normalize a raw transcript under `control`.
    ///
    /// A transcript over [`INPUT_TOKEN_BUDGET`] is cut into pieces, each
    /// normalized in turn and the results joined. `cancel` is polled once per
    /// generated token; when it is raised the decode stops and this returns an
    /// error, so a request the daemon gave up on does not keep the device
    /// busy.
    ///
    /// An empty result is a valid answer — a transcript of nothing but filler
    /// normalizes to nothing — and is returned as such.
    ///
    /// # Errors
    /// If the transcript cannot be tokenized or the output decoded, or the
    /// decode was cancelled.
    pub fn normalize(
        &mut self,
        text: &str,
        control: Control,
        cancel: &AtomicBool,
    ) -> Result<String> {
        let pieces = chunk::chunk(text, INPUT_TOKEN_BUDGET, |s| self.count_tokens(s));
        if pieces.len() > 1 {
            log::info!(
                "transcript over {INPUT_TOKEN_BUDGET} tokens; normalizing in {} pieces",
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
    /// Every request is a fresh conversation: the prefill writes the caches
    /// from position 0, and nothing of the last request is attended to.
    fn generate(&mut self, prompt: &str, cancel: &AtomicBool) -> Result<String> {
        // A cancel between the pieces of a long transcript spares the rest
        // their prefills.
        if cancel.load(Ordering::SeqCst) {
            bail!("cancelled");
        }
        let prompt_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("tokenize prompt: {e}"))?
            .get_ids()
            .to_vec();

        // Generation is bounded by the input: a cleanup is never much longer
        // than what it cleans, so a piece's ceiling follows its own length
        // while the model card's absolute cap still applies.
        let max_new = (2 * prompt_tokens.len() + 64).min(self.max_new_tokens);

        let start = std::time::Instant::now();
        let mut cancelled = false;
        let generated = self
            .model
            .generate(&prompt_tokens, max_new, &self.stop_tokens, |_| {
                if cancel.load(Ordering::SeqCst) {
                    cancelled = true;
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
        if cancelled {
            bail!("cancelled");
        }
        if generated.len() >= max_new {
            log::warn!(
                "decode hit the {max_new}-token cap without an end-of-turn token; \
                 returning what it produced"
            );
        }
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode output: {e}"))?;
        let elapsed = start.elapsed().as_secs_f64();
        let generated_count = u32::try_from(generated.len()).unwrap_or(u32::MAX);
        log::debug!(
            "{} prompt tokens, {generated_count} generated in {elapsed:.2}s ({:.1} tok/s)",
            prompt_tokens.len(),
            f64::from(generated_count) / elapsed.max(f64::EPSILON)
        );
        Ok(text.trim().to_string())
    }
}

/// The stop tokens of a Qwen3 tokenizer, which must have `<|im_end|>`.
fn stop_tokens(tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let stop_tokens: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    anyhow::ensure!(
        tokenizer
            .token_to_id("<|im_end|>")
            .is_some_and(|id| stop_tokens.contains(&id)),
        "tokenizer has no <|im_end|> token; this is not a Qwen3 tokenizer"
    );
    Ok(stop_tokens)
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| anyhow!("load {}: {e}", path.display()))
}

fn read_config(path: &Path) -> Result<TextConfig> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

/// `relative` in `dir`, which the manifest provisions there.
fn model_file(dir: &Path, relative: &str) -> Result<PathBuf> {
    let path = dir.join(relative);
    anyhow::ensure!(path.is_file(), "{relative} not found in {}", dir.display());
    Ok(path)
}

#[derive(Debug)]
struct ModelFiles {
    tokenizer: PathBuf,
    config: PathBuf,
    weights: PathBuf,
}

/// Find the `tokenizer.json`, the `config.json` and the one `*.gguf` the
/// manifest provisions into a model directory. The weights are found by
/// extension rather than by exact filename so the same engine serves every
/// quantization the manifest could list.
fn resolve_files(dir: &Path) -> Result<ModelFiles> {
    let tokenizer = model_file(dir, "tokenizer.json")?;
    let config = model_file(dir, "config.json")?;
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
    Ok(ModelFiles {
        tokenizer,
        config,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    #[test]
    fn a_model_dir_needs_a_tokenizer_a_config_and_exactly_one_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_files(dir.path()).unwrap_err().to_string();
        assert!(err.contains("tokenizer.json"), "{err}");

        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let err = resolve_files(dir.path()).unwrap_err().to_string();
        assert!(err.contains("config.json"), "{err}");

        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let err = resolve_files(dir.path()).unwrap_err().to_string();
        assert!(err.contains("no .gguf"), "{err}");

        std::fs::write(dir.path().join("s1-mini-q4_k_m.gguf"), b"").unwrap();
        let files = resolve_files(dir.path()).unwrap();
        assert!(files.weights.ends_with("s1-mini-q4_k_m.gguf"));

        std::fs::write(dir.path().join("another.gguf"), b"").unwrap();
        let err = resolve_files(dir.path()).unwrap_err().to_string();
        assert!(err.contains("expected one .gguf"), "{err}");
    }

    #[test]
    fn the_model_list_is_the_manifests() {
        let manifest = include_str!("../backend.toml");
        let declared: Vec<&str> = crate::manifest_probe::model_blocks(manifest)
            .iter()
            .filter_map(|block| crate::manifest_probe::string_field(block, "name"))
            .collect();
        assert_eq!(declared, MODELS);
    }

    /// The manifest's `prepare_files` of `model`, and the destinations of its
    /// `files`.
    fn prepare_files(model: &str) -> (Vec<&'static str>, Vec<&'static str>) {
        let manifest = include_str!("../backend.toml");
        let block = crate::manifest_probe::model_blocks(manifest)
            .into_iter()
            .find(|block| crate::manifest_probe::string_field(block, "name") == Some(model))
            .unwrap();
        let prepare = crate::manifest_probe::string_list(&block, "prepare_files").unwrap();
        let destinations = block
            .iter()
            .filter_map(|line| line.split_once("destination = \""))
            .filter_map(|(_, rest)| rest.split('"').next())
            .collect();
        (prepare, destinations)
    }

    #[test]
    fn every_model_prepares_from_files_it_downloads() {
        for model in MODELS {
            let (prepare, destinations) = prepare_files(model);
            assert!(!prepare.is_empty(), "{model}");
            for file in prepare {
                assert!(destinations.contains(&file), "{file} is not a destination");
            }
        }
    }

    // Naming the device creates nothing, so these run without a GPU.
    #[test]
    fn a_request_for_another_accelerator_gets_this_builds() {
        let (_, name) = select_device(Some("some-other-accelerator"));
        assert_eq!(name, BUILT_FOR);
        let (_, name) = select_device(Some(BUILT_FOR));
        assert_eq!(name, BUILT_FOR);
        let (_, name) = select_device(Some(""));
        assert_eq!(name, BUILT_FOR);
    }

    #[test]
    fn a_missing_checkpoint_fails_the_load_naming_the_file() {
        let err = Engine::load(Path::new("/nonexistent"), MODELS[0], None, &|_| {}).unwrap_err();
        assert!(matches!(err, LoadError::Failed(_)));
        assert!(err.to_string().contains("tokenizer.json"), "{err}");
    }

    #[test]
    fn load_errors_say_what_went_wrong() {
        let unavailable = LoadError::DeviceUnavailable("no driver".into());
        assert!(unavailable.to_string().contains("no driver"));
        assert!(unavailable.to_string().contains(BUILT_FOR));
        let failed = LoadError::from(anyhow!("broken"));
        assert!(failed.to_string().contains("broken"));
    }

    #[test]
    fn a_panic_is_caught_with_its_message() {
        assert_eq!(catching(|| 1), Ok(1));
        assert_eq!(
            catching(|| panic!("{}", String::from("owned"))),
            Err::<(), _>("owned".into())
        );
        assert_eq!(catching(|| panic!("static")), Err::<(), _>("static".into()));
        assert_eq!(
            catching(|| std::panic::panic_any(7)),
            Err::<(), _>("panicked".into())
        );
    }

    fn load_fixture(backend: &fixture::Backend) -> Result<Engine, LoadError> {
        Engine::load(&backend.dir, fixture::MODEL, None, &|_| {})
    }

    #[test]
    fn a_load_reports_each_step_and_ends_ready_to_normalize() {
        let backend = fixture::backend();
        let reports = std::sync::Mutex::new(Vec::new());
        let mut engine = Engine::load(&backend.dir, fixture::MODEL, None, &|r| {
            reports.lock().unwrap().push(r);
        })
        .unwrap();
        assert_eq!(engine.device_name(), BUILT_FOR);
        assert!(format!("{engine:?}").contains("Engine"));
        let reports = reports.into_inner().unwrap();
        // A CPU build compiles nothing, so every load is a plain one; a GPU
        // build with no cache directory to find a marker in builds kernels.
        let (phase, warming) = if ON_GPU {
            (Phase::InitialSetup, Step::BuildingKernels)
        } else {
            (Phase::Loading, Step::WarmingUp)
        };
        assert!(reports.iter().all(|r| r.phase == Some(phase)));
        let steps: Vec<Step> = reports.iter().filter_map(|r| r.step).collect();
        let first_warm = steps.iter().position(|s| *s == warming).unwrap();
        assert!(
            steps[..first_warm]
                .iter()
                .all(|s| *s == Step::LoadingWeights)
        );
        assert_ne!(steps[..first_warm], []);
        assert!(reports.iter().filter_map(|r| r.progress).all(|p| p < 1.0));

        let text = engine
            .normalize(
                "so um hello there",
                Control::default(),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert!(text.len() < 4 * MAX_NEW_TOKENS, "{text:?}");
        engine.release();
    }

    /// A transcript over the budget is cut, every piece is decoded, and a
    /// raised cancel stops the first piece at its first token.
    #[test]
    fn a_long_transcript_goes_in_pieces_and_a_cancel_stops_it() {
        let backend = fixture::backend();
        let mut engine = load_fixture(&backend).unwrap();
        engine.max_new_tokens = 16;
        // Byte tokens, four to a word: over the budget, in two pieces.
        let long = "word ".repeat(INPUT_TOKEN_BUDGET / 4 + 8);
        assert_eq!(
            chunk::chunk(&long, INPUT_TOKEN_BUDGET, |s| engine.count_tokens(s)).len(),
            2
        );
        engine
            .normalize(&long, Control::default(), &AtomicBool::new(false))
            .unwrap();
        let err = engine
            .normalize(&long, Control::default(), &AtomicBool::new(true))
            .unwrap_err();
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn kernels_build_from_the_prepare_files_alone() {
        let backend = fixture::backend();
        let (prepare, _) = prepare_files(fixture::MODEL);
        for entry in std::fs::read_dir(backend.model_dir()).unwrap() {
            let path = entry.unwrap().path();
            let name = format!(
                "models/{}/{}",
                fixture::MODEL,
                path.file_name().unwrap().display()
            );
            if !prepare.contains(&name.as_str()) {
                std::fs::remove_file(path).unwrap();
            }
        }
        backend.edit_config(|c| c["num_hidden_layers"] = 28.into());
        let reports = std::sync::Mutex::new(Vec::new());
        Engine::build_kernels(&backend.dir, fixture::MODEL, None, &|r| {
            reports.lock().unwrap().push(r);
        })
        .unwrap();
        let reports = reports.into_inner().unwrap();
        assert!(reports.iter().all(|r| r.phase == Some(Phase::InitialSetup)));
        let steps: Vec<Step> = reports.iter().filter_map(|r| r.step).collect();
        assert_ne!(steps, []);
        assert!(steps.iter().all(|s| *s == Step::BuildingKernels));
        // No cache directory in a test, so nothing records the build.
        assert_eq!(Engine::kernels_built(fixture::MODEL), !ON_GPU);
    }

    #[test]
    fn a_kernel_build_without_its_files_fails() {
        let err =
            Engine::build_kernels(Path::new("/nonexistent"), MODELS[0], None, &|_| {}).unwrap_err();
        assert!(format!("{err:#}").contains("tokenizer.json"), "{err:#}");
    }

    /// A load that fails says why.
    fn load_error(backend: &fixture::Backend) -> String {
        let err = load_fixture(backend).unwrap_err();
        assert!(matches!(err, LoadError::Failed(_)));
        err.to_string()
    }

    #[test]
    fn a_config_this_port_cannot_run_fails_the_load() {
        let backend = fixture::backend();
        backend.edit_config(|c| c["attention_bias"] = true.into());
        let err = load_error(&backend);
        assert!(err.contains("attention_bias"), "{err}");
    }

    #[test]
    fn an_unreadable_config_fails_the_load() {
        let backend = fixture::backend();
        std::fs::write(backend.model_dir().join("config.json"), "{").unwrap();
        let err = load_error(&backend);
        assert!(err.contains("config.json"), "{err}");
    }

    #[test]
    fn a_tokenizer_without_an_end_of_turn_fails_the_load() {
        let backend = fixture::backend();
        let path = backend.model_dir().join("tokenizer.json");
        let tokenizer = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, tokenizer.replace("<|im_end|>", "<|im_stop|>")).unwrap();
        let err = load_error(&backend);
        assert!(err.contains("<|im_end|>"), "{err}");
    }

    #[test]
    fn a_checkpoint_missing_a_tensor_fails_the_load() {
        let backend = fixture::backend();
        let names: Vec<String> = fixture::tensor_names()
            .into_iter()
            .filter(|n| n != "blk.1.ffn_up.weight")
            .collect();
        fixture::write_checkpoint(&backend.model_dir().join(fixture::GGUF), &names);
        let err = load_error(&backend);
        assert!(err.contains("not found in the checkpoint"), "{err}");
        assert!(
            err.contains("transformer.layers.1.mlp.up_proj.weight"),
            "{err}"
        );
    }

    #[test]
    fn tensors_no_parameter_takes_fail_the_load() {
        let backend = fixture::backend();
        // One decoder layer fewer than the checkpoint holds.
        backend.edit_config(|c| c["num_hidden_layers"] = 1.into());
        let err = load_error(&backend);
        assert!(err.contains("belong to no parameter"), "{err}");
        assert!(err.contains("transformer.layers.1."), "{err}");
    }

    #[test]
    fn a_tensor_of_the_wrong_shape_fails_the_load() {
        let backend = fixture::backend();
        backend.edit_config(|c| c["vocab_size"] = 300.into());
        let err = load_error(&backend);
        assert!(err.contains("embed_tokens"), "{err}");
    }

    /// Without an output head of its own, the head is the embedding.
    #[test]
    fn a_checkpoint_without_a_head_ties_it_to_the_embedding() {
        let backend = fixture::backend();
        let names: Vec<String> = fixture::tensor_names()
            .into_iter()
            .filter(|n| n != "output.weight")
            .collect();
        fixture::write_checkpoint(&backend.model_dir().join(fixture::GGUF), &names);
        load_fixture(&backend).unwrap();
    }
}
