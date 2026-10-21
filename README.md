# S1-mini — a Super STT post-processor

[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/super-stt-s1-mini/coverage.json)](https://jorge-menjivar.github.io/super-stt-s1-mini/)

Runs [S1-mini by Superwhisper](https://huggingface.co/superwhisper/s1-mini)
on your transcripts before they are typed. S1-mini is a 0.6B-parameter text
normalizer: it takes the raw output of a speech model and rewrites it as clean
written text.

```
so um i need to like send the the report by uh friday no wait make that thursday
                                   ↓
I need to send the report by Thursday.
```

Fillers go, false starts and self-corrections resolve to the value you landed
on, punctuation and capitalization are applied, and spoken numbers, dates,
times, currency and email addresses come out written (`twenty three thousand
four hundred and fifty dollars` → `$23,450`; `support at superwhisper dot com`
→ `support@superwhisper.com`).

This is a **post-processor** backend for
[Super STT](https://github.com/jorge-menjivar/super-stt): the daemon hands it a
finished transcript on `POST /v1/process` and it hands back a cleaned one. It
does not transcribe audio. It runs entirely on your machine — the model is
downloaded once, and the backend has no network at all.

## What it does

The model is steered with three settings, each a closed set it was trained on.
Set them in **Settings → Models → Post-processing → Configure**.

| Setting | Values | Default | What it does |
|---|---|---|---|
| Styling | `casual`, `semi-casual`, `semi-formal`, `formal` | `semi-formal` | The register. `casual` is all lowercase with apostrophes stripped; `semi-casual` keeps your phrasing but capitalizes `I` and its contractions; `semi-formal` is standard written English with contractions kept and `gonna` smoothed to `going to`; `formal` also expands contractions. |
| Structure | `prose`, `lists` | `prose` | Whether an enumeration of three or more items may become a Markdown bullet list. |
| Context | `general`, `email` | `general` | `email` lays the text out as a greeting line, the body, and a sign-off block, separated by blank lines. |

The same input under each register:

| Styling | `hmm im gonna be late theres a cute dog outside i cant just walk past him` |
|---|---|
| `casual` | `hmm im gonna be late. theres a cute dog outside. i cant just walk past him` |
| `semi-casual` | `hmm, I'm gonna be late. there's a cute dog outside. I can't just walk past him` |
| `semi-formal` | `I'm going to be late. There's a cute dog outside. I can't just walk past him.` |
| `formal` | `I am going to be late. There is a cute dog outside. I cannot just walk past him.` |

A value outside the trained set is not sent to the model — the model card
warns it garbles the output — so a typo in a setting falls back to the default
and is logged.

## The model

One model, `s1-mini-q4_k_m`: release v1 of S1-mini in the Q4_K_M build
(484 MB), the one the model card recommends and measured its accuracy on
(94.8% token accuracy on a 7,519-case held-out set). `backend.toml` pins it,
and the tokenizer and `config.json` from the model's own repository, to the
Hugging Face commits that published them and to the bytes' SHA-256. The model
is **English only**.

A GPU is not required. On a GPU a short dictation takes about a tenth of a
second; on a desktop CPU, two to four seconds.

Long transcripts are handled: S1-mini is trained on inputs of up to about a
thousand tokens, so a dictation that runs past that is cut at sentence
boundaries (word boundaries when there is no punctuation yet), each piece
normalized on its own, and the results rejoined.

## Install

**From the registry:** Settings → Library → Browse → *S1-mini* → Install.
Then Models → Post-processing, select it as the backend and press
**Load model**. The first load downloads the model files.

**From this directory:**

```bash
just stage                                          # CPU build
just stage --no-default-features --features cuda    # CUDA build (needs the CUDA headers)
just stage --no-default-features --features vulkan  # Vulkan build (needs nothing)
```

Then Settings → Library → Installed → *Import from dir*, and point it here.

The daemon picks the release build for your machine from the ones
`backend.toml` lists — CPU, CUDA 12 or 13, ROCm, Vulkan, or Metal on Apple
Silicon — and downloads the model files on first load.

## What's in here

The model runs on [Burn](https://github.com/tracel-ai/burn). Its GPU kernels
are compiled at runtime by CubeCL, so one binary per accelerator covers every
GPU generation the driver can compile for, which is also what reaches ROCm,
Vulkan and Metal. The CPU build runs on Burn's pure-Rust CPU backend, flex.
The decoder is the Qwen3 stack the Qwen ASR backend runs.

| Path | What it holds |
|---|---|
| `src/prompt.rs` | The prompt S1-mini was trained on, and the control line that steers it. |
| `src/chunk.rs` | Cutting a long transcript into pieces the model can take. |
| `src/gguf.rs` | The GGUF's tensor directory, and dequantizing its Q4_K and Q6_K blocks. |
| `src/qwen3/` | The Qwen3 decoder, its config, loading the GGUF into it, and greedy decoding. |
| `src/engine.rs` | Device and precision, the load and its warm-up, building the kernels ahead of a load, and normalizing a request. |
| `src/progress.rs` | What a load reports while it runs. |
| `src/model_thread.rs` | The one thread the model lives on. |
| `src/main.rs` | The `/v1` server. |
| `src/fixture.rs` | A tiny S1-mini the tests write on the fly, so the suite needs no weights. |

The backend dequantizes the GGUF as it loads it, to f16 on a GPU and to f32
on the CPU, in about a second. That keeps the download at the 484 MB build the
accuracy was measured on, where the model's 1.5 GB BF16 safetensors would have
loaded without it, and it keeps the files a candle-era install already has.

### Performance

Measured on an RTX 3090 and a Ryzen 9 5900X against the candle build this
backend shipped before (v0.1.0), with requests back to back. The machine was
running other work at the time.

| | candle CUDA | Burn CUDA | Burn Vulkan |
|---|---:|---:|---:|
| short request, median of 30 | 0.10 s | 0.11 s | 0.15 s |
| 120-sentence transcript, four pieces | 22.2 s | 15.4 s | 22.0 s |
| first request after a load | 0.24 s | 0.06 s | 0.09 s |
| load, kernel cache warm | 1.4 s | 3.0 s | 3.4 s |
| load, first ever on the machine | 7.5 s | 181–223 s | 24–56 s |
| GPU memory, idle | 1.33 GiB | 2.05–2.15 GiB | 1.81–1.84 GiB |
| GPU memory, during requests | 1.97–2.15 GiB | 2.08–2.18 GiB | 1.84–1.88 GiB |
| GPU memory, peak of the first load | — | 4.12 GiB | 3.81 GiB |

candle's column assumes the NVIDIA driver could keep the kernels it
translated from candle's PTX in `$HOME/.nv`. The daemon's sandbox leaves that
directory read-only, and without the driver's cache every candle load took
7.5 s and its first request 6.5 s. This backend keeps the driver's cache
beside its own, in `SUPER_STT_BACKEND_CACHE_DIR`.

Burn holds more at idle because it multiplies dense f16 weights, 1.5 GB of
them, where candle kept the Q4_K blocks. `estimated_vram_bytes` in
`backend.toml` is set from the CUDA figure.

**The CPU build is slower than candle's**, by 1.6 to 1.8 times: 3.2 s
against 1.9 s for the median short request, 91 s against 56 s for the 30, and
490 s against 277 s for the long transcript. Each token streams the dense f32
weights, 1.8 GB of them, through flex's single-threaded small matmuls, where
candle read 0.5 GB of Q4_K a token. The process also holds 3.1 GiB of RAM
where candle held 1.2. f16 weights would win back a quarter of the time and
half the memory, but they changed the punctuation of 3 of the 30 outputs, so
the CPU stays on f32. `docs/port-notes.md` has the details.

The decoder's key/value caches are allocated up front, and each decoding step
attends over a power-of-two window of them and replays a graph captured for
that window. A request meets only a handful of shapes, and the warm-up has
built all of them.

### Kernel cache

CubeCL compiles the kernels a request needs the first time it meets them and
tunes each operation for its shape, so a GPU load ends with a warm-up that
runs every prefill and decode shape a request can reach: the first real
request then finds everything ready. Compiled kernels are kept in
`SUPER_STT_BACKEND_CACHE_DIR` when the daemon grants one, so only the first
load of a build pays; without it the backend logs a warning and recompiles on
every load. The CPU build has no kernels to build; its load runs one short
request only to prove the whole path works before it reports `ready`.

Super STT can have the kernels built before the weights arrive. `POST
/v1/prepare`, with the body of `/v1/load`, builds them from the model's
`config.json` and `tokenizer.json` alone, which the manifest lists as its
`prepare_files`, on a two-layer model whose weights are zeros. The status
reads `preparing`, then `prepared`. A load sent in the meantime waits for the
build and reports both as one initial setup. On the RTX 3090 the build
takes 175 s on CUDA and 23 s on Vulkan, and the load after it is ready in
3 s. It peaks as high as a plain first load, so what it saves is time, not
memory: the kernels build while the GGUF downloads.

## Develop

Requires [`just`](https://github.com/casey/just).

```bash
just test            # unit tests + the protocol tests over the socket (no weights)
just ci              # format, lint, build, and test
just build-cuda      # needs the CUDA headers — no GPU, no compute capability
just build-vulkan    # needs nothing, the loader is found at runtime
just build-rocm      # needs the ROCm headers
just build-metal     # on macOS; needs only the Xcode command-line tools
```

The default build is the CPU one. Each build carries exactly one accelerator,
which is why the GPU recipes pass `--no-default-features`: cargo features are
additive, so `--features cuda` alone would keep the CPU backend too.

The real-model test downloads what `backend.toml` lists into
`target/test-backend` and normalizes the model card's examples:

```bash
just test-e2e                                        # CPU
just test-e2e --no-default-features --features cuda  # GPU
```

**Burn comes from a fork.** `Cargo.toml` pins `jorge-menjivar/burn` at the
revision the Voxtral and Qwen backends build on: upstream Burn plus fixes to
Burn's fusion crates and to tensor reads that are not upstream yet, with the
fork's CubeCL patched in to match. The CUDA build compiles its kernels through
CubeCL's C++ backend rather than its default, LLVM.

## Checking the port against candle

This backend ran on candle before it ran on Burn, and the port was checked
against that build (`docs/port-notes.md` has the numbers):

- **The weights.** Dequantizing every tensor of the shipped GGUF with this
  crate and with candle gave bit-identical values: 311 tensors, 751,632,384
  values.
- **The text.** A corpus of 31 requests — the model card's examples, every
  setting, dictation of the kind the daemon sends, and a 120-sentence
  transcript — came back identical from Burn on CUDA, Vulkan and the CPU.
  Against candle it matched 30 of 31 on CUDA and 29 of 31 on the CPU, and
  candle's two builds disagree with each other on two: candle quantizes the
  activations to 8 bits inside its matmuls, so near-ties flip differently.
  Where the model card shows the full-precision output, Burn's matches it.

## Protocol

Implements the [Super STT backend contract](https://github.com/jorge-menjivar/super-stt/blob/main/docs/protocol/backend/contract.md)
v4 for the post-processor role, as a subprocess backend:

| Route | Behavior |
|---|---|
| `GET /v1/ping` | `pong`. |
| `GET /v1/status` | `starting` → `loading` → `ready`, or `error` with a `reason`; `preparing` → `prepared` for a prepare. While loading or preparing it reports the `phase`, `step` and `progress`. |
| `POST /v1/prepare` | Builds the kernels of the named model from its `prepare_files`; `prepared` at once on the CPU, or when this build already has them. `400 invalid_model` for a model it does not declare, `409 already_loading` once a load has begun. |
| `POST /v1/load` | Loads the named model from `SUPER_STT_BACKEND_DIR/models/<name>`; `400 invalid_model` for a model it does not declare, `409 already_loading` while one is in flight. |
| `POST /v1/process` | Normalizes `text` under the control line read from the `x-stt-option-*` headers; `400 invalid_text` when it is missing or blank, `409 not_ready` before a load, `500 processing_failed` on a decode error or a cancel. |
| `POST /v1/cancel` | Stops the decode in flight; `409 nothing_in_progress` otherwise. |
| `POST /v1/transcribe` | `400 unsupported_role` — this backend does not transcribe. |

A build runs on the one accelerator it was built for. The daemon's `device`
in a load is checked against it, and a mismatch is logged rather than obeyed.

An empty result is a valid one: a transcript of nothing but filler normalizes
to nothing, and the daemon keeps your raw text in that case.

## License

The backend is GPL-3.0-only; see [LICENSE](LICENSE).

The model, S1-mini by Superwhisper, is a derivative of Qwen3-0.6B (Alibaba
Cloud) and is distributed under Apache 2.0 with an additional naming term:
anything that uses it must keep identifying it as "S1-mini" by "Superwhisper".
See the [model license](https://huggingface.co/superwhisper/s1-mini/blob/main/LICENSE).
The backend downloads the model at load time and does not redistribute it.
