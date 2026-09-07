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
(94.8% token accuracy on a 7,519-case held-out set). It runs comfortably on a
laptop CPU. `backend.toml` pins it to the Hugging Face commit that published
it and to the bytes' SHA-256. The model is **English only**.

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
just stage                   # CPU build
just stage --features cuda   # CUDA build (needs the CUDA toolkit)
```

Then Settings → Library → Installed → *Import from dir*, and point it here.

The daemon picks the release build for your machine — CPU, or one of the
per-compute-capability CUDA builds `backend.toml` lists — and downloads the
model files on first load. A GPU is not required.

## Develop

```bash
just test         # unit tests + the protocol test over the socket (no weights)
just ci           # fmt + lint + build + test
```

The prompt S1-mini was trained on lives in [`src/prompt.rs`](src/prompt.rs)
and the chunker in [`src/chunk.rs`](src/chunk.rs); both are pure and tested
without a model. [`src/engine.rs`](src/engine.rs) loads the GGUF through
candle's Qwen3 support and decodes greedily, which is what the model is trained
for. [`src/main.rs`](src/main.rs) is the `/v1` server.

To run the real-model test, put the two files `backend.toml` lists for
`s1-mini-q4_k_m` under `<dir>/models/s1-mini-q4_k_m/` and:

```bash
just test-e2e <dir>                            # CPU
just test-e2e <dir> --release --features cuda  # GPU
```

## Protocol

Implements the [Super STT backend contract](https://github.com/jorge-menjivar/super-stt/blob/main/docs/protocol/backend/contract.md)
for the post-processor role, as a subprocess backend:

| Route | Behavior |
|---|---|
| `GET /v1/ping` | `pong`. |
| `GET /v1/status` | `starting` → `loading` → `ready`, or `error` with a `reason`. |
| `POST /v1/load` | Loads the named model from `SUPER_STT_BACKEND_DIR/models/<name>`; `409 already_loading` while one is in flight. |
| `POST /v1/process` | Normalizes `text` under the control line read from the `x-stt-option-*` headers; `400 invalid_text` when it is missing or blank, `409 not_ready` before a load, `500 processing_failed` on a decode error or a cancel. |
| `POST /v1/cancel` | Stops the decode in flight; `409 nothing_in_progress` otherwise. |
| `POST /v1/transcribe` | `400 unsupported_role` — this backend does not transcribe. |

An empty result is a valid one: a transcript of nothing but filler normalizes
to nothing, and the daemon keeps your raw text in that case.

## License

The backend is GPL-3.0-only; see [LICENSE](LICENSE).

The model, S1-mini by Superwhisper, is a derivative of Qwen3-0.6B (Alibaba
Cloud) and is distributed under Apache 2.0 with an additional naming term:
anything that uses it must keep identifying it as "S1-mini" by "Superwhisper".
See the [model license](https://huggingface.co/superwhisper/s1-mini/blob/main/LICENSE).
The backend downloads the model at load time and does not redistribute it.
