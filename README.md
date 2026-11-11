# Super Voice — S1-mini backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-voice-s1-mini/coverage.json)](https://super-libre.github.io/super-voice-s1-mini/)

[S1-mini by Superwhisper](https://huggingface.co/superwhisper/s1-mini), a
0.6B-parameter text normalizer, as a post-processor backend for
[Super Voice](https://github.com/jorge-menjivar/super-voice). It rewrites a
finished transcript as clean written text: fillers and false starts go,
punctuation and casing are applied, and numbers and dates come out written.
English only, on Linux and macOS, on the CPU or on any GPU that CUDA, ROCm,
Vulkan or Metal can drive.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single Rust binary that runs the model
through [Burn](https://github.com/tracel-ai/burn) and answers
`POST /v1/process`.

## License

GPL-3.0-only. See `LICENSE`. The model the daemon fetches at runtime, S1-mini
by Superwhisper, derives from Qwen3-0.6B and is under Apache 2.0 with a naming
term: anything that uses it must keep identifying it as "S1-mini" by
"Superwhisper". See the
[model license](https://huggingface.co/superwhisper/s1-mini/blob/main/LICENSE).
