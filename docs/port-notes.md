# Notes on the Burn port

The measurements and findings behind choices in the code. The code's comments
say what it does and why; the numbers, and how they were found, are here.

Measured on 2026-10-20 on an RTX 3090 (NVIDIA driver 610.57) and a Ryzen 9
5900X, a desktop that was running other work at the time: a Voxtral backend
held 9 GB of the card and a Qwen TTS backend 2.4 GB, and other builds came
and went on the CPU. Timings vary between runs for it, by up to about 30% on
the CPU, and a first load's kernel build took up to twice as long on a busy
CPU as on a quieter one. The candle build compared against is this
repository's previous commit, `3d37855`, built for CUDA with
`CUDA_COMPUTE_CAP=86` and for the CPU.

## The weights

The backend still downloads the Q4_K_M GGUF and dequantizes it as it loads
(`src/gguf.rs`), into f16 on a GPU and f32 on the CPU.

- **Bit for bit.** A throwaway program linking candle-core at the revision
  the candle backend pinned dequantized every tensor of the shipped file with
  both: all 311 tensors, 751,632,384 values, came out bit-identical.
- **The output head is its own tensor.** The GGUF stores `output.weight` in
  Q6_K beside a Q4_K `token_embd.weight`, a materialized copy of the tied
  embedding, quantized differently. candle used it as the head, and so does
  this port, which is 311 MB of the 1.5 GB of f16 weights.
- **Time.** Reading and dequantizing the file takes 0.9–1.1 s on CUDA and
  Vulkan and 1.3 s on flex, and 2.6–2.8 s in a CUDA load into an empty
  kernel cache.
- **Download.** The BF16 safetensors in the model's own repository are 1.5 GB
  and would load through burn-store unchanged. The Q4_K_M is 484 MB, it is
  the build the published accuracy was measured on, and it is what installed
  backends already have; dequantizing it costs a second of the load.

## Precision

Every GPU computes in f16 and the CPU in f32 (`engine::compute_dtype`).

A corpus of 31 requests, sent over the socket the way the daemon sends them
(the model card's examples, every styling, structure and context value,
dictation of the kind the daemon sends, and a 120-sentence transcript the
chunker cuts into four pieces), gave:

- **Burn against itself:** CUDA in f16, Vulkan in f16 and flex in f32 wrote
  the same text for all 31, the long transcript included.
- **Burn against candle:** 30 of 31 identical on CUDA, 29 of 31 on the CPU.
  candle's two builds disagree with each other on two. Its quantized matmuls
  quantize the activations to 8 bits as well, Q8_1 on CUDA and Q8_K on the
  CPU, where Burn multiplies dense dequantized weights in f16 or f32, so where
  a near-tie flips, Burn's text is the one the dequantized model writes.

| Input | candle CUDA | candle CPU | Burn, every build |
|---|---|---|---|
| `let's meet at half past two tomorrow uh actually make it three fifteen p m` | `Let's meet at 3:15pm.` | `Let's meet at 3:15pm tomorrow.` | `Let's meet at 3:15pm tomorrow.` |
| the cute-dog example, `formal` | `I am going to be late. …` | `Hmm, I am going to be late. …` | `I am going to be late. …` |
| `so for the trip we need to pack sunscreen …` | `So for the trip we need …` | `So for the trip, we need …` | `So for the trip we need …` |

The model card's BF16 outputs are the Burn column on the first two rows and
candle CPU's on the third.

- **f16 on flex** changed the punctuation of 3 of 30 short requests against
  f32, where CUDA's and Vulkan's f16 changed none: flex computes more of the
  pass in half precision than CubeCL's kernels, which accumulate in f32. With
  its slower prefill (below), that kept the CPU on f32.
- **RoPE.** candle's `quantized_qwen3` builds its rotary tables in f16, the
  dtype it reads from the GGUF's missing `general.dtype`; this port builds
  them in f32, as `transformers` does.

## Speed

Requests back to back, after the load's warm-up: the 30 short requests and
the long transcript. On a GPU, the minimum of three runs of each. On the CPU,
the better of two rounds for the short requests, and one round for the long
transcript.

| | candle CUDA | Burn CUDA | Burn Vulkan | candle CPU | Burn CPU (flex) |
|---|---:|---:|---:|---:|---:|
| first request after a load | 0.24 s | 0.06 s | 0.09 s | 1.7 s | 1.9 s |
| short request, median | 0.104 s | 0.110 s | 0.153 s | 1.9 s | 3.2 s |
| 30 short requests | 2.9 s | 3.0 s | 4.3 s | 56 s | 91 s |
| long transcript, four pieces | 22.2 s | 15.4 s | 22.0 s | 277 s | 490 s |
| load, kernel cache warm | 1.4 s | 3.0 s | 3.4 s | 1.2 s | 3.6 s |

candle has no warm-up. Its kernels are PTX, which the NVIDIA driver
translates on first use and caches under `$HOME/.nv`. With that cache off
(`CUDA_CACHE_DISABLE=1`), its load took 7.5 s and its first request 6.5 s.
The daemon's sandbox leaves `$HOME/.nv` read-only, so that is what candle paid
there on every load. This backend points `CUDA_CACHE_PATH` into
`SUPER_STT_BACKEND_CACHE_DIR`, beside CubeCL's own cache.

### The CPU

The flex build is the slow one: a 90-token prompt prefills in 0.6 s, against
candle's 1.3 s, but each token then takes 110–140 ms, against about 60 ms.

- **Why.** flex runs a matmul on one thread below 192³ multiply-adds, and
  every projection of a decode step is one row times a 1024- to 3072-wide
  weight, 1 to 3 million. So each token streams 1.8 GB of f32 weights
  through one core, and the output head's 0.6 GB on all of them; candle reads
  0.5 GB of Q4_K a token.
- **Column blocks.** Viewing each weight as column blocks, `[blocks, d_in,
  d_out / blocks]`, reaches flex's batch-parallel path. With 8 to 32 blocks
  a token took 129–171 ms instead of 109–118.
- **f16 weights** halve what a token streams: 76 ms a token, and 25% off the
  30 short requests, but the 90-token prefill took 1.45 s, and 3 of 30
  outputs changed (above).

## Memory

GPU memory of the process, from `nvidia-smi`. Each Burn range has a later load
at its low end and a first load at its high end; a first load leaves
0.1 GiB more behind.

| | candle CUDA | Burn CUDA | Burn Vulkan |
|---|---:|---:|---:|
| idle after the load | 1.33 GiB | 2.05–2.15 GiB | 1.81–1.84 GiB |
| peak during requests | 1.97–2.15 GiB | 2.08–2.18 GiB | 1.84–1.88 GiB |
| peak of a first load | — | 4.12 GiB | 3.81 GiB |
| peak of a later load | 1.33 GiB | 2.05 GiB | 1.81 GiB |

The dense f16 weights are 1.5 GB where candle held 0.5 GB of Q4_K. The
key/value caches, sized for 2048 positions by the warm-up, take 235 MB more. A
first load's peak is CubeCL tuning: it allocates the buffers of every
candidate it benchmarks.

On the CPU the process holds 3.1 GiB, the f32 weights, and peaks at 3.8 GiB
while it loads. candle's CPU build held 1.1–1.2 GiB.

## Kernel cache and load progress

The warm-up's progress is measured in kernel-cache entries, against a
per-accelerator estimate (`WARM_UP_CACHE_ENTRIES`), the larger of the counts
below. Each count is from a first load into an empty cache; they vary by a few
entries between runs. The larger figure in each range was measured while
other builds were using the CPU.

| Accelerator | entries | first load | later load | longest stall |
|---|---:|---:|---:|---:|
| CUDA, RTX 3090 | 1122–1131 | 181–223 s | 3.0 s | 6.9–8.1 s |
| Vulkan, RTX 3090 | 564–584 | 24–56 s | 3.3–3.4 s | 1.6–2.2 s |
| ROCm | not measured; uses Vulkan's count | | | |
| Metal | not measured; uses Vulkan's count | | | |

The daemon fails a load whose step and progress stand still for 120 s. The
warm-up covers every prefill and decode window a request reaches
(`engine::WARM_UP_PROMPTS`), so no request after it compiles anything: the
first request after a load took 0.06 s on CUDA and 0.09 s on Vulkan.

## Building the kernels before the weights arrive

`POST /v1/prepare` (contract v4) builds the kernels from the two files the
manifest lists as the model's `prepare_files`, `config.json` and
`tokenizer.json`, while the GGUF downloads. It runs the load's warm-up on a
model with two decoder layers and every weight zero.

| | CUDA | Vulkan |
|---|---:|---:|
| prepare | 175 s | 23 s |
| load after it | 3.0 s | 3.1 s |
| peak during the prepare | 4.12 GiB | 3.86 GiB |
| left when the prepare hands over | 0.34 GiB | 0.09 GiB |
| load sent 10 s into the prepare, ready after | 176 s | 25 s |

- **The load after it** takes as long as a later load. It writes the 2 or 3
  entries the zero model did not reach.
- **A load sent during the prepare** waits for it and is ready when a load
  after it would have been.
- **The peak** is a plain first load's. The two-layer model holds 0.7 GB of
  weights to the real one's 1.5 GB, but it tunes the same full-size
  embedding, output head and prefills, and the tuning sets the peak. Unlike
  Voxtral's, this prepare buys time, not memory: the build runs while the
  484 MB file downloads.

## The long transcript

The 120-sentence transcript the E2E test sends is cut into four pieces, and
both candle and Burn decode two of them to the 1024-token cap, repeating a
dozen of the sentences: 11,664 characters come back for 9,972, identical
between the two. That is the model running on past the end of a repetitive
piece, not the port.

## Provenance of details kept from the candle backend

- `src/prompt.rs` and `src/chunk.rs` are unchanged.
- `INPUT_TOKEN_BUDGET`, `MAX_NEW_TOKENS`, the per-piece cap of twice the
  prompt plus 64, the stop tokens and the trimming of the output are the
  candle backend's.
