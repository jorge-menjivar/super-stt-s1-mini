# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone S1-mini backend. Mirrors the recipe names used
# by the main super-stt repo (`just check`, `just ci`, etc.).
#
# `just --list` shows only the last comment line before a recipe, so any recipe
# whose comment runs to more than one line carries a [doc("…")] summary for the
# listing and keeps the prose above it for whoever opens this file.
#
# The model runs on Burn, pinned by revision to a fork in Cargo.toml. The first
# build fetches and compiles it, which is slow; nothing else about the build is
# unusual and there is no C toolchain to install.
#
# Every recipe that builds for a GPU passes `--no-default-features` with one
# accelerator named, matching the release workflow: features are additive, so
# a build that kept the default `flex` alongside `cuda` would carry two
# backends.

# Default: build release
default: build-release

# Remove everything this repo's recipes generate: the cargo target dir, the
# binary staged into the repo root by `just stage`, and the coverage artifacts.
# All are gitignored and all are one recipe away from coming back.
[doc("Remove the target dir, the staged binary, and coverage output")]
clean:
    cargo clean
    rm -f super-stt-backend-s1-mini lcov.info

# Compiles with debug profile. Usage: just build-debug [args]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile — the pure-Rust CPU backend, the default.
# Usage: just build-release [args]
build-release *args:
    cargo build --release --locked {{ args }}

# Build with CUDA. Needs the CUDA toolkit headers on the host; no GPU and no
# compute capability are needed, since CubeCL compiles the kernels at runtime.
[doc("Build with CUDA (needs the toolkit headers, no GPU)")]
build-cuda *args:
    cargo build --release --locked --no-default-features --features cuda {{ args }}

# Build with ROCm. Needs the ROCm headers `cubecl-hip-sys` binds against.
build-rocm *args:
    cargo build --release --locked --no-default-features --features rocm {{ args }}

# Build with Vulkan — the vendor-neutral GPU path. Needs no SDK to build; the
# loader is found at runtime.
[doc("Build with Vulkan (needs no SDK)")]
build-vulkan *args:
    cargo build --release --locked --no-default-features --features vulkan {{ args }}

# Build with Metal, on macOS. Needs Xcode's command line tools; the kernels
# are compiled at runtime.
build-metal *args:
    cargo build --release --locked --no-default-features --features metal {{ args }}

# Build, then copy the binary to the entrypoint name `backend.toml` declares, so
# this directory can be installed with the daemon's Import-from-dir path. Cargo
# already names the artifact that, and the release workflow tarballs it under
# the same name, so a local install and a published one stage the same bytes
# under the same name. Usage: just stage [--no-default-features --features cuda]
[doc("Build and stage the binary for Import-from-dir. Usage: just stage [--no-default-features --features cuda]")]
stage *args: (build-release args)
    cp target/release/super-stt-backend-s1-mini super-stt-backend-s1-mini
    @echo "staged super-stt-backend-s1-mini — this directory is now installable with Import from dir"

# Runs a clippy check — mirrors super-stt's lint. Default features lint the
# flex build, which compiles every line of the backend's own code but the
# other accelerators' arms of `select_device`; they differ only in which Burn
# feature is on. Run `just check --no-default-features --features cuda` to
# lint the CUDA build.
[doc("Lint with clippy (pedantic, -D warnings)")]
check *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Runs a clippy check with JSON message format (consumed by clippy-sarif in CI)
check-json: (check '--message-format=json')

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# The suite needs no weights: the load, the warm-up and the decode run on a
# tiny checkpoint the tests write themselves (`src/fixture.rs`).
#
# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

# Everything `backend.toml` declares for the model, laid out as the daemon
# lays it out, so `target/test-backend` works as `SUPER_STT_BACKEND_DIR`.
[doc("Fetch the model's files into target/test-backend")]
fetch-model:
    #!/usr/bin/env bash
    set -euo pipefail
    grep -o 'url = "[^"]*", destination = "models/s1-mini-q4_k_m/[^"]*"' backend.toml |
    while read -r line; do
        url=$(echo "$line" | sed 's/url = "\([^"]*\)".*/\1/')
        dest="target/test-backend/$(echo "$line" | sed 's/.*destination = "\([^"]*\)"/\1/')"
        mkdir -p "$(dirname "$dest")"
        [ -f "$dest" ] || curl -fL --retry 3 -o "$dest" "$url"
    done
    echo "s1-mini-q4_k_m in target/test-backend/models/s1-mini-q4_k_m"

# Run the gated real-model test against the files `fetch-model` provisions,
# on whatever accelerator the args name:
#   just test-e2e [--no-default-features --features cuda]
[doc("Run the real-model E2E test. Usage: just test-e2e [--no-default-features --features cuda]")]
test-e2e *args: fetch-model
    SUPER_STT_TEST_S1_MINI=1 SUPER_STT_BACKEND_DIR={{ justfile_directory() }}/target/test-backend \
        cargo test --release --locked {{ args }} --test process_e2e -- --nocapture

# Measure code coverage (requires cargo-llvm-cov). --remap-path-prefix keeps the
# report paths relative (src/...), and tests/ is excluded so only product code
# is counted. Usage: just coverage [--html]
[doc("Measure code coverage. Usage: just coverage [--html]")]
coverage *args:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex 'tests/'

# Render the browsable HTML report to target/llvm-cov/html/, which the Pages
# job uploads verbatim. cargo-llvm-cov's index is a single flat file list;
# per-file pages show line-level coverage.
[doc("Render the browsable HTML report to target/llvm-cov/html/")]
coverage-html *args:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' --html {{ args }}

# Drop a shields.io "endpoint" JSON beside the rendered HTML so the README badge
# reads the latest line-% straight from Pages — no Gist or PAT. `report` reuses
# the profile data coverage-html just collected, so run it after that recipe.
[doc("Write the shields.io badge JSON beside the rendered HTML")]
coverage-badge:
    #!/usr/bin/env bash
    set -euo pipefail
    pct=$(cargo llvm-cov report --json --summary-only --ignore-filename-regex 'tests/' \
      | jq -r '.data[0].totals.lines.percent')
    if   awk "BEGIN{exit !($pct>=90)}"; then color=brightgreen
    elif awk "BEGIN{exit !($pct>=75)}"; then color=green
    elif awk "BEGIN{exit !($pct>=60)}"; then color=yellow
    elif awk "BEGIN{exit !($pct>=40)}"; then color=orange
    else color=red
    fi
    printf '{"schemaVersion":1,"label":"coverage","message":"%.1f%%","color":"%s"}\n' \
      "$pct" "$color" > target/llvm-cov/html/coverage.json
    cat target/llvm-cov/html/coverage.json

# Full local CI gate: format, lint, build, test. No doctests — this is a
# binary-only crate, so `cargo test --doc` has no lib target.
[doc("Full local CI gate: format, lint, build, test")]
ci: fmt-check check build-release test
