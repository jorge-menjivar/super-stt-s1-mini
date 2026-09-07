# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone S1-mini backend. Mirrors the recipe names used
# by the main super-stt repo (`just check`, `just ci`, etc.).
#
# `just --list` shows only the last comment line before a recipe, so any recipe
# whose comment runs to more than one line carries a [doc("…")] summary for the
# listing and keeps the prose above it for whoever opens this file.

# Default: build release
default: build-release

# Remove everything this repo's recipes generate: the cargo target dir, the
# binary staged into the repo root by `just stage`, and the coverage artifacts.
# All are gitignored and all are one recipe away from coming back.
[doc("Remove the target dir, the staged binary, and coverage output")]
clean:
    cargo clean
    rm -f super-stt-backend-s1-mini lcov.info

# Compiles with debug profile. Usage: just build-debug [--features cuda]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile. Usage: just build-release [--features cuda]
build-release *args:
    cargo build --release --locked {{ args }}

# Build, then copy the binary to the entrypoint name `backend.toml` declares, so
# this directory can be installed with the daemon's Import-from-dir path. Cargo
# already names the artifact that, and the release workflow tarballs it under
# the same name, so a local install and a published one stage the same bytes
# under the same name. Usage: just stage [--features cuda]
[doc("Build and stage the binary for Import-from-dir. Usage: just stage [--features cuda]")]
stage *args: (build-release args)
    cp target/release/super-stt-backend-s1-mini super-stt-backend-s1-mini
    @echo "staged super-stt-backend-s1-mini — this directory is now installable with Import from dir"

# One release asset: cross-compile for `target` with `features`, then tarball
# the binary under the name `backend.toml` declares for that (target, suffix).
# CUDA builds run inside an `nvidia/cuda:*-devel` image (carries nvcc), picked
# with `cuda_image`; CUDA_COMPUTE_CAP is forwarded into the container by the
# `cross` passthrough in Cargo.toml. The release workflow calls this once per
# matrix entry. Usage: just release-asset <target> <suffix> [features] [cuda_image]
[doc("Cross-build one release tarball. Usage: just release-asset <target> <suffix> [features] [cuda_image]")]
release-asset target suffix features="" cuda_image="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{ cuda_image }}" ]; then
        export CROSS_TARGET_X86_64_UNKNOWN_LINUX_GNU_IMAGE="{{ cuda_image }}"
    fi
    cross build --release --locked --target {{ target }} \
        --features "{{ features }}" --bin super-stt-backend-s1-mini
    tar -czf "s1-mini-{{ target }}-{{ suffix }}.tar.gz" \
        -C "target/{{ target }}/release" super-stt-backend-s1-mini
    ls -lh s1-mini-*.tar.gz

# Runs a clippy check — mirrors super-stt's lint. Default (CPU) features: the
# CUDA feature needs a toolkit the hosted runner lacks, and every line of this
# crate's own code is compiled either way. Run `just check --features cuda`
# locally to additionally lint the candle CUDA backend.
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

# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

# Run the gated real-model test. Provision `models/s1-mini-q4_k_m` under the
# given backend dir first (the two files `backend.toml` lists for it), then:
#   just test-e2e <backend-dir> [--features cuda]
[doc("Run the real-model E2E test. Usage: just test-e2e <backend-dir> [--features cuda]")]
test-e2e backend_dir *args:
    SUPER_STT_TEST_S1_MINI=1 SUPER_STT_BACKEND_DIR={{ backend_dir }} \
        cargo test --locked {{ args }} --test process_e2e -- --nocapture

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
