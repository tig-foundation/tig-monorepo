#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/tmp/cur-experiment-cargo
export RUSTUP_HOME=/tmp/cur-experiment-rustup

apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates curl git libssl-dev pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$CARGO_HOME/env"

rustc --version
nvcc --version
nvidia-smi

cargo run --release --example cur_stage_experiment --features cur_decomposition -- \
  tig-algorithms/lib/cur_decomposition/ptx/combined.ptx \
  --m 8193 \
  --n 8193 \
  --poly poly \
  --seeds 1 \
  --sketch-extra 20 \
  --sv-threshold 1e-6 \
  --output-dir "$C3_ARTIFACTS_DIR"
