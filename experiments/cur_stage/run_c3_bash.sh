#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/tmp/cur-experiment-cargo
export RUSTUP_HOME=/tmp/cur-experiment-rustup
CUR_CUDA_PACKAGE_SERIES=${CUR_CUDA_PACKAGE_SERIES:-12-2}
CUR_CUDA_DIRECTORY=${CUR_CUDA_DIRECTORY:-12.2}

apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates curl git libssl-dev pkg-config

if ! command -v nvcc >/dev/null 2>&1; then
  curl -fsSLo /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update
  apt-get install -y --no-install-recommends \
    "cuda-compiler-$CUR_CUDA_PACKAGE_SERIES" \
    "cuda-cudart-dev-$CUR_CUDA_PACKAGE_SERIES" \
    "libcublas-dev-$CUR_CUDA_PACKAGE_SERIES" \
    "libcusolver-dev-$CUR_CUDA_PACKAGE_SERIES"
fi

export PATH="/usr/local/cuda-$CUR_CUDA_DIRECTORY/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-$CUR_CUDA_DIRECTORY/lib64:${LD_LIBRARY_PATH:-}"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$CARGO_HOME/env"

rustc --version
nvcc --version
nvidia-smi

CUR_M=${CUR_M:-8193}
CUR_N=${CUR_N:-8193}
CUR_POLY=${CUR_POLY:-poly}
CUR_SEEDS=${CUR_SEEDS:-1}
CUR_SKETCH_EXTRA=${CUR_SKETCH_EXTRA:-20}
CUR_POWER_ITERS=${CUR_POWER_ITERS:-1}
CUR_RESTARTS=${CUR_RESTARTS:-2}
CUR_MAXVOL_SWAPS=${CUR_MAXVOL_SWAPS:-8}
CUR_MAXVOL_TOLERANCE=${CUR_MAXVOL_TOLERANCE:-1.05}

cargo run --release --example cur_stage_experiment --features cur_decomposition -- \
  tig-algorithms/lib/cur_decomposition/ptx/combined.ptx \
  --m "$CUR_M" \
  --n "$CUR_N" \
  --poly "$CUR_POLY" \
  --seeds "$CUR_SEEDS" \
  --sketch-extra "$CUR_SKETCH_EXTRA" \
  --power-iters "$CUR_POWER_ITERS" \
  --sophisticated-restarts "$CUR_RESTARTS" \
  --maxvol-swaps "$CUR_MAXVOL_SWAPS" \
  --maxvol-tolerance "$CUR_MAXVOL_TOLERANCE" \
  --sv-threshold 1e-6 \
  --output-dir "$C3_ARTIFACTS_DIR"
