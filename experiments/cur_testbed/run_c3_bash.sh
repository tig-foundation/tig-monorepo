#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/tmp/cur-testbed-cargo
export RUSTUP_HOME=/tmp/cur-testbed-rustup
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

CUR_TESTBED_PTX=${CUR_TESTBED_PTX:-/tmp/cur-testbed-portable.ptx}
nvcc --ptx experiments/cur_testbed/portable_kernels.cu \
  --output-file "$CUR_TESTBED_PTX" \
  --gpu-architecture compute_70 \
  --use_fast_math \
  --optimize 3

CUR_M=${CUR_M:-2000}
CUR_N=${CUR_N:-3000}
CUR_DELTA=${CUR_DELTA:-10000}
CUR_POLY=${CUR_POLY:-poly}
CUR_SPECTRUM_A=${CUR_SPECTRUM_A:-13}
CUR_SEEDS=${CUR_SEEDS:-1}
CUR_SKETCH_EXTRA=${CUR_SKETCH_EXTRA:-20}
CUR_POWER_ITERS=${CUR_POWER_ITERS:-1}
CUR_RESTARTS=${CUR_RESTARTS:-2}
CUR_MAXVOL_SWAPS=${CUR_MAXVOL_SWAPS:-8}
CUR_MAXVOL_TOLERANCE=${CUR_MAXVOL_TOLERANCE:-1.05}
CUR_SV_THRESHOLD=${CUR_SV_THRESHOLD:-1e-6}

RUSTFLAGS='--cfg feature="cuda-12020"' \
cargo run --release --example cur_testbed --features cur_decomposition -- \
  "$CUR_TESTBED_PTX" \
  --m "$CUR_M" \
  --n "$CUR_N" \
  --delta "$CUR_DELTA" \
  --poly "$CUR_POLY" \
  --spectrum-a "$CUR_SPECTRUM_A" \
  --seeds "$CUR_SEEDS" \
  --sketch-extra "$CUR_SKETCH_EXTRA" \
  --power-iters "$CUR_POWER_ITERS" \
  --sophisticated-restarts "$CUR_RESTARTS" \
  --maxvol-swaps "$CUR_MAXVOL_SWAPS" \
  --maxvol-tolerance "$CUR_MAXVOL_TOLERANCE" \
  --sv-threshold "$CUR_SV_THRESHOLD" \
  --output-dir "$C3_ARTIFACTS_DIR"

cp "$CUR_TESTBED_PTX" "$C3_ARTIFACTS_DIR/cur_testbed_portable.ptx"
