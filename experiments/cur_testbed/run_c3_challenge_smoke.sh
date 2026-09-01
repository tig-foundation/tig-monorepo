#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/tmp/cur-challenge-cargo
export RUSTUP_HOME=/tmp/cur-challenge-rustup

if ! apt-get update; then
  # C3 base images may include unrelated third-party repositories. A transient
  # mirror-sync failure there should not prevent the CUR validation job from
  # installing packages from Ubuntu and NVIDIA.
  for source in /etc/apt/sources.list.d/*; do
    if [ -f "$source" ] && grep -q 'packages.fluentbit.io' "$source"; then
      mv "$source" "$source.disabled"
    fi
  done
  apt-get update
fi
apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl git libssl-dev pkg-config

if ! command -v nvcc >/dev/null 2>&1; then
  curl -fsSLo /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update
  apt-get install -y --no-install-recommends \
    cuda-compiler-12-2 cuda-cudart-dev-12-2 libcublas-dev-12-2 libcusolver-dev-12-2
fi

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --profile minimal
. "$CARGO_HOME/env"

nvcc --ptx experiments/cur_testbed/portable_kernels.cu \
  --output-file /tmp/cur-challenge.ptx \
  --gpu-architecture compute_70 \
  --use_fast_math \
  --optimize 3

export RUSTFLAGS='--cfg feature="cuda-12020"'
cargo test -p tig-challenges --features c009 --lib
cargo check -p tig-runtime --features c009
cargo check -p tig-verifier --features c009
cargo check -p tig-algorithms --features cur_decomposition --examples
cargo run --release -p tig-challenges \
  --example cur_decomposition_smoke \
  --features c009 -- \
  /tmp/cur-challenge.ptx

# Exercise an innovator algorithm across the complete index-only solution
# format. The verifier's shared fast-QR routine computes U for all eight
# sub-instances. `--seeds 0` performs one warm-up nonce without a benchmark sweep.
cargo run --release -p tig-algorithms \
  --example test_multi_instance \
  --features cur_decomposition -- \
  /tmp/cur-challenge.ptx \
  --seeds 0 \
  --algos 'leverage (1t)' \
  --sizes 2000x2000 \
  --poly

cp /tmp/cur-challenge.ptx "$C3_ARTIFACTS_DIR/cur_challenge_smoke.ptx"
