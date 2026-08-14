#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/tmp/cur-algorithm-cargo
export RUSTUP_HOME=/tmp/cur-algorithm-rustup

apt-get update
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

export RUSTFLAGS='--cfg feature="cuda-12020"'
cargo check -p tig-algorithms --features cur_decomposition --examples
