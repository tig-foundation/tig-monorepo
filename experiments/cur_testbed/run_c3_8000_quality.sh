#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/tmp/cur-quality-cargo
export RUSTUP_HOME=/tmp/cur-quality-rustup
CUR_M=${CUR_M:-8000}
CUR_N=${CUR_N:-8000}
CUR_REPORT_NAME=${CUR_REPORT_NAME:-cur_quality_8000_report.json}
CUR_REPORT_PREFIX=${CUR_REPORT_PREFIX:-cur_quality_8000}
CUR_SEED_START=${CUR_SEED_START:-0}
CUR_SEEDS=${CUR_SEEDS:-1}
CUR_APT_OPTIONS=(
  -o Acquire::ForceIPv4=true
  -o Acquire::http::Timeout=15
  -o Acquire::https::Timeout=15
  -o Acquire::Retries=1
)

if ! apt-get "${CUR_APT_OPTIONS[@]}" update; then
  for source in /etc/apt/sources.list.d/*; do
    if [ -f "$source" ] && grep -q 'packages.fluentbit.io' "$source"; then
      mv "$source" "$source.disabled"
    fi
  done
  apt-get "${CUR_APT_OPTIONS[@]}" update
fi
apt-get "${CUR_APT_OPTIONS[@]}" install -y --no-install-recommends \
  build-essential ca-certificates curl git libssl-dev pkg-config

if ! command -v nvcc >/dev/null 2>&1; then
  curl -fsSLo /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get "${CUR_APT_OPTIONS[@]}" update
  apt-get "${CUR_APT_OPTIONS[@]}" install -y --no-install-recommends \
    cuda-compiler-12-2 cuda-cudart-dev-12-2 libcublas-dev-12-2 libcusolver-dev-12-2
fi

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --profile minimal
. "$CARGO_HOME/env"

nvcc --ptx experiments/cur_testbed/portable_kernels.cu \
  --output-file /tmp/cur-quality.ptx \
  --gpu-architecture compute_70 \
  --use_fast_math \
  --optimize 3

export RUSTFLAGS='--cfg feature="cuda-12020"'
cargo check -p tig-algorithms \
  --features cur_decomposition \
  --example cur_quality_benchmark
for ((offset = 0; offset < CUR_SEEDS; offset++)); do
  seed_index=$((CUR_SEED_START + offset))
  if [ "$CUR_SEEDS" -eq 1 ]; then
    report_path="$C3_ARTIFACTS_DIR/$CUR_REPORT_NAME"
  else
    report_path="$C3_ARTIFACTS_DIR/${CUR_REPORT_PREFIX}_seed${seed_index}.json"
  fi
  cargo run --release -p tig-algorithms \
    --features cur_decomposition \
    --example cur_quality_benchmark -- \
    /tmp/cur-quality.ptx \
    "$report_path" \
    "$CUR_M" \
    "$CUR_N" \
    "$seed_index"
done

cp /tmp/cur-quality.ptx "$C3_ARTIFACTS_DIR/cur_quality_8000.ptx"
