#!/bin/bash
set -euo pipefail

export CUR_M=8000
export CUR_N=8000
export CUR_SEED_START=0
export CUR_SEEDS=3
export CUR_REPORT_PREFIX=cur_quality_8000

exec bash experiments/cur_testbed/run_c3_8000_quality.sh
