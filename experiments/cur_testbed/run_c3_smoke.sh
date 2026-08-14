#!/bin/bash
set -euo pipefail

export CUR_M=512
export CUR_N=640
export CUR_POLY=poly
export CUR_SEEDS=1
export CUR_SKETCH_EXTRA=12
export CUR_POWER_ITERS=1
export CUR_RESTARTS=2
export CUR_MAXVOL_SWAPS=4

exec experiments/cur_testbed/run_c3_bash.sh
