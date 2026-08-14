#!/bin/bash
set -euo pipefail

export CUR_M=1025
export CUR_N=1100
export CUR_POLY=poly
export CUR_SEEDS=1
export CUR_SKETCH_EXTRA=12
export CUR_POWER_ITERS=1
export CUR_RESTARTS=2
export CUR_MAXVOL_SWAPS=4

exec experiments/cur_stage/run_c3_bash.sh
