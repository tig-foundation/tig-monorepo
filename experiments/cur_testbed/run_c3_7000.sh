#!/bin/bash
set -euo pipefail

# Keep every experiment parameter at the main runner's portable defaults and
# override only the square matrix dimensions.
export CUR_M=7000
export CUR_N=7000

exec experiments/cur_testbed/run_c3_bash.sh
