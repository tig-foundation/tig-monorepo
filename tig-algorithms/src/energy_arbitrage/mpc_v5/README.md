# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** mpc_v5
* **Copyright:** 2026 0x1E29106d0fA67Aa8848eFDa24BA12B0cDb42Da90
* **Identity of Submitter:** 0x1E29106d0fA67Aa8848eFDa24BA12B0cDb42Da90
* **Identity of Creator of Algorithmic Method:** Anonymous. Based on established Model Predictive Control principles for networked battery storage arbitrage, plus oracle scheduling for low-volatility regimes.
* **Unique Algorithm Identifier (UAI):** null
* **Predecessor:** mpc_v4


## Approach

Four-tier composite architecture based on challenge size (num_batteries × num_steps):

**Branch Baseline — Per-node Oracle Greedy** (size ≤ 1000): NEW in mpc_v5.
- Full remaining-horizon DA price average per battery node (vs 12-step window in built-in greedy).
- Per-node prices (vs node 0 for all batteries in built-in greedy).
- Charge at steps where DA price < remaining_avg − $5; discharge where > remaining_avg + $5.
- Per-line softening + binary-search flow enforcement for feasibility.
- Handles BASELINE (10 batteries × 96 steps = 960).

**Branch A — Full MPC** (1000 < size ≤ 3000): unchanged from mpc_v4.
- H=96, N candidates (effort-dependent), coordinate descent over battery order.
- Binary-search flow enforcement. Handles CONGESTED (1920).

**Branch B — v5 Greedy Fallback** (3000 < size ≤ 15000): unchanged from mpc_v4.
- RT+DA sanity check (0.95/1.05 thresholds), 50% scaling, 3-step DA window.
- Handles MULTIDAY (7680), DENSE (11520).

**Branch C — H=1 MPC with DA override** (size > 15000): unchanged from mpc_v4.
- N candidates per battery (effort-dependent), in-call warm-start at effort=4.
- DA price override at h=0 eliminates RT spike misdirection.
- Terminal SOC value provides one-step forward-looking signal.
- Handles CAPSTONE (100 batteries × 192 steps = 19200).

**Routing thresholds**: BASELINE_THRESHOLD=1000, FULL_MPC_THRESHOLD=3000, SHALLOW_MPC_THRESHOLD=15000.


## Hyperparameters

Single hyperparameter `effort` (integer 0–4, **default 3**):

| effort | Branch A | Branch C | Warm-start | Notes |
|--------|----------|----------|------------|-------|
| 0–2    | H=96, N=5 | N=3 | off | mpc_v3-equivalent quality |
| **3 (default)** | H=96, N=9 | N=5 | off | Recommended — +9.8% CAPSTONE median |
| 4      | H=128, N=17 | N=5 | on | Maximum quality, slower |

Note: branch_baseline (size ≤ 1000) is **effort-agnostic** — uses oracle scheduling regardless of effort.

Recommended submission:
```json
{"effort": 3}
```
or `null` (uses default effort=3).


## Measured Quality (5000-seed validation, mainnet baseline)

| Track | Qualifying | Median Quality | × baseline |
|---|---|---|---|
| BASELINE | 17% | 0.561 | 1.56× |
| CONGESTED | 99.9% | 3.550 | 4.55× |
| MULTIDAY | 86% | 1.686 | 2.69× |
| DENSE | 88% | 1.753 | 2.75× |
| CAPSTONE | 100% | 7.052 | 8.05× |


### Key improvements over mpc_v4

- **BASELINE**: 7% → 17% qualifying (+2.4×) via branch_baseline with per-node full-horizon scheduling.
- **CAPSTONE median**: 6.79 → 7.05 (+3.8%) via effort=3 (n_cand_c=5 vs 3).
- Other tracks: unchanged.


### Architecture insight

BASELINE qualifying ceiling is fundamentally bounded at ~17% by built-in greedy baseline performance. The branch_baseline module achieves >99% of oracle profit (LP with perfect foresight) on all BASELINE nonces. The built-in greedy baseline captures 60-96% of oracle profit on 83% of scenarios, making 2×-baseline qualifying mathematically infeasible on those nonces.


## References and Acknowledgments

### 1. Predecessor

mpc_v4 — three-branch composite (Full MPC + Greedy fallback + H=1 MPC DA-override). mpc_v5 adds branch_baseline for low-volatility small problems (BASELINE track).

### 2. Academic

TIG Foundation. *"Optimal Arbitrage of Networked Energy Storage."* 2026.

Rawlings, J.B., Mayne, D.Q. & Diehl, M. (2017) — "Model Predictive Control: Theory, Computation, and Design." 2nd ed., Nob Hill Publishing.

### 3. Protocol

TIG Protocol Update 0.0.5 (2026-02-05).


## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses
