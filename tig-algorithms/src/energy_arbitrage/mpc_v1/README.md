# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** mpc_v1
* **Copyright:** 2026 0x1E29106d0fA67Aa8848eFDa24BA12B0cDb42Da90
* **Identity of Submitter:** 0x1E29106d0fA67Aa8848eFDa24BA12B0cDb42Da90
* **Identity of Creator of Algorithmic Method:** Anonymous. Based on established Model Predictive Control principles for networked battery storage arbitrage.
* **Unique Algorithm Identifier (UAI):** null


## Approach

Composite architecture inspired by The Architect's post-0.0.5 guidance (Protocol Update 0.0.5 announcement, 2026-02-05): *"composite algorithms that detect the track and apply specialized subroutines."*

Algorithm detects challenge size (`num_batteries × num_steps`) at start of `solve_challenge()` and routes to one of two policy branches:

**Branch A — Rolling-Horizon MPC** (for tractable track sizes ≤ 3000):
- 96-step lookahead using `challenge.take_step` with DA price override
- 5 candidates per battery, sequential coordinate descent sorted by capacity descending
- Joint flow enforcement via 32-iteration binary search uniform scaling
- Used for BASELINE (size=960) and CONGESTED (size=1920)

**Branch B — Completion-Focused Fallback** (for large track sizes > 3000):
- Single-step greedy with 3-step DA-forward averaging
- Conservative 50% action scaling vs `action_bounds`
- Flow feasibility via scaled retry [1.0, 0.5, 0.25, 0.0]
- Used for MULTIDAY (size=7680), DENSE (size=11520), CAPSTONE (size=19200)
- Ensures bundle completion within fuel budget on larger scenarios


## Measured Quality

**Multi-seed Variance Analysis**

Reference hardware: AMD Ryzen 9 7945HX, Rust release build.
Quality values are grand averages across multiple seed batches (TIG scale).

| Track | grand_avg | batch_CV | n | Notes |
|---|---|---|---|---|
| BASELINE | +1.20 | 36% | 300 (10×30) | Qualifying via bundle median |
| CONGESTED | +4.19 | 11% | 300 (10×30) | Strong, lowest variance |
| MULTIDAY | +3.67 | 38% | 100 (5×20) | High avg, variance from RT spikes |
| DENSE | +2.64 | 21% | 100 (5×20) | Stable fallback performance |
| CAPSTONE | +1.53 | 23% | 100 (5×20) | Positive, threshold-passing |

**Stability Disclosure**

All 5 tracks exhibit coefficient of variation > 20% at nonce level. This is inherent to the challenge: real-time prices include heavy-tailed Pareto jumps. Bundle-level aggregation (median per 10 nonces, TIG protocol Continuous `quality_type`) smooths per-nonce variance significantly. Benchmarkers should expect stable qualification at the -1.0 `min_active_quality` threshold currently active.


## Positioning

This algorithm is tuned for CONGESTED track performance (median quality ~3.2, batch CV 11%). It is also viable on MULTIDAY and DENSE via the fallback branch (completion-focused, positive quality). On BASELINE and CAPSTONE, current median quality (~0.3) qualifies under the starter threshold (-1.0) but represents opportunity for future iteration via algorithmic improvements — not protocol limitations.

Fuel utilization is efficient: maximum observed 16.5B instructions on CONGESTED (0.33% of the 5T `max_fuel_budget`), leaving substantial headroom for deeper horizons or scenario-based methods in future submissions.


## Known Limitations

- BASELINE median quality (~0.34) lower than CONGESTED (~3.16) despite both using full MPC. Under investigation: likely coordination sub-optimum with only 10 batteries. Future work.
- DENSE/CAPSTONE use fallback (not full MPC) due to v1 fuel budget caution. With measured headroom, future versions can upgrade these to full MPC.
- No stochastic scenario handling. RT price tail events (Pareto, α≈2.5 on CAPSTONE) occasionally produce negative nonce quality. Scenario-based MPC (K=10 price paths) is planned v2 direction.


## References and Acknowledgments

### 1. Academic

TIG Foundation. *"Optimal Arbitrage of Networked Energy Storage: Challenge Specification for TIG."* 2026.
https://docs.tig.foundation/static/energy_grid_optimization.pdf

### 2. Protocol

TIG Protocol Update 0.0.5 (2026-02-05) — Randomized Track Assignment.
Architecture of this algorithm derives from The Architect's guidance on composite algorithms.

### 3. Other

The Architect [TIG] Team — Protocol design enabling composite architectures.


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
