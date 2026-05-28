# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** mpc_router_v3
* **Copyright:** 2026 0x1E29106d0fA67Aa8848eFDa24BA12B0cDb42Da90
* **Identity of Submitter:** 0x1E29106d0fA67Aa8848eFDa24BA12B0cDb42Da90
* **Identity of Creator of Algorithmic Method:** Anonymous.
* **Unique Algorithm Identifier (UAI):** null
* **Predecessor:** mpc_v8


## Approach

Track-aware router dispatching to purpose-built solvers per battery-count band:

- **BASELINE** (n ≤ 15): AMPC v2 with ASCA + Bellman DP, ordering=0.
- **CONGESTED** (n ≤ 30): Online joint LP simplex (mpc_v8). Per-battery DP value tables
  provide continuation values; a two-phase full-tableau primal simplex runs at each RT step.
- **MULTIDAY** (n ≤ 50): AMPC v2 with shadow pricing + joint pair refinement, soc=41, asca=35.
- **DENSE** (n ≤ 80): AMPC v2 with shadow pricing + joint pair refinement, soc=41, asca=45.
- **CAPSTONE** (n > 80): AMPC v2 with shadow pricing + joint pair refinement, soc=41, asca=60.

On the three quality-capped tracks (MULTIDAY / DENSE / CAPSTONE) all algorithms converge to
the 10 M quality cap; the only meaningful differentiator is fuel throughput. mpc_router_v3
achieves **3.1× (MULTIDAY), 2.6× (DENSE), and 1.7× (CAPSTONE)** more bundles per compute
unit than titan_v1, because fuel per nonce is 4.77 B / 9.34 B / 26.88 B vs 15.02 B /
24.30 B / 46.62 B for titan respectively.


## Measured Performance

Fuel values are from LLVM-instrumented `.so` runs (deterministic, mirrors on-chain parity).
Source: `fuel_parity_records.csv`, 16 nonces/track, SHA b20950f8 (same code as this bundle;
solutions bit-for-bit identical; bundled fuel is 0.0–0.3% lower due to single-unit
compilation). Titan_v1 reference fuel from independent benchmark.

| Track     | Quality med | Threshold | Live cutoff¹ | Fuel med | Fuel max | Ratio vs titan_v1     |
|-----------|------------|-----------|--------------|----------|----------|-----------------------|
| BASELINE  | 1,006,880  | 700,000   | ~1,010,000   | 1.70 B   | 1.71 B   | 0.98× (parity)        |
| CONGESTED | 5,229,605  | 5,000,000 | ~6,284,000   | 2.52 B   | 2.59 B   | 0.70×                 |
| MULTIDAY  | 10,000,000 | 7,000,000 | 10,000,000   | 4.77 B   | 5.02 B   | 0.318× → **3.1×**     |
| DENSE     | 10,000,000 | 8,000,000 | 10,000,000   | 9.34 B   | 10.05 B  | 0.384× → **2.6×**     |
| CAPSTONE  | 10,000,000 | 8,000,000 | 10,000,000   | 26.88 B  | 28.74 B  | 0.577× → **1.7×**     |

¹ Live cutoff as of 2026-05-27. All fuel maxima are well under the 5 T cap (max 28.74 B =
0.57% of 5 T).

**Caveats:**
- BASELINE and CONGESTED pass the static threshold but the live top-100 cutoff exceeds our
  quality median → no qualifiers expected from these tracks at current competition levels.
- MULTIDAY minimum bundle-median cap-rate = 55.8% across 20 seeds (soc=41, full 500-nonce
  bundles). Functional, but narrowest of the three capped tracks.
- DENSE minimum cap-rate = 72.6%; CAPSTONE = 81.5% (both comfortable).


## Hyperparameters

All parameters are optional; defaults produce the measured results above.

| Parameter          | Type   | Default    | Description                                      |
|--------------------|--------|------------|--------------------------------------------------|
| `soc_levels`       | usize  | 41         | SOC grid size for MULTIDAY / DENSE / CAPSTONE    |
| `asca_iters`       | usize  | per-track  | ASCA iteration override for the dispatched track |
| `congested_method` | string | `"mpc_v8"` | CONGESTED branch: `mpc_v8` (default) \| `v8c`   |


## Acknowledgments

Builds on MPC/ADMM literature (Boyd et al. 2011; Bertsekas 2017; Rawlings et al. 2017),
two-phase LP simplex (Dantzig 1963), and TIG Protocol 0.0.5.


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
