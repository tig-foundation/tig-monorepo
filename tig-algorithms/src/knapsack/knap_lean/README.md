# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** knap_lean
* **Copyright:** 2026 ChervovNikita
* **Identity of Submitter:** ChervovNikita (gcnikitachervov@gmail.com)
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track | Fuel Budget | Nonces/Bundle |
|---|---|---|
| n_items=1000,budget=5 | 5T | 1000 |
| n_items=1000,budget=10 | 5T | 1000 |
| n_items=1000,budget=25 | 5T | 1000 |
| n_items=5000,budget=10 | 5T | 50 |
| n_items=5000,budget=25 | 5T | 50 |

## Measured Quality

Bundle median over full protocol bundles, two independent seeds, `hyperparameters = null`,
0 invalid nonces.

| Track | Median (seed c003r127a) | Median (seed c003r128b) | Max Fuel | % of cap |
|---|---:|---:|---:|---:|
| n_items=1000,budget=5 | 455,826 | 457,473 | 1.14e9 | 0.023% |
| n_items=1000,budget=10 | 231,930 | 230,567 | 2.16e9 | 0.043% |
| n_items=1000,budget=25 | 52,660 | 52,623 | 1.86e9 | 0.037% |
| n_items=5000,budget=10 | 155,106 | 147,533 | 8.19e9 | 0.164% |
| n_items=5000,budget=25 | 48,572 | 45,498 | 1.53e10 | 0.307% |

Max Fuel is the worst single nonce over both seeds (2000 nonces per 1000-item track, 100 per
5000-item track).

The two n=1000 low-budget figures fell by more than half against an earlier revision of this file
(2.70e9 → 1.14e9 and 4.81e9 → 2.16e9). That is the bound prunes: fuel is a basic-block counter, so
eliminating loop iterations lowers it directly. The three unchanged rows are the tracks those patches
do not touch. **Both figures were recomputed from the gated bundles of this exact artifact**, not
carried over.

All tracks are far inside the fuel cap — the largest observed use is 0.31% of the 5e12 budget.

Determinism verified on every track: re-running the same 30 nonces reproduces per-nonce identical
`quality` AND `fuel_consumed`.

## References and Acknowledgments

### 1. Academic Papers
- Hochbaum, Baumann, Goldschmidt & Zhang, *A Fast and Effective Breakpoints Heuristic Algorithm for
  the Quadratic Knapsack Problem*, EJOR 2025 (arXiv:2408.12183)

### 2. Code References
- `knap_master_v2` (c003_a140)
- `superfast_knap_v1` (c003_a137)
- `knap_quality_opt_v11` (c003_a133)
- TIG baseline `tabu_search`

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
