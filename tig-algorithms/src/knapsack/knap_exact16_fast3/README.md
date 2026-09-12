# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** knap_exact16_fast3
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

## Measured Against the Previous Submission

Reference: `knap_exact16` (on-chain `c003_a150`, source `submission005_exactsearch16`). Full protocol
bundles, `hyperparameters = null`, same machine, same batch, ABBA-counterbalanced.

| Track | Speedup | Per-nonce output identical | Median (seed c003r300a) | Max Fuel | % of cap |
|---|---:|---|---:|---:|---:|
| n_items=1000,budget=5 | 1.346× | 4000/4000 | 463,494 | 3.76e8 | 0.008% |
| n_items=1000,budget=10 | 1.117× | 4000/4000 | 231,250 | 2.42e8 | 0.005% |
| n_items=1000,budget=25 | 1.065× | 4000/4000 | 51,721 | 9.00e8 | 0.018% |
| n_items=5000,budget=10 | 1.009× | 200/200 | 148,542 | 2.35e9 | 0.047% |
| n_items=5000,budget=25 | 1.036× | 200/200 | 46,993 | 3.41e9 | 0.068% |

Speedups pool two seeds measured in mirrored orders (ABBA and BAAB); an A/A control in the same batch
read +0.10%. **Output is identical to the previous submission on every one of 12,400 per-nonce
comparisons**, and the bundle medians are byte-identical to it. Solve
outcomes are deterministic, so each nonce is an independent observation, and quality is unchanged by
construction. See `PROOF.md`.

## References and Acknowledgments

### 1. Academic Papers
- Hochbaum, Baumann, Goldschmidt & Zhang, *A Fast and Effective Breakpoints Heuristic Algorithm for
  the Quadratic Knapsack Problem*, EJOR 2025 (arXiv:2408.12183)

### 2. Code References
- `knap_exact16` (c003_a150) — the direct predecessor of this submission
- `knap_lean` (c003_a144)
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
