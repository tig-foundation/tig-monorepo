# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** knap_exact16
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

Bundle median over full protocol bundles, `hyperparameters = null`, 0 invalid nonces.

| Track | Median (seed c003r218a) | Max Fuel | % of cap |
|---|---:|---:|---:|
| n_items=1000,budget=5 | 458,872 | 1.14e9 | 0.023% |
| n_items=1000,budget=10 | 230,375 | 4.73e8 | 0.009% |
| n_items=1000,budget=25 | 51,469 | 1.19e9 | 0.024% |
| n_items=5000,budget=10 | 157,138 | 2.44e9 | 0.049% |
| n_items=5000,budget=25 | 46,072 | 4.08e9 | 0.082% |

All tracks are far inside the fuel cap — the largest observed use is 0.082% of the 5e12 budget,
a headroom of roughly 1,200×.

**Output is identical to the previous submission on 6,250 per-nonce comparisons across three
independent seeds, with zero differences.** Solve outcomes are deterministic, so
each nonce is an independent observation, not a summary statistic. The `n_items=1000,budget=5` track
is served by `track1.rs`, which is byte-identical (md5) to the previous submission's.

Determinism verified on every track and every cell: the harness re-runs the same 5 nonces and
requires per-nonce identical `quality` before a bundle is scored.

Speed against the previous submission, full protocol bundles, ABBA-counterbalanced
(`ref, cand, cand, ref`), same machine, same batch:

| Track | Speedup |
|---|---:|
| n_items=1000,budget=10 | 1.357× |
| n_items=1000,budget=25 | 1.263× |
| n_items=5000,budget=10 | 1.137× |
| n_items=5000,budget=25 | 1.178× |

A wider replication is in progress; `PROOF.md` states precisely which figures are single-seed and
which are not.

## References and Acknowledgments

### 1. Academic Papers
- Hochbaum, Baumann, Goldschmidt & Zhang, *A Fast and Effective Breakpoints Heuristic Algorithm for
  the Quadratic Knapsack Problem*, EJOR 2025 (arXiv:2408.12183)

### 2. Code References
- `knap_lean` (c003_a144) — the direct predecessor of this submission
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
