# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** zofogin
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|---|---|---|
| n_h_edges=10000 | 5T | `{"clusters": 4, "effort": 5, "fm_rounds": 48, "ils_iterations": 28, "ils_quick_refine": 240, "post_ils_polish": 300, "refinement": 80000, "swap_scale": 120}` (same output as mica_clintonite) |
| n_h_edges=20000 | 5T | `{"clusters": 4, "effort": 5, "fm_rounds": 41, "ils_iterations": 8, "ils_quick_refine": 260, "post_ils_polish": 90, "refinement": 100000, "swap_scale": 65, "aim_target": T}` |
| n_h_edges=50000 | 5T | `{"clusters": 4, "refinement": 100000, "aim_target": T}` |
| n_h_edges=100000 | 5T | `{"clusters": 4, "refinement": 100000, "aim_target": T}` |
| n_h_edges=200000 | 5T | `{"clusters": 4, "refinement": 100000, "aim_target": T}` |

- `aim_target`: the qualifying quality T of the track, as a fraction. T is the lowest value of
  `qualifier_qualities_by_track[<track>]` in `/get-challenges` divided by 1e6 (280037 on chain is 0.280037).
  Update it when the line moves by more than the stop margin (table below). A stale value costs qualifiers
  both ways: above the line it gives up nonces that would qualify, below it, nonces can stop too close to the
  line. A missing or non-numeric value takes the default, the line plus 0.0008 (mean over blocks 1395245-1397665;
  20000: 1395245-1398139; 100000: block 1399558): 20000: 0.2850, 50000: 0.2844, 100000: 0.280343,
  200000: 0.2830. Not used on 10000.
- All mica_clintonite hyperparameters still apply. Stopping below T is used only with the
  recommended settings: clusters 4, refinement 80000 to 120000, no other keys except `aim_target`, `effort` 5 and, on 20000,
  the six other keys of its row at those values.
- Additional exact recipes, with optional `aim_target` and no other keys:
  - 50000: `{"clusters":4,"effort":5,"fm_rounds":10,"ils_iterations":16,"ils_quick_refine":150,"post_ils_polish":260,"refinement":100000,"swap_scale":120}`.
  - 100000: `{"clusters":4,"refinement":50000}`.
  - 200000: `{"clusters":4,"refinement":60000}`.

## Changes from mica_clintonite

mica_clintonite (c005_a039) with stopping on 20k-200k. A bundle qualifies when its median nonce quality reaches T.

A nonce stops once its estimated quality is clearly above T, or once T is out of reach.

| Track | 20000 | 50000 | 100000 | 200000 |
|---|---|---|---|---|
| stop margin, recommended | 0.0076 | 0.00375 | 0.0021 | 0.00135 |
| stop margin, additional exact recipe | - | 0.0048 | 0.0039 | 0.0033 |

With other settings only the stop above T is used. On 10k the code runs mica_clintonite unchanged.
Runs stay deterministic: every random draw is seeded from `challenge.seed`.

## Results

Recommended settings, fresh paired nonces, interleaved official tig-runtime / tig-verifier runs. T is the
line at block 1395762 (20k: 1398139). Compute includes verifier time. Qualifiers per compute is the qualifier
ratio divided by the compute ratio.

| Track | 10000 | 20000 | 50000 | 100000 | 200000 |
|---|---|---|---|---|---|
| Compared with | mica_clintonite | mica_clintonite | mica_clintonite | mica_clintonite | mica_clintonite |
| Nonces | 3 | 60 | 30 | 24 | 24 |
| At or above T | same output | 30 / 30 | 19 / 19 | 8 / 8 | 9 / 9 |
| Compute ratio | 1.00 | 0.85 | 0.79 | 0.75 | 0.81 |
| Qualifiers per compute | 1.00 | 1.18 | 1.26 | 1.33 | 1.24 |

## References and Acknowledgments

### Academic Papers
- Kernighan & Lin, "An Efficient Heuristic Procedure for Partitioning Graphs", Bell System Technical Journal 49(2), 1970, https://doi.org/10.1002/j.1538-7305.1970.tb01770.x - pair swaps of the block bisection.
- The references of mica_clintonite apply to the partitioner.

### Code References
- mica_clintonite (c005_a039), FP Labs. Copyright of the mica_clintonite files remains with FP Labs.
- TIG greedy baseline (tig-challenges hypergraph kernels.cu).

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
