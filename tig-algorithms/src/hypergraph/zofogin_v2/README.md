# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** zofogin_v2
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

- `aim_target`: the qualifying quality T of the track. T is the lowest value of
  `qualifier_qualities_by_track[<track>]` in `/get-challenges` divided by 1e6 (280037 on chain is 0.280037).
  Update it when the line moves by more than the stop margin (table below). A stale value costs qualifiers
  both ways: above the line it gives up nonces that would qualify, below it, nonces can stop too close to the
  line. Without a numeric value T defaults to the line minus 0.0005, with stop margin 0.009 on 20000 and 0.005
  on 50000-200000: 20000: 0.28523 (block 1405500), 50000: 0.284982, 100000: 0.280511 (block 1405450),
  200000: 0.282474 (block 1405859). Not used on 10000.
- All mica_clintonite hyperparameters still apply. Stopping below T on 20000 to 200000 is used only with the
  recommended settings: clusters 4, refinement 80000 to 120000, no other keys except `aim_target`, `effort` 5 and,
  on 20000, the six other keys of its row at those values.
- Additional exact recipes, with optional `aim_target` and no other keys:
  - 50000: `{"clusters":4,"effort":5,"fm_rounds":10,"ils_iterations":16,"ils_quick_refine":150,"post_ils_polish":260,"refinement":100000,"swap_scale":120}`.
  - 100000: `{"clusters":4,"refinement":50000}`.
  - 200000: `{"clusters":4,"refinement":60000}`.

## Changes from mica_clintonite

mica_clintonite (c005_a039) with stopping and extra refinement near T on 20k-200k. A bundle qualifies when its
median nonce quality reaches T.

A nonce stops once its estimated quality is clearly above T, or once T is out of reach. A nonce that ends just
below T gets extra refinement until it is above T or the extra rounds run out (on 50000 to 200000, short reruns of
the coldest refinement rounds before the passes); it only keeps a better partition, so no nonce ends lower than with
mica_clintonite.

| Track | 20000 | 50000 | 100000 | 200000 |
|---|---|---|---|---|
| stop margin, recommended | 0.0025 | 0.002 | 0.001 | 0.0009 |
| stop margin, additional exact recipe | - | 0.0045 | 0.0028 | 0.0028 |

With other settings the stop above T and the extra passes are used. On 10k the code runs mica_clintonite unchanged.
Runs stay deterministic: every random draw is seeded from `challenge.seed`.

## Results

Recommended settings with `aim_target` = T, compared with mica_clintonite at the same settings without
`aim_target`, on the same fresh nonces, interleaved official tig-runtime / tig-verifier runs. T is the line at
block 1403680. Compute includes verifier time. Bundle qualifiers per compute uses the protocol's median rule and
bundle size, averaged over lines within 0.001 of T.

| Track | 10000 | 20000 | 50000 | 100000 | 200000 |
|---|---|---|---|---|---|
| Compared with | mica_clintonite | mica_clintonite | mica_clintonite | mica_clintonite | mica_clintonite |
| Nonces | 2 | 48 | 48 | 48 | 48 |
| At or above T | same output | 20 / 20 | 14 / 17 | 17 / 21 | 15 / 19 |
| Qualifiers lost against mica_clintonite | 0 | 0 | 0 | 0 | 0 |
| Compute ratio | 1.00 | 0.86 | 0.77 | 0.71 | 0.60 |
| Qualifying nonces per compute | 1.00 | 1.17 | 1.58 | 1.74 | 2.09 |
| Bundle qualifiers per compute | 1.00 | 1.72 | 4.37 | 3.51 | 3.33 |

## References and Acknowledgments

### Academic Papers
- Kernighan & Lin, "An Efficient Heuristic Procedure for Partitioning Graphs", Bell System Technical Journal 49(2), 1970, https://doi.org/10.1002/j.1538-7305.1970.tb01770.x - pair swaps of the block bisection.
- The references of mica_clintonite apply to the partitioner.

### Code References
- zofogin (c005_a042), tilk: the stopping this code extends.
- mica_clintonite (c005_a039), FP Labs. Copyright of the mica_clintonite files remains with FP Labs.
- ember_stromboli (c007_a040) also stops its search once the result is good enough.
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
