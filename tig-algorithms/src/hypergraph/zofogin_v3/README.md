# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** zofogin_v3
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|---|---|---|
| n_h_edges=10000 | 5T | `{"clusters": 4, "effort": 5, "fm_rounds": 48, "ils_iterations": 28, "ils_quick_refine": 240, "post_ils_polish": 300, "refinement": 80000, "swap_scale": 120}` |
| n_h_edges=20000 | 5T | `{"clusters": 4, "effort": 5, "fm_rounds": 41, "ils_iterations": 8, "ils_quick_refine": 260, "post_ils_polish": 90, "refinement": 100000, "swap_scale": 65}` |
| n_h_edges=50000 | 5T | `{"clusters": 4, "refinement": 100000}` |
| n_h_edges=100000 | 5T | `{}` |
| n_h_edges=200000 | 5T | `{"clusters": 4, "refinement": 100000}` |

- `aim_target`: the track's qualifying quality T, the lowest
  `qualifier_qualities_by_track[<track>]` in `/get-challenges` divided by 1e6.
  Without it T defaults to the line at block 1414645: 10000: 0.288542, 20000: 0.285567, 50000: 0.284779, 100000:
  0.280911, 200000: 0.282733. Stop margin 0.0083 on 10000, 0.00725 on 20000, 0.001 on 100000, 0.0015 on 50000 and
  200000: update T when the line moves more.
- All mica_bityite hyperparameters apply with their defaults. Nonces also end early at the extra settings `help()`
  lists.
- 20000 option: the recommended settings plus `"pz": 0`: 1.14 [1.08, 1.21] bundle qualifiers
  per compute against mica_xanthophyllite, 0.96 [0.34, 5.60] against the recommended settings (48 nonces); those
  stay first.

## Changes from mica_bityite

mica_bityite (c005_a045) with extra refinement near T on 50000 and 200000, and an earlier end
on every track. A nonce that ends just below T gets up to six more passes of the host refinement chain, stopping once
clearly above T. The passes keep only a better feasible partition, so no nonce ends lower; other
nonces run as in mica_bityite.

On 10000 (recommended settings, null or `{"refinement": 60000}`) and 20000 (also with `"pz": 0`) the quality is estimated once the grouped refinement ends, before
the iterated local search and polish, about half of the time. A nonce clearly above T (not at 60000), or too
far below T to reach it, ends there.
On 20000 to 200000 at the recommended or extra settings it also runs at fixed
points during the grouped refinement; on 50000, 200000 and the extra 100000 (clusters/effort/refinement) and 16-ILS 20000 settings it only ends nonces far below T.

## Results

Recommended settings, default T, official runtime and verifier on one GPU, fresh nonces, arms
interleaved per nonce. Compute includes verifier time. Bundle qualifiers per compute uses the protocol's median
and bundle size at default T, with 95% paired
bootstrap intervals.

| Track | 10000 | 20000 | 50000 | 100000 | 200000 |
|---|---|---|---|---|---|
| Nonces | 96 | 96 | 48 | 96 | 48 |
| Bundle qualifiers per compute against mica_bityite | 1.24 [1.14, 1.37] | 1.18 [1.11, 1.26] | 1.44 [0.93, 2.22] | 1.05 [1.00, 1.11] | 1.84 [1.06, 4.96] |
| Speed step (same outputs, 16-32 nonces) | 1.02 [1.01, 1.03] | 1.02 [1.01, 1.03] | 1.04 [1.03, 1.04] | 1.06 [1.05, 1.07] | 1.04 [1.03, 1.04] |

- 10000, 20000 and 100000: 34, 43 and 34 of 96 nonces ended early, at 0.48, 0.70 and 0.85 of mica_bityite's
  time on those nonces; no nonce crossed the line (35, 37 and 47 qualifying nonces);
  other outputs equal mica_bityite's.
- 200000 and 50000: the extension raised 10 and 4 of 48 nonces (+750 to +1480 and +380 to +890 ppm);
  no 50000 nonce crossed the line. Against zofogin_v2: 200000 1.74 [1.22, 2.06].
- Speed step, 10000: null 1.03 [1.02, 1.04]; `{"refinement": 60000}` (96 nonces) 0.99 [0.96, 1.04].
- Later steps, each against the previous: late end (50000, 200000) 1.14 [1.09, 1.19], 1.07 [1.02, 1.12] (96, 56
  nonces), ending only nonces over 0.0058 below T; mid-refinement checks 20000 1.08 [1.04, 1.12], 50000 1.02
  [1.01, 1.04], 200000 1.10 [1.05, 1.16] (96, 32, 48); same outputs: 10000 0.99, 100000 1.00; ends and checks
  at extra settings (24-32 nonces; 100000/20000 inferred): 50000 `"effort": 5, "refinement": 80000` 1.20 [1.08, 1.38];
  200000 `"effort": 5` 1.15 [1.05, 1.28], `"refinement": 80000` 1.19 [1.06, 1.37]; 100000
  `"clusters": 4, "refinement": 100000` 1.04 [1.01, 1.07]; 20000 16-ILS 1.05 [0.99, 1.14].

## References and Acknowledgments

### Academic Papers
- The references of mica_bityite apply to the partitioner.
- The quality estimate uses the greedy recursive bisection baseline of tig-challenges (hypergraph/mod.rs).

### Code References
- mica_bityite (c005_a045), FP Labs: the partitioner this code extends. Copyright of the mica_bityite files remains with FP Labs.
- zofogin_v2 (tilk): the quality estimate.

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
