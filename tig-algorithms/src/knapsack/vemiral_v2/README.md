# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** vemiral_v2
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|---|---|---|
| n_items=1000,budget=5 | 5T | `{"aim_target": T}` |
| n_items=1000,budget=10 | 5T | `{"aim_target": T}` |
| n_items=1000,budget=25 | 5T | `{"aim_target": T}` |
| n_items=5000,budget=10 | 5T | `{"aim_target": T}` |
| n_items=5000,budget=25 | 5T | `{"aim_target": T}` |

- `aim_target`: the qualifying quality T of the track, as a fraction. T is the lowest value of
  `qualifier_qualities_by_track[<track>]` in `/get-challenges` divided by 1e6 (164722 on chain is 0.164722).
  Update it when the line moves by more than 0.0005 (0.001 on 5000/10); a stale value loses qualifiers
  either way. A missing or non-numeric value takes the default:

  | Track | 1000, b=5 | 1000, b=10 | 1000, b=25 | 5000, b=10 | 5000, b=25 |
  |---|---|---|---|---|---|
  | default `aim_target` | 0.4679 | 0.2357 | 0.0534 | 0.165 | 0.051 |

- `{"aim": 0}` gives the packrun output.
- All packrun hyperparameters still apply. Stopping margins were set at packrun's defaults.

## Changes from vemiral

vemiral (c003_a159) with less work on nonces whose outcome is already decided, and no change to any nonce's
qualification at T - 0.0025 or above:
- 1000/25: the flow bound runs before the search. A nonce stops at once when its flow solution is already at
  least T + 0.003 or its certified upper bound is below T - 0.0025. Nonces that continue reuse its
  breakpoints and flow solution as the search seed.
- 1000/5 and 1000/10: the band rerun, block exchanges, exact core search and 135-round rerun are skipped when
  the certified upper bound is below T - 0.0025, or when a short exact search proves that no solution reaches
  T - 0.0025 (tried before the band rerun and, on 1000/10, before the 135-round rerun). A solution it finds
  is kept if better.
- Reruns of nonces just below T continue the first pass instead of starting again, with the same output as a
  rerun from scratch: on 1000/25 and 1000/10 each 135-round restart whose input is unchanged resumes where its
  first-pass run stopped; on 1000/5 and 1000/10 the heavier core rerun keeps the first pass's flow pass and
  local search and runs only the larger core search.
- 1000/5: the flow pass reads the interaction rows directly and sums them in integers; the search matrix is
  built only for nonces that continue past it. On 1000/5 and 1000/25 each node's arc list is allocated once.
- The baseline is computed once per instance. 5000/10 and 5000/25 run vemiral's code unchanged.

vemiral's own changes from packrun (c003_a156) are unchanged: the effort of each nonce is set by the
qualifying quality T of its track, since a bundle qualifies when its median nonce quality reaches T.

## Results

Same fresh nonces for both codes, runs interleaved on the same core, official tig-runtime plus tig-verifier
CPU per nonce, both codes with `{"aim_target": T}` at the line of block 1418937. Qualifiers per compute =
(nonces at or above T) ratio / compute ratio, with its 95% interval.

| Track | 1000, b=5 | 1000, b=10 | 1000, b=25 |
|---|---|---|---|
| Compared with | vemiral | vemiral | vemiral |
| `aim_target` | 0.467914 | 0.235644 | 0.053399 |
| Nonces | 960 | 2880 | 1440 |
| At or above T: vemiral_v2 / vemiral | 410 / 410 | 1336 / 1336 | 658 / 658 |
| Compute ratio, vemiral_v2 / vemiral | 0.985 | 0.967 | 0.964 |
| Qualifiers per compute | 1.015 [1.007, 1.024] | 1.034 [1.023, 1.046] | 1.037 [1.026, 1.047] |

No nonce changed sides of T, T +- 0.001 or T +- 0.0025. On the 5000-item tracks the
output and compute are vemiral's.

## References and Acknowledgments

### Academic Papers
- D. Pisinger, "The quadratic knapsack problem - a survey", Discrete Applied Mathematics, 2007.
- A. Caprara, D. Pisinger, P. Toth, "Exact Solution of the Quadratic Knapsack Problem", INFORMS Journal on Computing, 1999.
- H. R. Lourenco, O. C. Martin, T. Stutzle, "Iterated Local Search", Handbook of Metaheuristics, 2003.

### Code References
- packrun (c003_a156), NVX. Copyright remains with NVX.
- TIG baseline (knapsack/baselines/tabu_search.rs).
- ember_stromboli (job_scheduling) also stops its search once the result is good enough.

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
