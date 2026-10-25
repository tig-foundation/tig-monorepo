# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** vemiral
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

## Changes from packrun

packrun (c003_a156) with the effort of each nonce set by the qualifying quality T of its track. A bundle
qualifies when its median nonce quality reaches T, so a nonce stops once it is clearly above T or cannot
reach it, and nonces that end just below T get a further search towards it.

## Results

Same fresh nonces for all codes: 2000 per 1000-item track, 240 on 5000/25, 500 on 5000/10 (compute from
200). satchel with defaults and with the `ils_rounds` some benchmarkers send (135 on 1000/10 and 1000/25,
226 on 5000/10, 141 on 5000/25), vemiral with defaults. Compute: tig-runtime plus tig-verifier CPU per
nonce. T at block 1399520.

| | 1000, b=5 | 1000, b=10 | 1000, b=25 | 5000, b=10 | 5000, b=25 |
|---|---|---|---|---|---|
| nonces at or above T: satchel / satchel ils / vemiral | 898 / - / 899 | 933 / 933 / 937 | 928 / 929 / 929 | 157 / 157 / 158 | 71 / 72 / 72 |
| compute, vemiral / satchel | 0.98 | 0.97 | 0.94 | 0.91 | 0.89 |
| compute, vemiral / satchel ils | - | 0.55 | 0.83 | 0.66 | 0.90 |

No nonce at or above T with satchel fell below T with vemiral. Within T +- 0.001 single nonces can cross
either way on 1000/25 and the 5000-item tracks. Weighted by the track mix and settings of the three largest
c003 benchmarkers (block 1399518), vemiral gives 1.22x, 1.31x and 1.42x the qualifying bundles of satchel
per unit of compute, after the 0.975 age multiplier of a newly live code (95% lower bounds 1.03, 1.08, 1.22).

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
