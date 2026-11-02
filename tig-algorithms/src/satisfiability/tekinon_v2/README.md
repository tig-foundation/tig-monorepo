# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** tekinon_v2
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|---|---|---|
| n_vars=5000,ratio=4267 | 5T | `null` |
| n_vars=7500,ratio=4267 | 5T | `null` |
| n_vars=10000,ratio=4267 | 5T | `null` |
| n_vars=100000,ratio=4150 | 5T | `null` |
| n_vars=100000,ratio=4200 | 5T | `null` |

Settings are fixed per track; any map is ignored.

## Method

n_vars=100000,ratio=4150: reinforced survey propagation, then probSAT from the decoded assignment, as in tekinon, with
a faster reinforcement schedule, so survey propagation needs fewer iterations. The solution found can differ from
tekinon's. n_vars=100000,ratio=4200: as in tekinon, except that survey propagation stops sooner once it no longer
improves.

ratio=4267 tracks: probSAT from the majority polarity of each variable, giving up on nonces unlikely to be solved.
Against tekinon, runs that stay far from a solution are given up earlier. A run is the same flip sequence as in
tekinon up to the point where one of them stops, so a nonce tekinon solves is solved with the same solution unless the
run is given up before that point. The local search step is faster than tekinon's, with the same flips. Runs are
deterministic.

## Results

Against tekinon, settings null, fuel 5T, fresh nonces, official runtime and verifier. Qualifiers per compute on the
ratio=4267 tracks (13-nonce bundles at the current qualifying lines) measured before the faster local search step,
which is listed in its own row; CPU time per nonce on n_vars=100000, where both codes solve every nonce.

| | n_vars=5000 | n_vars=7500 | n_vars=10000 | n_vars=100000,ratio=4150 | n_vars=100000,ratio=4200 |
|---|---|---|---|---|---|
| Nonces | 192 | 192 | 192 | 48 | 384 |
| Solved, tekinon_v2 / tekinon | 52 / 52 | 29 / 29 | 13 / 13 | 48 / 48 | 384 / 384 |
| Qualifiers per compute, x tekinon | 1.00 | 1.09 | 1.11 | - | - |
| CPU time per nonce, x tekinon | - | - | - | 0.89 | 0.97 |
| CPU time per nonce, faster local search step x without it (96 nonces, same flips) | 0.95 | 0.97 | 0.97 | - | - |

Against other codes, see tekinon's README (sat_tailwalk_v7 18-34x, sat_exact_sp128 18-39x, clausal_ember 87x
tekinon's time on n_vars=100000; 3.1-3.7x on ratio=4267).

## References and Acknowledgments

### 1. Academic Papers
- J. Chavas, C. Furtlehner, M. Mezard, R. Zecchina, "Survey-propagation decimation through distributed local computations", J. Stat. Mech. (2005) P11016.
- A. Braunstein, M. Mezard, R. Zecchina, "Survey propagation: an algorithm for satisfiability", Random Structures & Algorithms 27 (2005) 201-226.
- A. Balint, U. Schoening, "Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break", SAT 2012.

### 2. Code References
- tekinon (c001_a135, tilk): the base of every track.
- The SP seed for ratio=4200 was first used in TIG by sat_hybrid_sp128 (c001_a123); no code is reused from it.

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
