# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** tekinon
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

Settings are fixed per track; any map is ignored. n_vars=10000 runs use up to 1.4T fuel; with 0.5T, 4 of 22
solves on nonces 128-319 were cut short. Less time per nonce on every track measured, largest on
n_vars=100000.

## Method

n_vars=100000 (ratio 4150 and 4200): reinforced survey propagation, then probSAT from the decoded assignment.

ratio=4267 tracks: probSAT from the majority polarity of each variable, giving up early on nonces unlikely to be solved.

## Results

Same instances for all codes, official runtime and verifier. Rival CPU time per nonce as a multiple of tekinon's
on the same nonces. Rival settings:
sat_tailwalk_v7 and sat_exact_sp128 {"max_fuel_high":400000000000} on n_vars=5000 and 7500;
on n_vars=10000, sat_tailwalk_v7 null on nonces 128-319 and {"max_fuel_high":125000000000} on nonces 32-127,
sat_exact_sp128 {"max_fuel_high":200000000000}; null on n_vars=100000.

| Track | tekinon | sat_tailwalk_v7 | sat_exact_sp128 | clausal_ember |
|---|---|---|---|---|
| n_vars=100000,ratio=4150 | 1 | 34x | 39x | 87x |
| n_vars=100000,ratio=4200 | 1 | 18x | 18x | |

All codes solved 12/12 on both tracks. tekinon solved 300/300 per track on a separate run.

n_vars=5000 (nonces 0-63): 11/64 solved, as sat_tailwalk_v7. On each unsolved nonce of 32-38, sat_tailwalk_v7
took 3.7x and sat_exact_sp128 3.1x tekinon's mean time per nonce over 0-63.

n_vars=7500 (nonces 32-127): 18/96 solved, as sat_tailwalk_v7 and sat_exact_sp128. On 24 of these nonces run side
by side, sat_tailwalk_v7 took 3.1x tekinon's time.

n_vars=10000 (nonces 128-319): 22/192 solved; sat_tailwalk_v7 solved 24/192 and took 3.4x tekinon's time. On
nonces 32-127: tekinon 9/96 solved, sat_tailwalk_v7 11/96, sat_exact_sp128 9/96.

## References and Acknowledgments

### 1. Academic Papers
- J. Chavas, C. Furtlehner, M. Mezard, R. Zecchina, "Survey-propagation decimation through distributed local computations", J. Stat. Mech. (2005) P11016.
- A. Braunstein, M. Mezard, R. Zecchina, "Survey propagation: an algorithm for satisfiability", Random Structures & Algorithms 27 (2005) 201-226.
- A. Balint, U. Schoening, "Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break", SAT 2012.

### 2. Code References
- None. The SP seed for ratio=4200 was first used in TIG by sat_hybrid_sp128 (c001_a123); no code is reused from it.

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
