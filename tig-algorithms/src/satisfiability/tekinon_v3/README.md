# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** tekinon_v3
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

n_vars=100000 tracks: tekinon_v2's reinforced survey propagation and probSAT, with a faster survey propagation
update. Solutions can differ from tekinon_v2's; both solve every tested nonce.

ratio=4267 tracks: before the local search, a short survey propagation pass estimates how many solution clusters the
instance has. A nonce whose estimate shows it is unlikely to be solved, or where survey propagation does not
converge, is given up at once and saved as unsolved. Every other nonce runs tekinon_v2's probSAT from the majority
polarity of each variable, unchanged: the same flips and the same solution. A nonce tekinon_v2 solves is therefore
solved with the same solution unless the screen gives it up. The screen uses its own random stream, so runs stay
deterministic.

## Results

Against tekinon_v2, settings null, fuel 5T, fresh nonces, official runtime and verifier, 95% intervals. Qualifiers
per compute on the ratio=4267 tracks for 13-nonce bundles at the qualifying lines of blocks 1,424,035-1,424,905
(5000: 6 or 7 of 13, 7500: 4 of 13, 10000: 4 of 13); CPU time per nonce on n_vars=100000, where both codes solve
every nonce.

| | n_vars=5000 | n_vars=7500 | n_vars=10000 | n_vars=100000,ratio=4150 | n_vars=100000,ratio=4200 |
|---|---|---|---|---|---|
| Nonces | 384 | 96 | 96 | 4800 | 4800 |
| Solved, tekinon_v3 / tekinon_v2 | 86 / 92 | 18 / 18 | 8 / 8 | 4800 / 4800 | 4800 / 4800 |
| CPU time, x tekinon_v2 | 0.47 [0.36, 0.58] | 0.62 [0.33, 0.83] | 0.44 [0.22, 0.62] | 0.955 [0.944, 0.966] | 0.944 [0.923, 0.966] |
| Qualifiers per compute, x tekinon_v2 | 1.58 [1.16, 2.19] (6 of 13), 1.46 [1.01, 2.07] (7 of 13) | 1.63 [1.21, 2.99] | 2.27 [1.61, 4.46] | - | - |

On n_vars=5000 the screen gives up 6 of the 92 nonces tekinon_v2 solves; on the other tracks no solved nonce is lost.
Against other codes, see tekinon's and tekinon_v2's READMEs.

## References and Acknowledgments

### 1. Academic Papers
- J. Chavas, C. Furtlehner, M. Mezard, R. Zecchina, "Survey-propagation decimation through distributed local computations", J. Stat. Mech. (2005) P11016.
- A. Braunstein, M. Mezard, R. Zecchina, "Survey propagation: an algorithm for satisfiability", Random Structures & Algorithms 27 (2005) 201-226.
- M. Mezard, G. Parisi, R. Zecchina, "Analytic and algorithmic solution of random satisfiability problems", Science 297 (2002) 812-815.
- A. Balint, U. Schoening, "Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break", SAT 2012.

### 2. Code References
- tekinon_v2 (tilk): the base of every track; tekinon (c001_a135, tilk) before it.
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
