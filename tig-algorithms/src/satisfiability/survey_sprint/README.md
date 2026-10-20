# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** survey_sprint
* **Copyright:** 2026 Fred; donor copyrights remain with ChervovNikita (sat_exact / c001_a125 routes and donor kernel), zhenglcc (sat_tailwalk_v7 / c001_a128 track2 route) and NVX (flipwave / c001_a130)
* **Identity of Submitter:** Fred
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

Per-track code for c001. The n_vars 5000 and 10000 tracks run the routes of c001_a127 `sat_exact_sp128`
unchanged (same files, same solutions, same fuel). The n_vars 7500 track runs the track2 route of c001_a128
`sat_tailwalk_v7` (zhenglcc) unchanged: its `track2.rs` and its donor kernel file, byte for byte. The two n_vars=100000 tracks (ratio 4150 and 4200) run
`fast100k.rs`: CSR preprocessing, a survey-propagation reinforcement seed (Chavas, Furtlehner, Mézard, Zecchina 2005;
the update equations of `engine_spr` in c001_a123/a127) handed over to a probSAT local search (Balint & Schöning 2012)
once the assignment leaves at most `spr_target` unsatisfied clauses, with two fallbacks (a second seed with another RNG
stream, then an uncapped probSAT walk). Randomness is seeded from the public instance seed only.

## Hyperparameters

All optional; `null` is the shipped operating point. n_vars=100000 keys: `spr_rate`, `spr_gmax`, `spr_lam`, `spr_damp`,
`spr_check`, `spr_patience`, `spr_maxit`, `spr_target`, `spr_async`, `spr_resync`, `cb`, `flip_cap0`, `flip_cap1`,
`walk_restart`; unknown keys are ignored. The small tracks keep the keys and defaults of c001_a127.

## References and Acknowledgments

### Academic Papers
- A. Braunstein, M. Mézard, R. Zecchina, "Survey propagation: an algorithm for satisfiability", Random Structures & Algorithms 27 (2005) 201-226.
- J. Chavas, C. Furtlehner, M. Mézard, R. Zecchina, "Survey-propagation decimation through distributed local computations", J. Stat. Mech. (2005) P11016.
- A. Balint, U. Schöning, "Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break", SAT 2012, LNCS 7317, pp. 16-29.
- B. Selman, H. Kautz, B. Cohen, "Local Search Strategies for Satisfiability Testing", DIMACS Series 26 (1996).

### Code References
- c001_a127 `sat_exact_sp128` (Fred): base of this package (small-track routes of c001_a125 `sat_exact` by ChervovNikita and the helper modules of its `mod.rs`).
- c001_a123 `sat_hybrid_sp128` (Fred): `engine_spr` reinforcement kernel.
- c001_a130 `flipwave` (NVX, TIG Innovator Outbound Game License): seed-to-walk hand-off on the unsatisfied-clause count, unrolled scatter pass, biased initialisation.
- c001_a128 `sat_tailwalk_v7` (zhenglcc, TIG Innovator Outbound Game License): the whole n_vars=7500 route (`tw7_track2.rs`, `tw7_donor_kernel.rs`; the donor kernel carries the helper modules of c001_a125 by ChervovNikita) and the three-literal specialisation of the survey update.
- TIG baseline (satisfiability).

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
