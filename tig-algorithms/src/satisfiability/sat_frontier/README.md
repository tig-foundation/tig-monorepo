# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_frontier
* **Copyright:** 2026 Yash Shah (portfolio integration and modifications); upstream contributors retain their respective copyrights
* **Identity of Submitter:** Yash Shah
* **Identity of Creator of Algorithmic Method:** zhenglcc (for the c001_a128 components)
* **Unique Algorithm Identifier (UAI):** null

## Method

This is a derivative portfolio of public C001 code. It starts from
`sat_exact_sp128` (c001_a127), the adoption leader at the time of
construction. Its `n_vars=5000,ratio=4267` route specializes the common
three-literal local-search choice while preserving its selection logic.
Optionally, `t4_fuel_b` enables the independent complementary search from
`sat_tailwalk_v7` (c001_a128) *after* a failed primary search. Only a complete
satisfying assignment from that rescue search is forwarded to
`save_solution`. This preserves the primary search trajectory and gives the
optional portfolio the union of the two solve sets whenever the external fuel
budget permits both paths to finish.

On `n_vars=100000,ratio=4150`, it uses c001_a128's cached local-search route.
On `n_vars=100000,ratio=4200`, it specializes the three-literal survey
propagation update from c001_a128 while preserving the original floating-point
operation order. The 7,500- and 10,000-variable tracks use c001_a127's routes
unchanged.
These choices were checked with paired local tests against the public binaries.
No new SAT method is claimed.

## Hyperparameters

Use `{}` or `null` on all tracks except `n_vars=10000,ratio=4267`, where
`{"max_fuel_high":180000000000}` is the public c001_a127 recommendation.
On the 5000-variable track, `{"t4_fuel_b":50000000000}` enables the optional
rescue search. The default `t4_fuel_b=0` skips it and preserves the faster
primary path. The external fuel budget should be 5 trillion, the active C001
maximum at construction time.

## Public Sources and Research

* [c001_a127 / sat_exact_sp128](https://github.com/tig-foundation/tig-monorepo/tree/satisfiability/sat_exact_sp128/tig-algorithms/src/satisfiability/sat_exact_sp128), used for four tracks and fallback on the 5000-variable track.
* [c001_a128 / sat_tailwalk_v7](https://github.com/tig-foundation/tig-monorepo/tree/satisfiability/sat_tailwalk_v7/tig-algorithms/src/satisfiability/sat_tailwalk_v7), source of `tw7_donor_kernel.rs`, `tw7_engine_iv.rs`, `tw7_track4.rs`, the extracted Track1 helpers, and the specialized update in `sp128_engine_spr.rs`.
* The 100000-variable ratio-4200 route retains the survey-propagation seeding and local search described in [Braunstein, Mézard, and Zecchina, *Survey propagation: An algorithm for satisfiability*](https://arxiv.org/abs/cs/0212002) and attributed by the upstream submission.

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
