# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_imp_v6
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional details

Per-track solvers (track1–track5) building on `sat_imp_v5`. Changes in v6:

* **Track 1 (n_vars=5000):** revised diversification kick — candidates are sampled across several unsatisfied clauses instead of one, scored by make−break. Higher quality than v5.
* **Track 2 (n_vars=7500):** trajectory-based early abandonment. Attempt 0 is stopped if its best unsat count is still above 12 at 90M flips or above 8 at 130M flips; on 200 traced instances every eventually-solved instance was at or below 5 and 4 respectively at those points, while ~55% of instances never get close. Instances that pass the checkpoints run exactly as before. Two additional restart attempts (same 300M-flip budget, relaxed biased init with phase-save from the best assignment) are appended for the survivors, paid for by the abandonment. Net: one more instance solved per 100 than v5 at ~27% lower runtime. Fuel per surviving instance is around 1 trillion (v5: around 650 billion).
* **Tracks 3–5:** unchanged from v5.

Hyperparameters (track 2): `abandon_checkpoints_mflips` / `abandon_thresholds` (defaults `[90,130]` / `[12,8]`; empty arrays disable), `extra_attempts` (default 2; 0 restores the v5 schedule), `restart_attempts` (default 4).

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