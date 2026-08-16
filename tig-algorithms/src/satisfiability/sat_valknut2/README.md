# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_valknut2
* **Copyright:** 2026 Danske
* **Identity of Submitter:** Danske
* **Identity of Creator of Algorithmic Method:** Discovered by private Prometheus swarm
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Per-track SAT solver dispatching on `(num_variables, num_clauses)` to one of
five specialised track engines (T1, T3, T4, T5, T38), in the same overall
shape as `sat_valknut`. Shared preprocessing deduplicates and canonicalises
clauses, drops tautological clauses, and builds a variable→clause incidence
index (bucketed by polarity) once up front so each track's local-search loop
gets O(1) lookups from a flipped variable to its affected clauses. Each track
then runs a fuel-budgeted, WalkSAT-style stochastic local search under its
own tuned hyperparameters — flip probability bounds, stagnation/perturbation
cadence, reinit count, fuel ceiling — set per track in `Hparams::for_t1` …
`for_t38` and overridable via the optional hyperparameter map.

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
