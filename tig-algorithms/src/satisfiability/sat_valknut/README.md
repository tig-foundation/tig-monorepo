# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_valknut
* **Copyright:** 2026 Danske
* **Identity of Submitter:** Danske
* **Identity of Creator of Algorithmic Method:** Discovered by private Prometheus swarm (agent Danskesat-1, model Qwen3-Coder-Next)
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Per-track SAT solver dispatching on `(num_variables, num_clauses)` to one of
five specialised track engines (T1, T3, T4, T5, T38). Shared preprocessing
scans the clause set once to build per-variable positive/negative literal
occurrence counts (dropping tautological clauses), seeds an RNG from the
challenge seed, and saves an all-`false` solution up front as a feasibility
floor before any search begins. Each track then runs a fuel-budgeted,
WalkSAT-style stochastic local search (flip-based, with periodic restarts)
under its own tuned hyperparameters — fuel budget, stagnation limit, reinit
count, and flip-selection probability — set per track in `Hparams::for_t1`
… `for_t38` and overridable via the optional hyperparameter map.

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
