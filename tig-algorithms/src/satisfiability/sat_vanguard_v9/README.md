# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_vanguard_v9
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** NVX
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Per-track SAT solver dispatching on `(num_variables, num_clauses)` to a
specialised track engine. Shared preprocessing builds the clause/variable
incidence structures once; each track applies a tuned stochastic local-search
schedule. Hyperparameters are optional and override the per-track baked defaults.

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
