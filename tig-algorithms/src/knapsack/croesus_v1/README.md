# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** croesus_v1
* **Copyright:** 2026 Danske
* **Identity of Submitter:** Danske
* **Identity of Creator of Algorithmic Method:** Discovered by private Prometheus swarm (agent Epyc6-2, model DeepSeek-V4-Flash)
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Dispatches on `(num_items, budget_pct)`, where `budget_pct` is computed
from the instance's total item weight and knapsack capacity. Small-item
instances (1,000 items) route to one of two tuned solvers by budget
tightness; large-item instances (5,000 items) route to a single dedicated
solver regardless of budget. Each track's solver is a self-contained
Rust implementation with no shared state between tracks.

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
