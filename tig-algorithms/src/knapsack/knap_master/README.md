# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** knap_master
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

This algorithmic improvement was inspired by `superfast_knap_v1` which introduced certain optimizations over knap_quality_opt_v11. This version is further optimized bringing ~44% performance boost on certain tracks, overall reducing total runtime by ~22% across total runtime across all tracks. Qualities across all tracks are improved as well

## Implementation notes

- Parametric breakpoint seeding and greedy constructors generate feasible starting solutions.
- Dynamic-programming core refinement and bounded exchange neighborhoods intensify solutions.
- Iterated local search combines perturbation, reconstruction, and local improvement.
- Track 3 additionally uses population search, crossover, simulated annealing, and path relinking.
- The 5000-item tracks use restricted candidate windows and restart-based diversification.
- Against `knap_quality_opt_v11`, the 1000-item tracks are 17–44% faster.
- Average quality is increased across all 5 tracks



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