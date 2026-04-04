# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** first_submission
* **Copyright:** 2026 sdiamond
* **Identity of Submitter:** sdiamond
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

### 1. Code References
- Based on adaptive_js_v3 by the TIG community, extended with enhanced local search phases and multi-candidate greedy reassignment.

### 2. Academic Papers
- Mastrolilli, M. and Gambardella, L.M., "Effective Neighbourhood Functions for the Flexible Job Shop Problem", Journal of Scheduling, 2000.


## Additional Notes

Improvements over adaptive_js_v3:
- **job_shop track:** Expanded solution pool (18 vs 15), additional tabu search start point (11 vs 10), added critical-block local search phase with insertion moves and machine rerouting after tabu search.
- **fjsp_medium track:** Added critical-block local search phase after tabu search, multi-candidate greedy machine reassignment (top 3 solutions vs single best).
- **hybrid_flow_shop track:** Second round of enhanced local search with perturbation, multi-candidate greedy machine reassignment (top 5 solutions vs single best).
- **fjsp_high track:** Multi-candidate greedy machine reassignment (top 5 solutions vs single best).

Hyperparameters (same as v3):
- `track` (required): "flow_shop" | "hybrid_flow_shop" | "job_shop" | "fjsp_medium" | "fjsp_high"
- `job_shop_iters`: integer, default 10000, max 100000
- `hybrid_flow_shop_iters`: integer, default 2000, max 50000
- `fjsp_medium_iters`: integer, default 2000, max 50000


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
