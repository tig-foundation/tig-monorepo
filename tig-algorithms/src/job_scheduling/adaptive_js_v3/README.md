# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** adaptive_js_v3
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Highly optimised hybrid solver combining adaptive dispatching rules, learned biases, critical-path local search, and bottleneck-calibrated feature normalisation.

Hyperparameters:

- `{"track":"<value>"}` *(required)* - One of: `flow_shop`, `hybrid_flow_shop`, `job_shop`, `fjsp_medium`, `fjsp_high`
- `{"job_shop_iters":<value>}` - Tabu search depth (iterations per solution) for job_shop track (default 10,000, max 100,000). Scales strongly — higher values give meaningfully better quality at the cost of runtime.
- `{"hybrid_flow_shop_iters":<value>}` - Local search iterations for hybrid_flow_shop track (default 2,000, max 50,000)
- `{"fjsp_medium_iters":<value>}` - Local search iterations for fjsp_medium track (default 2,000, max 50,000)

flow_shop and fjsp_high have fixed internal iteration counts optimised by testing — no tunable iters parameters for those tracks.

Increasing `job_shop_iters` WILL increase runtime significantly, please tune for your individual compute budget.

Key methods used:
- Adaptive priority dispatching rules (multiple strategies per track)
- NEH constructive heuristic with learned job biases
- Disjunctive graph representation
- Critical path-based local search
- Iterated Local Search (ILS) with perturbation cycles
- N5 neighborhood (insert moves within machine sequences)
- Solution-based learning for job ordering and machine penalties

References:
- Nawaz, M., Enscore, E. E., & Ham, I. (1983). A heuristic algorithm for the m-machine, n-job flow-shop sequencing problem. Omega, 11(1), 91-95.
- Balas, E., & Vazacopoulos, A. (1998). Guided local search with shifting bottleneck for job shop scheduling. Management Science Research Report, MSRR-609.
- Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). Iterated local search. In Handbook of metaheuristics (pp. 320-353). Springer.
- Nowicki, E., & Smutnicki, C. (1996). A fast taboo search algorithm for the job shop problem. Management Science, 42(6), 797-813.

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