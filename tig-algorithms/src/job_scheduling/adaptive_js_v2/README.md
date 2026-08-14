# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** adaptive_js_v2
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Further refined and highly tuneable hybrid solver combining adaptive dispatching rules, learned biases, and critical-path local search.

Hyperparameters:

- `{"effort":"default"}` - 500 restarts, 3000/2600/2000/2000/2000 iters (js/fs/hfs/fjspM/fjspH)
- `{"effort":"medium"}` - 1,000 restarts, 4000/4000/3000/3000/3000 iters
- `{"effort":"high"}` - 1,500 restarts, 5000/6000/4000/4000/4000 iters
- `{"effort":"extreme"}` - 2,000 restarts, 6000/10000/5000/5000/5000 iters

- `{"num_restarts":<value>}` - Override restart count (1 to 20,000)
- `{"job_shop_iters":<value>}` - Tabu search iterations for job_shop track (100 to 50,000)
- `{"flow_shop_iters":<value>}` - Iterated greedy iterations for flow_shop track (100 to 50,000)
- `{"hybrid_flow_shop_iters":<value>}` - Local search iterations for hybrid_flow_shop track (100 to 50,000)
- `{"fjsp_medium_iters":<value>}` - Local search iterations for fjsp_medium track (100 to 50,000)
- `{"fjsp_high_iters":<value>}` - Local search iterations for fjsp_high track (100 to 50,000)

Increasing the effort level WILL increase runtime, please tune for your individual needs.

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