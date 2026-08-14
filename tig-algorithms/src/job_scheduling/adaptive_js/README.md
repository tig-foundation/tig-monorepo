# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** adaptive_js
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Draft version for initial mainnet testing. Hybrid solver combining adaptive dispatching rules, learned biases, and critical-path local search.

Hyperparameters - 

- '{"effort":medium}' - More restarts and LS rounds
- '{"effort":high}' - Even more restarts and LS rounds
- '{"effort":extreme}' - Maximum restarts and LS rounds

- '{"num_restarts":<value>} - Benchmarker enforced value (Default : 500) Maximum value : 20000

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