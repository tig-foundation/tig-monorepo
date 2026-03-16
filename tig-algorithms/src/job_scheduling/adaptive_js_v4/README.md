# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** adaptive_js_v4
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

You MUST use the correct track hyperparameter for each specific track. For example - 

- For the flow_shop track, you MUST use the hyperparameter `{"track":"flow_shop"}`
- For the job_shop track, you MUST use the hyperparameter `{"track":"job_shop"}`
- If you forget to do this, or use an incorrect hyperparameter, it will default to             `{"track":"fjsp_high"}` which will not give optimal results.

In addition, these are the tuneable hyperparameters for specific tracks only - 

- `{"job_shop_iters":<value>}` — Tabu search depth for the `job_shop` track (default 10,000, min 100, max 100,000). Scales strongly, higher values give meaningfully better quality at the cost of runtime. Increasing this parameter WILL increase runtime significantly, tune carefully.

- `{"hybrid_flow_shop_iters":<value>}` — Number of GRASP construction restarts for the `hybrid_flow_shop` track (default 2,000, min 100, max 50,000). 

- `{"fjsp_medium_iters":<value>}` — Number of GRASP construction restarts for the `fjsp_medium` track (default 2,000, min 100, max 50,000). Values above 300 also unlock additional ALNS and ILS rounds. 

`flow_shop` and `fjsp_high` have fixed internal iteration counts, there are no tunable iters parameters for those tracks.

### References

- Johnson, S. M. (1954). Optimal two‐ and three‐stage production schedules with setup times included. *Naval Research Logistics Quarterly*, 1(1), 61–68.
- Palmer, D. S. (1965). Sequencing jobs through a multi-stage process in the minimum total time — a quick method of obtaining a near optimum. *Operational Research Quarterly*, 16(1), 101–107.
- Campbell, H. G., Dudek, R. A., & Smith, M. L. (1970). A heuristic algorithm for the n-job, m-machine sequencing problem. *Management Science*, 16(10), B630–B637.
- Glover, F. (1989). Tabu search — Part I. *ORSA Journal on Computing*, 1(3), 190–206.
- Nawaz, M., Enscore, E. E., & Ham, I. (1983). A heuristic algorithm for the m-machine, n-job flow-shop sequencing problem. *Omega*, 11(1), 91–95.
- Taillard, E. (1990). Some efficient heuristic methods for the flow shop sequencing problem. *European Journal of Operational Research*, 47(1), 65–74.
- Nowicki, E., & Smutnicki, C. (1996). A fast taboo search algorithm for the job shop problem. *Management Science*, 42(6), 797–813.
- Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). Iterated local search. In *Handbook of Metaheuristics* (pp. 320–353). Springer.
- Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science*, 40(4), 455–472.
- Ruiz, R., & Stützle, T. (2007). A simple and effective iterated greedy algorithm for the permutation flowshop scheduling problem. *European Journal of Operational Research*, 177(3), 2033–2049.

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
