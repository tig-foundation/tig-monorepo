# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** adaptive_js_v9
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Adaptive multi-track job scheduling solver. The current defaults were selected from benchmark sweeps and, at the time of packaging, produced the best observed overall results for the intended tracks. Benchmarkers should treat the defaults as the primary recommended configuration and only override hyperparameters when doing controlled sweeps.

You MUST use the correct track hyperparameter for each specific track. For example:

- For the flow_shop track, you MUST use the hyperparameter `{"track":"flow_shop"}`
- For the hybrid_flow_shop track, you MUST use the hyperparameter `{"track":"hybrid_flow_shop"}`
- For the job_shop track, you MUST use the hyperparameter `{"track":"job_shop"}`
- For the fjsp_medium track, you MUST use the hyperparameter `{"track":"fjsp_medium"}`
- For the fjsp_high track, you MUST use the hyperparameter `{"track":"fjsp_high"}`
- If you forget to do this, or use an incorrect hyperparameter, it defaults to `{"track":"fjsp_high"}` which will not give optimal results for other tracks.

Recommended baseline commands:

- `{"track":"flow_shop"}`
- `{"track":"hybrid_flow_shop"}`
- `{"track":"job_shop"}`
- `{"track":"fjsp_medium"}`
- `{"track":"fjsp_high"}`

Tuneable hyperparameters:

- `{"job_shop_iters":<value>}` — Tabu search depth for the `job_shop` track. Default `25000`, clamp `100..200000`. Higher values can improve quality but may significantly increase runtime.
- `{"hybrid_flow_shop_iters":<value>}` — Constructor restart budget for the `hybrid_flow_shop` track. Default `3500`, clamp `100..100000`. The packaged default performed best in current testing; change only for controlled sweeps.
- `{"hybrid_flow_shop_tabu_iters":<value>}` — Final CPDT tabu iteration budget for the `hybrid_flow_shop` track. Default `1800`, clamp `100..100000`. Sweeps showed quality plateaued quickly, so the default is recommended.
- `{"hybrid_flow_shop_tabu_seeds":<value>}` — Number of final CPDT tabu seed solutions for the `hybrid_flow_shop` track. Default `5`, clamp `1..16`. Testing showed quality improved up to about `5` seeds and then plateaued.
- `{"hybrid_flow_shop_tabu_stagnation":<value>}` — Final CPDT tabu stagnation/reset limit for the `hybrid_flow_shop` track. Default `400`, clamp `20..10000`. Current sweeps found `400` strongest among tested values, however finer tuning may produce better results.
- `{"hybrid_flow_shop_tabu_reassign_every":<value>}` — Frequency of final CPDT machine-reassignment moves for the `hybrid_flow_shop` track. Default `1`, clamp `1..32`. Current sweeps favored reassignment every iteration.
- `{"fjsp_medium_iters":<value>}` — Restart budget for the `fjsp_medium` track. Default `2000`, clamp `100..100000`. Also scales tabu, critical-block, ALNS, and ILS budgets.

Benchmarking guidance:

- Start with only the correct `track` key and no extra tuning keys.
- If tuning, change one hyperparameter at a time and compare against the default for the same nonce count and worker count.
- For `hybrid_flow_shop`, the strongest tested profile is the default profile: constructor restarts `3500`, final tabu iters `1800`, tabu seeds `5`, stagnation `400`, and reassignment every iteration.
- The `flow_shop` and `fjsp_high` tracks have no recommended tunable budget in this package; their main budgets are fixed internally.
- Any unsupported hyperparameter keys are ignored.

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
