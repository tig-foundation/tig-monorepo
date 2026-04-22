# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** job_eight
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track ID | Track | Recommended Fuel | Nonces/Bundle | Recommended HP |
|----------|-------|-----------------|---------------|----------------|
| T44 | n=50,s=fjsp_high | 100B | 40 | `{}` |
| T45 | n=50,s=fjsp_medium | 1T | 40 | `{}` |
| T46 | n=50,s=flow_shop | 100B | 40 | `{}` |
| T47 | n=50,s=hybrid_flow_shop | 100B | 40 | `{}` |
| T48 | n=50,s=job_shop | 1T | 40 | `{"job_shop_iters":100000}` |

> All tracks are auto-detected. No `track` HP override needed. Default `job_shop_iters=100000`.

## References and Acknowledgments

### 1. Academic Papers
- Nowicki, E. and Smutnicki, C., *"A Fast Taboo Search Algorithm for the Job Shop Problem"*, Management Science, 1996
- Mastrolilli, M. and Gambardella, L.M., *"Effective Neighbourhood Functions for the Flexible Job Shop Problem"*, Journal of Scheduling, 2000
- Taillard, E., *"Some efficient heuristic methods for the flow shop sequencing problem"*, European Journal of Operational Research, 1990
- Nawaz, M., Enscore, E. and Ham, I., *"A Heuristic Algorithm for the m-Machine, n-Job Flow-shop Sequencing Problem"*, OMEGA, 1983

### 2. Code References
- TIG baseline

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
