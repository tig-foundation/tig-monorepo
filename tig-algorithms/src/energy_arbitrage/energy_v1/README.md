# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** energy_v1
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

# Additional Information

Please make sure you use the correct hyperparameter for the specific track, for example -

`{"track":"baseline"}`
`{"track":"capstone"}`

ect.

This algorithm draws on standard ideas from battery arbitrage, degradation-aware dispatch, and PTDF-based network feasibility. 

The implementation is a custom challenge-specific heuristic combination rather than a direct reproduction of any single paper. It also includes several experimental, non-standard draft heuristics that are not taken directly from the cited literature, including region-based congestion allocation, proactive PTDF action bounds, post-feasibility value recovery, and day-ahead/real-time congestion-aware action correction.

Relevant references:

* Marek Petrik and Xiaojian Wu. "Optimal Threshold Control for Energy Arbitrage with Degradable Battery Storage." Conference on Uncertainty in Artificial Intelligence (UAI), 2015. https://mlanthology.org/uai/2015/petrik2015uai-optimal/
* Dheepak Krishnamurthy, Canan Uckun, Zhi Zhou, Prakash Thimmapuram, and Audun Botterud. "Energy Storage Arbitrage Under Day-Ahead and Real-Time Price Uncertainty." IEEE Transactions on Power Systems, 2017. https://www.osti.gov/pages/biblio/1358239-energy-storage-arbitrage-under-day-ahead-real-time-price-uncertainty
* Ray D. Zimmerman, Carlos E. Murillo-Sanchez, and Robert J. Thomas. "MATPOWER: Steady-State Operations, Planning and Analysis Tools for Power Systems Research and Education." IEEE Transactions on Power Systems, 2011. https://matpower.org/docs/MATPOWER-paper.pdf
* Bolun Xu. "Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets." 2017. https://arxiv.org/abs/1707.04567
* Ningkun Zheng, Joshua Jaworski, and Bolun Xu. "Arbitraging Variable Efficiency Energy Storage using Analytical Stochastic Dynamic Programming." 2021. https://arxiv.org/abs/2108.06000

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