# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** prometheus_hgs_adv
* **Copyright:** 2026 prometheus-tig-swarm  
* **Identity of Submitter:** prometheus-tig-swarm 
* **Identity of Creator of Algorithmic Method:** Thibaut Vidal
* **Unique Algorithm Identifier (UAI):** c002_a110

## References and Acknowledgments

This folder contains a Rust implementation of an evolved Hybrid Genetic Search (HGS) for the Vehicle Routing Problem with Time Windows (VRPTW) and the Capacitated Vehicle Routing Problem (CVRP). The algorithm, called `prometheus_hgs_adv`, is a descendant of the active advance [`hgs_advance`](https://github.com/tig-foundation/tig-monorepo/tree/vehicle_routing/hgs_advance/tig-algorithms/src/vehicle_routing/hgs_advance) (UAI c002_a110, Thibaut Vidal), and inherits its full architecture: evolutionary consensus compression, reverse-mode decomposition, and the high-performance local-search engine, all built on the HGS family exemplified by [HGS-CVRP](https://github.com/vidalt/HGS-CVRP) and the TIG baseline [`hgs_v1`](https://github.com/tig-foundation/tig-monorepo/tree/main/tig-algorithms/src/vehicle_routing/hgs_v1) [1, 2, 3].

The variant was produced by the Prometheus TIG swarm — a population of AI coding agents iterating an edit → benchmark → publish loop against freshly generated instance sets (600–1000 clients), with every change selected on measured solution quality under a fixed per-instance compute budget. The surviving iterations layer four coordinated intensification/diversification mechanisms onto the inherited search:

1. **Ruin-and-recreate perturbation (SISR).**  
   Periodic string-removal phases based on Slack Induction by String Removals [4] remove spatially adjacent customer strings from the incumbent, bounded by average route cardinality, and reinsert them greedily with blink-based position skipping. This injects structured diversity that survives the education pass, complementing the population's crossover-driven exploration.

2. **Ejection-pool perturbation.**  
   Separate perturbation phases eject a spatial cluster of clients (or, with small probability, a whole route) around a random seed customer into an ejection pool, then repair via chained best-insertions in the spirit of guided ejection search [5], allowing bounded chains of secondary ejections before reinsertion.

3. **Route-pool set-partitioning recombination.**  
   Distinct feasible routes discovered during the run are harvested into a fixed-width bitmask route pool. At a configurable cadence, a depth-bounded branch-and-bound solves a set-partitioning problem over the pool, recombining the best routes from different individuals into a single covering solution [6]. Routes re-entering from the pool are re-sequenced exactly when small enough by a Pareto-label dynamic program for the TSP with time windows, on route-cache misses.

4. **Route-subset decomposition (POPMUSIC).**  
   The incumbent is periodically split into proximity-ranked route subsets — ranking blends centroid distance with temporal gap — and each subset is re-optimized by a short sub-GA with its own iteration and patience budgets, in the manner of POPMUSIC matheuristics [7]. This concentrates population-based effort on small, coherent subproblems, complementing the inherited reverse-mode decomposition.

These mechanisms interact with the inherited architecture throughout the run: perturbation phases feed new route structures into the route pool; set-partitioning recombination consolidates them into the master solution; and decomposition re-optimizes the consolidated incumbent at a finer granularity. The local-search engine is extended with a queue-driven move-descent pass over per-customer best moves with stale-entry invalidation, and the run finishes with dedicated polish rounds and a final full-scale GA budget. The exploration presets are extended to `exploration_level` 6 (deep HGS); all parameters documented in `params.rs` remain individually overridable through the standard hyperparameters interface.

## Implementation Map

The main components specific to this submission are implemented in the following files:

* `genetic.rs`: main HGS loop, extended with the SISR and ejection-pool perturbation phases, route harvesting into the bitmask route pool, the set-partitioning branch-and-bound (`sp_branch`), the exact TSPTW reordering DP (`tsptw_optimal_order`), and the POPMUSIC route-subset decomposition with its sub-GA (`route_subset_decomposition`).

* `local_search.rs`: inherited high-performance local-search engine, extended with a per-customer best-move descent pass (`smd_pass`) using stale-entry pruning and cross-route move application.

* `params.rs`: parameter definitions and presets, including the perturbation/recombination/decomposition cadences, the new `polish_rounds`, `final_ga_iters`, and `seed_immediate` controls, and the extended `exploration_level` 0–6 presets.

* `solver.rs`, `reverse_mode.rs`, `compression.rs`, `population.rs`: inherited from `hgs_advance` — mode selection, reverse-mode decomposition workflow, evolutionary consensus compression, and subpopulation/diversity/penalty management — with adaptations where the new mechanisms hook into the incumbent trajectory.

Additional support files include `sequence.rs`, `individual.rs`, `pred_queue.rs`, `constructive.rs`, and `problem.rs`, which provide route evaluation, individual representation, priority-queue machinery, initial-solution construction, and CVRP/VRPTW feasibility machinery. The `loader_tig.rs` module implements the standard TIG I/O; the `benchmark_io`-gated loader modules (`loader_cvrp.rs`, `loader_vrptw.rs`) are inert on mainnet builds.

### Academic Papers

[1] Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., & Rei, W. (2012). *A hybrid genetic algorithm for multidepot and periodic vehicle routing problems*. Operations Research, 60(3), 611–624. https://doi.org/10.1287/opre.1120.1048

[2] Vidal, T., Crainic, T. G., Gendreau, M., & Prins, C. (2013). *A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time windows*. Computers & Operations Research, 40(1), 475–489. https://doi.org/10.1016/j.cor.2012.07.018

[3] Vidal, T. (2022). *Hybrid genetic search for the CVRP: Open-source implementation and SWAP\* neighborhood*. Computers & Operations Research, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643

[4] Christiaens, J., & Vanden Berghe, G. (2020). *Slack induction by string removals for vehicle routing problems*. Transportation Science, 54(2), 417–433. https://doi.org/10.1287/trsc.2019.0914

[5] Nagata, Y., & Bräysy, O. (2009). *A powerful route minimization heuristic for the vehicle routing problem with time windows*. Operations Research Letters, 37(5), 333–338.

[6] Rochat, Y., & Taillard, É. D. (1995). *Probabilistic diversification and intensification in local search for vehicle routing*. Journal of Heuristics, 1(1), 147–167.

[7] Queiroga, E., Sadykov, R., & Uchoa, E. (2021). *A POPMUSIC matheuristic for the capacitated vehicle routing problem*. Computers & Operations Research, 136, 105475. https://doi.org/10.1016/j.cor.2021.105475

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
