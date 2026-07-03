# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** hgs_advance
* **Copyright:** 2026 Thibaut Vidal
* **Identity of Submitter:** Thibaut Vidal
* **Identity of Creator of Algorithmic Method:** Thibaut Vidal
* **Unique Algorithm Identifier (UAI):** c002_a110

## References and Acknowledgments

This folder contains a Rust implementation of an "advanced" and scalable Hybrid Genetic Search (HGS) for the Vehicle Routing Problem with Time Windows (VRPTW) and the Capacitated Vehicle Routing Problem (CVRP). The algorithm, called `hgs_advance`, accompanies an advance submission for the TIG `vehicle_routing` challenge.

The method builds on the HGS family of algorithms, as exemplified by the open-source [HGS-CVRP](https://github.com/vidalt/HGS-CVRP) implementation and the TIG baseline [`hgs_v1`](https://github.com/tig-foundation/tig-monorepo/tree/main/tig-algorithms/src/vehicle_routing/hgs_v1). HGS is highly effective because it combines population-based exploration with aggressive local-search education. However, this same strength becomes a bottleneck on large instances: offspring are repeatedly improved through expensive full-dimensional local searches, and the population may spend substantial effort refining regions of the search space where many route structures are already stable.

`hgs_advance` targets this scalability bottleneck through three coordinated mechanisms:

1. **Evolutionary consensus compression.**  
   The algorithm detects predecessor-successor relations that remain stable across successive feasible individuals in the population. These stable arcs are used to compress chains of clients into equivalent macro-clients, while preserving the demand, travel, service-time, and time-window information needed for valid CVRP/VRPTW evaluation.

2. **Reverse-mode decomposition.**  
   Instead of maintaining a full master-level population throughout the run, the algorithm follows a single global incumbent trajectory. This master solution is decomposed into spatially coherent route-cluster subproblems, which are solved with HGS at increasing exploration depth and then reintegrated into the master solution. This moves population-based search to smaller and more focused subproblems.

3. **High-performance local search for large instances.**  
   The local-search engine combines customer-level best-move selection, systematic lower-bound prefilters, route/customer timestamps to avoid redundant evaluations, inherited-route handling, and a bounded first-loop deterioration mechanism for controlled diversification.

Together, these components define a scalable HGS architecture in which consensus compression reduces the active problem size, reverse mode reduces the active search region, and the local-search engine reduces wasted move evaluations. The implementation preserves the core strengths of HGS while making the method better suited to large-scale CVRP/VRPTW instances under limited computational budgets.

Together, these components define a scalable HGS architecture in which consensus compression reduces the active problem size, reverse mode reduces the active search region, and the local-search engine reduces wasted move evaluations. These mechanisms are synergistic and interact throughout the run: the master solution induces high-quality seed solutions for the route-cluster subproblems; these seeds are deliberately inserted later to preserve early subproblem diversity; and, even before insertion, they participate in the consensus process to ensure that compression decisions are consistent with the inherited master structure. The implementation preserves the core strengths of HGS while making the method better suited to large-scale CVRP/VRPTW instances under limited computational budgets.

## Implementation Map

The main advance-specific components are implemented in the following files:

* `solver.rs`: top-level entry point and mode selection. It dispatches to reverse mode when `params.decomp_nb_phases > 0`; otherwise, it runs the standard HGS flow.

* `genetic.rs`: main HGS loop, including crossover, education, repair, compression triggers, population remapping, and decompression of final solutions.

* `population.rs`: feasible and infeasible subpopulation management, diversity tracking, penalty adaptation, and evolutionary consensus tracking. This also includes consensus checks that account for the reserved delayed seed in subproblem runs.

* `compression.rs`: consensus-chain contraction into compact instances. It builds macro-clients from stable predecessor-successor chains while preserving demand, travel, service-time, and time-window semantics.

* `reverse_mode.rs`: reverse-mode decomposition workflow. It builds route-cluster subproblems from the master solution, maps clients between master and subproblem indices, runs phased subproblem HGS, merges subproblem routes, and applies master-level local search and repair.

* `local_search.rs`: high-performance local-search engine, including customer-level best-move selection, lower-bound move prefilters, route/customer timestamps, inherited-route handling, and bounded first-loop deterioration.

* `params.rs`: parameter definitions and presets controlling exploration level, compression cadence, decomposition phases, local-search behavior, and scalability-oriented options.

Additional support files include `sequence.rs`, `individual.rs`, and `problem.rs`, which provide route evaluation, individual representation, problem data, and CVRP/VRPTW feasibility machinery used by the advance-specific components.

### Academic Papers

[1] Vidal, T., Crainic, T. G., Gendreau, M., and Prins, C. (2013). *A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time windows*. Computers & Operations Research, 40(1), 475-489. https://doi.org/10.1016/j.cor.2012.07.018

[2] Vidal, T. (2022). *Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood*. Computers & Operations Research, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643

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