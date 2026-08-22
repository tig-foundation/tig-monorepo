# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** ember_stromboli
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n=50,s=fjsp_high        | 5T | `{}` |
| n=50,s=fjsp_medium      | 5T | `{}` |
| n=50,s=flow_shop        | 5T | `{}` |
| n=50,s=hybrid_flow_shop | 5T | `{}` |
| n=50,s=job_shop         | 5T | `{}` |

> Tracks are auto-detected from the challenge parameters. **Hyperparameters passed on the command line are ignored**: every setting is baked in per track, so no override is required and none has any effect. The engine is anytime and deterministic — a valid solution is saved as soon as one exists and only overwritten on improvement, so an interrupted run still returns its best schedule. There is no clock, no thread, no environment variable and no filesystem access; the only randomness is a `SmallRng` seeded from `challenge.seed`.
>
> **Each track stops on its own work ceiling rather than on the granted budget.** At the recommended fuel a larger grant buys no additional quality, and wall clock is roughly constant per track instead of proportional to the fuel. On a much smaller grant the run becomes fuel-bound and returns the best schedule found so far.


## References and Acknowledgments

### Academic Papers
- Pearce & Kelly, *"A Dynamic Topological Sort Algorithm for Directed Acyclic Graphs"*, DOI: https://doi.org/10.1145/1187436.1210590 — the incremental topological order that lets the `graph` module of `hybrid_engine.rs` evaluate a move by repairing the order and repropagating only the affected cone instead of re-running a full topological sort; the repair touches 3.7 to 4.8 operations out of ~1650.
- Nowicki & Smutnicki, *"A Fast Taboo Search Algorithm for the Job Shop Problem"*, DOI: https://doi.org/10.1287/mnsc.42.6.797 — the critical-block (N5) neighbourhood driving the tabu searches of `solve_pool` and `job_shop_engine.rs`.
- Taillard, *"Parallel Taboo Search Techniques for the Job Shop Scheduling Problem"*, DOI: https://doi.org/10.1287/ijoc.6.2.108 — the head/tail path bounds behind every constant-time swap estimate (`estimate_swap`, `swap_estimate`, `estimate_swap_mk`).
- Nawaz, Enscore & Ham, *"A Heuristic Algorithm for the m-Machine, n-Job Flow-Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0305-0483(83)90088-9 — the NEH insertion construction of `flow_shop_engine.rs`.
- Taillard, *"Some Efficient Heuristic Methods for the Flow Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0377-2217(90)90090-X — the O(n·m) insertion acceleration for NEH; this submission generalises it exactly to re-entrant routes (`ReentrantInsBuf`).
- Ruiz & Stützle, *"A Simple and Effective Iterated Greedy Algorithm for the Permutation Flowshop Scheduling Problem"*, DOI: https://doi.org/10.1016/j.ejor.2005.12.009 — the destruction/reconstruction search on permutation flow instances.
- Johnson, *"Optimal Two- and Three-Stage Production Schedules with Setup Times Included"*, DOI: https://doi.org/10.1002/nav.3800010110 — Johnson's rule, used inside the CDS order candidates.
- Palmer, *"Sequencing Jobs Through a Multi-Stage Process in the Minimum Total Time — A Quick Method of Obtaining a Near Optimum"*, DOI: https://doi.org/10.1057/jors.1965.8 — the slope-index order candidate.
- Campbell, Dudek & Smith, *"A Heuristic Algorithm for the n Job, m Machine Sequencing Problem"*, DOI: https://doi.org/10.1287/mnsc.16.10.B630 — the CDS family of order candidates.
- Carlier, *"The One-Machine Sequencing Problem"*, DOI: https://doi.org/10.1016/S0377-2217(82)80007-6 — the one-machine branch-and-bound (with Schrage's heuristic) that re-sequences bottleneck machines in `job_shop_engine.rs`.
- Adams, Balas & Zawack, *"The Shifting Bottleneck Procedure for Job Shop Scheduling"*, DOI: https://doi.org/10.1287/mnsc.34.3.391 — the shifting-bottleneck seed construction and re-optimisation cycles.
- Kacem, Hammadi & Borne, *"Approach by Localization and Multiobjective Evolutionary Optimization for Flexible Job-Shop Scheduling Problems"*, DOI: https://doi.org/10.1109/TSMCC.2002.1009117 — the load-balancing assignment heuristic behind `kacem_construct`.

### Code References
- TIG baseline (`dispatching_rules`), replicated by `gate_floor.rs` so that the submitted makespan is never worse than the verifier's greedy floor.
- Portions of `flow_shop_engine.rs` and `job_shop_engine.rs` derive from engines published on-chain as `task_tree_h` (`c007_a032`) and `task_tree_j` (`c007_a036`).
- `hybrid_engine.rs` is vendored from the engine published on-chain as `shuttle_bell` (`c007_a037`); this submission rewrites its evaluation incrementally (Pearce & Kelly above), replaces its critical-set scan by a root propagation, and tunes its work ceiling and restart patience.
- The fjsp_high and fjsp_medium tracks use an original engine (`mod.rs`, `solve_pool`): disjunctive-graph tabu search over an N5 swap neighbourhood plus machine-reassignment moves, seeded by a pool of dispatch constructions.
- The evaluation paths of all five engines were rewritten for fuel cost without changing behaviour. Each rewrite is proven by replaying the search and comparing the hash of the entire improvement trace, with a negative control that must diverge; the hybrid path additionally carries a shadow that recomputes heads, tails and makespan by a full topological sort at every evaluation, and its critical-set propagation was proven against the original scan over 400,000 side-by-side calls.


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
