# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** ember_etna
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

The track is detected from the instance itself and every setting is a constant of the source, fixed per track; the hyperparameter map passed to `solve_challenge` is accepted and ignored, so an empty map (`{}`) and any other map give the same run. The solver is anytime and deterministic: a valid schedule is saved as soon as one exists and is replaced only by one that is no worse, so an interrupted run still returns the best schedule it has saved. There is no clock, no thread, no environment variable and no filesystem access; all pseudo-randomness derives from `challenge.seed`.

Each track stops on its own work ceiling rather than on the granted budget. At the recommended fuel every track finishes below its grant, so a larger grant does not change the result; on a much smaller grant the run becomes fuel-bound and returns the best schedule saved so far.


## Structure

| File | Role |
|------|------|
| `mod.rs` | Entry point, track detection, safety floors, and the engine of the two flexible tracks (fjsp_high, fjsp_medium) |
| `flow_shop_engine.rs` | Engine of the flow_shop track |
| `job_shop_engine.rs` | Engine of the job_shop track |
| `hybrid_engine.rs` | Engine of the hybrid_flow_shop track, and the incremental disjunctive graph also used by the job_shop and flexible engines |
| `hfs_construct.rs` | Alternative start of the hybrid_flow_shop track |
| `gate_floor.rs` | Dispatching-rule schedules offered to the host before any search |


## References and Acknowledgments

### Academic Papers
- Pearce & Kelly, *"A Dynamic Topological Sort Algorithm for Directed Acyclic Graphs"*, DOI: https://doi.org/10.1145/1187436.1210590 — incremental topological order of the disjunctive graph (`hybrid_engine.rs`).
- Glover & Laguna, *"Tabu Search"*, DOI: https://doi.org/10.1007/978-1-4615-6089-0 — tabu search and path relinking (`job_shop_engine.rs`, `mod.rs`, `hybrid_engine.rs`).
- Nowicki & Smutnicki, *"A Fast Taboo Search Algorithm for the Job Shop Problem"*, DOI: https://doi.org/10.1287/mnsc.42.6.797 — critical-block (N5) neighbourhood of the tabu searches.
- Taillard, *"Parallel Taboo Search Techniques for the Job Shop Scheduling Problem"*, DOI: https://doi.org/10.1287/ijoc.6.2.108 — head/tail bounds used to estimate a swap in constant time.
- Dueck, *"New Optimization Heuristics: The Great Deluge Algorithm and the Record-to-Record Travel"*, DOI: https://doi.org/10.1006/jcph.1993.1010 — record-to-record acceptance of the critical-block descent (`flow_shop_engine.rs`).
- Feo & Resende, *"Greedy Randomized Adaptive Search Procedures"*, DOI: https://doi.org/10.1007/BF01096763 — randomised list-scheduling constructions with a local descent (`flow_shop_engine.rs`).
- Nawaz, Enscore & Ham, *"A Heuristic Algorithm for the m-Machine, n-Job Flow-Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0305-0483(83)90088-9 — NEH insertion construction (`flow_shop_engine.rs`, `job_shop_engine.rs`).
- Taillard, *"Some Efficient Heuristic Methods for the Flow Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0377-2217(90)90090-X — insertion acceleration for NEH, extended here to re-entrant routes.
- Ruiz & Stützle, *"A Simple and Effective Iterated Greedy Algorithm for the Permutation Flowshop Scheduling Problem"*, DOI: https://doi.org/10.1016/j.ejor.2005.12.009 — destruction/reconstruction search on permutation routes.
- Johnson, *"Optimal Two- and Three-Stage Production Schedules with Setup Times Included"*, DOI: https://doi.org/10.1002/nav.3800010110 — Johnson's rule, inside the CDS candidates.
- Palmer, *"Sequencing Jobs Through a Multi-Stage Process in the Minimum Total Time — A Quick Method of Obtaining a Near Optimum"*, DOI: https://doi.org/10.1057/jors.1965.8 — slope-index order candidate.
- Campbell, Dudek & Smith, *"A Heuristic Algorithm for the n Job, m Machine Sequencing Problem"*, DOI: https://doi.org/10.1287/mnsc.16.10.B630 — CDS order candidates.
- Carlier, *"The One-Machine Sequencing Problem"*, DOI: https://doi.org/10.1016/S0377-2217(82)80007-6 — one-machine branch-and-bound with Schrage's heuristic (`job_shop_engine.rs`).
- Adams, Balas & Zawack, *"The Shifting Bottleneck Procedure for Job Shop Scheduling"*, DOI: https://doi.org/10.1287/mnsc.34.3.391 — shifting-bottleneck seed construction and re-optimisation (`job_shop_engine.rs`).
- Kacem, Hammadi & Borne, *"Approach by Localization and Multiobjective Evolutionary Optimization for Flexible Job-Shop Scheduling Problems"*, DOI: https://doi.org/10.1109/TSMCC.2002.1009117 — load-balancing assignment heuristic of the flexible engine (`mod.rs`).

### Code References
- TIG baseline (`dispatching_rules`), replicated by `gate_floor.rs`.
- This submission derives from `ember_stromboli` (`c007_a040`, FP Labs).
- Portions of `flow_shop_engine.rs` and `job_shop_engine.rs` derive from the engines published on-chain as `task_tree_h` (`c007_a032`) and `task_tree_j` (`c007_a036`, NVX).
- `hfs_construct.rs` derives from the hybrid_flow_shop construction phase of `task_tree_j` (`c007_a036`, NVX).
- `hybrid_engine.rs` derives from the engine published on-chain as `shuttle_bell` (`c007_a037`).
- The engine of the fjsp_high and fjsp_medium tracks (`solve_pool` in `mod.rs`) is original work.


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
