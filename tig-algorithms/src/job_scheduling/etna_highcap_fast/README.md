# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** etna_highcap_fast
* **Copyright:** 2026 Fred; donor copyrights remain with FP Labs, NVX and stinger
* **Identity of Submitter:** Fred
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n=50,s=fjsp_high        | 5T | null |
| n=50,s=fjsp_medium      | 5T | null |
| n=50,s=flow_shop        | 5T | null |
| n=50,s=hybrid_flow_shop | 5T | null |
| n=50,s=job_shop         | 5T | null |

Leave hyperparameters null. `etna_highcap_fast` is `etna_highcap` (`c007_a046`) with three rewrites of the job_shop engine that change no result and remove work (next paragraph). `etna_highcap` is `ember_etna` (`c007_a044`, FP Labs) with a single change of behaviour: the work cap of the fjsp_high pool search is raised from 6.5e10 to 2e11 fuel (2e11 is already the fjsp_medium cap, in both codes). With null hyperparameters, or with any map that has no `"v1lab"` key, every track runs `ember_etna`'s sequence of operations (track detection, engines, work ceilings, saves) with `ember_etna`'s settings, except that cap; on fjsp_medium, flow_shop and hybrid_flow_shop the run is `ember_etna`'s, and on job_shop the saved schedules are `ember_etna`'s, found with less work. The solver is anytime and deterministic, without clock, thread, environment variable or filesystem access; all pseudo-randomness derives from `challenge.seed`.

The three job_shop rewrites are all in `job_shop_engine.rs`, which runs only on the job_shop track; apart from the name and this README, it is the only file that differs from `etna_highcap`. Each keeps every save of the run identical (same schedules, in the same order): the topological pass that settles an ambiguous critical path no longer branches on absent links (an absent link points to a spare record); in the critical-block descent of the iterated local search, a probe whose resulting sequence was already probed in the same step is skipped, and an adjacent-swap probe is skipped when a valid lower bound of its makespan (the longest path through the two swapped operations, from the heads and tails of the current schedule) already reaches the best probe of the step. A skipped probe could not have been retained: its makespan would equal that of the earlier probe of the same sequence, or be at least the bound, or the swap would create a cycle. On job_shop the run uses about 0.986 times the fuel of `etna_highcap`; at the recommended fuel no track is stopped by its budget, so the result does not depend on it.

The fjsp_high cap is a maximum, not a target: the pool search spends at most `min(fuel left when it starts - 1e8, cap)`, so a smaller fuel budget on that track lowers the spend, and a larger one never raises it above 2e11. Measured: a 1e11 fjsp_high fuel budget gives the same quality as a 1e11 cap on 12 of 12 nonces. Measured on 16 nonces: with a 6.5e10 fjsp_high fuel budget, the quality was within 213 of that of `ember_etna` with 5e12 (higher on 1, equal on 14, lower on 1; same solution on 14 of 16 nonces), for about 0.98 times the CPU time of `ember_etna`.

Study settings are read only under the key `"v1lab"`, named after the laboratory build that was benchmarked; each one is optional and a missing, mistyped or out-of-range value keeps its default:

| key | effect | default (`ember_etna` value if different) |
|-----|--------|--------------------|
| `caps.hybrid` | fuel the hybrid_flow_shop construction and tabu search may spend (1e10 to 4.5e12) | 2e11 |
| `caps.fjsp_high` / `caps.fjsp_medium` | work cap of the pool search (1e10 to 4.5e12; `caps.fjsp_high` 6.5e10 sets `ember_etna`'s cap) | 2e11 / 2e11 (`ember_etna`: 6.5e10 / 2e11) |
| `js_iters` | iterations of each full job_shop tabu search (1e3 to 3e6) | 300 000 |
| `js_starts` | pooled schedules the first job_shop tabu searches start from (1 to 15) | 4 |
| `js_extra_sbp` | extra shifting-bottleneck seeds, each followed by a full tabu search, after the last job_shop save (0 to 4) | 0 |
| `hyb_alt_iters` | restarts of the hybrid_flow_shop alternative-start construction (100 to 20 000) | 1000 |
| `y2` | hybrid_flow_shop: the full `task_tree_j` t47 pipeline builds the alternative start | false |
| `y2_tabu_cap` | hybrid_flow_shop: the tabu budget is set after the alternative-start construction | false |
| `stop_lb` | flow_shop / job_shop: stop once the saved makespan reaches a lower bound (longest job, preemptive one-machine bound) | false |


## Local Benchmark

TIG runtime 0.0.8, round-136 track settings, 5e12 fuel budget, null hyperparameters.

**This submission:** the binary built locally from these `.rs` files (this README is not compiled), against `etna_highcap` (`c007_a046`; the binary built locally from its submitted files). 188 runs: 12 nonces per track, plus the nonces reserved for the final check of `etna_highcap` (64 fjsp_high nonces, 16 nonces of each other track). Each solution was compared by hash with that of `etna_highcap` on the same track and nonce, directly or through reference runs that the `etna_highcap` binary reproduces exactly. On the 16 reserved job_shop nonces, each run was also paired with a run of `etna_highcap` on the same nonce to compare CPU time.

| Track | Solutions vs `etna_highcap` | CPU time vs `etna_highcap` |
|-------|-----------------------------|----------------------------|
| n=50,s=job_shop | identical: same solution hash on 28 of 28 nonces | x0.870 (16 paired nonces, sum of CPU time; 95% bootstrap CI x0.841 to x0.899; lower on 16 of 16 nonces) |
| the four other tracks | identical: same solution hash on 160 of 160 runs (76 fjsp_high, 28 on each other track) | unchanged code; fuel equal to that of `etna_highcap` within the run-to-run noise |

Reruns of 5 of these runs (one per track) gave the same solution 5 times out of 5. On job_shop this binary used 0.986 times the fuel of `etna_highcap`. Before this check, a laboratory build of the same code under another module name gave, on 32 other job_shop nonces, the same solutions as `etna_highcap` on 32 of 32 nonces, for x0.863 the CPU time of a neutral rebuild of `etna_highcap` (95% CI x0.835 to x0.891); that rebuild costs x0.9985 the CPU time of the `etna_highcap` binary, so the gain does not come from where the code lands in the binary.

**Against `ember_etna`:** the reserved nonces are those of the final check of `etna_highcap`, in which each run was paired with a run of the official `ember_etna` binary on the same nonce; the solutions being the same, its results hold for this submission. On fjsp_high: +1,930.7 per nonce on average (95% CI +1,675.5 to +2,185.9; bootstrap +1,684.7 to +2,180.8), better on 62 of 64 nonces, equal on 1, worse on 1, for x3.1 the CPU time of `ember_etna` (sum over the same 64 nonces; on fjsp_high this submission runs the code of `etna_highcap`), with 5.2e10 to 2.0e11 fuel per nonce (`ember_etna`: 2.8e10 to 6.5e10). On the four other tracks: same solution hash as `ember_etna` on 64 of 64 nonces (16 per track).

On fjsp_high the extra quality costs about three times the CPU time of `ember_etna`; whether that pays depends on the qualifying threshold of the round.


## Recommended settings

- **Fuel budget and hyperparameters:** 5e12 fuel and null hyperparameters on every track, as for the other job_scheduling codes. On fjsp_medium, flow_shop and hybrid_flow_shop the run, and therefore its cost, is `ember_etna`'s; on job_shop the solutions are `ember_etna`'s, for about 0.87 times the CPU time of `etna_highcap`, whose job_shop run is `ember_etna`'s.
- **fjsp_high `num_bundles`:** each fjsp_high nonce costs about 3 times the CPU time of `ember_etna` (x3.1 above) and about 1.7 times that of `ember_stromboli` (`c007_a040`; in a separate paired run on this track, `ember_etna` used 0.57 times the CPU time of `ember_stromboli`). To keep the same benchmark duration, reduce the fjsp_high `num_bundles` in proportion: about 1/3 of the number used with `ember_etna`, or about 60% of the number used with `ember_stromboli`.
- **fjsp_high `fuel_budget`:** the 2e11 work cap is a maximum, so a lower fjsp_high `fuel_budget` lowers the cap (see above): each nonce then costs less and keeps only part of the quality gain. Measured: with a 1e11 budget, a fjsp_high nonce costs about 1.6 times the CPU time of `ember_etna`.
- **job_shop `num_bundles`:** at the same benchmark duration, the job_shop `num_bundles` can be raised by about 15%: a job_shop nonce costs about 0.87 times the CPU time of `etna_highcap` (x0.870 above), for the same solutions.


## Structure

| File | Role |
|------|------|
| `mod.rs` | Entry point, track detection, safety floors, and the engine of the two flexible tracks (fjsp_high, fjsp_medium) |
| `lab.rs` | Parsing of the `"v1lab"` settings, lower bounds |
| `flow_shop_engine.rs` | Engine of the flow_shop track |
| `job_shop_engine.rs` | Engine of the job_shop track, with the three rewrites described above |
| `hybrid_engine.rs` | Engine of the hybrid_flow_shop track, and the incremental disjunctive graph also used by the job_shop and flexible engines |
| `hfs_construct.rs` | Alternative start of the hybrid_flow_shop track |
| `t47.rs` | Full hybrid_flow_shop pipeline of `task_tree_j` (`track_t47.rs`), used only with `y2` |
| `gate_floor.rs` | Dispatching-rule schedules offered to the host before any search |


## References and Acknowledgments

### Academic Papers
- Pearce & Kelly, *"A Dynamic Topological Sort Algorithm for Directed Acyclic Graphs"*, DOI: https://doi.org/10.1145/1187436.1210590 — incremental topological order of the disjunctive graph (`hybrid_engine.rs`).
- Glover & Laguna, *"Tabu Search"*, DOI: https://doi.org/10.1007/978-1-4615-6089-0 — tabu search and path relinking (`job_shop_engine.rs`, `mod.rs`, `hybrid_engine.rs`).
- Nowicki & Smutnicki, *"A Fast Taboo Search Algorithm for the Job Shop Problem"*, DOI: https://doi.org/10.1287/mnsc.42.6.797 — critical-block (N5) neighbourhood of the tabu searches.
- Taillard, *"Parallel Taboo Search Techniques for the Job Shop Scheduling Problem"*, DOI: https://doi.org/10.1287/ijoc.6.2.108 — head/tail bounds used to estimate a swap in constant time, and to skip the descent probes of the job_shop engine that cannot be retained.
- Dueck, *"New Optimization Heuristics: The Great Deluge Algorithm and the Record-to-Record Travel"*, DOI: https://doi.org/10.1006/jcph.1993.1010 — record-to-record acceptance of the critical-block descent (`flow_shop_engine.rs`).
- Feo & Resende, *"Greedy Randomized Adaptive Search Procedures"*, DOI: https://doi.org/10.1007/BF01096763 — randomised list-scheduling constructions with a local descent (`flow_shop_engine.rs`).
- Nawaz, Enscore & Ham, *"A Heuristic Algorithm for the m-Machine, n-Job Flow-Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0305-0483(83)90088-9 — NEH insertion construction (`flow_shop_engine.rs`, `job_shop_engine.rs`).
- Taillard, *"Some Efficient Heuristic Methods for the Flow Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0377-2217(90)90090-X — insertion acceleration for NEH, extended here to re-entrant routes.
- Ruiz & Stützle, *"A Simple and Effective Iterated Greedy Algorithm for the Permutation Flowshop Scheduling Problem"*, DOI: https://doi.org/10.1016/j.ejor.2005.12.009 — destruction/reconstruction search on permutation routes.
- Johnson, *"Optimal Two- and Three-Stage Production Schedules with Setup Times Included"*, DOI: https://doi.org/10.1002/nav.3800010110 — Johnson's rule, inside the CDS candidates.
- Palmer, *"Sequencing Jobs Through a Multi-Stage Process in the Minimum Total Time — A Quick Method of Obtaining a Near Optimum"*, DOI: https://doi.org/10.1057/jors.1965.8 — slope-index order candidate.
- Campbell, Dudek & Smith, *"A Heuristic Algorithm for the n Job, m Machine Sequencing Problem"*, DOI: https://doi.org/10.1287/mnsc.16.10.B630 — CDS order candidates.
- Carlier, *"The One-Machine Sequencing Problem"*, DOI: https://doi.org/10.1016/S0377-2217(82)80007-6 — one-machine branch-and-bound with Schrage's heuristic (`job_shop_engine.rs`); preemptive one-machine bound (`lab.rs`).
- Jackson, *"Scheduling a Production Line to Minimize Maximum Tardiness"*, Research Report 43, Management Science Research Project, UCLA, 1955 — preemptive schedule behind the one-machine lower bound (`lab.rs`).
- Adams, Balas & Zawack, *"The Shifting Bottleneck Procedure for Job Shop Scheduling"*, DOI: https://doi.org/10.1287/mnsc.34.3.391 — shifting-bottleneck seed construction and re-optimisation (`job_shop_engine.rs`).
- Kacem, Hammadi & Borne, *"Approach by Localization and Multiobjective Evolutionary Optimization for Flexible Job-Shop Scheduling Problems"*, DOI: https://doi.org/10.1109/TSMCC.2002.1009117 — load-balancing assignment heuristic of the flexible engine (`mod.rs`).
- Mastrolilli & Gambardella, *"Effective Neighbourhood Functions for the Flexible Job Shop Problem"*, Journal of Scheduling 3(1), 2000 — reassignment moves of the graph tabu search.

### Code References
- This code is `etna_highcap` (`c007_a046`, Fred) with the three job_shop rewrites described above, all in `job_shop_engine.rs`; its other `.rs` files are `etna_highcap`'s, with only the name changed in the `help()` text (`mod.rs`) and in the first comment of `lab.rs`.
- `etna_highcap` is `ember_etna` (`c007_a044`, FP Labs) with study settings added: `flow_shop_engine.rs`, `gate_floor.rs` and `hfs_construct.rs` are `ember_etna`'s files unchanged; `mod.rs`, `hybrid_engine.rs` and `job_shop_engine.rs` are `ember_etna`'s with the settings and phase marks passed to their orchestration functions (`solve_challenge`, `solve_pool`, `job_shop::solve`). The one change of default is the fjsp_high work cap in `solve_challenge` (`mod.rs`): 2e11 instead of `ember_etna`'s 6.5e10.
- The engine of the fjsp_high and fjsp_medium tracks (`solve_pool` in `mod.rs`) is FP Labs' original work in `ember_etna`; this submission only raises the default of its fjsp_high work cap.
- The job_shop engine (`job_shop_engine.rs`) is `ember_etna`'s; this submission rewrites three of its parts without changing any of its results.
- `ember_etna` derives from `ember_stromboli` (`c007_a040`); both are by FP Labs (player `0x26c3c880f00daee5f6857837b19499d833338474`).
- Portions of `flow_shop_engine.rs` and `job_shop_engine.rs` derive from the engines published on-chain as `task_tree_h` (`c007_a032`) and `task_tree_j` (`c007_a036`, NVX).
- `hfs_construct.rs` derives from the hybrid_flow_shop construction phase of `task_tree_j` (`c007_a036`, NVX).
- `t47.rs` is `track_t47.rs` of `task_tree_j` (`c007_a036`, NVX), unchanged, as also carried by `ttj_shuttle_ember` (`c007_a042`).
- `hybrid_engine.rs` derives from the engine published on-chain as `shuttle_bell` (`c007_a037`, stinger).
- The use of the t47 pipeline as the start of a graph tabu search, and its hyperparameters, come from `ttj_shuttle_ember` (`c007_a042`, Fred).
- TIG baseline (`dispatching_rules`), replicated by `gate_floor.rs`.


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
