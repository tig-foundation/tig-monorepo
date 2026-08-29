# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** mica_muscovite
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n_h_edges=10000  | 5T | `{}` |
| n_h_edges=20000  | 5T | `{}` |
| n_h_edges=50000  | 5T | `{}` |
| n_h_edges=100000 | 5T | `{}` |
| n_h_edges=200000 | 5T | `{}` |

> Tracks are dispatched on `num_hyperedges`. **The defaults are the intended operating point** — `{}` selects `effort=5` and the per-track settings that were tuned for it; every hyperparameter simply overrides them, so no override is required. Note that `effort` defaults to 5 here, not 3: the defaults aim at quality. For a faster run at lower quality, pass `{"effort": 3}` or `{"effort": 1}`.
>
> The engine is anytime and deterministic. A valid partition is saved as soon as one exists, and the final save is guarded by a feasibility check so a valid solution is never overwritten by an invalid one. There is no thread, no clock, no hash map and no filesystem access in the control flow; every hot-path comparator is a total order, and the only randomness is a counter-based generator seeded from constants and from `challenge.seed`.
>
> **Actual fuel consumption is far below the recommended grant** — roughly 40 G to 190 G depending on the track, against a 5T budget. The recommendation is conservative: the internal round cap is derived from a cost model, and a much smaller grant makes the run cap its refinement rounds rather than fail.


## References and Acknowledgments

### Academic Papers
- Fiduccia & Mattheyses, *"A Linear-Time Heuristic for Improving Network Partitions"*, 19th Design Automation Conference, 1982 — https://limsk.ece.gatech.edu/book/papers/fm.pdf — the move-based refinement with gain buckets and rollback implemented in `fm.rs`.
- Kernighan & Lin, *"An Efficient Heuristic Procedure for Partitioning Graphs"*, Bell System Technical Journal 49(2), 1970 — the pairwise-exchange idea generalised by the swap and cycle phases of `track.rs`.
- Schlag et al., *"Memetic Multilevel Hypergraph Partitioning"*, GECCO 2018 — the elite pool and the per-hyperedge consensus recombination used by the ILS stage.
- *"Jet: Multilevel Graph Partitioning on GPUs"*, Sandia National Laboratories — the deterministic, lock-free refinement round that `jet.rs` follows: propose every move at once, then filter, rather than applying moves one at a time under a priority queue.
- Raghavan, Albert & Kumara, *"Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks"*, Physical Review E 76, 2007 — the label propagation of `lp.rs`.
- Lourenço, Martin & Stützle, *"Iterated Local Search"*, Handbook of Metaheuristics, 2003 — the perturbation / re-optimisation / acceptance loop around the refinement chain.

### Code References
- **The starting partition is original to this submission.** It is derived deterministically from data the challenge already exposes, rather than from a random or purely greedy seed. It is guarded: if the assumptions it rests on stop holding, the solver falls back to its constructed partition, so the fallback costs quality and never validity.
- The GPU stages (hyperedge clustering, node preferences, move proposal, connectivity reduction, swap gains, elite voting, final balance) and the host refinement chain are original to this submission.
- No third-party solver is vendored. The frozen dependency set is used as-is.


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
