# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** satchel_cluster
* **Copyright:** 2026 Lucas Monnin
* **Identity of Submitter:** Lucas Monnin
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- D. Pisinger, *"The quadratic knapsack problem — a survey"*, Discrete Applied Mathematics, 2007.
- A. Caprara, D. Pisinger, P. Toth, *"Exact Solution of the Quadratic Knapsack Problem"*, INFORMS Journal on Computing, 1999.
- H. R. Lourenço, O. C. Martin, T. Stützle, *"Iterated Local Search"*, Handbook of Metaheuristics, 2003.

### 2. Code References
- `satchel` (c003_a154, NVX) — `track1.rs` … `track5.rs` and the dispatcher are satchel's, unchanged.

## Additional Notes

`satchel_cluster` runs satchel and then, on the tracks with budget ≤ 17 % (n_items=1000 at budget 5 and 10,
n_items=5000 at budget 10), an exact local-search polish of satchel's packing (`polish.rs`).

The generator assigns participants to project subsets, so items form clusters: strong pair values inside a cluster,
weak ones to the rest. Where satchel stops short, the better packing almost always exchanges a whole cluster for
another of the same total weight — a move per-item scoring against the current packing does not see, because an
unselected cluster's value lies in its internal pairs. The polish keeps a contribution vector
contrib[i] = Σ_{j∈S} V[i][j] and searches, best improvement first: add; 1-1, 2-1 and 1-2 swaps (exactly pruned, since
V ≥ 0 bounds every gain by the best fitting outside contribution); *cluster-in* (take the unselected part of a
cluster of the graph V ≥ 150, eject the lowest contribution per weight until it fits, refill); and *cluster swap /
drop-and-refill* (exchange the selected part of one cluster for the unselected part of another, ranked by an
exact-before-repair estimate). Every move is applied exactly and undone unless the value rises; the polished
packing is saved only if strictly better, so results are never below satchel's.

Offline, against satchel on fresh seeds with the protocol's generator and quality rule, never worse on any
instance, extra solve time ≈ 2 % (1000 items) and ≈ 7 % (5000 items) of the benchmarker's per-nonce cost.
Tracks with budget > 17 % run satchel unchanged.

Deterministic (no clock, threads or randomness); no hyperparameters needed. Recommended fuel: as satchel.

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
