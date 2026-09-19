# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** mica_inf
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


Minimises the connectivity metric km1 = sum over hyperedges of (blocks touched - 1), under the
challenge's block-size constraint.

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n_h_edges=10000  | 5T | `{}` |
| n_h_edges=20000  | 5T | `{}` |
| n_h_edges=50000  | 5T | `{}` |
| n_h_edges=100000 | 5T | `{}` |
| n_h_edges=200000 | 5T | `{}` |

> Tracks are dispatched on `num_hyperedges`. **The defaults are the intended operating point**:
> `{}` selects `effort=5` and the per-track settings that go with it. Every hyperparameter listed
> by `help()` overrides its default; an integer (representable as i64) outside its range is
> clamped, anything else takes the default. A lower `effort` (0 to 4) trades quality for time.
>
> The solver first builds a partition on the device, then infers a starting partition from the
> hypergraph's connectivity alone (device-side recursive bisection) and uses it to initialise the
> refinement. The constructed partition is kept as the fallback when the inference is not usable.
> Refinement then alternates device rounds (move proposal, filtering and quota selection on the
> device, execution on the host), pairwise and cyclic block exchanges, and an iterated local search
> with an elite pool, before a host refinement chain finishes the run.
>
> The engine is anytime and deterministic. A feasible partition is saved as soon as one exists, and
> every later save is guarded by a feasibility check and an objective comparison, so a saved
> solution is only ever replaced by a feasible better one. There is no thread, no clock, no hash map
> and no filesystem access in the control flow, and every random draw is seeded from constants and
> from `challenge.seed`, which is used only as a random-number seed. Grid-wide synchronisation occurs
> only at launch boundaries, so `fuel_consumed` and `runtime_signature` repeat across runs of the
> same nonce.
>
> The fuel recommendation is conservative: the main refinement loop caps its round count from the
> granted fuel, and the last saved partition stands if the fuel runs out later.


## References and Acknowledgments

### Academic Papers
- Fiduccia & Mattheyses, *"A Linear-Time Heuristic for Improving Network Partitions"*, 19th Design Automation Conference, 1982 — https://doi.org/10.1109/DAC.1982.1585498 — the move-based refinement with rollback of `fm.rs`.
- Osipov & Sanders, *"n-Level Graph Partitioning"*, ESA 2010 — https://doi.org/10.1007/978-3-642-15775-2_24 — the adaptive stopping rule of the localised searches in `fm.rs`.
- Akhremtsev, Heuer, Sanders & Schlag, *"Engineering a Direct k-way Hypergraph Partitioning Algorithm"*, ALENEX 2017 — https://doi.org/10.1137/1.9781611974768.3 — the localised k-way FM search and the km1 gain cache of `gain.rs`.
- Karypis, Aggarwal, Kumar & Shekhar, *"Multilevel Hypergraph Partitioning: Applications in VLSI Domain"*, IEEE Transactions on VLSI Systems 7(1), 1999 — https://doi.org/10.1109/92.748202 — the contraction and projection step of `coarsen.rs`.
- Kernighan & Lin, *"An Efficient Heuristic Procedure for Partitioning Graphs"*, Bell System Technical Journal 49(2), 1970 — https://doi.org/10.1002/j.1538-7305.1970.tb01770.x — the pairwise exchanges generalised by the swap and cycle phases.
- Zhou, Huang & Schölkopf, *"Learning with Hypergraphs: Clustering, Classification, and Embedding"*, NIPS 2006 — the hypergraph random walk whose fixed-point iteration drives the device-side bisection of `kernels_inf.cu`.
- Pothen, Simon & Liou, *"Partitioning Sparse Matrices with Eigenvectors of Graphs"*, SIAM Journal on Matrix Analysis and Applications 11(3), 1990 — https://doi.org/10.1137/0611030 — recursive spectral bisection with a balanced threshold split.
- Gilbert, Madduri, Boman & Rajamanickam, *"Jet: Multilevel Graph Partitioning on Graphics Processing Units"*, SIAM Journal on Scientific Computing 46(5), 2024 — https://doi.org/10.1137/23M1559129 — the batched propose-then-filter refinement round of `jet.rs`.
- Raghavan, Albert & Kumara, *"Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks"*, Physical Review E 76, 036106, 2007 — https://doi.org/10.1103/PhysRevE.76.036106 — the label propagation of `lp.rs`.
- Lourenço, Martin & Stützle, *"Iterated Local Search"*, Handbook of Metaheuristics, 2003 — https://doi.org/10.1007/0-306-48056-5_11 — the perturbation / re-optimisation / acceptance loop around the refinement.
- Andre, Schlag & Schulz, *"Memetic Multilevel Hypergraph Partitioning"*, GECCO 2018 — https://doi.org/10.1145/3205455.3205475 — the elite pool with per-hyperedge consensus recombination, and the crossover of the 10k track.
- Kirkpatrick, Gelatt & Vecchi, *"Optimization by Simulated Annealing"*, Science 220(4598), 1983 — https://doi.org/10.1126/science.220.4598.671 — the annealed acceptance of the 10k track's iterated local search.

### Code References
- The starting partition inferred from connectivity, the device-side move selection
  (`kernels_filter.cu`), the device stages of each track and the host refinement chain are original
  to this submission. The device stages and the host chain continue the submitter's earlier
  `mica_muscovite` line, which they extend.
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
