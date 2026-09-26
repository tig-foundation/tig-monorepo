# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** mica_clintonite
* **Copyright:** 2026 FP Labs
* **Identity of Submitter:** FP Labs
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

Minimises the connectivity metric km1 (sum over hyperedges of the number of blocks touched minus one) under the
challenge's block-size constraint, with device-side refinement rounds followed by a host refinement chain.

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n_h_edges=10000  | 5T | `{}` |
| n_h_edges=20000  | 5T | `{}` |
| n_h_edges=50000  | 5T | `{}` |
| n_h_edges=100000 | 5T | `{}` |
| n_h_edges=200000 | 5T | `{}` |

> Tracks are dispatched on `num_hyperedges`, and `{}` selects the per-track defaults, which are the intended
> operating point. Every hyperparameter listed by `help()` overrides its default; a missing, non-integer or
> out-of-range value takes the default. `effort` (0 to 5) and `refinement` set the size of the refinement budget.
>
> The engine is anytime and deterministic. A feasible partition is saved as soon as one exists, and a saved solution is
> only ever replaced by a feasible better one. There is no thread, no clock, no hash map and no filesystem access in
> the control flow. Every random draw is seeded from constants and from `challenge.seed`, which is used only as a
> random-number seed.

## References and Acknowledgments

### Academic Papers
- Fiduccia & Mattheyses, *"A Linear-Time Heuristic for Improving Network Partitions"*, 19th Design Automation Conference, 1982 — https://doi.org/10.1109/DAC.1982.1585498
- Osipov & Sanders, *"n-Level Graph Partitioning"*, ESA 2010 — https://doi.org/10.1007/978-3-642-15775-2_24
- Akhremtsev, Heuer, Sanders & Schlag, *"Engineering a Direct k-way Hypergraph Partitioning Algorithm"*, ALENEX 2017 — https://doi.org/10.1137/1.9781611974768.3
- Karypis, Aggarwal, Kumar & Shekhar, *"Multilevel Hypergraph Partitioning: Applications in VLSI Domain"*, IEEE Transactions on VLSI Systems 7(1), 1999 — https://doi.org/10.1109/92.748202
- Schlag, Heuer, Gottesbüren, Akhremtsev, Schulz & Sanders, *"High-Quality Hypergraph Partitioning"*, ACM Journal of Experimental Algorithmics 27, 2022 — https://doi.org/10.1145/3529090
- Kernighan & Lin, *"An Efficient Heuristic Procedure for Partitioning Graphs"*, Bell System Technical Journal 49(2), 1970 — https://doi.org/10.1002/j.1538-7305.1970.tb01770.x
- Zhou, Huang & Schölkopf, *"Learning with Hypergraphs: Clustering, Classification, and Embedding"*, NIPS 2006
- Pothen, Simon & Liou, *"Partitioning Sparse Matrices with Eigenvectors of Graphs"*, SIAM Journal on Matrix Analysis and Applications 11(3), 1990 — https://doi.org/10.1137/0611030
- Gilbert, Madduri, Boman & Rajamanickam, *"Jet: Multilevel Graph Partitioning on Graphics Processing Units"*, SIAM Journal on Scientific Computing 46(5), 2024 — https://doi.org/10.1137/23M1559129
- Raghavan, Albert & Kumara, *"Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks"*, Physical Review E 76, 036106, 2007 — https://doi.org/10.1103/PhysRevE.76.036106
- Lourenço, Martin & Stützle, *"Iterated Local Search"*, Handbook of Metaheuristics, 2003 — https://doi.org/10.1007/0-306-48056-5_11
- Andre, Schlag & Schulz, *"Memetic Multilevel Hypergraph Partitioning"*, GECCO 2018 — https://doi.org/10.1145/3205455.3205475
- Kirkpatrick, Gelatt & Vecchi, *"Optimization by Simulated Annealing"*, Science 220(4598), 1983 — https://doi.org/10.1126/science.220.4598.671

### Code References
- The device stages, the device-side move selection and execution, and the host refinement chain are original to
  this submission. They continue the submitter's earlier `mica` line, from `mica_muscovite` to `mica_inf`.
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
