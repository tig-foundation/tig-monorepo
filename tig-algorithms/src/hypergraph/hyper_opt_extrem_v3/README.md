# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** hyper_opt_extrem_v3
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** NVX
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

hyper_opt_extrem_v3 is a GPU-accelerated hypergraph k-way partitioning algorithm featuring guided perturbation, boundary-only refinement, and per-track adaptive strategies.

Key innovations over v2:
- **Guided perturbation**: targets the 500 highest-connectivity hyperedges to focus ILS exploration
- **Boundary-only refinement**: skips interior nodes on small graphs for 5x speedup
- **Incremental edge flag updates**: only recomputes flags for affected hyperedges
- **Per-track tiebreaker**: pseudo-random move ordering diversity for small/medium graphs
- **Adaptive stagnation detection**: faster escape from local minima

Hyperparameters (all optional, pass `null` for defaults):
- `refinement`: max refinement rounds (default: per-track 3000-8000)
- `ils`: ILS iterations (default: per-track 30-150)
- `ils_refine`: refine rounds per ILS iteration (default: per-track 60-120)
- `polish`: final polish rounds (default: per-track 200-600)
- `post_balance`: post-balance cleanup rounds (default: per-track 80-120)
- `move_limit`: max moves per round (default: 200000)

See `help_algorithm hyper_opt_extrem_v3` for more details.

## References and Acknowledgments

### 1. Academic Papers
- Fiduccia, Mattheyses, *"A Linear-Time Heuristic for Improving Network Partitions"*, DAC 1982
- Karypis, Kumar, *"Multilevel k-way Hypergraph Partitioning"*, VLSI Design 2000
- Lourenco, Martin, Stutzle, *"Iterated Local Search"*, Handbook of Metaheuristics 2003
- Hendrickson, Leland, *"A Multi-Level Algorithm For Partitioning Graphs"*, SC 1995
- Gilbert et al., *"Jet: Multilevel Graph Partitioning on GPUs"*, SIAM J. Sci. Comput. 2023
- Schlag et al., *"High-Quality Hypergraph Partitioning"*, ACM J. Exp. Algorithmics 2023
- Glover, *"Tabu Search — Part I & II"*, ORSA J. Computing 1989/1990

### 2. Code References
- hyper_opt_extrem_v2 (TIG) - https://github.com/tig-foundation/tig-monorepo

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
