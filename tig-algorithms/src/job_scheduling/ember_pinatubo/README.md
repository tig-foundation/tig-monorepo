# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** ember_pinatubo
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

`{}` selects the per-track defaults, which are the intended operating point. The solver is deterministic and anytime.

## References and Acknowledgments

### Academic Papers
- Pearce & Kelly, *"A Dynamic Topological Sort Algorithm for Directed Acyclic Graphs"*, DOI: https://doi.org/10.1145/1187436.1210590
- Glover & Laguna, *"Tabu Search"*, DOI: https://doi.org/10.1007/978-1-4615-6089-0
- Nowicki & Smutnicki, *"A Fast Taboo Search Algorithm for the Job Shop Problem"*, DOI: https://doi.org/10.1287/mnsc.42.6.797
- Taillard, *"Parallel Taboo Search Techniques for the Job Shop Scheduling Problem"*, DOI: https://doi.org/10.1287/ijoc.6.2.108
- Dueck, *"New Optimization Heuristics: The Great Deluge Algorithm and the Record-to-Record Travel"*, DOI: https://doi.org/10.1006/jcph.1993.1010
- Feo & Resende, *"Greedy Randomized Adaptive Search Procedures"*, DOI: https://doi.org/10.1007/BF01096763
- Nawaz, Enscore & Ham, *"A Heuristic Algorithm for the m-Machine, n-Job Flow-Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0305-0483(83)90088-9
- Taillard, *"Some Efficient Heuristic Methods for the Flow Shop Sequencing Problem"*, DOI: https://doi.org/10.1016/0377-2217(90)90090-X
- Ruiz & Stützle, *"A Simple and Effective Iterated Greedy Algorithm for the Permutation Flowshop Scheduling Problem"*, DOI: https://doi.org/10.1016/j.ejor.2005.12.009
- Johnson, *"Optimal Two- and Three-Stage Production Schedules with Setup Times Included"*, DOI: https://doi.org/10.1002/nav.3800010110
- Palmer, *"Sequencing Jobs Through a Multi-Stage Process in the Minimum Total Time - A Quick Method of Obtaining a Near Optimum"*, DOI: https://doi.org/10.1057/jors.1965.8
- Campbell, Dudek & Smith, *"A Heuristic Algorithm for the n Job, m Machine Sequencing Problem"*, DOI: https://doi.org/10.1287/mnsc.16.10.B630
- Carlier, *"The One-Machine Sequencing Problem"*, DOI: https://doi.org/10.1016/S0377-2217(82)80007-6
- Adams, Balas & Zawack, *"The Shifting Bottleneck Procedure for Job Shop Scheduling"*, DOI: https://doi.org/10.1287/mnsc.34.3.391
- Kacem, Hammadi & Borne, *"Approach by Localization and Multiobjective Evolutionary Optimization for Flexible Job-Shop Scheduling Problems"*, DOI: https://doi.org/10.1109/TSMCC.2002.1009117

### Code References
- TIG baseline `dispatching_rules`.
- This submission continues the submitter's `ember` line.

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
