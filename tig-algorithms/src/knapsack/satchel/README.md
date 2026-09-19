# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** satchel
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

Solver for the quadratic knapsack problem, where item pairs carry interaction
values in addition to individual profits. The entry point routes each instance by
item count and budget fraction to a dedicated per-track engine, so the search
strategy matches the instance regime. Each engine builds a strong initial packing
(a parametric-breakpoint / greedy seed), then improves it with iterated local
search over exchange neighbourhoods and a dynamic-programming refinement core; the
harder regimes add a population with crossover and an exact branch-and-bound stage
on a reduced core to close the last of the gap. All randomisation is seeded only
from the public instance seed. Effort is bounded by deterministic work counters,
never wall-clock, so throughput is reproducible. All tuning is exposed as
hyperparameters read in `from_map`; the defaults are the shipped operating point.

## References and Acknowledgments

### Academic Papers
- D. Pisinger, *"The quadratic knapsack problem — a survey"*, Discrete Applied Mathematics, 2007.
- A. Caprara, D. Pisinger, P. Toth, *"Exact Solution of the Quadratic Knapsack Problem"*, INFORMS Journal on Computing, 1999 (branch-and-bound core).
- H. R. Lourenço, O. C. Martin, T. Stützle, *"Iterated Local Search"*, Handbook of Metaheuristics, 2003.
- R. Bellman, *"Dynamic Programming"*, Princeton University Press, 1957 (dynamic-programming refinement).

### Code References
- TIG baseline (knapsack).

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
