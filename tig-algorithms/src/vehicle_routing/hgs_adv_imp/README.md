# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** hgs_adv_imp
* **Copyright:** 2026 Karl Shand
* **Identity of Submitter:** Karl Shand
* **Identity of Creator of Algorithmic Method:** Thibaut Vidal
* **Unique Algorithm Identifier (UAI):** c002_a110

## Additional Information

### Base Algorithm

This algorithm is derived from `hgs_advance`, a Hybrid Genetic Search (HGS) solver
for the Vehicle Routing Problem with Time Windows (VRPTW), originally authored by
**Thibaut Vidal**. The HGS framework combines a genetic algorithm with a powerful
local search engine, using SREX and OX crossover operators, adaptive penalty
management, and a hierarchical decomposition strategy (reverse mode) to solve large
VRPTW instances by breaking them into spatially coherent subproblems.

### Improvements in hgs_adv_imp

Two targeted improvements were made over the baseline `hgs_advance`:

#### 1. Route-Split Escape in Decomposition (`reverse_mode.rs`)

**What it does:** During the hierarchical decomposition loop, if a phase completes
without improving the incumbent solution cost, the algorithm identifies the longest
route in the current solution, splits its clients into two halves (verifying capacity
feasibility), and uses this modified route set as the input for clustering in the
next phase.

**Why it works:** The decomposition clusters routes spatially and solves each cluster
as an independent subproblem. When the same clustering structure repeats across
phases, the sub-GAs converge to the same local optima. Splitting the longest route
forces a different partition of the customer space, exposing the sub-GAs to new
neighbourhood relationships they could not explore under the original route
boundaries. This diversification mechanism is cheap (O(n) to split) and only
activates when stagnation is detected, so it adds no overhead on improving phases.

#### 2. Squared Diversity Rank in Population Management (`population.rs`)

**What it does:** The biased fitness function that governs parent selection and
survivor elimination combines a cost rank with a diversity rank. In the original,
diversity rank is linear (position / (n-1)). In hgs_adv_imp, for populations of
10 or more individuals, the diversity rank is squared (rank²).

**Why it works:** Squaring the diversity rank creates a nonlinear selection pressure:
individuals in the middle of the diversity spectrum receive a smaller diversity bonus
(0.5² = 0.25 vs 0.5 linear), while genuinely isolated individuals at the top of the
diversity ranking retain a strong bonus (0.9² = 0.81 vs 0.9 linear — the relative
gap widens). This sharpens the algorithm's preference for truly novel solutions over
incrementally different ones, which reduces premature convergence on larger population
sizes without sacrificing the cost-based selection pressure that drives quality.

**Test results:** Testing locally using an AMD 7950 CPU shows that qualities increased overall for all 5 tracks, with the biggest gains coming from larger n_nodes tracks, which is expected.

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
