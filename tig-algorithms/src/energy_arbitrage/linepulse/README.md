# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** linepulse
* **Copyright:** 2026 Iain Jackson
* **Identity of Submitter:** Iain Jackson
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

The solver plans with value tables and decides each step with an exact linear program.

1. **Value tables.** For every battery and time step, a backward dynamic program computes the value of holding each state of charge (a 25-point grid). It uses a small set of candidate charge and discharge levels: 13, 9 or 7, depending on fleet size (5 in the intermediate fixed-point tables). The tables average over the public real-time price model. A 5-point Gauss-Hermite rule covers the Gaussian price noise, and conditional-mean bins cover the heavy-tailed price jumps.
2. **Deliverability and terminal value.** Future power in the tables is scaled down by how much of a simultaneous fleet move the battery's network lines can carry. The floors are tuned by fleet size, with a stronger end-of-horizon drain on the largest fleets. On fleets of 60 or more batteries, energy left at the horizon gets a small salvage value.
3. **Congestion-priced tables.** The fleet is rolled forward on day-ahead prices, solving the step LP at every step. Each line's dual price, multiplied by the battery's network sensitivity (PTDF), becomes a per-battery price shift, smoothed over a few steps. The tables are rebuilt on the shifted prices, and this repeats as a damped fixed point: 3-4 rounds, depending on fleet size. In the last round, small fleets average two rollouts on randomly perturbed prices, and the largest fleets sample real-time prices over the final 12 hours.
4. **Fleet-aggregate correction.** A second dynamic program treats the whole fleet as one battery, limited by the largest move the network can deliver. A small share of the gap between its marginal value of energy and the batteries' average marginal value is added to every battery's decision.
5. **Step decisions.** Each step solves an exact LP that maximises this step's revenue plus the change in table value, subject to every line limit. Line constraints are added only when violated (constraint generation), starting from the lines that bound during the rollouts. Over the last 4-6 steps, one LP plans all remaining steps together with state-of-charge and line constraints. If no LP answer exists, a merit-order allocator grants each battery's one-step request in order of value per MW until a line binds. This path did not run on any of 57,600 test decision steps.
6. **Efficiency.** Batteries at the same node with the same normalised parameters share one table, scaled by capacity. The LPs use a bounded dual simplex with a long-step (bound-flipping) ratio test (a revised simplex for the multi-step LP). Rollouts use a coarser state-of-charge grid than decisions.

Only causal information is used: day-ahead prices, the published real-time price model parameters, and the current realised state. The algorithm has no hyperparameters, and it clears all of its caches at the start of each instance.

## References and Acknowledgments

### 1. Academic Papers
- D. P. Bertsekas, *Dynamic Programming and Optimal Control*, Athena Scientific (backward induction for the value tables).

### 2. Code References
- NVX, the `gridflow` and `peakshift` energy_arbitrage submissions in tig-monorepo. Several techniques here belong to the same family as that engine's:
  - stochastic DP with Gauss-Hermite quadrature over the RT price model;
  - deliverability-aware value of stored energy and an end-of-horizon drain;
  - rebuilding the DP tables with PTDF-based congestion prices;
  - LP-based step dispatch.

  No code was copied. The implementations here were written independently.

  The fleet-aggregate correction, the multi-step endgame LP, the merit allocator and the efficiency work are original to this submission.

### 3. Other
- None.

## Additional Notes

Validated on the author's hardware against the peakshift submission over 375 instances across the five tracks plus three independent 250-instance seeds: about +1.3% mean quality at roughly a quarter of the wall-clock, zero invalid solutions, and a maximum of 16.8 Gfuel per instance (budget 500 Gfuel).

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
