# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** shiftforge
* **Copyright:** 2026 Kernelsmith
* **Identity of Submitter:** Kernelsmith
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

`shiftforge` is a clean-room flexible job-shop solver. It is deterministic (one `SmallRng` seeded from
`challenge.seed`, no clock, no threads, no file access), anytime (a valid schedule is saved as soon as
one exists and only overwritten on improvement), and it stops on its own work ceiling rather than on the
granted fuel budget.

**Pipeline.** The challenge's greedy reference procedure is reproduced exactly first, so the first saved
schedule is never worse than the greedy validity bound. Then: randomised multi-start non-delay
dispatching (Giffler & Thompson 1960); for single-machine routes an NEH construction (Nawaz, Enscore &
Ham 1983) with iterated greedy (Ruiz & Stuetzle 2007) on the job permutation; for rigid instances a
shifting-bottleneck construction (see below); and finally an elite cycle of short tabu searches on the
disjunctive graph (Nowicki & Smutnicki 1996 critical-block neighbourhood, Taillard 1994 head/tail
bounds for constant-time move estimates, Mastrolilli & Gambardella 2000 machine re-insertion inside the
feasible window, evaluation through a maintained topological order in the style of Pearce & Kelly 2007).

Three things are specific to this submission, and each was chosen by measurement.

**1. Exact one-machine re-sequencing (Carlier 1982).** Schrage's heuristic plus branch and bound with a
node budget, on heads and tails taken from the graph *without* the arcs of the machine being solved.

**2. Shifting bottleneck construction (Adams, Balas & Zawack 1988).** Machine sequences are not guessed
from random dispatching orders but fixed one at a time - always the machine whose exactly solved
one-machine problem has the largest value - after which the already fixed machines are re-solved and
kept when the makespan drops. On `job_shop` this reaches 11.25 % above the lower bound in **1.44 s**,
where 300 randomised dispatching starts need 11.89 s to reach 18.28 %. It is used only on rigid
instances: on flexible ones it produced 11,793 against 10,466 from the dispatching pool, because it
re-orders machines but does not touch the assignment.

**3. A measured work ceiling instead of spending the grant.** The solver stops after ~350 G
instructions. Measured on `job_shop`: 100 G gives quality 81,309 (141 s), 200 G gives 84,156 (263 s),
350 G gives 85,020 (471 s), 700 G gives 86,154 (896 s). A benchmarker computes 40 nonces per bundle, so
spending the whole 5e12 grant would cost over 20 h per bundle for a few tenths of a percent.

The number of assignment moves generated per iteration is also set per track class, because the optimum
differs: with average flexibility ~2.5 a cap of 16 gives 159,584 against 135,478 for unlimited; on
same-route flexible instances 32 is best (80,162 against 72,240); with average flexibility ~10 unlimited
wins (26,206 against 16,710 for a cap of 16). A single global cap would have cost 37 % on that last
track. An exact re-sequencing pass *after* the tabu search was implemented and then switched off: it
finds a different order on 29 of 30 machines and improves the makespan on none.

**Measured against the highest-adoption algorithm of this challenge** (`ember_stromboli`), 8 instances per track,
pairwise on identical instances, both solvers in their delivered configuration with the real 5e12 grant,
official `tig-runtime` / `tig-verifier`, hyperparameters `{}`, 0 invalid. Positive = this submission has
the lower makespan:

| Track | makespan mean | median | better on | runtime here / leader |
|---|---|---|---|---|
| n=50,s=hybrid_flow_shop | **+0.74 %** | **+0.94 %** | 5 of 8 | 540 s / 601 s |
| n=50,s=fjsp_medium | +0.25 % | +0.23 % | 4 of 8 | 628 s / 342 s |
| n=50,s=job_shop | +1.55 % | +0.20 % | 4 of 8 | 555 s / 479 s |

Honest about the rest: on `n=50,s=flow_shop` quality is level but the leader gets there 24 times faster
(28 s against 672 s), and on `n=50,s=fjsp_high` this submission is 4.5 % behind in makespan. On those two
tracks it is not the better choice.

Hyperparameters are optional (`pool`, `cycle`, `floor_pct`, `rcap`); the defaults are the measured
configuration.

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
