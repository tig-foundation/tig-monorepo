# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** shuttle_bell
* **Copyright:** 2026 stinger
* **Identity of Submitter:** stinger
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Summary

A shape-aware anytime tabu search for the flexible job shop, specialised for the
**hybrid_flow_shop** and **fjsp_medium** tracks.

Three design points distinguish it from the current field:

**1. Fuel-aware anytime search.** The algorithm reads `__fuel_remaining` and keeps improving
until the marginal gain no longer justifies the fuel, saving on every improvement, with a
`fuel_cap` hyperparameter (default 5e11). Existing job_scheduling algorithms run fixed
iteration budgets, so at a 5e12 grant they continue spending long after their quality curve
has flattened. This is a throughput gain for the Benchmarker, not only a quality gain.

**2. Guaranteed validity floor.** The effort-0 dispatching-rules baseline is replayed and saved
first; every subsequent save is gated on `makespan <= floor`. Across 1,230 nonces spanning all
five tracks and six fuel levels: **0 invalid nonces, 0 verifier failures.** Since a single
invalid nonce discards an entire benchmark, this is a material reliability property.

**3. Instance-shape detection driving the search.** A detector separates rigid from flexible
and single-route from multi-route instances (validated on all five tracks) and selects the
restart policy and neighbourhood accordingly — including a Mastrolilli–Gambardella machine
reassignment move that relocates a critical operation to the best insertion position on an
alternative eligible machine, found by a single head/tail pass.

Measured against the incumbent on identical nonce sets, real production seeds, `fuel_budget` 5e11
(bar = live 100th-best bundle):

| Track | Bar | shuttle_bell | task_tree_g |
|---|---|---|---|
| hybrid_flow_shop | 74,242 | **74,421** | 59,412 |
| fjsp_medium | 108,479 | **126,553** | ~107,650 |
| job_shop | 93,466 | 86,434 | 90,336 |
| fjsp_high | 66,367 | 29,240 | 66,222 |
| flow_shop | 34,241 | 8,428 | 23,770 |

It is a specialist and does not claim otherwise: it is behind on flow_shop, job_shop and
fjsp_high. Two of those are structural. A full Ruiz–Stützle Iterated Greedy was implemented and
measured for flow_shop and came out 43–189% worse than the disjunctive-graph schedule, because
reentrance makes a permutation representation the wrong encoding — the `route_is_unique()` gate
present in existing algorithms is sound engineering rather than an oversight.

Benchmarkers running this algorithm should weight `num_bundles` toward hybrid_flow_shop and
fjsp_medium.

## Target Tracks & Recommended Hyperparameters

| Track ID | Track | Recommended Fuel | Recommended HP |
|----------|-------|------------------|----------------|
| T44 | n=50,s=fjsp_high        | 1T | `{}` |
| T45 | n=50,s=fjsp_medium      | 5T | `{}` |
| T46 | n=50,s=flow_shop        | 5T | `{}` |
| T47 | n=50,s=hybrid_flow_shop | 5T | `{}` |
| T48 | n=50,s=job_shop         | 1T | `{}` |

> Track shape is auto-detected; no `track` hyperparameter is required. Defaults are baked in.
> `fuel_cap` (u64, default 5e11) caps fuel spend regardless of the grant — below 2e11,
> fjsp_medium falls under its qualifying bar. job_shop and fjsp_high saturate by 1e11.
> Run `help_algorithm shuttle_bell` for the full hyperparameter list.

## References and Acknowledgments

### 1. Academic Papers
- Nowicki, E. & Smutnicki, C. (1996). *A Fast Taboo Search Algorithm for the Job Shop Problem.*
  Management Science 42(6) — critical-block neighbourhood and tabu structure.
- Mastrolilli, M. & Gambardella, L.M. (2000). *Effective Neighbourhood Functions for the Flexible
  Job Shop Problem.* Journal of Scheduling 3(1) — machine reassignment with best-position insertion.
- Taillard, E. (1990). *Some Efficient Heuristic Methods for the Flow Shop Sequencing Problem.*
  EJOR 47(1) — accelerated insertion evaluation.
- Ruiz, R. & Stützle, T. (2007). *A Simple and Effective Iterated Greedy Algorithm for the
  Permutation Flowshop Scheduling Problem.* EJOR 177(3) — implemented and measured for flow_shop;
  reported here as a negative result (see Summary).

### 2. Code References
- TIG baseline (`dispatching_rules`) — used as the guaranteed validity floor.

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
