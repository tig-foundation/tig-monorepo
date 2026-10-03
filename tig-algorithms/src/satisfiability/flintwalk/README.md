# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** flintwalk
* **Copyright:** 2026 Kernelsmith
* **Identity of Submitter:** Kernelsmith
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

`flintwalk` is a stochastic local search solver, written from scratch. The search is WalkSAT/SKC
(Selman, Kautz & Cohen 1994) with a freebie move, optional adaptive noise (Hoos 2002, off by default)
and escalating random kicks on stagnation; the initial assignment comes from a literal-count bias,
pure literals and unit propagation. Deterministic throughout: the only randomness is an xoshiro256++
stream seeded from `challenge.seed`.

Two things are specific to this submission.

**1. Cost per step.** The network runs the fuel-instrumented binary, and the instrumentation is
charged per basic block, so a short loop with an exit branch pays a lot of instrumentation per useful
instruction. The three hot loops therefore process a fixed, branch-free block of eight occurrence-list
entries and mask the overhang instead of branching per entry: break counting scans the full list with
no early exit and accumulates branch-free, and the two flip loops update the clause counters over the
same fixed block (entries past the end of a list are rewritten with their own value, a no-op, and the
clauses that just became empty are collected through a bit mask). `occ` carries exactly eight reserve
slots at the end, so a block may always read - and harmlessly rewrite - past the end of one list.
This costs about 1.35x the memory accesses and saves far more in branches. It does not change the
search: against the previous version the solver produces bit-identical solutions and identical flip
counts (nonce 19301 on n_vars=100000/ratio=4150: flips = 34,206,069, freebies = 2,962,795 in both).

Measured 2026-09-20 with the official `tig-runtime` / `tig-verifier` (fuel-instrumented `.so` built
with `build_so` and the LLVM fuel plugin, monorepo `a4db8c5`), hyperparameters `{}` = defaults, fresh
instances, 25 instances on `n_vars=100000,ratio=4150`, every solver on the same instances, 0 invalid:

| Algorithm | solved | median | mean | p90 | median fuel |
|---|---:|---:|---:|---:|---:|
| **flintwalk** | 25/25 | **12.20 s** | 12.64 s | 14.81 s | 13.96 G |
| previous version of this solver | 25/25 | 25.54 s | 25.95 s | 29.79 s | 11.87 G |

Pairwise, the new loops win 25 of 25. Fuel stays at 0.28 % of the 5e12 budget; it rises slightly
because the fixed block reads `max(8, len)` entries instead of `len` - the gain is in branches.

**2. Two regimes, detected from the instance.** The ratio of clauses to variables says whether the
instance sits below the satisfiability phase transition or at it, and the two need opposite settings.
Below (`ratio < 4.25`) instances are large and effectively always satisfiable: long uninterrupted runs
win. Restarting hurts there, measured pairwise per nonce on `ratio=4200`: fixed attempt length
74 / 447 / 93 s, with Luby restarts 649 / 628 / 240 s, with shorter attempts partly not solved at all.
At the transition (`ratio >= 4.25`) instances are small and a good share of them is unsatisfiable, so
two things change:

**The solver limits its own number of steps** (50,000 flips per variable at the transition, 80,000
below it). This matters: the runtime's fuel limit is not enforced inside a tight compute loop. Same
seed, same budget, only one hyperparameter apart: with a periodic write the run stops at the budget
(status 87, 0.27 s), without it the run keeps going past 100 s. A solver with no step limit of its
own therefore never gives up on an unsatisfiable instance, which is useless for a benchmarker
computing 100 nonces per bundle, and the fuel it finally reports may exceed the budget. Below the
transition the limit never binds (340 flips per variable are needed there).

**The break weights become polynomial** - probSAT with `(eps + break)^-cb`, eps 0.9, cb 2.38
(Balint & Schoening 2012) - instead of the exponential `cb^-break`.

Measured 2026-09-21 on `n_vars=5000,ratio=4267`, 10 instances, 50,000 flips per variable, scoring the
smallest number of unsatisfied clauses reached (lower is better): polynomial weights with fixed
attempt length 4.5; the same with restarts on stagnation 39.5; WalkSAT with restarts on stagnation
20.6; WalkSAT with an age-based tie-break 135.8. Only the polynomial weights help in this solver, so
only they are switched on. In the delivered configuration the solver reaches a mean of 2.1 remaining
clauses and solves 3 of 10 instances in 65 s and 113 G fuel (2.3 % of the budget) per instance; under
an equal wall-clock budget the three public network solvers solve exactly the same instances.

Hyperparameters are optional JSON overrides (`noise`, `phi`, `theta`, `kick_len0`, `kick_max`,
`try_flips_per_var`, `max_flips_per_var`, `pick` = `skc` | `probsat` | `probsat_poly`, `cb`, `eps`,
`init`, `luby`, `restart`, `kick_reset`, `best_keep`, `ratio_split`, `stats`, `trace`). The defaults
are the measured configuration; in particular adaptive noise is off (`phi = 0`, fixed p = 0.5),
because on the 100k tracks, which sit below the phase transition, adapting the noise drives p down to
~0.11 and the search stalls.

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
