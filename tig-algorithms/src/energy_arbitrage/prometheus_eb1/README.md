# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** prometheus_eb1
* **Copyright:** 2026 Prometheus Early-Beta Swarm
* **Identity of Submitter:** Prometheus Early-Beta Swarm
* **Identity of Creator of Algorithmic Method:** Discovered by the Prometheus early-beta multi-agent LLM swarm (143 contributing agents across multiple model providers). 
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

This algorithm was **evolved by an autonomous multi-agent LLM swarm** rather than ported from a single prior work; the techniques it composes are, however, standard tools from convex optimization, optimal control, and power-systems engineering:

### 1. Academic Papers
- Boyle, J. P. & Dykstra, R. L., *"A Method for Finding Projections onto the Intersection of Convex Sets in Hilbert Spaces"*, Lecture Notes in Statistics, vol. 37, pp. 28–47 (1986). — basis for the alternating/Dykstra-style projection onto the feasible polytope.
- Dykstra, R. L., *"An Algorithm for Restricted Least Squares Regression"*, Journal of the American Statistical Association, 78(384), pp. 837–842 (1983).

### 2. Code References
- None. The implementation is original swarm-generated Rust; no external source was copied.

### 3. Other
- **PTDF (Power Transfer Distribution Factors)** linearized DC network-flow model — standard transmission-constraint formulation in power-systems dispatch.
- **Receding-horizon Model Predictive Control (MPC)** with a per-battery dynamic-programming value function — standard stochastic-control framing for storage arbitrage.


## Additional Notes

### Runtime / fuel behavior

The solver is bounded by the runtime's **fuel** counter (`__fuel_remaining`), not by
wall-clock time, so its stopping point is deterministic across machines:

- At startup it saves an all-zeros schedule as a safety net, then runs the optimizing
  rollout.
- It spends fuel on the per-step optimization until the remaining fuel reaches a small
  reserve (~1/28 of what was available), after which the remaining steps fall back to
  zero actions so the rollout always completes and saves a valid solution. It never
  intentionally exhausts fuel (no out-of-fuel exit).
- **More fuel ⇒ more optimization ⇒ higher quality**, up to the 10,000,000 cap. Easy
  tracks finish full optimization using almost no fuel; the largest track (`capstone`)
  needs roughly **240–250B fuel consumed** to reach the maximum score.


### Hyperparameters

All hyperparameters are optional integers passed as a JSON object; any omitted field
uses its default. Values are clamped to safe minimums where noted so the solver cannot
be misconfigured into a panic. Example:

```json
{"dp_soc_levels": 41, "policy_action_levels": 97, "grad_outer_iters": 40}
```

| Hyperparameter         | Default | Min | Effect of increasing it |
|------------------------|---------|-----|-------------------------|
| `dp_soc_levels`        | 33      | 2   | Finer SOC grid for the DP value function → better long-horizon decisions, more setup fuel. |
| `dp_action_levels`     | 17      | 3   | Finer action grid while *building* the DP → smoother value function, more setup fuel. |
| `policy_action_levels` | 65      | 3   | Finer action grid when *querying* the DP at runtime → better per-step actions, more per-step fuel. |
| `proj_max_iters`       | 80      | 1   | More alternating-projection iterations onto the feasible flow polytope → tighter feasibility, more per-step fuel. |
| `grad_outer_iters`     | 25      | 0   | More projected-gradient outer iterations → higher profit per step, more per-step fuel. 0 skips the gradient stage. |
| `grad_ls_iters`        | 6       | 1   | More backtracking line-search steps per gradient iteration → better step sizes, more per-step fuel. |
| `bisect_iters`         | 30      | 1   | More feasibility-scaling bisection iterations in the fallback → finer feasible scaling, more per-step fuel. |
| `coord_polish_passes`  | 1       | 0   | More PTDF-aware coordinate-polish passes over binding lines → extra profit, more per-step fuel. 0 disables the polish. |
| `lookahead_horizon`    | 24      | 1   | Longer day-ahead window for the quantile-threshold target (steps). |
| `fuel_budget`          | 0       | —   | Absolute cap (in fuel units) on how much fuel the rollout may spend before falling back to zeros. `0` = spend all available fuel (minus the reserve). Set a positive value to trade quality for lower fuel cost; it is always clamped to the available fuel so it cannot cause an out-of-fuel exit. |

**Tuning guidance.** There is a direct **quality-vs-fuel** trade-off: every "finer / more
iterations" knob raises solution quality but costs more fuel per step, and on
fuel-limited tracks (e.g. `capstone`) spending more fuel per step means fewer steps get
optimized before the reserve is hit. Practical recipes:

- *Maximize quality when fuel is plentiful:* leave `fuel_budget = 0` (use all available)
  and increase `policy_action_levels`, `grad_outer_iters`, and `coord_polish_passes`.
- *Reduce fuel cost on easy tracks (baseline/congested) without losing the max score:*
  set `fuel_budget` to a modest value, or lower the iteration counts (`grad_outer_iters`,
  `proj_max_iters`, `bisect_iters`) and `policy_action_levels`.
- *Improve a fuel-limited track:* the binding constraint is the runtime fuel cap, so
  raising the cap helps most; otherwise lower per-step cost (smaller
  `policy_action_levels` / `grad_outer_iters`) so more steps get optimized within budget.





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
