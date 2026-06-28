# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** dale_v1
* **Copyright:** 2026 Dale
* **Identity of Submitter:** Dale
* **Identity of Creator of Algorithmic Method:** Dale, building on the `prometheus_eb1` architecture (see References)
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

`dale_v1` is a tuned and extended fork of `prometheus_eb1`. The core architecture — a
per-battery dynamic-programming value function used as a receding-horizon control
target, combined with a joint PTDF-projected gradient step over all batteries — is
inherited from that algorithm; this submission adds the improvements described below
rather than introducing a new architecture.

### 1. Academic Papers
- Boyle, J. P. & Dykstra, R. L., *"A Method for Finding Projections onto the Intersection of Convex Sets in Hilbert Spaces"*, Lecture Notes in Statistics, vol. 37, pp. 28–47 (1986). — basis for the alternating projection onto the feasible flow polytope.
- Dykstra, R. L., *"An Algorithm for Restricted Least Squares Regression"*, Journal of the American Statistical Association, 78(384), pp. 837–842 (1983).
- Holt, C. C., *"Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages"*, ONR Memorandum 52, Carnegie Institute of Technology (1957). — basis for the level+trend residual forecast described below.

### 2. Code References
- `tig-algorithms/src/energy_arbitrage/prometheus_eb1/` — base architecture this submission forks and tunes.

### 3. Other
- **PTDF (Power Transfer Distribution Factors)** linearized DC network-flow model — standard transmission-constraint formulation in power-systems dispatch.
- **Receding-horizon Model Predictive Control (MPC)** with a per-battery dynamic-programming value function — standard stochastic-control framing for storage arbitrage.


## Additional Notes

### What changed vs. `prometheus_eb1`

`dale_v1` keeps the prometheus architecture (per-battery DP value function MPC + joint
PTDF-projected gradient ascent + PTDF-aware coordinate polish) and tightens it in four
main ways:

1. **Finer DP grids.** `dp_soc_levels` 33 → 49 and `dp_action_levels` 17 → 33, so the
   value function and the policy's query grid are resolved at comparable resolution
   instead of the value function being the coarser of the two.
2. **More coordinate-polish passes.** `coord_polish_passes` 1 → 4, so the PTDF-aware
   per-battery sweep has more chances to converge to the joint optimum on binding lines.
3. **More diverse gradient-ascent seeds.** The joint projected-gradient ascent runs from
   6 seeds instead of 3 (heuristic target, DP seed, zero baseline, plus three scaled
   variants) to better escape PTDF-induced local maxima, with a fuel-aware "light" mode
   that drops back to the original 3 seeds and halves the gradient iterations as the
   runtime fuel counter nears its floor.
4. **Holt exponential smoothing for the RT-vs-DA residual forecast**, replacing the
   static 0.65/0.35 median blend. The level+trend Holt state widens or narrows the
   day-ahead lookahead band per node based on recently observed real-time residuals,
   with a wider forecast clamp and bounded ring buffers feeding the percentile spike/dip
   bands.

Smaller changes layered on top: a feasible greedy-drain fallback at fuel exhaustion
(discharges stored energy at decent prices instead of abandoning it as zeros) with a
stronger horizon-based terminal drain; asymmetric (upward-skewed) jump weighting in the
DP to reflect the Pareto-tailed price spikes; single-counted transaction friction in the
policy-time thresholds (the DP's break-even margin double-counts the round-trip cost,
which is appropriate for the value function but overly conservative for the per-step
policy gate); and a central-difference SOC shadow-value gradient in place of the
one-sided difference.

### Runtime / fuel behavior

Identical model to `prometheus_eb1`: the solver is bounded by the runtime's **fuel**
counter (`__fuel_remaining`), not wall-clock time. It saves an all-zeros schedule before
any optimization, spends fuel on the per-step optimization until a reserve (~1/20 of
what was available — slightly larger than prometheus's 1/28 to cover the now-nonzero
drain tail) is reached, then switches to the feasible greedy-drain fallback for the
remaining steps so the rollout always completes with a valid, non-trivial solution
rather than zeroing out the tail.

### Test results

Verified with `test_algorithm dale_v1 's=<TRACK>' null --nonces 20` against all five
energy_arbitrage tracks. 20/20 nonces finished, 0 invalid solutions on every track:

| Track     | Nonces finished | Invalid | Avg quality |
|-----------|------------------|---------|--------------|
| BASELINE  | 20/20            | 0       | 207,188      |
| CONGESTED | 20/20            | 0       | 405,112      |
| MULTIDAY  | 20/20            | 0       | 2,543,895    |
| DENSE     | 20/20            | 0       | 1,309,609    |
| CAPSTONE  | 20/20            | 0       | 320,939      |

### Hyperparameters

All hyperparameters are optional and passed as a JSON object; any omitted field uses its
default. Values are clamped to safe minimums/ranges so the solver cannot be
misconfigured into a panic. Example:

```json
{"dp_soc_levels": 49, "coord_polish_passes": 4, "holt_alpha": 0.22}
```

| Hyperparameter         | Default | Min/Range  | Effect of increasing it |
|------------------------|---------|------------|--------------------------|
| `dp_soc_levels`        | 49      | 2          | Finer SOC grid for the DP value function → better long-horizon decisions, more setup fuel. |
| `dp_action_levels`     | 33      | 3          | Finer action grid while *building* the DP → smoother value function, more setup fuel. |
| `policy_action_levels` | 65      | 3          | Finer action grid when *querying* the DP at runtime → better per-step actions, more per-step fuel. |
| `proj_max_iters`       | 80      | 1          | More alternating-projection iterations onto the feasible flow polytope → tighter feasibility, more per-step fuel. |
| `grad_outer_iters`     | 25      | 0          | More projected-gradient outer iterations → higher profit per step, more per-step fuel. 0 skips the gradient stage. |
| `grad_ls_iters`        | 6       | 1          | More backtracking line-search steps per gradient iteration → better step sizes, more per-step fuel. |
| `bisect_iters`         | 30      | 1          | More feasibility-scaling bisection iterations in the fallback → finer feasible scaling, more per-step fuel. |
| `coord_polish_passes`  | 4       | 1          | More PTDF-aware coordinate-polish passes over binding lines → extra profit, more per-step fuel. |
| `lookahead_horizon`    | 24      | 1          | Longer day-ahead window for the quantile-threshold target (steps). |
| `fuel_budget`          | 0       | —          | Absolute cap (in fuel units) on how much fuel the rollout may spend before falling back to the drain policy. `0` = spend all available fuel (minus the reserve). |
| `holt_alpha`           | 0.22    | 0.02–0.8   | Holt level-smoothing weight for the RT-vs-DA residual forecast → higher values track recent residuals more closely (less smoothing). |
| `holt_beta`            | 0.12    | 0.02–0.8   | Holt trend-smoothing weight for the residual forecast → higher values let the trend term adapt faster. |

**Tuning guidance.** As with `prometheus_eb1`, there is a direct **quality-vs-fuel**
trade-off: every "finer / more iterations" knob raises solution quality but costs more
fuel per step, and on fuel-limited tracks (e.g. `CAPSTONE`) spending more fuel per step
means fewer steps get optimized before the reserve is hit.

- *Maximize quality when fuel is plentiful:* leave `fuel_budget = 0` and increase
  `policy_action_levels`, `grad_outer_iters`, and `coord_polish_passes`.
- *Reduce fuel cost on easy tracks (BASELINE/CONGESTED) without losing the max score:*
  set `fuel_budget` to a modest value, or lower the iteration counts.
- *Improve a fuel-limited track:* the binding constraint is the runtime fuel cap, so
  lowering per-step cost (smaller `policy_action_levels` / `grad_outer_iters`) lets more
  steps get optimized within budget before the greedy-drain fallback engages.


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
