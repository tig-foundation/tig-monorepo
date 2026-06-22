# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** prom_inspired
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Bellman, R., *"Dynamic Programming"*, Princeton University Press, 1957
- Bertsekas, D.P., *"Dynamic Programming and Optimal Control"*, Athena Scientific, 2017
- Wood, A.J., Wollenberg, B.F. and Sheblé, G.B., *"Power Generation, Operation, and Control"* (3rd ed.), Wiley, 2013
- Boyd, S. and Vandenberghe, L., *"Convex Optimization"*, Cambridge University Press, 2004
- Mohsenian-Rad, H., *"Optimal Bidding, Scheduling, and Deployment of Battery Systems in California Day-Ahead Energy Market"*, IEEE Transactions on Power Systems, 2016

### 2. Code References

Inspired by `prometheus_eb1`, then extended into a track-specialised folder that mixes a stronger PTDF / ASCA / LP baseline path with fuel-budgeted projected-gradient solvers for the larger tracks. Further refined using a combination of idea injection into `tacit_knowledge.md` and GPT 5.5.

### 3. Other

Top-level dispatch now happens in `mod.rs` on `num_batteries` (BASELINE ≤ 15 / CONGESTED ≤ 30 / MULTIDAY ≤ 50 / DENSE ≤ 80 / CAPSTONE ≤ 150), routing to five top-level track files. Hyperparameters are track-dependent and overridable via the TIG `Hyperparameters` JSON map rather than one single shared knob set with identical behaviour across all scenarios.

**Implementation details:**

- **Routing / structure:** `mod.rs` is the folder entry point and routes to `track_baseline.rs`, `track_congested.rs`, `track_multiday.rs`, `track_dense.rs`, and `track_capstone.rs`, allowing each scenario to evolve independently at the top level.
- **Two solver families:** The folder now contains two distinct families. `track_baseline.rs` keeps a Prometheus-style PTDF / ASCA / LP / column-generation engine, while `track_congested.rs`, `track_multiday.rs`, `track_dense.rs`, and `track_capstone.rs` are standalone projected-search solvers built around DP seeds, PTDF-aware projection, and fuel budgeting.
- **Network model:** The baseline path builds a sparse cache by injecting one unit per battery and recording line impacts (`ptdf_sparse`, `b_to_lines`). The other tracks build reusable PTDF sensitivity matrices and use them to project joint actions back into the feasible network polytope.
- **Value function:** All tracks solve backward single-battery DPs over SOC with efficiency, transaction fees, and degradation included. Baseline uses a cached Bellman table inside its shared engine; congested / multiday / dense use near/far horizon DP tables with cubic Hermite interpolation; capstone additionally precomputes a per-battery degradation factor for cheaper repeated value queries.
- **Baseline-specific upgrades:** The baseline solver supports optional stochastic DP scenarios (`use_sdp`), LP dispatch warm starts (`use_lp`, `lp_refine_sweeps`), and column generation (`use_cg`, `cg_iters`) on top of ASCA refinement and optional exogenous-congestion LMP anticipation.
- **Target construction for larger tracks:** Congested / dense / capstone first build heuristic targets from day-ahead quantiles, RT spike/dip bands, residual-shifted prices, and terminal SOC pressure. These targets are then compared or blended with DP-preferred actions before joint refinement.
- **Projected-gradient refinement:** Congested / multiday / dense / capstone all perform joint PTDF-aware projected-gradient ascent over one or more seeds, then run a one-battery-at-a-time coordinate polish step over the currently feasible action interval.
- **Feasibility repair:** The projected-search tracks avoid expensive brute-force repair. Congested uses projection plus a Gauss-Seidel style `restore_feasibility` sweep, while dense / capstone rely on projection plus fallback scaling; all tracks keep zero actions as the final guaranteed-feasible fallback.
- **Fuel-aware execution:** The non-baseline tracks budget directly against `__fuel_remaining`, reserving a safety floor so they can degrade gracefully to direct actions or zero actions instead of risking an out-of-fuel failure near the end of rollout.
- **Multiday-specific regime gating:** `track_multiday.rs` adds congestion-shadow adjusted effective prices plus explicit policy modes (`Zero`, `DirectDp`, `DirectTarget`, `Full`). Easy states can skip expensive joint search, while congested or terminally urgent states still receive the full optimization path.
- **Congested-specific refinement:** `track_congested.rs` adds future-only top-2 DP seed diversification and a momentum-adapted projected-gradient step so it can search more aggressively without paying for a large backtracking loop every iteration.
- **Track-dependent defaults:** Default grids, polish counts, fuel gates, and price-shaping logic differ materially by track. In particular, baseline keeps LP / CG / SDP toggles active, while the larger-track solvers lean on DP seeding, PTDF projection, and fuel-aware projected search instead.

## Hyperparameter Guide

The current defaults are intended to be the recommended baseline for this solver family. Most knobs are already track-tuned, so overrides should mainly be used for focused testing on one track or for an explicit quality/runtime trade-off.

### Baseline (`s=baseline`, up to 15 batteries)

- `anticipate_lmp` with `lmp_premium_scale=0.75` is now part of the tuned default baseline path.
- `use_lp`, `use_cg`, `use_sdp`: enable the stronger warm-start / search paths. These are generally worth leaving at their defaults.
- `lp_refine_sweeps`, `cg_iters`, `asca_iters`: increase local refinement depth. Higher values can improve dispatch quality but add runtime.
- `soc_levels`, `action_grid`: control DP/action resolution. Higher resolution can help fine decisions but is fuel-expensive.
- `network_derating`, `flow_margin`: make feasibility more conservative when line limits are tight.

### Congested (`s=congested`, up to 30 batteries)

- Tuned defaults now use `grad_outer_iters=120` and `lookahead_horizon=16`.
- `grad_outer_iters`: the main quality/runtime knob for joint projected-gradient search.
- `coord_polish_passes`: final one-battery-at-a-time polish. It can improve feasibility/quality, but it is runtime-heavy.
- `lookahead_horizon`: controls day-ahead context used when constructing charge/discharge targets. Shortening it helped more than increasing projection depth in recent tests.
- `fuel_budget`: optional fuel cap. `0` means use the internal fuel-aware default.

### Multiday (`s=multiday`, up to 50 batteries)

- Tuned defaults now use `grad_outer_iters=60`.
- `grad_outer_iters`: primary search-depth knob once the policy chooses the full optimization path.
- `coord_polish_passes`: local polish after projected search; useful when full refinement is active.
- `lookahead_horizon`: affects regime gating and price-context decisions across the longer horizon, but recent tests showed weaker gains than increasing outer iterations.
- `fuel_budget`: caps work for long rollouts; `0` lets the solver manage fuel automatically.

### Dense (`s=dense`, up to 80 batteries)

- `grad_outer_iters`: most impactful dense-track knob from recent tests. The default is now `40`, which captured most of the quality gain without the full runtime cost of `60`.
- `coord_polish_passes`: tested at `2` and `3` with very small quality movement and noticeable runtime cost; default remains preferred.
- `lookahead_horizon`: tested at `16` and `32` with low impact; default remains preferred.
- `dp_soc_levels`, `dp_action_levels`, `policy_action_levels`: expensive resolution knobs. Leave defaults unless doing a focused dense-track sweep.

### Capstone (`s=capstone`, up to 150 batteries)

- Current capstone defaults are still the conservative starting point.
- `grad_outer_iters`: likely the first knob to test, since capstone uses the same general projected-search family as dense / multiday.
- `proj_max_iters`, `grad_ls_iters`, `bisect_iters`: feasibility/projection precision knobs. Higher values can improve feasibility handling but cost fuel.
- `coord_polish_passes`: adds local refinement after joint search; tune cautiously on large instances.
- `fuel_budget`: important for preventing over-spend on the largest rollouts. `0` uses the solver's internal safety budget.

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