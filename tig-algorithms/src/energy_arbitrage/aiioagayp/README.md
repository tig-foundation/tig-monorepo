# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** aiioagayp
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

Inspired by "titan", improved qualities over iycbtjt. Use fuel_budget 500b to avoid invalid solutions.

### 3. Other

Track-specialised dispatch on `num_batteries` (BASELINE ≤ 15 / CONGESTED ≤ 30 / MULTIDAY ≤ 50 / DENSE ≤ 80 / CAPSTONE ≤ 150), shared core in `helpers.rs`. All hyperparameters (13) overridable via the TIG `Hyperparameters` JSON map.

**Implementation details:**

- **Network model:** At cache construction, a unit impulse is injected per battery to extract sparse PTDF sensitivity vectors (`ptdf_sparse`, `b_to_lines`) and infer each battery's grid injection node.
- **Value function:** Single-battery Bellman DP backward in time over a 201-level SOC grid with configurable action discretisation (`action_grid`), accounting for round-trip efficiency, degradation cost, and transaction fee. Network derating (`network_derating`) scales power limits before the DP sweep.
- **LMP anticipation oracle:** Optional pre-computation of expected nodal price premiums from exogenous injection stress on each line, modulated by a congestion threshold and scale factor, added to the day-ahead price signal used in the DP.
- **Coordinate ascent (ASCA):** `run_asca_candidates` performs sequential coordinate ascent over a selected battery subset. Each step computes a feasible action window from PTDF half-space constraints under current flows, then evaluates endpoints, zero, and ternary search over charge/discharge intervals to handle the kinked profit function. Battery ordering is refreshed each sweep by `dynamic_order`, which scores upside divided by a stress-weighted footprint.
- **Candidate pruning:** `run_asca` optionally prunes the lowest-capacity batteries using a network-and-potential-aware ranking before passing candidates to `run_asca_candidates`.
- **Multi-start construction (BASELINE / CONGESTED / MULTIDAY):** `build_scaled_greedy_seed` constructs an alternative starting point by independently solving each battery's best unconstrained action, then scaling all actions toward zero to restore feasibility; a sequential greedy variant (`build_sequential_greedy_seed`) picks batteries one-by-one by merit/stress ratio. The better feasible seed is selected and refined before comparison with the zero-start solution.
- **Deflator:** `run_deflator` repairs line-flow violations by identifying culprit batteries (positive contribution to overloaded line, sorted by ascending ROI) and proportionally shrinking their actions to clear each overflow. A homothetic safety scale-down is applied if the iterative repair fails to achieve feasibility.
- **Post-deflator intensification:** After deflation, `run_deflator` computes a line-relief signal (`compute_line_relief_signal`) comparing utilisation before and after repair, then runs a bounded ASCA pass (`run_asca_candidates`) over changed batteries and stressed-line neighbours, followed by a `run_post_deflator_polish` stage that selects a wider candidate neighbourhood for a final improvement sweep.
- **Pairwise exchange polish (BASELINE / CONGESTED):** `run_pair_exchange_polish` evaluates asymmetric paired moves between batteries sharing a PTDF-overlapping line: one battery (anchor) is moved to each probe point, and the best response of its partner (responder) is computed under the resulting flows. The best improving pair is committed.
- **Destroy-repair polish (BASELINE / CONGESTED):** `run_small_destroy_repair_polish` identifies high-stress batteries ranked by their congestion contribution relative to retained value, zeros out 1–4 of them, re-runs a bounded ASCA and deflator pass, and keeps the result if it improves total feasible value.

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