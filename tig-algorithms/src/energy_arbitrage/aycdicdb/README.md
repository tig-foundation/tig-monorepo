# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** aycdicdb
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

Inspired by the AI evolved solvers `titan_v2` and `near_arb_v1` using Google Gemini Flash 2.5

### 3. Other

Use 500b fuel for full functionality

`aycdicdb` uses track-specialised dispatch selected by `num_batteries`:
- `<= 15`: baseline
- `<= 30`: congested
- `<= 50`: multiday
- `<= 80`: dense
- `<= 150`: capstone

The implementation is split across the five track files in this folder rather than a single shared `helpers.rs`. The exposed TIG hyperparameters are track-dependent rather than a single fixed set. Common exported knobs include `soc_levels`, `action_grid`, `asca_iters`, `ternary_iters`, `convergence_tol`, `anticipate_lmp`, `lmp_threshold`, `lmp_premium_scale`, `jump_premium`, `prune_ratio`, `deflator_iters`, `flow_margin`, and `network_derating`. Larger-track variants additionally expose congestion-oriented controls such as `ldd_iters`, `ldd_step_size`, and dense `screened_multistart_refine_cap`.

**Implementation details:**

- **Network model:** Each track constructs sparse PTDF sensitivity tables by applying a unit battery injection and measuring the induced line-flow change. These tables are reused to score actions, build feasible windows, and repair overloads.
- **Value function:** The core signal is a single-battery backward dynamic program over state of charge, using day-ahead prices together with efficiency losses, transaction costs, degradation penalties, and optional congestion-aware premium adjustments.
- **Congestion anticipation:** When enabled, exogenous network flows are converted into expected nodal premiums before the DP sweep so the value function can prefer actions that are less likely to worsen stressed lines.
- **Local refinement:** Candidate dispatches are refined with PTDF-constrained coordinate updates. For each battery, the solver computes a feasible action interval under current flows, then evaluates zero, interval endpoints, and continuous charge/discharge candidates with ternary search.
- **Feasibility repair:** If a candidate overloads any line, a deflator-style repair stage reduces or removes the most harmful actions and falls back to a final global scale-down if needed to recover feasibility.
- **Track-specific extensions:** Smaller tracks stay close to the DP + local-refinement + repair template. Larger tracks can additionally enable LDD, KKT/dual-ascent style congestion pricing, ADMM repair, and soft-constrained LP dispatch. The dense track also includes a screened multistart seeding pass, and the multiday track includes a joint pair-refinement step.

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