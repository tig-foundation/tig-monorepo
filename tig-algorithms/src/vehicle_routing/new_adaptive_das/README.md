# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** new_adaptive_das
* **Copyright:** 2026 Brent Beane 
* **Identity of Submitter:** Brent Beane
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

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

- **Purpose:** Briefly, adaptive DAS adjusts decisions as new information (forecasts, realizations) arrives, while a static/deterministic DAS produces a single plan from fixed inputs. Benchmarkers should pick the evaluation style that matches the goal: simulation of updates (adaptive) versus single-run comparison (static).

- **Evaluation protocol:** For adaptive methods, benchmarks must emulate time/forecast updates (rolling-horizon simulations). Decide how often updates occur, what data is revealed at each step, and whether solutions can be repaired or fully recomputed.

- **Reproducibility:** Record seeds, environment, and all non-deterministic sources. Adaptive runs often include stochastic forecasts or randomized tie-breaking — report distributions (mean, median, and variance) over multiple seeds or instances.

- **Runtime & budgets:** Adaptive approaches may require repeated solves (one per update window). Report per-update and total runtime, and enforce consistent time budgets across compared methods.

- **Feasibility & constraints:** If adaptation uses warm-starts or partial repairs, ensure feasibility 
checks are consistent with static baselines. Log any constraint violations and how they were corrected.

- **Metrics to collect:** Beyond final objective value, collect intermediate metrics (regret vs. hindsight, per-step cost, number of repairs), solution stability (how much routes/assignments change), and rollout performance under different forecast realizations.

- **Instance generation:** When using synthetic or perturbed forecasts, document generation process and parameter settings so others can reproduce and extend the benchmark.

- **Practical tips:** Run multiple seeds, include deterministic baselines, fix random sources where appropriate, and publish raw logs so results can be re-aggregated. Consider providing a `help()` utility that prints recommended evaluation parameters and example commands to run benchmarks.

**Benchmark Parameters**

Below are recommended configuration parameters for running reproducible benchmarks and their typical defaults. Adjust these to match your experimental design and note any deviations when publishing results.

- `update_frequency`: How often the solver receives new data (seconds or simulation steps). Default: `60` (seconds) or `1` (step).
- `forecast_horizon`: Number of future steps considered when planning. Default: `4` (steps).
- `forecast_update_times`: Explicit times or intervals when forecasts are revealed (e.g., `[0, 15, 30, 60]`). Default: rolling window matching `update_frequency`.
- `num_instances`: Number of problem instances to evaluate. Default: `100`.
- `num_seeds`: Number of seeds per instance for randomness. Default: `10`.
- `seed_list`: Optional explicit seeds to use for reproducibility. Default: random but logged.
- `time_budget_per_update`: Max solver time per update (seconds). Default: `10`.
- `total_time_budget`: Cumulative time budget for entire run. Default: `time_budget_per_update * number_of_updates`.
- `solver_timeout`: Solver-level timeout to enforce (seconds). Default: same as `time_budget_per_update`.
- `solver_threads`: Number of threads/CPUs for solver. Default: `1` (single-threaded) unless comparing parallel performance.
- `use_warm_start`: Whether to warm-start from previous solution (`true`/`false`). Default: `true` for adaptive methods.
- `repair_mode`: How updates are applied: `none` (recompute fully), `partial` (repair affected routes), or `full` (full re-opt). Default: `partial`.
- `recompute_on_update`: If `true`, always run optimizer on each update; if `false`, apply greedy repairs. Default: `true`.
- `tie_breaking`: `deterministic` or `random`. Default: `deterministic` for reproducibility.
- `sample_size`: Number of forecast samples / Monte Carlo rollouts when evaluating stochastic policies. Default: `50`.
- `forecast_noise_level`: Parameter controlling synthetic forecast perturbation (e.g., std dev). Default: document dataset-specific value.
- `instance_perturbation`: Type and magnitude of instance perturbations (drop/late arrivals/etc.). Default: none unless studying robustness.
- `metrics_to_collect`: List of metrics (e.g., `final_cost, regret_vs_hindsight, per_step_cost, runtime_per_update, repairs_count, route_changes`). Default: include all.
- `report_stats`: Which aggregations to report (`mean, median, std, min, max`). Default: `mean, median, std`.
- `max_memory_mb`: Memory cap for solver/process monitoring. Default: unset (system default).
