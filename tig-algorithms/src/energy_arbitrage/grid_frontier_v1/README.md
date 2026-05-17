# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** grid_frontier_v1
* **Copyright:** 2026 Joan Vidal Llauradó
* **Identity of Submitter:** Joan Vidal Llauradó
* **Identity of Creator of Algorithmic Method:** Joan Vidal Llauradó
* **Unique Algorithm Identifier (UAI):** null

## Algorithm Summary

`grid_frontier_v1` is an adaptive controller for the TIG Energy Arbitrage challenge. It uses solver-visible state and challenge data: current real-time prices, battery states of charge, action bounds, day-ahead prices, exogenous injections, battery parameters, and network/PTDF data.

The controller combines shadow-price dispatch, expected real-time valuation, compact policy selection, and network-feasible schedule repair. The default configuration is intentionally conservative so benchmark runs stay within the official fuel budget across the Energy Arbitrage tracks. The solver saves a valid conservative schedule before rollout and refreshes saved schedules during evaluation so a valid output is available if execution stops early.

## Recommended Hyperparameters

Use the default configuration for benchmark submission. It is configured for validity and runtime safety with `null` hyperparameters.

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
