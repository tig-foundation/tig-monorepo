# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** near_arb_v1
* **Copyright:** 2026 CodeAlchemist
* **Identity of Submitter:** CodeAlchemist
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


### 3. Other
Track-specialised dispatch on `num_batteries` (BASELINE/CONGESTED/MULTIDAY/DENSE/CAPSTONE), shared core in `helpers.rs`: PTDF impulse extraction, network-derated single-battery Bellman DP on a 201-level SOC grid, directional LMP oracle, Analytical Sequential Coordinate Ascent (kink-aware ternary search per charge/discharge segment) under PTDF half-spaces, surgical deflator that ranks offending batteries by ROI before falling back to a homothetic safety scale-down. All hyperparameters (13) overridable via the TIG `Hyperparameters` JSON map.

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
