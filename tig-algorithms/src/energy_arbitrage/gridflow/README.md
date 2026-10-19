# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** gridflow
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

Battery energy-arbitrage solver that schedules charge/discharge across a fleet of
storage units to maximise revenue against a time-varying price signal under
power, capacity and round-trip-efficiency constraints. The entry point dispatches
each instance by battery count to a dedicated per-track engine, so the effort and
the solution structure match the instance scale. Small fleets are solved close to
optimality with a linear-programming / dynamic-programming dispatch; larger fleets
use congestion-aware pricing (a shadow price on shared grid capacity) followed by
rolling-horizon refinement that re-optimises overlapping windows. All tuning is
exposed as hyperparameters read in `from_map`; the defaults are the shipped
operating point.

## References and Acknowledgments

### Academic Papers
- R. Sioshansi, P. Denholm, T. Jenkin, J. Weiss, *"Estimating the value of electricity storage in PJM: Arbitrage and some welfare effects"*, Energy Economics, 2009.
- M. R. Bradbury, L. Pratson, D. Patiño-Echeverri, *"Economic viability of energy storage systems based on price arbitrage potential in real-time U.S. electricity markets"*, Applied Energy, 2014.
- D. P. Bertsekas, *"Dynamic Programming and Optimal Control"*, Athena Scientific (dynamic-programming dispatch).
- D. Bertsimas, J. N. Tsitsiklis, *"Introduction to Linear Optimization"*, Athena Scientific, 1997 (linear-programming formulation and shadow prices).

### Code References
- TIG baseline (energy_arbitrage).

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
