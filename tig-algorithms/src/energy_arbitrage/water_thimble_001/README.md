# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** water_thimble_001
* **Copyright:** 2026 Ozymandias (water_dp.rs, mod.rs); 2026 NVX (gridflow-derived files: hyperparameters.rs, track_t1.rs, track_t4.rs, track_t5.rs)
* **Identity of Submitter:** Ozymandias
* **Identity of Creator of Algorithmic Method:** Ozymandias for the water-value dynamic program (water_dp.rs); NVX for the gridflow engines (track_t1.rs, track_t4.rs, track_t5.rs)
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

The entry point dispatches each instance by battery count.

* **16 to 50 batteries (the congested and multiday tracks): `water_dp.rs`.**
  - Offline, once per instance, each battery gets a backward stochastic dynamic program over a
    discretised state of charge. The program uses the day-ahead price path at the battery's node,
    and takes the expectation over the real-time price distribution: Gaussian spread, Pareto
    jumps and a two-branch congestion premium. The result is a water-value table: the expected
    profit still achievable from each time step and state of charge.
  - The planning power and the probability of being served are measured by replaying the online
    allocator at plan time, so stored energy is valued at what the network will actually let the
    battery deliver.
  - Online, at each step, every battery scores candidate actions by realised step profit plus
    water value. A network-feasible fleet allocator then reconciles those actions with the line
    limits, choosing among four candidates:
    - value-ordered greedy, ranked by value per unit of scarce line headroom;
    - uniform scaling with a top-up;
    - counter-flow pairs, in which a charge and a discharge cancel on a binding line;
    - a relief allocation.
* **Every other fleet size (the baseline, dense and capstone tracks):** gridflow's per-track
  engines by NVX, unchanged apart from one removal (see Additional Notes).

## References and Acknowledgments

### 1. Academic Papers
- M. V. F. Pereira, L. M. V. G. Pinto, *"Multi-stage stochastic optimization applied to energy planning"*, Mathematical Programming 52, 359–375 (1991). Water values for stochastic storage scheduling.
- D. P. Bertsekas, *"Dynamic Programming and Optimal Control"*, Athena Scientific.
- Cited by gridflow for its engines:
  - R. Sioshansi, P. Denholm, T. Jenkin, J. Weiss, *"Estimating the value of electricity storage in PJM: Arbitrage and some welfare effects"*, Energy Economics, 2009.
  - M. R. Bradbury, L. Pratson, D. Patiño-Echeverri, *"Economic viability of energy storage systems based on price arbitrage potential in real-time U.S. electricity markets"*, Applied Energy, 2014.
  - D. Bertsimas, J. N. Tsitsiklis, *"Introduction to Linear Optimization"*, Athena Scientific, 1997.

### 2. Code References
- NVX – gridflow (energy_arbitrage, c008_a052): https://github.com/tig-foundation/tig-monorepo/tree/main/tig-algorithms/src/energy_arbitrage/gridflow. Used under the TIG Innovator Outbound Game License. hyperparameters.rs, track_t1.rs, track_t4.rs and track_t5.rs, and the hyperparameter merge in mod.rs, are derived from it.
- TIG baseline (energy_arbitrage), as referenced by gridflow.

## Additional Notes

- The gridflow engines are used as published, with one change: their lookup of the challenge's
  private price seed is removed. Since TIG's seed obfuscation on 2026-09-18 that lookup has
  returned nothing in any fresh build, so every call site takes the branch it already took. On 72
  test nonces of the baseline, dense and capstone tracks, the change left gridflow's results
  identical, nonce for nonce.
- Each solver reads its own hyperparameters and ignores the other's:
  - water_dp: `dp_budget`, `flow_safety`, `jump_optimism`, `cong_optimism`, `power_share`;
  - gridflow engines: as documented in gridflow's README, with the same defaults.

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
