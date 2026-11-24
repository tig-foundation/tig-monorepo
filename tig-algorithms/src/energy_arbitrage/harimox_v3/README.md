# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** harimox_v3
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP | Engine |
|---|---|---|---|
| s=baseline | 5T | `null` | value tables (linepulse c008_a058 base) |
| s=congested | 5T | `null` | value tables |
| s=multiday | 5T | `null` | value tables |
| s=dense | 5T | `null` | value tables |
| s=capstone | 5T | `{"target_quality": T}` | harimox_v2 capstone engine, unchanged |

Baseline, congested, multiday and dense take no hyperparameters; any keys sent there are ignored.

`target_quality` (capstone only): T = the lowest value of `qualifier_qualities_by_track["s=capstone"]` in
`/get-challenges` divided by 1e6 (3806936 on chain is 3.806936). Default 3.85, as in harimox_v2. Stop margin 1.0
below T. A missing key uses the default; 0 turns the target off and gives the harimox solutions. Update T when the
line moves by more than 0.1. A stale value costs qualifiers both ways: above the line it gives up nonces that would
qualify, below it spends full effort on nonces that would qualify anyway. The other harimox_v2 capstone keys still
apply on capstone.

## Method

Instances are routed by num_batteries: up to 80 batteries (baseline, congested, multiday, dense) run the value-table
engine, 81 to 150 (capstone) run the harimox_v2 capstone engine with the same solutions as harimox_v2.

The value-table engine starts from linepulse: per-battery stochastic DP tables, re-priced for network congestion,
with an exact step LP over every line limit. Changes from linepulse: the tables model the whole public real-time
price process, including the congestion premium at nodes whose lines run near their limits; the table update
evaluates every state-of-charge breakpoint a battery can reach in one step instead of a fixed set of power levels;
the step LP uses finer state-of-charge breakpoints on baseline and dense. Only causal information is used
(day-ahead prices, the public price model parameters, the current state), and runs are deterministic.

## Results

Runs: `tig-runtime` and `tig-verifier` built with the c008 feature, the codes run on the same nonces in alternating
order on the same cores, 5T fuel, null HP (capstone: harimox_v2 at its default, peakshift at
`{"dp_action_levels":8,"dp_soc_levels":41}`). Quality is harimox_v3 minus the other code (mean over nonces), time is
the harimox_v3 / other code runtime CPU ratio. All solutions valid.

| | baseline | congested | multiday | dense | capstone |
|---|---|---|---|---|---|
| nonces | 960 | 576 | 192 | 960 | 768 |
| quality vs linepulse | +0.0070 | +0.0037 | +0.0047 | +0.0068 | +0.0126 |
| time vs linepulse | 0.95 | 0.84 | 0.91 | 1.07 | 0.95 |
| quality vs harimox | +0.0059 | - | -0.0020 | - | - |
| time vs harimox | 1.04 | - | 0.34 | - | - |
| quality vs peakshift | - | - | - | - | +0.0038 |
| time vs peakshift | - | - | - | - | 0.47 |

Quality is above linepulse on every track; dense costs 7% more time than linepulse for its quality gain, and
multiday quality is slightly below harimox (within noise) at about a third of its time.

## References and Acknowledgments

### Code References
- linepulse (c008_a058), Iain Jackson: base of the value-table engine (stochastic DP tables, congestion re-pricing,
  fleet correction, step and endgame LPs, simplex codes). Copyright remains with Iain Jackson.
- gridflow and peakshift (c008_a052, c008_a053), NVX: the technique family linepulse credits (Gauss-Hermite DP over
  the price model, deliverability-aware storage value, PTDF congestion prices, LP dispatch).
- harimox_v2 (c008_a059), tilk: capstone engine, unchanged; it builds on voltpath (c008_a055), NVX, whose
  copyright remains with NVX.
- energy_v1 capstone baseline policy (tig-challenges energy_arbitrage/baselines/energy_v1/capstone.rs), TIG
  Foundation: used by the capstone engine to estimate quality.
- Price and congestion model: tig-challenges energy_arbitrage/market.rs and network.rs.

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
