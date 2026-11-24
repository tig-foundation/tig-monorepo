# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** arb_omega_c2
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

`arb_omega_c2` dispatches by `num_batteries` and enforces every schedule step through `strict_schedule` (exact SOC feasibility replay and convex line-flow scaling within `1e-8` margin of every transmission limit):
- `10` or `60` batteries (`baseline` / `dense`): `refined` congestion-priced stochastic DP + dispersion quadrature + step/terminal LPs with degradation secants.
- `20` batteries (`congested`): `linepulse_base` congestion-priced stochastic DP with aggregate fleet correction and exact step LPs.
- `40` batteries (`multiday`): `classic` congestion-priced stochastic DP with calibrated quadrature and stride/secant LPs.
- Other sizes (`capstone` / fallback): `donor_harimox` (`voltpath`/`wasserwert` adaptive lookahead and local polish).

### Code References
- TIG `c008` public submissions (`linepulse` / `c008_a058` by Iain Jackson, `harimox` / `c008_a057` by tilk, `voltpath` / `c008_a055` by NVX, `wasserwert` / `c008_a054` by Kernelsmith) under the TIG Open Data License and TIG Innovator Outbound Game License.

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
