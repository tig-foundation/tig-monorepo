# IC Energy v1

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** ic_energy_v1
* **Copyright:** 2026 Illuminati Congo / Jahn Hooks
* **Identity of Submitter:** Illuminati Congo / Jahn Hooks
* **Identity of Creator of Algorithmic Method:** Illuminati Congo / Jahn Hooks
* **Unique Algorithm Identifier (UAI):** null

## Overview

IC Energy v1 is an `energy_arbitrage` solver that keeps a robust general dispatch layer and adds congestion-aware handling for medium battery-count scenarios. The implementation routes congested-grid instances through a dedicated congestion coordination path while retaining a stable baseline policy for smaller and larger scenarios.

The submitted variant is tuned for the `CONGESTED` scenario family and uses a PTDF-aware congestion-safety signal to adjust local dispatch decisions under line-loading pressure. The intended effect is to improve objective quality on congested network instances without introducing feasibility failures.

## Suggested Hyperparameters

The validation run used the following congestion-safety settings:

```json
{
  "use_congestion_safety": true,
  "congestion_safety_threshold": 0.50,
  "congestion_safety_strength": 0.2
}
```

The solver also accepts the standard hyperparameters exposed by the base implementation. If no hyperparameters are provided, the algorithm uses its built-in defaults.

## Local Validation Summary

Official TIG development container validation was run with:

* Image: `ghcr.io/tig-foundation/tig-monorepo/energy_arbitrage/dev:0.0.7`
* Scenario: `s=congested`
* Nonces: 100 paired nonces
* Workers: 4
* Candidate: `c008_ic001` / IC Energy v1
* Baseline comparison: `c008_a011` / titan_v2

Observed paired validation result:

* Candidate average quality: `559089`
* Baseline average quality: `546069`
* Average quality delta: `+13020` (`+2.384%`)
* Candidate invalid solutions: `0 / 100`
* Baseline invalid solutions: `0 / 100`
* Candidate elapsed time: `153.54s`
* Baseline elapsed time: `141.04s`
* Runtime ratio: `1.089x`

This validation summary is provided for reproducibility context only. TIG benchmark performance depends on live benchmarker adoption, scenario mix, hardware, and protocol conditions.

## References and Acknowledgments

### Code References

* TIG Foundation, `tig-monorepo`, `energy_arbitrage` challenge and public algorithm implementations: https://github.com/tig-foundation/tig-monorepo
* This implementation builds on the public TIG `titan_v2` energy-arbitrage structure and introduces congestion-specific routing and PTDF-aware congestion-safety behavior for the congested scenario family.

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
