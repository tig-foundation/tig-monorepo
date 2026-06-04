# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** water_threshold_v1
* **Copyright:** 2026 hoon1983
* **Identity of Submitter:** hoon1983
* **Identity of Creator of Algorithmic Method:** hoon1983
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

A receding-horizon water-value threshold policy for the battery-arbitrage challenge.
Per node, charge/discharge price thresholds are taken from quantiles of the fully-known
day-ahead price horizon. At each step the policy acts on the observed real-time price
(charging below the charge threshold, discharging above the discharge threshold, idling
between), so real-time price spikes and crashes are exploited directly. An optional
state-of-charge feedback term shifts the thresholds to self-regulate the daily cycle.
Joint actions are projected onto the network-feasible set (line-flow softening with a
global-bisection fallback), guaranteeing no line-limit violation.

Tunable hyperparameters (with defaults): `charge_q` (0.25), `discharge_q` (0.75),
`soc_bias` (0.0).

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
