# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** codex_ea_hybrid
* **Copyright:** 2026 Carlos
* **Identity of Submitter:** Carlos
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments  

This implementation is a track-routed engineering variant based on public TIG code submissions for the `energy_arbitrage` challenge:

* `titan_v2`, used for the smaller `baseline` and `congested` tracks.
* `near_arb_v1`, used for the larger `multiday`, `dense`, and `capstone` tracks.

Local testing showed that this routing improves average quality over using `titan` uniformly and over using either component uniformly across all tracks.


## Additional Notes

Use `null` hyperparameters for the recommended default behavior.

The policy routes by `challenge.num_batteries`, which maps to TIG's current c008 tracks:

* `10` and `20` batteries: `titan_v2`
* `40`, `60`, and `100` batteries: `near_arb_v1`


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
