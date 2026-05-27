# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** ea_frontier_router
* **Copyright:** 2026 Carlos
* **Identity of Submitter:** Carlos
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

This implementation is a track-routed engineering variant based on public TIG code submissions for the `energy_arbitrage` challenge:

* `titan_v2`, used as the robust path for `baseline`, `congested`, and `capstone`.
* `near_arb_v1`, used for `multiday` and `dense`, where local validation showed stronger quality.

## Additional Notes

Use `null` hyperparameters for the recommended default behavior.

The main goal is robustness across the five observed instance families. Local validation found that `near_arb_v1` is strong on `multiday` and `dense`, but fails on `capstone`. In the 60-nonce comparison below, `near_arb_v1` produced 60 invalid capstone solutions, while this router completed capstone with 60 valid solutions by selecting the titan path there.

After comparing against available public c008 binaries, `iycbtjt` was the strongest capstone comparator. The final router therefore keeps the same track routing, enables the `titan_v2` KKT capstone path by default in `sub_t53`, and uses validated per-track derating defaults for `congested`, `multiday`, and `dense`. The capstone KKT change improved capstone from 9,774,197 to 9,829,027 on the same 60-nonce validation window.

Comparison was run in the official TIG `energy_arbitrage/dev:0.0.5` Docker image with `--fuel 100000000000`, `--workers 8`, and 60 nonces per track:

| Track | titan_v2 avg | near_arb_v1 avg | ea_frontier_router avg | Router invalid |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1,993,325 | 1,915,125 | 1,993,325 | 0 |
| congested | 6,062,764 | 5,910,195 | 6,108,826 | 0 |
| multiday | 8,574,702 | 9,040,047 | 9,125,520 | 0 |
| dense | 9,444,523 | 9,784,506 | 9,811,135 | 0 |
| capstone | 9,774,197 | 60 invalid | 9,829,027 | 0 |

Public-binary comparison on the same 60-nonce harness:

| Track | iycbtjt avg | ea_frontier_router avg | Delta |
| --- | ---: | ---: | ---: |
| baseline | 1,938,496 | 1,993,325 | +2.83% |
| congested | 5,913,715 | 6,108,826 | +3.30% |
| multiday | 8,769,680 | 9,125,520 | +4.06% |
| dense | 9,630,626 | 9,811,135 | +1.87% |
| capstone | 9,824,876 | 9,829,027 | +0.04% |

The final package bundles only the public `titan_v2` modules used by the router (`sub_t49`, `sub_t50`, and `sub_t53`) plus `near_arb_v1`. The unused titan mid-track modules were removed from the submission payload after validation, reducing request size without changing behavior. The corrected capstone route uses the public `sub_t53` implementation with the KKT capstone path enabled by default.

An additional 200-nonce-per-track validation of the earlier routed implementation was run in the same official Docker image with `--fuel 100000000000` and `--workers 8` before the full `titan_v2` capstone source was bundled:

| Track | Valid | Invalid | Avg quality |
| --- | ---: | ---: | ---: |
| baseline | 200 | 0 | 1,979,770 |
| congested | 200 | 0 | 6,177,154 |
| multiday | 200 | 0 | 9,132,206 |
| dense | 200 | 0 | 9,700,571 |
| capstone | 200 | 0 | 9,768,023 |

Total extended validation: 1000 solutions, 0 invalid.

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
