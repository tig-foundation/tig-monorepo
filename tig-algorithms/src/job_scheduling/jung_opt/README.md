# TIG Code Submission

## Submission Details

* **Challenge Name:** job_scheduling
* **Algorithm Name:** jung_opt
* **Copyright:** 2026 ChervovNikita
* **Identity of Submitter:** ChervovNikita
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track | Fuel Budget | Nonces/Bundle |
|---|---|---|
| n=50,s=job_shop | 5T | 40 |
| n=50,s=fjsp_medium | 5T | 40 |
| n=50,s=fjsp_high | 5T | 40 |
| n=50,s=hybrid_flow_shop | 5T | 40 |
| n=50,s=flow_shop | 5T | 40 |

## Measured Quality

Bundle median over 40 nonces, seed `rand_hash`.

| Track | Bundle Median | Max Fuel | Max Runtime |
|---|---:|---:|---:|
| n=50,s=fjsp_medium | 130,748 | 49.9% | 58 min |
| n=50,s=job_shop | 104,635 | 52.3% | 72 min |
| n=50,s=fjsp_high | 67,620 | 24.9% | 31 min |
| n=50,s=hybrid_flow_shop | 65,192 | 1.9% | 2 min |
| n=50,s=flow_shop | 18,108 | 0.9% | 2 min |

Runtime figures are upper bounds measured under concurrent load. All tracks are inside the fuel cap
and the 2 h per-nonce limit.

Determinism verified on every track: re-running the same nonces reproduces per-nonce identical results.

## References and Acknowledgments

### 1. Academic Papers
- N/A

### 2. Code References
- TIG baseline

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
