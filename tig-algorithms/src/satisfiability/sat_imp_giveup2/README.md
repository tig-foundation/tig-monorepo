# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_imp_giveup2
* **Copyright:** 2026 ChervovNikita
* **Identity of Submitter:** ChervovNikita
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track | Fuel Budget | Nonces/Bundle |
|---|---|---|
| n_vars=7500,ratio=4267 | 5T | 100 |
| n_vars=10000,ratio=4267 | 5T | 100 |
| n_vars=5000,ratio=4267 | 5T | 100 |
| n_vars=100000,ratio=4200 | 5T | 100 |
| n_vars=100000,ratio=4150 | 5T | 100 |

## Measured Quality

Solves per 100-nonce bundle, seed `rand_hash`. Quality is binary per nonce, so the bundle
score is the solve count.

| Track | Solves / 100 | Max Fuel | Max Runtime |
|---|---:|---:|---:|
| n_vars=7500,ratio=4267 | 11.67 | <0.1% | 20 min |
| n_vars=10000,ratio=4267 | 9.67 | <0.1% | 24 min |

Runtime figures are per-nonce upper bounds measured under concurrent load. All tracks are
inside the fuel cap and the per-bundle wall limit.

Determinism verified on every track: re-running the same nonces reproduces per-nonce identical
results and identical `runtime_signature`.

## References and Acknowledgments

### 1. Academic Papers
- N/A

### 2. Code References
- Derived from the TIG code submission `satisfiability/sat_imp_v4`, which carries no Unique
  Algorithm Identifier and embodies no registered Advance Submission.

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
