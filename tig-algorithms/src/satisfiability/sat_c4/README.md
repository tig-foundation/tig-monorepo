# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_c4
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

Quality is binary per nonce, so the bundle score is the solve count. Measured on a
32-physical-core Xeon E5-2682 v4, sole-tenant, one batch per track. **Solve rates are
reported over the nonce count actually run**, not extrapolated to 100.

| Track | Solves | Nonces run | Max Fuel | Max Runtime (per nonce) |
|---|---:|---:|---:|---:|
| n_vars=7500,ratio=4267 | 12 | 100 | <0.1% | 18 min |
| n_vars=10000,ratio=4267 | 5 | 32 | <0.1% | 17.4 min |
| n_vars=5000,ratio=4267 | 20 | 96 | <0.1% | 8.4 min |
| n_vars=100000,ratio=4200 | 32 | 32 | <0.1% | 8.2 min |
| n_vars=100000,ratio=4150 | 96 | 96 | <0.1% | 1.2 min |

Runtime figures are per-nonce upper bounds measured under 32-way concurrent load. All
tracks are inside the fuel cap and the per-bundle wall limit.

n=7500 is carried from `sat_il1`: `track2.rs` is byte-identical between the two, so the
measurement transfers exactly.

Determinism verified on every changed track: re-running the same nonces reproduces
per-nonce identical `(nonce, quality, fuel_used)`.

## Changes vs `sat_il1`

Two files differ; `mod.rs`, `track2.rs`, `track3.rs` and `track5.rs` are byte-identical.

| file | change |
|---|---|
| `track1.rs` | **IL1** — occurrence-run bounds interleaved into one array (`bnd[2v]`, `bnd[2v+1]`, `bnd[2v+2]`) instead of two separate arrays, so both polarities of a variable share a cache line. **FLAT4** — clause literals zero-padded to a stride-4 block, so a clause is one aligned 16-byte read and the `co` offset array leaves the hot loop. |
| `track4.rs` | **FLAT4**, as above. |

Both are pure layout changes: same values, same reads, same order. The intra-clause
swap is deliberately retained so the transformation is bit-identical, which the
determinism check above verifies rather than assumes.

**Verification at a stressed fuel cap.** `n_vars=100000,ratio=4150` saturates at the
production cap (96/96), where a solve-set identity check is degenerate — any variant
trivially matches. At `max_fuel_low=5e9` only 72 of 96 solve, and the solve sets are
still identical to the predecessor.

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
