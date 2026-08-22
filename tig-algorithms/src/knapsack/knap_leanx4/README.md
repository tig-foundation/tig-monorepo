# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** knap_leanx4
* **Copyright:** 2026 ChervovNikita
* **Identity of Submitter:** ChervovNikita
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track | Fuel Budget | Nonces/Bundle |
|---|---|---|
| n_items=1000,budget=5 | 5T | 1000 |
| n_items=1000,budget=10 | 5T | 1000 |
| n_items=1000,budget=25 | 5T | 1000 |
| n_items=5000,budget=10 | 5T | 50 |
| n_items=5000,budget=25 | 5T | 50 |

## Measured Quality

Bundle score is the integer median of per-nonce qualities. Measured on a 72-core RTX4090 host,
one batch, live per-track hyperparameter blobs.

| Track | Median | Nonces run | Max Fuel | Wall (per nonce) |
|---|---:|---:|---:|---:|
| n_items=1000,budget=5 | 455826 | 1000 | <0.1% | 1.30 s |
| n_items=1000,budget=10 | 231930 | 1000 | <0.1% | 2.14 s |
| n_items=1000,budget=25 | 52660 | 1000 | 0.07% | 3.41 s |
| n_items=5000,budget=10 | 154979 | 50 | 0.39% | 28.0 s |
| n_items=5000,budget=25 | 48814 | 50 | 0.43% | 48.0 s |

All tracks are inside the fuel cap and the per-bundle wall limit. Wall figures are per-nonce
under 30-way concurrent load.

**Output is unchanged from the predecessor.** Per-nonce score identity against
`submission001_knap_lean` verified on 3,100 nonces across all five tracks: zero differing nonces.

Determinism verified on the same 3,100 nonces: re-running reproduces per-nonce identical
`(nonce, quality, fuel_consumed)` on every track, on both fields.

## Changes vs `submission001_knap_lean`

All five track files differ; `mod.rs` is byte-identical. Every change is bit-identical by
construction — same values, same order — which the determinism and identity checks above verify
rather than assume.

| file | change |
|---|---|
| `track1.rs` | Bounds-check elimination on provably in-range row slices; cached bounded-buffer argmin; removal of two unreachable HPF guards and three dead fields; deletion of 12 provably-zero diagonal terms; per-endpoint triple bound in `apply_best_swap23`. |
| `track2.rs` | Bounds-check elimination on flat matrix reads; one-shot swap22 gating hoisted out of the inner trip. |
| `track3.rs` | Capacity-exhausted early break in the DP scan; branchless bounds-check-free zobrist hash; three branchless rewrites; bounded strong-bucket scan cursor. |
| `track4.rs` | Bounds-check elimination on double-indexed reads; integer density key in `build_windows` replacing a float comparator; 4-pass 2-for-2 exchange fused into one. |
| `track5.rs` | Fused `replace_item` (one combined pass over `contrib` instead of two); symmetric-triangle interaction sum; per-restart constant hoisted across restarts; outer-loop gate lifting. |

Cost against the predecessor, one host, one session, production hyperparameters:

| Track | Ratio | Δ wall |
|---|---|---|
| n_items=1000,budget=5 | 1.0993 | −9.03% |
| n_items=1000,budget=10 | 1.0753 | −7.00% |
| n_items=1000,budget=25 | 1.0768 | −7.13% |
| n_items=5000,budget=10 | 1.0006 | −0.06% |
| n_items=5000,budget=25 | 1.0565 | −5.35% |

`n_items=5000,budget=10` is unchanged within measurement noise — four of the five tracks carry the
gain, and no improvement is claimed on the fifth.

Full evidence, including the quality comparison against every measurable rival and the caveats, is
in `PROOF.md` §11.

## References and Acknowledgments

### 1. Academic Papers
- N/A

### 2. Code References
- Derived from the TIG code submission `knapsack/knap_lean` (`c003_a144`), submitted by the same
  player, which in turn derives from the TIG code submission `knapsack/knap_master_v2`. This
  submission changes execution cost only; the algorithmic method is unchanged and embodies no
  registered Advance Submission.

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
