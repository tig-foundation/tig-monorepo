# L40 8000x8000 verifier-fast-U CUR run

This is the audited report for C3 job `job_1788264930438_7j65ny`, run on one
NVIDIA L40 with deterministic seed 0 on September 1, 2026. It repeats the
8000x8000 polynomial quality benchmark after the verifier-only redesign:
solutions contain only row and column indices, and verification computes the
canonical fast QR linking matrix for all eight subinstances.

The algorithm used three randomized block-Krylov/subspace trials, two power
iterations, a `k + 32` sketch, at most 16 max-volume swaps, and a max-volume
tolerance of 1.01. CUDA and the complete verifier-fast-U path were warmed on a
384x384 instance before measurement.

## Totals

| Measurement | Result |
|---|---:|
| Complete challenge generation | 288.991 ms |
| Solve all eight subinstances | 19,813.107 ms |
| Verify all eight solutions | 213.018 ms |
| Logical raw index payload | 29,752 bytes (0.028 MiB) |
| Exact serialized JSON solution | 24,693 bytes (0.024 MiB) |
| Mean continuous score | 0.707592197 |
| TIG integer quality | 707,592 |

Generation comprised 0.576 ms and 0.633 ms for the two Gaussian matrices,
231.811 ms for the two QR factorizations, 2.890 ms for basis extraction and
scaling, and 23.636 ms for the eight final matrix multiplications. The
generator's internal wall time was 276.213 ms; the externally measured total
also includes call and synchronization overhead.

## Comparison with the historical hybrid run

The earlier job `job_1787482766173_mk1p5w` used the same matrix dimensions,
seed, generator, and selector parameters, but submitted adaptive-SVD linking
matrices for four subinstances and used verifier fast U for the other four.

| Measurement | Historical hybrid | Verifier-only fast U | Change |
|---|---:|---:|---:|
| Raw solution payload | 588,352 bytes | 29,752 bytes | -94.94% (19.8x smaller) |
| Serialized JSON solution | 714,517 bytes | 24,693 bytes | -96.54% (28.9x smaller) |
| Verification | 199.770 ms | 213.018 ms | +6.63% |
| Solve | 18,618.454 ms | 19,813.107 ms | +6.42% |
| Mean score | 0.733399694 | 0.707592197 | -0.025807497 |

The score change is expected because the redesigned verifier replaces the four
submitted adaptive-SVD linking matrices with the canonical fast QR result.
Solve and verification timings are single-run wall-clock measurements and
include ordinary run-to-run variance.

## Per-subinstance results

| Sub | True rank | Target k | k / rank | Solve (ms) | Verify (ms) | Error ratio | Score | Serialized bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 263 | 145 | 0.5513 | 381.975 | 15.719 | 36.635709 | 0.580526663 | 1,050 |
| 1 | 510 | 250 | 0.4902 | 629.969 | 18.969 | 14.587277 | 0.673375642 | 1,714 |
| 2 | 753 | 170 | 0.2258 | 421.712 | 16.486 | 2.730347 | 0.836574406 | 1,222 |
| 3 | 882 | 616 | 0.6984 | 2,059.459 | 30.379 | 83.830380 | 0.591953899 | 4,034 |
| 4 | 1,217 | 443 | 0.3640 | 1,240.185 | 25.112 | 5.356067 | 0.784124241 | 2,938 |
| 5 | 1,432 | 165 | 0.1152 | 394.643 | 16.269 | 2.509382 | 0.847474642 | 1,182 |
| 6 | 1,511 | 619 | 0.4097 | 2,085.073 | 30.474 | 7.206676 | 0.765012264 | 4,054 |
| 7 | 1,711 | 1,311 | 0.7662 | 12,599.868 | 58.131 | 174.639388 | 0.581695822 | 8,490 |

The checked-in JSON report has SHA-256
`6304be5678cf048b09d5eeeed4324fd3841a05b0cb3fc0c0342c7d29b007e511`.
The original C3 artifacts also include the generated PTX, with SHA-256
`e99859e4100d5d016392d6f1556b1df6440b3e7db258219279b999ef7697dfd1`,
and can be retrieved with `c3 pull job_1788264930438_7j65ny`.
