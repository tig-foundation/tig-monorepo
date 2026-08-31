# L40 8000x8000 high-quality CUR run

This is the audited report for C3 job `job_1787482766173_mk1p5w`, run on one
NVIDIA L40 with deterministic seed 0. It exercises the current eight-subinstance
polynomial CUR track and the hybrid solution format: innovators submit `U` for
four subinstances, while the verifier reconstructs the four largest linking
matrices with the canonical fast method.

The algorithm uses three randomized block-Krylov/subspace trials, two power
iterations, a `k + 32` sketch, max-volume row and column refinement, and
residual-selected adaptive SVD linking matrices where `U` is submitted. CUDA
and both solution paths were warmed separately; compilation, package setup,
and warm-up are excluded from all timings below.

## Totals

| Measurement | Result |
|---|---:|
| Complete challenge generation | 288.990 ms |
| Solve all eight subinstances | 18,618.454 ms |
| Verify all eight solutions | 199.770 ms |
| Logical raw solution payload | 588,352 bytes (0.561 MiB) |
| Exact serialized JSON solution | 714,517 bytes (0.681 MiB) |
| Mean continuous score | 0.733399694 |
| TIG integer quality | 733,400 |

The logical payload consists of 558,600 bytes of submitted float32 linking
matrices and 29,752 bytes of int32 row/column indices.

Generation comprised 1.270 ms for the two Gaussian matrices, 230.097 ms for
the two QR factorizations, 3.104 ms for basis extraction and scaling, and
23.600 ms for the eight final matrix multiplications. The generator's internal
wall time was 276.054 ms; the externally measured end-to-end wall time above
also includes call and synchronization overhead.

## Per-subinstance results

| Sub | True rank | Target k | k / rank | U source | Solve (ms) | Verify (ms) | Error ratio | Score | Serialized bytes |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 263 | 145 | 0.5513 | Submitted | 590.902 | 8.987 | 7.944763 | 0.706281489 | 105,126 |
| 1 | 510 | 250 | 0.4902 | Submitted | 899.392 | 9.313 | 6.124035 | 0.753025125 | 310,550 |
| 2 | 753 | 170 | 0.2258 | Submitted | 647.447 | 8.871 | 2.709584 | 0.837614746 | 143,858 |
| 3 | 882 | 616 | 0.6984 | Verifier fast U | 1,900.507 | 31.109 | 83.830380 | 0.591953899 | 4,038 |
| 4 | 1,217 | 443 | 0.3640 | Verifier fast U | 1,160.456 | 25.530 | 5.356067 | 0.784124241 | 2,946 |
| 5 | 1,432 | 165 | 0.1152 | Submitted | 622.943 | 9.091 | 2.509109 | 0.847489968 | 135,430 |
| 6 | 1,511 | 619 | 0.4097 | Verifier fast U | 1,944.227 | 30.449 | 7.206676 | 0.765012264 | 4,062 |
| 7 | 1,711 | 1,311 | 0.7662 | Verifier fast U | 10,852.360 | 58.568 | 174.639388 | 0.581695822 | 8,498 |

The original raw C3 JSON artifact has SHA-256
`cfbcbb6ba25ca422722074bb97905839d927e3014c9965e0841f749ca0428f47`
and can be retrieved with `c3 pull job_1787482766173_mk1p5w`.
