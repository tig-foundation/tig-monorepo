# L40 10000x5000 high-quality CUR run

This is the audited report for C3 job `job_1787484828768_v7do0z`, run on one
NVIDIA L40 with deterministic seed 0. It uses the same polynomial spectrum,
eight-subinstance generator, solver parameters, warm-up, scoring, and hybrid
fast-U policy as the 8000x8000 quality run; only `m=10000` and `n=5000` differ.

Compilation, package setup, the L40 queue, and the separate 384x384 warm-up are
excluded from every workload timing.

## Totals

| Measurement | Result |
|---|---:|
| Complete challenge generation | 167.958 ms |
| Solve all eight subinstances | 7,504.894 ms |
| Verify all eight solutions | 123.304 ms |
| Logical raw solution payload | 235,708 bytes (0.225 MiB) |
| Exact serialized JSON solution | 284,909 bytes (0.272 MiB) |
| Mean continuous score | 0.718887205 |
| TIG integer quality | 718,887 |

The logical payload consists of 217,124 bytes of submitted float32 linking
matrices and 18,584 bytes of int32 row/column indices.

Generation comprised 0.798 ms for the two Gaussian matrices, 133.211 ms for
the two QR factorizations, 1.787 ms for basis extraction and scaling, and
12.770 ms for the eight final matrix multiplications. The generator's internal
wall time was 161.841 ms; the external total includes call and synchronization
overhead.

## Per-subinstance results

| Sub | True rank | Target k | k / rank | U source | Solve (ms) | Verify (ms) | Error ratio | Score | Serialized bytes |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 164 | 90 | 0.5488 | Submitted | 376.049 | 6.113 | 9.337054 | 0.668785711 | 40,878 |
| 1 | 318 | 156 | 0.4906 | Submitted | 560.575 | 6.354 | 6.384558 | 0.731715208 | 121,542 |
| 2 | 470 | 106 | 0.2255 | Submitted | 407.793 | 6.080 | 2.554272 | 0.832857849 | 56,682 |
| 3 | 551 | 385 | 0.6987 | Verifier fast U | 940.961 | 20.241 | 70.238572 | 0.583463166 | 2,562 |
| 4 | 761 | 277 | 0.3640 | Verifier fast U | 587.414 | 18.199 | 5.404378 | 0.769344019 | 1,870 |
| 5 | 895 | 103 | 0.1151 | Submitted | 399.418 | 6.335 | 2.377759 | 0.842818139 | 53,486 |
| 6 | 944 | 386 | 0.4089 | Verifier fast U | 935.578 | 19.787 | 6.883372 | 0.755423073 | 2,574 |
| 7 | 1,070 | 820 | 0.7664 | Verifier fast U | 3,296.890 | 32.987 | 169.199582 | 0.566690475 | 5,306 |

The original raw C3 JSON artifact has SHA-256
`957a46d3296267a1d558f838970e0fac49aec0c98c4e18dc4948112273775418`
and can be retrieved with `c3 pull job_1787484828768_v7do0z`.
