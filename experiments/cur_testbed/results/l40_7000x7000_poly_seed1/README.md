# L40 7000x7000 polynomial-spectrum run

> Historical calibration artifact: this run predates the verifier-only fast-U
> redesign. Its linking-matrix payloads and method comparisons are not the
> current index-only TIG solution protocol.

This is the audited report for C3 job `job_1786699035255_rxqh7h`, run on one
NVIDIA L40 with seed 0. The experiment used the portable Philox Gaussian
generator, six randomized shared-basis sub-instances, the `matrix_norms` and
`block_krylov_maxvol` selectors, and both linking-matrix methods.

All 24 measured selector/method/sub-instance combinations completed with
status `ok`. The pulled CSV contained exactly 24 rows and the JSON contained
one generation record and four six-result aggregates.

## Generation

| Stage | Time |
|---|---:|
| Gaussian U | 0.490 ms |
| Gaussian V | 0.564 ms |
| QR U | 81.679 ms |
| QR V | 85.383 ms |
| Six matrix multiplications | 16.548 ms |
| QR plus six multiplications | 183.610 ms |
| Complete generation wall time | 198.571 ms |

The six individual matrix-multiplication times were 5.858, 1.176, 1.882,
2.123, 2.419, and 3.091 ms.

## Aggregate solver results

| Selector | Linking-matrix method | Mean select | Mean U | Mean raw ratio | Raw U total | Serialized total |
|---|---|---:|---:|---:|---:|---:|
| `matrix_norms` | QR least squares | 3.927 ms | 12.350 ms | 104.509 | 5.831 MiB | 7.240 MiB |
| `matrix_norms` | Adaptive SVD | 3.927 ms | 1045.236 ms | 10.591 | 5.831 MiB | 7.214 MiB |
| `block_krylov_maxvol` | QR least squares | 892.856 ms | 12.281 ms | 28.006 | 5.831 MiB | 7.239 MiB |
| `block_krylov_maxvol` | Adaptive SVD | 892.856 ms | 994.176 ms | 6.679 | 5.831 MiB | 7.223 MiB |

All four aggregates scored 1.0 on every sub-instance under the thresholded
challenge scoring rule.

## Sophisticated selector by sub-instance

The selected rows and columns are reused between the two U methods, so the
selection time appears once below. `Fast ratio` uses QR least squares and
`accurate ratio` uses the residual-selected adaptive SVD method.

| Sub | True rank | k | k / true rank | Selection | Fast U | Fast ratio | Accurate U | Accurate ratio | U bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 367 | 109 | 0.297 | 149.080 ms | 3.379 ms | 3.243 | 175.267 ms | 3.243 | 47,524 |
| 1 | 473 | 275 | 0.581 | 383.744 ms | 8.252 ms | 33.279 | 512.225 ms | 8.178 | 302,500 |
| 2 | 820 | 175 | 0.213 | 209.057 ms | 5.246 ms | 2.757 | 258.053 ms | 2.757 | 122,500 |
| 3 | 1,024 | 757 | 0.739 | 1,815.040 ms | 21.414 ms | 100.948 | 1,973.913 ms | 12.597 | 2,292,196 |
| 4 | 1,146 | 414 | 0.361 | 663.101 ms | 11.680 ms | 5.620 | 894.300 ms | 4.671 | 685,584 |
| 5 | 1,499 | 816 | 0.544 | 2,137.113 ms | 23.718 ms | 22.191 | 2,151.300 ms | 8.625 | 2,663,424 |

The six raw float32 U matrices occupy 6,113,728 bytes (5.831 MiB) in total.
The result artifacts can be retrieved again with
`c3 pull job_1786699035255_rxqh7h`.
