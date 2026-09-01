# L40 7000x7000 eight-sub-instance run

> Historical calibration artifact: this run predates the verifier-only fast-U
> redesign. Its linking-matrix payloads and method comparisons are not the
> current index-only TIG solution protocol.

This is the audited report for C3 job `job_1786702545740_ohuj6v`, run on one
NVIDIA L40 with seed 0. It uses the revised `docs/cur.tex` design: eight
stratified shared-basis sub-instances and independent fixed multiplicative
singular-value noise drawn from `Uniform(-0.15, 0.15)`.

All 32 selector/method/sub-instance records completed with status `ok`. The
CSV contained all sub-instance, true-rank-stratum, and target-ratio-stratum
indices from 0 through 7. The JSON contained one generation record with eight
GEMM timings and four aggregates of eight successful results.

## Generation

| Stage | Time |
|---|---:|
| Gaussian U | 0.473 ms |
| Gaussian V | 0.524 ms |
| QR U | 83.690 ms |
| QR V | 84.319 ms |
| Eight matrix multiplications | 22.041 ms |
| QR plus eight multiplications | 190.050 ms |
| Complete generation wall time | 209.724 ms |

## Aggregate solver results

| Selector | Linking-matrix method | Mean select | Mean U | Mean raw ratio | Score | Raw U total | Serialized total |
|---|---|---:|---:|---:|---:|---:|---:|
| `matrix_norms` | QR least squares | 3.849 ms | 14.342 ms | 91.047 | 1.000 | 8.757 MiB | 10.857 MiB |
| `matrix_norms` | Adaptive SVD | 3.849 ms | 1111.738 ms | 10.652 | 1.000 | 8.757 MiB | 10.841 MiB |
| `block_krylov_maxvol` | QR least squares | 972.261 ms | 13.326 ms | 30.177 | 1.000 | 8.757 MiB | 10.855 MiB |
| `block_krylov_maxvol` | Adaptive SVD | 972.261 ms | 1103.096 ms | 6.890 | 1.000 | 8.757 MiB | 10.836 MiB |

## Sophisticated selector by sub-instance

| Sub | True rank | k | k / true rank | Selection | Fast U | Fast ratio | Accurate U | Accurate ratio | U bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 341 | 63 | 0.185 | 105.238 ms | 2.622 ms | 2.358 | 138.375 ms | 2.358 | 15,876 |
| 1 | 429 | 198 | 0.462 | 268.573 ms | 5.819 ms | 15.721 | 350.678 ms | 5.628 | 156,816 |
| 2 | 718 | 213 | 0.297 | 275.975 ms | 6.485 ms | 3.322 | 381.090 ms | 3.322 | 181,476 |
| 3 | 888 | 385 | 0.434 | 582.025 ms | 11.023 ms | 8.669 | 719.513 ms | 5.105 | 592,900 |
| 4 | 990 | 664 | 0.671 | 1,402.051 ms | 19.353 ms | 55.319 | 1,670.053 ms | 11.980 | 1,763,584 |
| 5 | 1,234 | 938 | 0.760 | 2,811.444 ms | 27.068 ms | 124.112 | 2,883.624 ms | 14.956 | 3,519,376 |
| 6 | 1,333 | 781 | 0.586 | 1,887.599 ms | 24.205 ms | 28.900 | 2,096.083 ms | 8.809 | 2,439,844 |
| 7 | 1,448 | 358 | 0.247 | 445.185 ms | 10.033 ms | 3.018 | 585.351 ms | 2.964 | 512,656 |

## TIG output-size implication

The eight raw float32 linking matrices occupy 9,182,528 bytes (8.757 MiB),
which is below 10 MiB. However, a complete JSON solution is between 10.836 and
10.857 MiB because decimal float serialization and row/column indices add
overhead. The smallest measured payload exceeds 10 MiB by 876,933 bytes, so
the current eight-sub-instance FP32 JSON solution does not meet a strict
less-than-10-MiB return limit at this sampled target-rank combination.

The artifacts can be retrieved with `c3 pull job_1786702545740_ohuj6v`.
