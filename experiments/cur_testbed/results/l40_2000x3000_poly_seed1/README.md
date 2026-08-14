# L40 result: 2000 x 3000 polynomial spectrum, seed 0

This directory contains the measured artifacts from C3 job
`job_1786643054555_2lrmhp`. The job ran on an NVIDIA L40 (48 GiB), CUDA 12.2,
with one measured seed, `delta=10000`, spectrum parameter `a=13`, spectrum
perturbation `+/-0.05`, one Krylov power iteration, two sophisticated-selector
restarts, and at most eight max-volume swaps.

The CSV is the authoritative per-run output. It has 24 successful rows: six
sub-instances times two selectors times two linking-matrix methods. The JSON
contains the generation record and four solver aggregates.

## Shared-basis generation

| Stage | Time (ms) |
|---|---:|
| Gaussian U pool | 12.106 |
| QR for U pool | 9.742 |
| Gaussian V pool | 17.957 |
| QR for V pool | 13.346 |
| Matrix multiplication, sub-instance 0 | 0.123 |
| Matrix multiplication, sub-instance 1 | 0.174 |
| Matrix multiplication, sub-instance 2 | 5.156 |
| Matrix multiplication, sub-instance 3 | 0.234 |
| Matrix multiplication, sub-instance 4 | 0.211 |
| Matrix multiplication, sub-instance 5 | 0.228 |
| Both QRs plus all six multiplications | 29.213 |
| Complete generation wall time | 64.516 |

The complete wall time also includes random sampling, allocations, extracting
and scaling basis columns, and Gaussian generation. The requested 29.213 ms
measure deliberately includes only both QR factorizations and all six final
matrix multiplications.

## Linking-matrix storage

Each submitted linking matrix contains `k^2` float32 values, requiring exactly
`4k^2` bytes. For the six target ranks in this run:

| Sub | k | Raw U bytes |
|---:|---:|---:|
| 0 | 31 | 3,844 |
| 1 | 79 | 24,964 |
| 2 | 50 | 10,000 |
| 3 | 216 | 186,624 |
| 4 | 119 | 56,644 |
| 5 | 233 | 217,156 |
| **Total** | | **499,232 bytes (0.476 MiB)** |

That total is for the six linking matrices returned by one algorithm and one
U method. The sophisticated selector's complete serialized six-solution JSON
payload was 623,047 bytes (0.594 MiB) with fast QR U and 622,023 bytes
(0.593 MiB) with adaptive-SVD U. Those payload figures include the selected
row and column indices and JSON serialization overhead. Storing both U-method
variants at once would take 998,464 raw bytes (0.952 MiB), but a protocol
submission would return only one set of six solutions.

## Sophisticated selector by sub-instance

The selection time is shared by both U rows because the selected indices are
frozen before U is computed. `ratio` is the unclipped
`||A-CUR||_F / ||A-SVD_k(A)||_F` value.

| Sub | True rank | k | k / true rank | Selection (ms) | U method | U time (ms) | Ratio | Score | Raw U bytes |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 105 | 31 | 0.295 | 24.837 | QR least squares | 1.333 | 4.554 | 1.000 | 3,844 |
| 0 | 105 | 31 | 0.295 | 24.837 | Adaptive SVD | 34.617 | 4.554 | 1.000 | 3,844 |
| 1 | 135 | 79 | 0.585 | 44.194 | QR least squares | 2.055 | 38.123 | 1.000 | 24,964 |
| 1 | 135 | 79 | 0.585 | 44.194 | Adaptive SVD | 74.606 | 10.739 | 1.000 | 24,964 |
| 2 | 234 | 50 | 0.214 | 32.675 | QR least squares | 1.717 | 2.379 | 1.000 | 10,000 |
| 2 | 234 | 50 | 0.214 | 32.675 | Adaptive SVD | 51.070 | 2.379 | 1.000 | 10,000 |
| 3 | 292 | 216 | 0.740 | 139.048 | QR least squares | 3.857 | 137.738 | 1.000 | 186,624 |
| 3 | 292 | 216 | 0.740 | 139.048 | Adaptive SVD | 247.050 | 15.623 | 1.000 | 186,624 |
| 4 | 328 | 119 | 0.363 | 74.352 | QR least squares | 2.416 | 4.747 | 1.000 | 56,644 |
| 4 | 328 | 119 | 0.363 | 74.352 | Adaptive SVD | 134.504 | 4.130 | 1.000 | 56,644 |
| 5 | 428 | 233 | 0.544 | 162.276 | QR least squares | 4.005 | 23.522 | 1.000 | 217,156 |
| 5 | 428 | 233 | 0.544 | 162.276 | Adaptive SVD | 279.291 | 8.575 | 1.000 | 217,156 |

Across the six sub-instances, the sophisticated selector averaged 79.563 ms.
Fast U averaged 2.564 ms and adaptive U averaged 136.856 ms. Their mean raw
ratios were 35.177 and 7.667 respectively; both received full score on all six
sub-instances under the target-rank-dependent scoring rule.

For comparison, the cheap matrix-norm selector averaged 1.665 ms. With fast U
it obtained mean ratio 193.878, mean score 0.685, and full score on three of six
sub-instances. With adaptive U it obtained mean ratio 13.587 and full score on
all six, at an average U time of 145.865 ms.
