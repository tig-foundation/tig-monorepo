# L40 result: 6000 x 6000 polynomial spectrum, seed 0

This directory contains the audited artifacts from C3 job
`job_1786695903213_6yz3oa`. The job ran on an NVIDIA L40 with CUDA 12.2, but
used only portable cuSOLVER, cuBLAS, and ordinary CUDA kernels: there is no
L40-specific algorithm, tuning, or reduced-precision requirement.

Only the matrix dimensions differ from the earlier 2000 x 3000 run. The other
parameters remain `delta=10000`, spectrum parameter `a=13`, spectrum
perturbation `+/-0.05`, one measured seed, one Krylov power iteration, two
sophisticated-selector restarts, and at most eight max-volume swaps.

The CSV contains 24 successful rows: six sub-instances times two selectors
times two linking-matrix methods. The JSON contains the generation record and
the four solver aggregates.

## Shared-basis generation

| Stage | Time (ms) |
|---|---:|
| Gaussian U pool | 115.722 |
| QR for U pool | 60.005 |
| Gaussian V pool | 117.024 |
| QR for V pool | 60.404 |
| Matrix multiplication, sub-instance 0 | 5.731 |
| Matrix multiplication, sub-instance 1 | 0.913 |
| Matrix multiplication, sub-instance 2 | 1.224 |
| Matrix multiplication, sub-instance 3 | 1.514 |
| Matrix multiplication, sub-instance 4 | 1.544 |
| Matrix multiplication, sub-instance 5 | 1.973 |
| Both QRs plus all six multiplications | 133.308 |
| Complete generation wall time | 377.659 |

The complete wall time also includes Gaussian generation, random sampling,
allocations, and extracting and scaling basis columns. The requested 133.308
ms measure deliberately includes only both QR factorizations and the six final
cuBLAS matrix multiplications.

## Linking-matrix storage

Each linking matrix contains `k^2` float32 values and therefore occupies
exactly `4k^2` raw bytes.

| Sub | k | Raw U bytes |
|---:|---:|---:|
| 0 | 93 | 34,596 |
| 1 | 236 | 222,784 |
| 2 | 150 | 90,000 |
| 3 | 649 | 1,684,804 |
| 4 | 355 | 504,100 |
| 5 | 700 | 1,960,000 |
| **Total** | | **4,496,284 bytes (4.288 MiB)** |

That total is for the six linking matrices returned by one algorithm and one
U method. For the sophisticated selector, the complete six-solution JSON was
5,586,159 bytes (5.327 MiB) with fast QR U and 5,569,195 bytes (5.311 MiB)
with adaptive-SVD U. These payloads include row/column indices and JSON
serialization overhead and remain below TIG's 10 MiB output limit.

## Sophisticated selector by sub-instance

The selection time is shared by both U rows because the selected indices are
frozen before comparing U methods. `Ratio` is the unclipped
`||A-CUR||_F / ||A-SVD_k(A)||_F` value.

| Sub | True rank | k | k / true rank | Selection (ms) | U method | U time (ms) | Ratio | Score |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 314 | 93 | 0.296 | 109.584 | QR least squares | 2.854 | 3.796 | 1.000 |
| 0 | 314 | 93 | 0.296 | 109.584 | Adaptive SVD | 149.520 | 3.796 | 1.000 |
| 1 | 405 | 236 | 0.583 | 270.167 | QR least squares | 6.486 | 35.635 | 1.000 |
| 1 | 405 | 236 | 0.583 | 270.167 | Adaptive SVD | 372.335 | 8.617 | 1.000 |
| 2 | 703 | 150 | 0.213 | 168.476 | QR least squares | 4.479 | 2.658 | 1.000 |
| 2 | 703 | 150 | 0.213 | 168.476 | Adaptive SVD | 242.328 | 2.650 | 1.000 |
| 3 | 877 | 649 | 0.740 | 1162.523 | QR least squares | 16.316 | 113.053 | 1.000 |
| 3 | 877 | 649 | 0.740 | 1162.523 | Adaptive SVD | 1257.684 | 12.191 | 1.000 |
| 4 | 983 | 355 | 0.361 | 457.111 | QR least squares | 9.370 | 5.492 | 1.000 |
| 4 | 983 | 355 | 0.361 | 457.111 | Adaptive SVD | 657.467 | 4.614 | 1.000 |
| 5 | 1285 | 700 | 0.545 | 1414.521 | QR least squares | 18.373 | 20.204 | 1.000 |
| 5 | 1285 | 700 | 0.545 | 1414.521 | Adaptive SVD | 1677.219 | 8.184 | 1.000 |

Across the six sub-instances, sophisticated selection averaged 597.064 ms.
Fast U averaged 9.646 ms with mean raw ratio 30.140; adaptive U averaged
726.092 ms with mean raw ratio 6.675. Both received full score on all six
under the target-rank-dependent scoring rule.

For comparison, the cheap matrix-norm selector averaged 3.666 ms. Its fast-U
and adaptive-U variants had mean raw ratios 112.086 and 10.436 respectively;
both also received full score on all six sub-instances.
