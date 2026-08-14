# L40 result: independent 8193x8193 polynomial matrices, one seed

This result was produced by C3 job `job_1786449626094_ihuphk` on an NVIDIA L40
with 48 GB VRAM. The host reported driver 535.183.06 and CUDA 12.2, and the
experiment was compiled with CUDA Toolkit 12.2. It uses `8193 = 2^13 + 1`, the
valid CUR challenge dimension nearest to 8000, one polynomial-spectrum seed,
and all 18 target-rank sub-instances. All 72 selector/U combinations succeeded.

Five independently seeded matrices A were generated, one for each true-rank
group. Target ranks in the same group reuse exactly the same A, allowing paired
comparisons across k without the production generator's dyadic nesting.

The controlled methods were:

- Simple selector: rows and columns with the largest squared matrix norms.
- Sophisticated selector: block-Krylov randomized SVD with `q=1`, sketch width
  `k+20`, max-volume row/column refinement, and the best residual from two
  deterministic restarts.
- Quick U: intersection inverse `U = W^-1`.
- Accurate U: truncated-SVD least squares `U = C^+ A R^+`, using relative
  singular-value threshold `1e-6`.

The selector is run once per sub-instance and its exact indices and extracted C
and R are reused for both U methods. The sophisticated selector's two internal
accurate-U/residual evaluations are included in `selection_ms` because they are
part of how it chooses its final indices. Every reported GPU stage is bounded
by a stream synchronization.

## Aggregate result

| Selector | U method | Select, all 18 | U, all 18 | Mean raw ratio | Mean score | Full scores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Matrix norms | Intersection inverse | 94.77 ms | 674.49 ms | 6903.0891 | 0.8138 | 10/18 |
| Matrix norms | SVD pseudoinverse | 94.77 ms | 2049.71 ms | 5.1001 | 1.0000 | 18/18 |
| Block Krylov + maxvol | Intersection inverse | 14846.21 ms | 679.20 ms | 105.5462 | 0.9595 | 16/18 |
| Block Krylov + maxvol | SVD pseudoinverse | 14846.21 ms | 2000.57 ms | 3.0524 | 1.0000 | 18/18 |

Accurate U averages 113.87 ms after the simple selector and 111.14 ms after the
sophisticated selector. The latter selector averages 824.79 ms, so verifier-side
accurate U is 13.5% of its selection time. For a deliberately cheap selector,
the relationship reverses: 113.87 ms for U versus 5.26 ms for selection.

With accurate U, the sophisticated selector lowers the mean unclipped error
ratio from 5.1001 to 3.0524, a 40.1% reduction. The protocol score hides this
difference: every accurate-U ratio is below its `k+1` threshold and therefore
receives score 1. Raw ratios should remain a primary calibration output even if
the clipped score is retained for the eventual challenge.

## Independent matrix generation

Generation is recorded once per independent true-rank group. `Copies` is the
time to make one separately owned GPU copy of A for every target k in the group.

| Matrix group | True rank | Q | Spectrum scaling | Q1 Sigma Q2^T | Copies | Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8193 | 750.63 ms | 0.90 ms | 33.50 ms | 2.88 ms | 788.05 ms |
| 1 | 4097 | 342.11 ms | 0.62 ms | 13.35 ms | 4.69 ms | 360.92 ms |
| 2 | 2049 | 182.39 ms | 0.45 ms | 7.05 ms | 6.88 ms | 196.95 ms |
| 3 | 1025 | 111.83 ms | 0.35 ms | 4.07 ms | 4.84 ms | 121.29 ms |
| 4 | 513 | 61.24 ms | 0.25 ms | 2.68 ms | 2.66 ms | 66.99 ms |
| **All five** | | **1448.20 ms** | **2.57 ms** | **60.65 ms** | **21.95 ms** | **1534.20 ms** |

Constructing the orthogonal factors dominates generation. The final dense
multiplications are only 60.65 ms in total for the five matrices.

## Per-sub-instance measurements

Each method cell is `U ms / raw error ratio / score`. Selection time is shown
separately and is shared by its quick- and accurate-U rows.

| Sub | True rank | k | k/tau | Simple select ms | Simple quick | Simple accurate | Soph. select ms | Soph. quick | Soph. accurate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8193 | 682 | 0.083 | 5.31 | 155.16 / 104.34 / 1.0000 | 323.40 / 2.81 / 1.0000 | 2342.07 | 161.47 / 55.79 / 1.0000 | 321.69 / 2.74 / 1.0000 |
| 1 | 8193 | 409 | 0.050 | 5.62 | 36.50 / 43.70 / 1.0000 | 150.18 / 2.81 / 1.0000 | 1075.13 | 37.65 / 107.25 / 1.0000 | 149.69 / 2.72 / 1.0000 |
| 2 | 4097 | 341 | 0.083 | 6.19 | 22.81 / 392.07 / 0.8723 | 121.50 / 2.83 / 1.0000 | 782.30 | 20.92 / 36.98 / 1.0000 | 95.13 / 2.70 / 1.0000 |
| 3 | 4097 | 204 | 0.050 | 5.24 | 5.22 / 48.46 / 1.0000 | 48.51 / 2.80 / 1.0000 | 336.37 | 5.47 / 34.88 / 1.0000 | 46.95 / 2.66 / 1.0000 |
| 4 | 4097 | 682 | 0.166 | 5.18 | 156.36 / 142.83 / 1.0000 | 308.44 / 3.04 / 1.0000 | 2313.38 | 156.84 / 79.46 / 1.0000 | 313.03 / 2.79 / 1.0000 |
| 5 | 4097 | 409 | 0.100 | 5.31 | 36.70 / 313.53 / 1.0000 | 146.03 / 2.84 / 1.0000 | 1075.07 | 37.37 / 103.19 / 1.0000 | 155.67 / 2.72 / 1.0000 |
| 6 | 2049 | 170 | 0.083 | 4.90 | 3.45 / 62.47 / 1.0000 | 43.59 / 2.83 / 1.0000 | 279.63 | 3.57 / 37.12 / 1.0000 | 35.35 / 2.60 / 1.0000 |
| 7 | 2049 | 102 | 0.050 | 5.24 | 0.93 / 19.00 / 1.0000 | 23.63 / 2.84 / 1.0000 | 174.47 | 1.20 / 29.24 / 1.0000 | 21.09 / 2.45 / 1.0000 |
| 8 | 2049 | 341 | 0.166 | 5.17 | 22.27 / 373.64 / 0.9153 | 110.34 / 3.00 / 1.0000 | 725.07 | 21.82 / 323.39 / 1.0000 | 93.87 / 2.73 / 1.0000 |
| 9 | 2049 | 204 | 0.100 | 5.38 | 4.99 / 242.76 / 0.8444 | 45.80 / 2.84 / 1.0000 | 342.26 | 5.54 / 38.70 / 1.0000 | 45.15 / 2.65 / 1.0000 |
| 10 | 2049 | 682 | 0.333 | 5.25 | 159.79 / 120346.42 / 0.0057 | 307.77 / 12.68 / 1.0000 | 2347.52 | 155.50 / 211.84 / 1.0000 | 315.47 / 5.27 / 1.0000 |
| 11 | 2049 | 409 | 0.200 | 5.30 | 35.11 / 710.30 / 0.5772 | 144.65 / 3.41 / 1.0000 | 1084.00 | 35.09 / 37.78 / 1.0000 | 154.96 / 2.80 / 1.0000 |
| 12 | 1025 | 170 | 0.166 | 4.84 | 3.37 / 24.45 / 1.0000 | 44.59 / 3.18 / 1.0000 | 281.20 | 3.38 / 49.48 / 1.0000 | 35.30 / 2.61 / 1.0000 |
| 13 | 1025 | 102 | 0.100 | 5.23 | 0.91 / 17.46 / 1.0000 | 19.96 / 2.84 / 1.0000 | 174.13 | 1.06 / 136.39 / 0.7552 | 20.50 / 2.43 / 1.0000 |
| 14 | 1025 | 341 | 0.333 | 5.19 | 21.43 / 603.11 / 0.5671 | 109.75 / 13.49 / 1.0000 | 722.67 | 22.32 / 164.81 / 1.0000 | 93.85 / 4.89 / 1.0000 |
| 15 | 1025 | 204 | 0.199 | 5.21 | 4.97 / 117.36 / 1.0000 | 46.28 / 3.67 / 1.0000 | 338.80 | 5.31 / 79.55 / 1.0000 | 46.44 / 2.69 / 1.0000 |
| 16 | 513 | 170 | 0.331 | 4.95 | 3.46 / 496.27 / 0.3446 | 34.72 / 19.33 / 1.0000 | 276.74 | 3.48 / 331.23 / 0.5163 | 35.51 / 4.87 / 1.0000 |
| 17 | 513 | 102 | 0.199 | 5.25 | 1.04 / 197.44 / 0.5217 | 20.58 / 4.58 / 1.0000 | 175.40 | 1.22 / 42.78 / 1.0000 | 20.92 / 2.60 / 1.0000 |

## U payload size

The 18 U matrices contain 2,488,818 f32 values: 9,955,272 bytes, or 9.494 MiB
raw. This is already close to the 10 MiB limit before indices and serialization.
The experiment serializes each real `Solution` with the repository's actual
bincode, gzip, base64, and JSON path, then measures the complete 18-item array.

| Selector | U method | Raw U | Actual serialized 18-solution payload |
| --- | --- | ---: | ---: |
| Matrix norms | Intersection inverse | 9.494 MiB | 11.879 MiB |
| Matrix norms | SVD pseudoinverse | 9.494 MiB | 11.744 MiB |
| Block Krylov + maxvol | Intersection inverse | 9.494 MiB | 11.849 MiB |
| Block Krylov + maxvol | SVD pseudoinverse | 9.494 MiB | 11.745 MiB |

All four actual payloads exceed 10 MiB. Returning only the row and column
indices would use 45,792 raw bytes (44.7 KiB) across all 18 sub-instances and
would eliminate this transport problem; U could then be reconstructed by the
fixed accurate method during verification.

The complete unrounded measurements are in `cur_stage_results.csv`, and the
machine-readable aggregate is in `cur_stage_summary.json`.
