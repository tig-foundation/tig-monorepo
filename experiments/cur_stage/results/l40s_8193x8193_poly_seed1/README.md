# L40S result: 8193x8193 polynomial track, one seed

This result was produced by C3 job `job_1786394390320_ibt623` on an NVIDIA
L40S with 48 GB VRAM at Crusoe. The VM reported driver 565.57.01 and CUDA 12.7;
the experiment was compiled with CUDA Toolkit 12.4. It uses `8193 = 2^13 + 1`,
the valid challenge dimension nearest to 8000, one polynomial-spectrum seed,
and all 18 target-rank sub-instances. All 72 selector/U combinations succeeded.

The controlled methods were:

- Simple selector: take the rows and columns with the largest squared norms.
- Sophisticated selector: randomized range sketch of width `k + 20`, GPU QR,
  then projected row/column leverage norms.
- Fast U: invert the selected intersection, `U = W^-1`, using f64
  Gauss-Jordan elimination.
- Accurate U: SVD-truncated pseudoinverse, `U = C^+ A R^+`, with relative
  singular-value threshold `1e-6`.

Each selector runs once per sub-instance. Its exact indices and extracted C and
R are reused for both U methods, and every reported GPU stage is bounded by a
stream synchronization.

## Aggregate timings and scores

| Selector | Selection, all 18 | Fast U, all 18 | Fast-U mean score | Accurate U, all 18 | Accurate-U mean score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple matrix norms | 83.11 ms | 449.44 ms | 0.8779 (15/18 full) | 1902.88 ms | 1.0000 (18/18 full) |
| Sophisticated randomized range | 572.23 ms | 452.76 ms | 0.8779 (15/18 full) | 1895.00 ms | 1.0000 (18/18 full) |

Per sub-instance, selection averages 4.62 ms for the simple method and 31.79 ms
for the sophisticated method. Fast U averages about 25.1 ms and accurate U
about 105.5 ms. Thus the sophisticated selector is 6.89 times slower but has
no material effect on score or raw approximation ratio for this generated
matrix family. Accurate U is 4.20 times slower than fast U and changes the
three failing cases to full score.

Including extraction and residual verification, the four complete 18-instance
pipelines take 736 ms (simple/fast), 2198 ms (simple/accurate), 1227 ms
(sophisticated/fast), and 2681 ms (sophisticated/accurate). Matrix generation,
which is shared setup rather than selector work, took 865 ms.

## Per-sub-instance results

Every matrix below is `8193x8193`; `true rank` describes the generated source
matrix. Entries of the form `time / score` report U-construction time in
milliseconds followed by the protocol score.

| Sub | True rank | k | U KiB | Simple select ms | Fast U ms / score | Accurate U ms / score | Soph. select ms | Fast U ms / score | Accurate U ms / score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8193 | 682 | 1816.89 | 4.62 | 105.41 / 1.0000 | 294.58 / 1.0000 | 63.76 | 106.13 / 1.0000 | 295.15 / 1.0000 |
| 1 | 8193 | 409 | 653.44 | 4.63 | 23.22 / 0.4374 | 133.37 / 1.0000 | 40.04 | 24.02 / 0.4374 | 134.13 / 1.0000 |
| 2 | 4097 | 341 | 454.22 | 4.59 | 14.04 / 0.2694 | 103.40 / 1.0000 | 33.41 | 14.10 / 0.2694 | 102.71 / 1.0000 |
| 3 | 4097 | 204 | 162.56 | 4.59 | 3.17 / 1.0000 | 52.38 / 1.0000 | 21.63 | 2.99 / 1.0000 | 45.51 / 1.0000 |
| 4 | 4097 | 682 | 1816.89 | 4.59 | 101.76 / 1.0000 | 290.66 / 1.0000 | 63.79 | 104.09 / 1.0000 | 300.84 / 1.0000 |
| 5 | 4097 | 409 | 653.44 | 4.77 | 28.22 / 1.0000 | 134.85 / 1.0000 | 40.18 | 29.37 / 1.0000 | 138.45 / 1.0000 |
| 6 | 2049 | 170 | 112.89 | 4.80 | 1.78 / 1.0000 | 38.14 / 1.0000 | 18.94 | 1.82 / 1.0000 | 32.73 / 1.0000 |
| 7 | 2049 | 102 | 40.64 | 4.61 | 0.51 / 1.0000 | 17.90 / 1.0000 | 13.11 | 0.55 / 1.0000 | 19.13 / 1.0000 |
| 8 | 2049 | 341 | 454.22 | 4.64 | 15.05 / 1.0000 | 101.17 / 1.0000 | 33.40 | 13.61 / 1.0000 | 104.86 / 1.0000 |
| 9 | 2049 | 204 | 162.56 | 4.58 | 3.02 / 1.0000 | 52.21 / 1.0000 | 21.68 | 3.01 / 1.0000 | 43.94 / 1.0000 |
| 10 | 2049 | 682 | 1816.89 | 4.62 | 108.32 / 1.0000 | 296.09 / 1.0000 | 63.65 | 108.02 / 1.0000 | 294.93 / 1.0000 |
| 11 | 2049 | 409 | 653.44 | 4.58 | 23.33 / 1.0000 | 133.03 / 1.0000 | 39.96 | 23.59 / 1.0000 | 133.04 / 1.0000 |
| 12 | 1025 | 170 | 112.89 | 4.57 | 1.85 / 1.0000 | 37.56 / 1.0000 | 18.91 | 1.83 / 1.0000 | 32.46 / 1.0000 |
| 13 | 1025 | 102 | 40.64 | 4.58 | 0.52 / 1.0000 | 17.93 / 1.0000 | 12.97 | 0.52 / 1.0000 | 17.87 / 1.0000 |
| 14 | 1025 | 341 | 454.22 | 4.54 | 13.86 / 1.0000 | 99.58 / 1.0000 | 33.30 | 13.72 / 1.0000 | 105.42 / 1.0000 |
| 15 | 1025 | 204 | 162.56 | 4.60 | 3.07 / 1.0000 | 49.68 / 1.0000 | 21.67 | 3.01 / 1.0000 | 43.42 / 1.0000 |
| 16 | 513 | 170 | 112.89 | 4.66 | 1.80 / 0.0950 | 32.72 / 1.0000 | 18.92 | 1.84 / 0.0950 | 32.76 / 1.0000 |
| 17 | 513 | 102 | 40.64 | 4.55 | 0.51 / 1.0000 | 17.64 / 1.0000 | 12.92 | 0.52 / 1.0000 | 17.64 / 1.0000 |

## U payload size

Each linking matrix contains `k^2` f32 values. Each target rank occurs three
times in the 18-sub-instance set.

| k | Count | Bytes per U | MiB per U | MiB across count |
| ---: | ---: | ---: | ---: | ---: |
| 102 | 3 | 41,616 | 0.0397 | 0.1191 |
| 170 | 3 | 115,600 | 0.1102 | 0.3307 |
| 204 | 3 | 166,464 | 0.1588 | 0.4763 |
| 341 | 3 | 465,124 | 0.4436 | 1.3307 |
| 409 | 3 | 669,124 | 0.6381 | 1.9144 |
| 682 | 3 | 1,860,496 | 1.7743 | 5.3229 |

Together, the 18 U matrices contain 2,488,818 f32 values: 9,955,272 bytes, or
9.494 MiB raw. Encoding each U separately in base64 requires approximately
12.659 MiB. For comparison, all row and column index vectors together are only
45,792 bytes (44.7 KiB) raw.

The complete unrounded measurements are in `cur_stage_results.csv`, and the
suite's machine-readable aggregate is in `cur_stage_summary.json`.
