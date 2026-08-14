# H100 result: 4097x4150, five seeds

This is the primary result from C3 job `job_1786391310142_fhhkfw` on an
NVIDIA H100 80 GB. It covers five measured seeds, both exponential and
polynomial spectra, all 18 sub-instances, and all four selector/U combinations.
All 720 measured cases succeeded. One sub-instance was run unmeasured as a GPU
warm-up, and every reported GPU stage is bounded by a stream synchronization.

## Aggregate result across both spectra

The score deviation below is the sample standard deviation of the ten
per-set means (five seeds times two spectra), rather than a deviation over the
180 individual sub-instances.

| Selector | U method | Selection ms/sub | U ms/sub | Mean ratio | Mean score + set SD | Full-score cases |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Matrix norms | Intersection inverse | 1.76 | 4.55 | 409.14 | 0.8532 +/- 0.0785 | 130/180 |
| Matrix norms | SVD pseudoinverse | 1.76 | 37.21 | 4.85 | 1.0000 +/- 0.0000 | 180/180 |
| Randomized range | Intersection inverse | 8.70 | 4.54 | 303.28 | 0.8616 +/- 0.0818 | 130/180 |
| Randomized range | SVD pseudoinverse | 8.70 | 37.01 | 4.85 | 1.0000 +/- 0.0000 | 180/180 |

For one complete 18-sub-instance set, the synchronized stage totals were:

| Selector | U method | Selection total ms | U total ms |
| --- | --- | ---: | ---: |
| Matrix norms | Intersection inverse | 31.72 +/- 0.22 | 81.83 +/- 1.56 |
| Matrix norms | SVD pseudoinverse | 31.72 +/- 0.22 | 669.72 +/- 8.09 |
| Randomized range | Intersection inverse | 156.62 +/- 0.36 | 81.74 +/- 1.30 |
| Randomized range | SVD pseudoinverse | 156.62 +/- 0.36 | 666.15 +/- 8.48 |

The robust U computation is therefore about 21 times the cost of cheap norm
selection and 4.3 times the cost of randomized-range selection. The cheap U
computation is about 2.6 times the cost of norm selection, but only half the
cost of randomized-range selection. Robust U timing is stable across the ten
sets: its set-level standard deviation is about 1.2% of its mean.

If U is sent by the solver, the measured residual/scoring verification takes
about 15--17 ms for all 18 cases in normal runs. Recomputing robust U in the
verifier adds about 0.67 seconds. It eliminates a raw U payload of 2,484,324
bytes (2.369 MiB, or about 3.159 MiB after base64 encoding), but it is not an
asymmetric verification step relative to either selector tested here.

## Quality and selector effects

The longer selector costs 4.94 times as much as the norm selector. With the
intersection U it improves mean score by only 0.00836; in the 180 paired cases,
five scores improve, five get worse, and 170 are unchanged. With robust U all
180 pairs have the same full score. On polynomial-spectrum matrices the two
selectors have exactly the same mean score for the cheap U path.

| Spectrum | Selector | Intersection-U score | SVD-U score | Intersection-U full cases |
| --- | --- | ---: | ---: | ---: |
| Exponential | Matrix norms | 0.9043 | 1.0000 | 73/90 |
| Exponential | Randomized range | 0.9210 | 1.0000 | 73/90 |
| Polynomial | Matrix norms | 0.8022 | 1.0000 | 57/90 |
| Polynomial | Randomized range | 0.8022 | 1.0000 | 57/90 |

The protocol score hides some large approximation differences. Intersection U
gets full score in 72.2% of cases even though its mean error ratio is 303--409
and individual ratios reach above 20,000. The raw ratio should remain in future
experiments even if the bounded protocol score is the official objective.

Using the median of the 18 sub-instance scores would remove nearly all
discrimination: nine of the ten weak-U sets have median score exactly 1, and
the remaining set has median 0.998938. All robust-U sets have median 1. The
mean is substantially more informative for this scoring rule.

## U scaling with target rank

These values combine both selectors, spectra, and all five seeds.

| Target k | Intersection inverse ms | SVD pseudoinverse ms |
| ---: | ---: | ---: |
| 51 | 0.14 | 7.96 |
| 85 | 0.39 | 15.17 |
| 102 | 0.61 | 18.51 |
| 170 | 2.42 | 34.37 |
| 204 | 4.04 | 49.47 |
| 341 | 19.67 | 97.16 |

The complete measurements are in `cur_stage_results.csv`; the direct
machine-readable aggregation emitted by the Rust suite is in
`cur_stage_summary.json`.
