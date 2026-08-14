# H100 result: 4097x4150, one seed

This result was produced by C3 job `job_1786390466681_k6f7ad` on an NVIDIA H100
80 GB HBM3. It covers both challenge spectra, all 18 sub-instances per spectrum,
and all four selector/U combinations: 144 detailed CSV rows. Every timing stage
is bounded by a CUDA stream synchronization.

## Aggregate result across both spectra

| Selector | U method | Selection ms | U ms | Verification ms | Mean ratio | Mean score | Full-score cases |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matrix norms | Intersection inverse | 1.77 | 4.58 | 0.82 | 698.98 | 0.8628 | 27/36 |
| Matrix norms | SVD pseudoinverse | 1.77 | 35.67 | 0.94 | 4.38 | 1.0000 | 36/36 |
| Randomized range | Intersection inverse | 8.72 | 4.58 | 0.82 | 686.85 | 0.8696 | 27/36 |
| Randomized range | SVD pseudoinverse | 8.72 | 35.25 | 0.93 | 4.38 | 1.0000 | 36/36 |

The long selector costs 4.9x the norm selector but changes mean score by only
0.0068 with intersection U and not at all with SVD U. On these generated
matrices, U quality matters much more than selector sophistication.

The robust SVD U costs 7.8x the intersection inverse on average. Across one set
of 18 sub-instances it takes about 0.63-0.66 seconds, versus about 0.082 seconds
for all intersection inverses. Norm selection takes about 0.032 seconds for all
18, while randomized range selection takes about 0.157 seconds.

U timing scales sharply with target rank:

| target k | Intersection inverse ms | SVD pseudoinverse ms |
| ---: | ---: | ---: |
| 51 | 0.14 | 7.67 |
| 85 | 0.39 | 14.66 |
| 102 | 0.62 | 17.83 |
| 170 | 2.48 | 32.33 |
| 204 | 4.01 | 47.22 |
| 341 | 19.85 | 93.05 |

## Interpretation

- Recomputing robust U in the verifier would not be free. At this track size it
  costs roughly 20x more than cheap norm selection, or 4x more than the longer
  randomized selector, across the 18 sub-instances.
- Recomputing intersection U is much cheaper, but its quality is inconsistent.
  The polynomial spectrum's mean score was 0.7904 and one sub-instance scored
  0.0051; the SVD U scored 1 on every sub-instance.
- The score threshold is forgiving: intersection U receives full score in 75%
  of cases even though its mean error ratio is about 699. The raw ratio column
  should therefore remain part of experiments even when protocol score is the
  official objective.
- This is one seed on one GPU model, so these numbers are a controlled timing
  result, not a hardware-independent confidence interval. The suite supports
  more seeds and dimensions for follow-up runs.

The complete measurements are in `cur_stage_results.csv`; machine-readable
aggregates are in `cur_stage_summary.json`.
