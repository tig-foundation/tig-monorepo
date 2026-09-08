# L40 8000x8000 sketchy_v2 CUR run

This is the audited result for C3 job `job_1788444962508_r5ezni`, run on one
NVIDIA L40 with deterministic seed 0 on September 3, 2026. It uses the
index-only CUR design: the algorithm may calculate the canonical fast linking
matrix while searching, but the submitted solution contains only row and
column indices and the verifier independently reconstructs `U`.

`sketchy_v2` first reproduces the three-trial `sketchy` search, retaining its
best solution as a fallback. It then generates five candidates with a `k + 64`
block-Krylov sketch, four power iterations, up to 48 max-volume swaps, and a
1.001 tolerance. Finally it evaluates the cross-product of the strongest row
and column selections using the verifier's canonical fast-`U` residual.

## Comparison with sketchy v1

Both runs use the same matrix dimensions, spectrum, seed, generator, and
verifier implementation.

| Measurement | sketchy v1 | sketchy_v2 | Change |
|---|---:|---:|---:|
| TIG integer quality | 707,592 | 713,358 | +5,766 |
| Mean continuous score | 0.707592197 | 0.713357999 | +0.005765801 |
| Solve all eight subinstances | 19,813.107 ms | 76,548.478 ms | 3.86x |
| Verify all eight solutions | 213.018 ms | 212.623 ms | -0.19% |
| Serialized JSON solution | 24,693 bytes | 24,729 bytes | +36 bytes |

Every subinstance improved:

| Sub | Target k | v1 ratio | v2 ratio | Ratio reduction | v1 score | v2 score |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 145 | 36.635709 | 32.433614 | 11.47% | 0.580526663 | 0.588883757 |
| 1 | 250 | 14.587277 | 12.253190 | 16.00% | 0.673375642 | 0.687995064 |
| 2 | 170 | 2.730347 | 2.609576 | 4.42% | 0.836574406 | 0.842778070 |
| 3 | 616 | 83.830380 | 82.058304 | 2.11% | 0.591953899 | 0.593121460 |
| 4 | 443 | 5.356067 | 5.059437 | 5.54% | 0.784124241 | 0.789913377 |
| 5 | 165 | 2.509382 | 2.446625 | 2.50% | 0.847474642 | 0.851048007 |
| 6 | 619 | 7.206676 | 7.031390 | 2.43% | 0.765012264 | 0.767260120 |
| 7 | 1,311 | 174.639388 | 159.958118 | 8.41% | 0.581695822 | 0.585864133 |

The checked-in JSON report has SHA-256
`7555e6dbc8bcdb4df91a71168b57a3ccd41b8e1bb47273089e0f8fdd7408ce3e`.
