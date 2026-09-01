# CUR design-calibration experiment

This experiment intentionally compares alternative linking-matrix methods for
design research. Those experimental U payloads are not valid TIG challenge
solutions: the production protocol accepts only row and column indices and the
verifier always computes U with its canonical fast QR method.

This is a controlled 2x2 experiment over the CUR challenge's 18 `(true rank,
target rank)` cases. Unlike the production generator, its five matrix groups are
independent: a fresh `A` is generated for every `(m, n, true_rank, spectrum,
seed)` group. Multiple target ranks in the same group intentionally reuse that
one `A`, giving paired comparisons across `k` without the old dyadic nesting.

| Component | Cheap variant | Longer variant |
| --- | --- | --- |
| Row/column selection | Top matrix row/column norms | Block-Krylov randomized SVD, max-volume refinement, and residual-scored restarts |
| U construction | Intersection inverse `U = W^-1` | SVD-based least-squares pseudoinverse `U = C^+ A R^+` |

For each selector and sub-instance, the suite selects indices once, extracts `C`
and `R` once, and reuses those exact objects for both U methods. This isolates U
construction from row/column quality. A stream synchronization brackets every
reported GPU stage.

The CSV has one row per `(seed, spectrum, sub-instance, selector, U method)`.
Generation is split into orthogonal-basis (`Q`) construction, spectral scaling,
the final `Q1 Sigma Q2^T` multiplication, and copies used to pair target ranks.
It also records selection, extraction, U, and verifier timings; `true_rank`,
`target_k`, and their ratio; the unclipped `fnorm / optimal_fnorm`; the protocol
score; raw U bytes; the exact JSON-serialized compressed solution bytes; and any
failure. The JSON file aggregates each of the four experiment cells, including
the raw and serialized size of a complete 18-solution submission.

The sophisticated selector obtains approximate left/right singular-vector
embeddings from one block-Krylov factorization, uses bounded max-volume swaps to
choose well-conditioned rows and columns, and retains the lowest-residual result
over `--sophisticated-restarts`. Its internal U/residual evaluations are part of
`selection_ms`; the two final U methods are still timed independently after the
winning indices have been frozen.

The long U path truncates singular values below `--sv-threshold` times the
largest singular value (default `1e-6`). It deliberately avoids forming or
inverting `C^T C` and `R R^T`, which is unstable for the challenge's
ill-conditioned spectra.

Run locally on a CUDA machine:

```bash
cargo run --release --example cur_stage_experiment --features cur_decomposition -- \
  tig-algorithms/lib/cur_decomposition/ptx/combined.ptx \
  --m 2049 --n 3000 --poly both --seeds 1 \
  --power-iters 1 --sophisticated-restarts 2 --maxvol-swaps 8 \
  --output-dir cur_experiment_results
```

The checked-in C3 job uses an NVIDIA L40 profile and an `8193x8193`
polynomial-spectrum stress track with one measured seed (18 sub-instances).
`8193 = 2^13 + 1` is the valid challenge size nearest to `8000`; CUR tracks
require `min(m, n)` to have the form `2^p + 1`.
Submit it from the
repository root with `c3 deploy`, then retrieve `cur_stage_results.csv` and
`cur_stage_summary.json` with `c3 pull <job-id>`.

`run_c3_bash.sh` runs directly on the C3 VM and installs the CUDA 12.2 compiler
and development libraries before invoking the experiment. Its defaults can be
overridden with `CUR_M`, `CUR_N`, `CUR_POLY`, `CUR_SEEDS`, `CUR_SKETCH_EXTRA`,
`CUR_POWER_ITERS`, `CUR_RESTARTS`, `CUR_MAXVOL_SWAPS`, and
`CUR_MAXVOL_TOLERANCE`. `CUR_CUDA_PACKAGE_SERIES` and `CUR_CUDA_DIRECTORY`
allow a newer toolkit on hosts whose driver supports it.

`selection_ms` and `extraction_ms` are intentionally repeated on both U-method
rows for a selector. They are attribution values for the same shared selection,
not evidence that selection ran twice.

The existing `results/l40s_8193x8193_poly_seed1/` and H100 result directories
are legacy nested-matrix runs and are retained for provenance. New independent-
matrix outputs are named explicitly to avoid mixing the two experiment designs.
The completed large independent-matrix run and analysis are in
`results/l40_8193x8193_independent_poly_seed1/`.
