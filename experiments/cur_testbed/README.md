# CUR shared-basis testbed

This standalone CUDA experiment implements the design in `docs/cur.tex`. It is
not coupled to the TIG runtime or Docker harness.

Its linking-matrix comparisons are calibration-only. Production TIG solutions
contain only row and column indices; verification computes the canonical fast
U for every sub-instance.

For each seed and spectrum type it generates exactly eight sub-instances from one
shared pair of orthonormal bases. It samples one true-rank ratio from each of the
eight specified strata, samples and shuffles one target-rank ratio from each of
the eight specified target strata, selects independently overlapping singular-
vector index sets, independently perturbs each singular value by a fixed
uniform `+/-15%`, and randomly re-pairs the left and right singular vectors.

The experiment compares two row/column selectors:

- `matrix_norms`: a cheap baseline taking the largest row and column norms.
- `block_krylov_maxvol`: randomized block-Krylov SVD embeddings, max-volume
  refinement, and residual-scored deterministic restarts.

The selected indices are frozen before comparing two linking-matrix methods:

- `qr_least_squares`: the fast method. Thin QR factorizations of C and R^T
  compute the least-squares linking matrix with triangular solves.
- `adaptive_svd_pseudoinverse`: the slower, rank-revealing method. It computes
  U = C^+ A R^+ over several SVD truncation thresholds, measures the actual
  residual for each, and returns the best. The QR result is included as a
  fallback, so this method cannot knowingly return a worse U for the same rows
  and columns.

Every GPU timing is synchronized. `cur_testbed_results.csv` contains one row
per sub-instance, selector, and U method. In particular it records:

- the U-pool and V-pool QR times separately;
- the U Sigma V^T multiplication time for each sub-instance;
- the requested full generation measure, equal to both QR times plus all eight
  matrix-multiplication times;
- complete generation wall time, including Gaussian generation, extraction,
  scaling, allocation, and sampling;
- row/column selection, extraction, U construction, and verification times;
- the raw ratio `||A-CUR||_F / ||A-SVD_k(A)||_F` and thresholded score; and
- raw and actually serialized U/solution sizes.

`cur_testbed_summary.json` contains the non-duplicated generation records and
aggregates for the four selector/U combinations. In each aggregate,
`total_raw_u_bytes` is the sum of the eight raw float32 linking matrices and
`serialized_solutions_bytes` is the complete eight-solution JSON payload,
including row/column indices and serialization overhead.

The standalone generator uses a portable counter-based Philox CUDA kernel for
the two Gaussian basis matrices. Each thread initializes one random stream and
emits batches of four normal values in a grid-stride loop. The grid is capped at
1024 blocks of 256 threads, avoiding one expensive `curand_init` call per matrix
entry while retaining deterministic, independent counter streams. The legacy
challenge kernel remains available for compatibility with existing TIG PTX
artifacts.

Audited L40 runs and their compact reports are in:

- `results/l40_2000x3000_poly_seed1/`
- `results/l40_6000x6000_poly_seed1/`
- `results/l40_7000x7000_poly_seed1/`
- `results/l40_7000x7000_poly_seed1_eight/`
- `results/l40_8000x8000_poly_seed0_verifier_fast_u/`

The 2000x3000, 6000x6000, and first 7000x7000 directories predate the
eight-sub-instance, fixed-15%-noise revision and are retained as historical
six-sub-instance results.

By default, a small unmeasured 64x80 generator and one solver sub-instance warm
the CUDA libraries before recorded work. This prevents one-time cuBLAS and
cuSOLVER initialization from being charged only to the first QR or first matrix
multiplication. Pass `--no-warmup` to measure cold-start behavior instead.

## Local CUDA run

```bash
nvcc --ptx experiments/cur_testbed/portable_kernels.cu \
  --output-file /tmp/cur-testbed-portable.ptx \
  --gpu-architecture compute_70 --use_fast_math --optimize 3

RUSTFLAGS='--cfg feature="cuda-12040"' \
cargo run --release --example cur_testbed --features cur_decomposition -- \
  /tmp/cur-testbed-portable.ptx \
  --m 2000 --n 3000 --poly both --delta 10000 \
  --spectrum-a 13 --seeds 1 \
  --power-iters 1 --sophisticated-restarts 2 --maxvol-swaps 8 \
  --output-dir cur_testbed_results
```

Use `--help` for every option. The dimensions are unrestricted positive matrix
sizes; unlike the legacy challenge, they need not have the form 2^p+1.

## C3 L40 run

The repository `.c3` file points at `run_c3_8000_quality.sh`, which runs the
production-design 8000x8000 polynomial benchmark with index-only solutions and
verifier-computed fast U. Submit with `c3 deploy` and retrieve its JSON report
and portable PTX with `c3 pull <job-id>`. The `run_c3_smoke.sh` and
`run_c3_challenge_smoke.sh` scripts provide smaller end-to-end checks.

The runner compiles the testbed kernel bundle from source for virtual
architecture `compute_70`. This is forward-compatible PTX rather than an
L40-specific binary, and runs on Volta-or-newer NVIDIA GPUs supported by the
CUDA libraries used by the Rust testbed.

## Gaussian-generation benchmark

Two warmed 6000x6000 L40 runs used the same testbed parameters. Replacing one
XORWOW initialization per matrix entry with the batched Philox kernel changed
the generation timings as follows:

| Measure | Legacy kernel | Philox kernel | Change |
|---|---:|---:|---:|
| Gaussian U | 115.722 ms | 0.384 ms | 301.7x faster |
| Gaussian V | 117.024 ms | 0.442 ms | 264.5x faster |
| Both Gaussian matrices | 232.746 ms | 0.826 ms | 281.8x faster |
| Both QR factorizations | 120.409 ms | 121.383 ms | within run variance |
| Six matrix multiplications | 12.899 ms | 12.633 ms | within run variance |
| Complete generation wall time | 377.659 ms | 146.558 ms | 2.58x faster |

The legacy result is C3 job `job_1786687346097_qnryyb`; the optimized result is
`job_1786698292617_ggccxq`. The random-number algorithm changes the particular
Gaussian samples, but not their distribution or the sampled rank metadata, so
solver-quality values are not intended as paired before/after measurements.
