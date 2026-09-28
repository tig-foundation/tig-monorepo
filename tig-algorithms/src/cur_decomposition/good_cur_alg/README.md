# good_cur_alg

A quality-first extension of `sketchy`. Registering `pub mod good_cur_alg;` in
the parent `mod.rs` makes it available to the normal TIG build scripts.

From the repository root in the CUR development environment:

```bash
export CHALLENGE=cur_decomposition
build_algorithm good_cur_alg
test_algorithm good_cur_alg 'm=2000,n=3000,poly=false' null --nonces 1 --fuel 1000000000000
```

The fuel value above is a starting allowance, not a measured requirement.
This solver intentionally spends more time and fuel than `sketchy`; the runtime
divides each nonce's fuel among its eight sub-instances. Increase `--fuel` if
necessary, or reduce trials/refinement rounds for a fixed budget.

## Search

1. Run the original `sketchy` solver as an initial incumbent. With the default
   three baseline trials, this uses the original default search unchanged.
2. Build a larger randomized SVD, with QR-stabilized subspace iterations and
   sketch size `k + max(sketch_extra, ceil(k * sketch_ratio))`.
3. Try GPU pivoted QR on both the leading singular-vector embeddings and the
   larger embeddings weighted by singular values. Pivots account for directions
   already selected instead of simply taking the largest leverage scores.
4. Try mixing the new row/column sets with the incumbent's opposite side.
5. Alternate conditional row and column selection. Each greedy pivot maximizes
   the captured energy of the opposite side's current projection in the
   compressed SVD space. Recompute target cross-products at every pivot to
   avoid cancellation in repeated rank-one downdates. Deflate twice, with
   double-precision reductions, and stop refinement when neither side improves.

Every proposed replacement is checked with `Challenge::evaluate_fast_fnorm`.
Only a finite, strictly lower residual replaces the incumbent. Thus a completed
default run retains at least the baseline quality; actual improvement and
runtime still need measurement on the target GPU. A limited fuel budget can
prevent the search from completing.

The baseline Rust implementation is reused from `../sketchy/mod.rs`; its three
CUDA kernels are also present locally because `build_ptx` concatenates only
the chosen algorithm directory's CUDA files. The original `sketchy` is unchanged.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `baseline_trials` | 3 | Original sketchy trials; 0 disables the baseline floor |
| `num_trials` | 2 | New randomized SVD searches |
| `sketch_extra` | 64 | Minimum extra sketch columns |
| `sketch_ratio` | 0.5 | Extra sketch columns as a fraction of k |
| `power_iters` | 3 | QR-stabilized subspace iterations |
| `refinement_rounds` | 3 | Maximum alternating row/column passes per trial |

For a longer search:

```bash
test_algorithm good_cur_alg 'm=2000,n=3000,poly=false' \
  '{"num_trials":4,"sketch_extra":128,"sketch_ratio":1.0,"power_iters":4,"refinement_rounds":6}' \
  --nonces 1 --fuel 1000000000000
```

## Paired quality benchmark

The comparison example runs both solvers on identical sub-instances, prints
CSV residuals/scores/times and aggregate protocol quality, and checks that
saved candidates and final results do not regress. It allows a small numerical
tolerance in assertions and does not enforce runtime fuel. The first solve may
include CUDA/library startup overhead, so timing is approximate.

```bash
cargo run --release -p tig-algorithms --features cur_decomposition \
  --example good_cur_compare -- \
  tig-algorithms/lib/cur_decomposition/ptx/good_cur_alg.ptx \
  'm=2000,n=3000,poly=false' 3
```

Use any of the five official track strings with this command.

Local validation: Rust compile checks (CUDA 12.6 bindings), NVRTC compilation
of the new selection kernels, threaded CPU emulation of those kernel bodies,
and independent least-squares checks of the pivot/conditional-objective math.
The math checks can be rerun with
`python3 experiments/good_cur_alg/check_selection_math.py` (requires NumPy).
No end-to-end CUDA quality measurements were available in the implementation
workspace because it had no working NVIDIA driver.
