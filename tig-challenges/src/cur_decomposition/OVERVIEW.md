# CUR Decomposition Challenge (c008)

## Problem Statement

Given a matrix **A** (m x n), find a CUR decomposition that approximates A:

```
A ~ C * U * R
```

where:
- **C** = selected columns of A (m x k)
- **U** = a linking matrix (k x k)
- **R** = selected rows of A (k x n)
- **k** = `target_k`, the number of columns/rows to select

The solver must choose which columns (`c_idxs`) and rows (`r_idxs`) to select, and compute the linking matrix (`u_mat`). Quality is measured by how small the reconstruction error `||A - CUR||_F` (Frobenius norm) is relative to the theoretical optimum.

## Differences from Standard TIG Challenges

Standard TIG challenges (c001-c006) follow a common pattern:

1. **One instance per nonce**: Each nonce generates a single challenge instance. The solver produces one solution, which is evaluated for a single quality score.
2. **Generic dispatch**: The runtime and verifier use a shared `dispatch_challenge!` macro that handles instance generation, solver invocation, and evaluation uniformly.
3. **Single fuel budget**: The solver gets the full fuel budget for its one instance.
4. **Direct solution serialization**: The solution is a single struct, serialized directly into OutputData.

c008 (CUR decomposition) differs in several ways:

1. **13 sub-instances per nonce**: Each nonce generates 5 matrices (from a single QR factorization), mapped to 13 (matrix, target_k) pairs. The solver is called 13 separate times, once per sub-instance, with no information transfer between calls.
2. **Custom dispatch blocks**: Both `tig-runtime` and `tig-verifier` have hand-written c008 match arms instead of using the generic `dispatch_challenge!` macro. This is needed to handle the sub-instance loop, per-instance fuel reset, and multi-solution aggregation.
3. **Split fuel budget**: The total fuel is divided evenly (`max_fuel / 13`), and each sub-instance gets its own independent budget. GPU fuel counters and CPU fuel counters are reset between sub-instances.
4. **Two CUDA streams**: Matrix generation runs on the default (non-fuel-checked) stream so it doesn't consume solver fuel. The solver runs on a fuel-checked stream.
5. **Aggregate quality**: Instead of a single quality score, per-instance qualities are combined via geometric mean. Any sub-instance with negative quality invalidates the entire nonce.
6. **Multi-solution output**: The serialized solution is a JSON array of 13 `Solution` structs (one per sub-instance), rather than a single solution.
7. **`MultiChallenge` container**: Generation produces a `MultiChallenge` struct holding 5 GPU matrices and 13 sub-instance descriptors, from which individual `Challenge` views are created on-the-fly for the solver.

## Multi-Instance Design

Each nonce produces **13 sub-instances** from **5 matrices**, all generated from a single expensive QR factorization. This amortizes the QR cost across many problems.

### Track Parameters

```rust
Track { n: i32, m: i32 }
```

Constraint: `min(m, n)` must equal `2^p + 1` for some `p >= 5` (e.g., 33, 65, 129, 257, ...).

### Matrix Generation

1. **QR Factorization** (one-time cost): Generate two random Gaussian matrices and extract orthogonal bases U (m x max_rank) and V (n x max_rank) via QR.

2. **Singular values**: Generate `max_rank` values using `sigma_j = exp(-2.0 * sqrt(j+1) / sqrt(max_rank))`, then shuffle with a seeded RNG. This produces a spectrum of decaying singular values.

3. **Dyadic index sets**: Build 5 nested subsets of `{0, 1, ..., 2^p}` using strides:

   | Matrix | Stride | Index set                     | Rank          |
   |--------|--------|-------------------------------|---------------|
   | 0      | 1      | {0, 1, 2, 3, ...}             | 2^p + 1       |
   | 1      | 2      | {0, 2, 4, 6, ...}             | 2^(p-1) + 1   |
   | 2      | 4      | {0, 4, 8, 12, ...}            | 2^(p-2) + 1   |
   | 3      | 8      | {0, 8, 16, 24, ...}           | 2^(p-3) + 1   |
   | 4      | 16     | {0, 16, 32, ...}              | 2^(p-4) + 1   |

   These sets are perfectly nested: I_4 ⊂ I_3 ⊂ I_2 ⊂ I_1 ⊂ I_0.

4. **Incremental matrix construction**: Each matrix A_i = U_i * diag(sigma) * V_i^T, where U_i and V_i are the columns of U and V at the corresponding index set. Built incrementally starting from the smallest:
   - A_4 (smallest rank, stride 16) computed from scratch
   - A_3 = A_4 + contribution from new indices in I_3 \ I_4
   - A_2 = A_3 + contribution from I_2 \ I_3
   - ... and so on up to A_0 (full rank, stride 1)

   This uses GEMM with `beta=1.0` to accumulate new components onto the previous matrix. Each matrix is stored directly at its final index (`matrices[0]` = largest rank through `matrices[4]` = smallest rank).

### Target Rank Mapping

Each matrix is paired with 2-3 target_k values (tau = min(m, n)):

| Matrix | Stride | Approx rank | target_k values              | Count |
|--------|--------|-------------|------------------------------|-------|
| 0      | 1      | tau         | tau/3, tau/5                 | 2     |
| 1      | 2      | tau/2       | tau/3, tau/5, tau/9          | 3     |
| 2      | 4      | tau/4       | tau/5, tau/9, tau/20         | 3     |
| 3      | 8      | tau/8       | tau/9, tau/17, tau/20        | 3     |
| 4      | 16     | tau/16      | tau/17, tau/20               | 2     |
| **Total** |     |             |                              | **13** |

### Optimal Error

For each sub-instance, the optimal Frobenius norm error is computed analytically:

```
optimal_fnorm = sqrt(sum of sigma_i^2 for i >= target_k)
```

where sigma values are sorted by magnitude descending, restricted to the index set of that matrix.

## What the Solver Sees

The solver function is called **13 times per nonce**, once per sub-instance. Each call receives:

```rust
pub struct Challenge {
    pub seed: [u8; 32],
    pub n: i32,
    pub m: i32,
    pub target_k: i32,
    pub d_a_mat: CudaSlice<f32>,  // the matrix A on GPU (m x n, column-major)
}
```

The solver has no knowledge of multi-instance structure. It simply receives a matrix and a target rank, and must produce:

```rust
pub struct Solution {
    pub c_idxs: Vec<i32>,   // which columns to select (length target_k)
    pub u_mat: Vec<f32>,    // linking matrix (target_k x target_k, column-major)
    pub r_idxs: Vec<i32>,   // which rows to select (length target_k)
}
```

**No information can be transferred between sub-instance calls.**

## Fuel Budget

The total fuel budget is split evenly across all 13 sub-instances:

```
fuel_per_instance = max_fuel / 13
```

- Matrix generation uses a non-fuel-checked stream (generation doesn't count toward fuel).
- Each sub-instance's solver runs on a fuel-checked stream with its own budget.
- If a sub-instance exhausts its fuel, execution moves to the next one.
- The solver should call `save_solution()` early with a "good enough" answer in case fuel runs out.

## Quality Scoring

### Per-instance quality

```
quality_i = (baseline - fnorm) / (baseline - optimal_fnorm)
```

where:
- `fnorm` = `||A - C*U*R||_F` (the solver's reconstruction error)
- `optimal_fnorm` = theoretical minimum error for that matrix and target_k
- `baseline = 50 * optimal_fnorm`

If `fnorm` exceeds the baseline for **any** sub-instance, that sub-instance's quality is negative and the **entire nonce submission is invalid**.

### Aggregate quality

The final quality is the **geometric mean** across all 13 sub-instances:

```
quality = exp(mean(ln(quality_i)))
```

This is then scaled to an integer: `round(quality * QUALITY_PRECISION)` where `QUALITY_PRECISION = 1_000_000`.

## Runtime Flow (tig-runtime)

1. Parse track, set up CUDA context
2. Load PTX with fuel limit = `fuel_per_instance * gpu_fuel_scale` (replacing `0xdeadbeefdeadbeef` placeholder)
3. Generate `MultiChallenge` on the non-fuel-checked stream
4. For each of 13 sub-instances:
   a. Launch `initialize_kernel` to reset GPU fuel counter and signature
   b. Reset CPU fuel counter
   c. Create `Challenge` view (device-to-device matrix copy on non-fuel-checked stream)
   d. Call solver's `entry_point` on fuel-checked stream
   e. Launch `finalize_kernel` to read GPU fuel/signature
   f. Accumulate fuel consumed and runtime signature
   g. Store solution (or empty solution if solver failed/didn't save)
5. Serialize all 13 solutions as JSON array into OutputData

## Verification Flow (tig-verifier)

1. Parse track, set up CUDA context
2. Generate `MultiChallenge` (same deterministic generation as runtime)
3. Deserialize solution as `Vec<Solution>` (13 entries)
4. Call `evaluate_solutions()`:
   - For each sub-instance: compute `||A - CUR||_F`
   - Compute per-instance quality, check for negatives
   - Return geometric mean quality

## Key Files

| File | Purpose |
|------|---------|
| `tig-challenges/src/cur_decomposition/mod.rs` | Challenge generation, evaluation, structs |
| `tig-challenges/src/cur_decomposition/kernels.cu` | CUDA kernels (matrix generation, column/row extraction) |
| `tig-algorithms/src/cur_decomposition/template.rs` | Solver template |
| `tig-runtime/src/main.rs` | Runtime execution (c008 dispatch block) |
| `tig-verifier/src/main.rs` | Solution verification (c008 dispatch block) |

## Open Review Items

1. **`finalize_kernel` on fuel-checked stream after fuel exhaustion**: `finalize_kernel` is launched on `solve_stream` (fuel-checked). If the solver exhausted fuel via `asm("trap")`, the stream may be in a bad state and `finalize_kernel` could fail to run. Need to verify that trap doesn't poison the stream for subsequent kernel launches. See `tig-runtime/src/main.rs` c008 dispatch block and `tig-binary/src/framework.cu`.

2. **GPU memory lifecycle of `challenge_view` allocations**: Each iteration of the runtime loop creates a `challenge_view` with an m x n `CudaSlice`. If cudarc's `Drop` for `CudaSlice` is deferred or pooled rather than immediate, GPU memory could accumulate across the 13 iterations. Worth testing with large matrices.

3. **`Vec<Solution>` serde round-trip**: The runtime serializes `Vec<c008::Solution>` via `serde_json::to_string`. Each `Solution` uses `impl_base64_serde!` for custom serde (bincode -> gzip -> base64). The verifier deserializes with `serde_json::from_str::<Vec<c008::Solution>>`. Need to verify this round-trips correctly — that the macro's custom `Serialize`/`Deserialize` impls work as expected when nested inside a `Vec` serialized by serde_json.
