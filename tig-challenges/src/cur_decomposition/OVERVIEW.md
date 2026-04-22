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

The solver must choose which columns (`c_idxs`) and rows (`r_idxs`) to select, and compute the linking matrix (`u_mat`). Quality is measured by how close the reconstruction error `||A - CUR||_F` is to the theoretical rank-k optimum.

## Differences from Standard TIG Challenges

Standard TIG challenges (c001-c006) follow a common pattern:

1. **One instance per nonce**: Each nonce generates a single challenge instance. The solver produces one solution, which is evaluated for a single quality score.
2. **Generic dispatch**: The runtime and verifier use a shared `dispatch_challenge!` macro.
3. **Single fuel budget**: The solver gets the full fuel budget for its one instance.
4. **Direct solution serialization**: The solution is a single struct.

c008 (CUR decomposition) differs:

1. **18 sub-instances per nonce**: Each nonce generates 5 matrices (from a single QR factorization), mapped to 18 (matrix, target_k) pairs. The solver is called 18 times per nonce, once per sub-instance, with no information transfer between calls.
2. **Custom dispatch blocks**: Both `tig-runtime` and `tig-verifier` have hand-written c008 match arms instead of the generic `dispatch_challenge!` macro — needed for the sub-instance loop, per-instance fuel reset, and multi-solution aggregation.
3. **Split fuel budget**: Total fuel divided as `max_fuel / 18`. GPU and CPU fuel counters reset between sub-instances.
4. **Two CUDA streams**: Matrix generation runs on the default (non-fuel-checked) stream; the solver runs on a fuel-checked stream.
5. **Aggregate quality**: Per-sub-instance scores are combined via **arithmetic mean**. A bad sub-instance only tanks its own slot; it does not invalidate the nonce.
6. **Multi-solution output**: The serialized solution is a JSON array of 18 `Solution` structs.

## Multi-Instance Design

Each nonce produces **18 sub-instances** from **5 matrices**, all from a single expensive QR factorization.

### Track Parameters

```rust
Track { n: i32, m: i32, poly: bool }
```

Constraint: `τ = min(m, n) = 2^p + 1` for some `p ≥ 5` (e.g., 33, 65, 129, 257, 513, ...).

`poly` selects the singular-value decay profile (exponential vs polynomial).

### Matrix Generation

Constants:
- `τ = min(m, n) = 2^p + 1`
- coherence `Δ = 10000.0` (passed as `max_d` to `gaussian_matrix_kernel`)
- exponential scale `L = 13.0` (in code; called `a` in the paper)

1. **Gaussian + coherence scaling**. Sample `G_U ∈ R^{m×τ}`, `G_V ∈ R^{n×τ}` with iid N(0,1), and column-scale by a linear interpolation `d_j` from `d_0 = Δ` down to `d_{τ-1} = 1`. This is baked into `gaussian_matrix_kernel`.

2. **QR factorization**. `U, _ = qr(G_U · D)`, `V, _ = qr(G_V · D)` via cuSolver (`Sgeqrf` + `Sorgqr`).

3. **Singular values** (length τ, descending, no shuffle).

   Exponential (`poly = false`):
   ```
   σ_j = exp( -L · sqrt(j+1) / sqrt(τ+1) )
   ```

   Polynomial (`poly = true`):
   ```
   σ_j = c / ( (j/τ)^2 + c ),   c = exp(-L) / (1 - exp(-L))
   ```
   giving σ_0 = 1 and σ_{τ-1} ≈ exp(-L).

4. **Dyadic nested index sets**. The paper defines levels `i ∈ {0..4}` with strides `2^(4-i)`:
   ```
   paper I_i = { 0, 2^(4-i), 2·2^(4-i), …, τ-1 },   |I_i| = 2^(p-4+i) + 1
   ```
   so paper `I_0` is coarsest (stride 16) and paper `I_4` is finest (stride 1).

   **Repo convention.** The code indexes the other way: `STRIDES = [1, 2, 4, 8, 16]` with repo matrix 0 being the largest. The mapping is:
   ```
   paper A_i           ↔  repo matrix index (4 - i)
   paper I_i stride 2^(4-i)  ↔  repo STRIDES[4 - i]
   ```

5. **Incremental construction** (building from smallest paper-rank to largest, i.e. repo index 4 down to 0):
   ```
   A_0 (paper) = Σ_{j ∈ I_0} σ_j · u_j v_j^T
   A_i (paper) = A_{i-1} + Σ_{j ∈ I_i \ I_{i-1}} σ_j · u_j v_j^T     for i = 1..4
   ```
   GEMM with `beta = 1.0` on the running accumulator. In the repo this is the `for level in (0..NUM_MATRICES).rev()` loop in `generate_multiple_instances`.

### Sub-Instance Mapping (18 sub-instances)

Define three disjoint 2-element target-rank sets for `k ∈ {0, 1, 2}`:
```
C_k = { ⌊2^(p-4+k) / 3⌋, ⌊2^(p-4+k) / 5⌋ }
```

Paper indexing:

| paper matrix | rank ≈ | targets       | count |
|--------------|--------|---------------|-------|
| A_0          | 2^(p-4)+1 | C_0         | 2     |
| A_1          | 2^(p-3)+1 | C_0 ∪ C_1   | 4     |
| A_2          | 2^(p-2)+1 | C_0 ∪ C_1 ∪ C_2 | 6 |
| A_3          | 2^(p-1)+1 | C_1 ∪ C_2   | 4     |
| A_4          | 2^p+1     | C_2         | 2     |
| **Total**    |        |               | **18** |

Each element of `C_k` is shared across exactly 3 matrices, so every target rank appears 3 times paired with different true ranks.

Translated to the repo (matrix 0 = largest = paper A_4), `TARGET_RANK_COUNTS = [2, 4, 6, 4, 2]`:

| repo matrix | stride | ≈ paper | targets                 | count |
|-------------|--------|---------|-------------------------|-------|
| 0           | 1      | A_4     | C_2                     | 2     |
| 1           | 2      | A_3     | C_1 ∪ C_2               | 4     |
| 2           | 4      | A_2     | C_0 ∪ C_1 ∪ C_2         | 6     |
| 3           | 8      | A_1     | C_0 ∪ C_1               | 4     |
| 4           | 16     | A_0     | C_0                     | 2     |

Since `⌊2^(p-4)/k⌋ < ⌊2^(p-3)/k⌋ < ⌊2^(p-2)/k⌋` for `p ≥ 5`, all six values across `C_0, C_1, C_2` are distinct, so no deduplication is needed.

### Optimal Error

For each sub-instance `(matrix, target_k)`, the optimal rank-`target_k` Frobenius error is closed-form:
```
optimal_fnorm = sqrt( Σ_{j ∈ I_i, rank(σ_j) ≥ target_k+1} σ_j^2 )
```
i.e. restrict σ to the indices of matrix `i`, sort descending, drop the top `target_k`, and take the Frobenius norm of the tail. Stored privately on each `Challenge` and used only at evaluation time.

## Challenge API

### Generation

```rust
let challenges: Vec<Challenge> = Challenge::generate_multiple_instances(
    &seed, &track, module, stream, &prop
)?;  // length = 18
```

One QR factorization, 5 matrices, 18 `Challenge` instances.

### Challenge struct

```rust
pub struct Challenge {
    pub seed: [u8; 32],
    pub n: i32,
    pub m: i32,
    pub target_k: i32,
    pub d_a_mat: CudaSlice<f32>, // m x n, column-major
    // optimal_fnorm is private — used only for verification
}
```

### Evaluation

```rust
let fnorm: f32 = challenge.evaluate_fnorm(&solution, module, stream, &prop)?;
let sub_score: f64 = challenge.evaluate_solution(&solution, module, stream, &prop)?;
```

`evaluate_solution` returns the per-sub-instance score in `(0, 1]`. The verifier averages across 18 calls.

## What the Solver Sees

Called **18 times per nonce**. Each call gets one `Challenge` and must return:

```rust
pub struct Solution {
    pub c_idxs: Vec<i32>,   // which columns to select, length target_k
    pub u_mat:  Vec<f32>,   // linking matrix, target_k x target_k (column-major)
    pub r_idxs: Vec<i32>,   // which rows to select, length target_k
}
```

No information transfers between sub-instance calls.

## Fuel Budget

```
fuel_per_instance = max_fuel / 18
```

- Matrix generation uses a non-fuel-checked stream (doesn't consume solver fuel).
- Each sub-instance solver runs on a fuel-checked stream with its own budget.
- If a sub-instance exhausts its fuel, execution moves to the next one.
- The solver should call `save_solution()` early with a "good enough" answer in case fuel runs out.

## Quality Scoring

### Per-sub-instance score

```
r = ||A - CUR||_F / optimal_fnorm          // always >= 1
T = target_k + 1
sub_score =
    1              if r <  T
    T / r          if r >= T
```

`sub_score` is always in `(0, 1]`. A solver that beats the threshold `T` for a given target rank gets a perfect `1.0` on that sub-instance; otherwise it is graded down proportionally.

### Aggregate score

```
score = (1 / 18) · Σ sub_score_i
```

arithmetic mean of the 18 sub-scores; bounded in `(0, 1]`.

Scaled to an integer: `round(score * QUALITY_PRECISION)` with `QUALITY_PRECISION = 1_000_000`.

**There is no "any negative quality invalidates the nonce" rule.** A poor sub-instance simply contributes a low score to the average; it does not poison the others. Validation errors from a malformed `Solution` (wrong lengths, out-of-bounds indices) still invalidate the nonce.

## Runtime Flow (tig-runtime)

1. Parse track, set up CUDA context.
2. Load PTX with fuel limit = `fuel_per_instance * gpu_fuel_scale` (replacing `0xdeadbeefdeadbeef`).
3. Call `Challenge::generate_multiple_instances()` on the non-fuel-checked stream → `Vec<Challenge>` (18 entries).
4. For each challenge:
   a. `initialize_kernel` to reset GPU fuel counter and signature.
   b. Reset CPU fuel counter.
   c. Call the solver's `entry_point` on the fuel-checked stream.
   d. `finalize_kernel` to read GPU fuel/signature.
   e. Accumulate fuel consumed and combine runtime signature.
   f. Store the solution (or an empty one if the solver failed/didn't save).
5. Serialize all 18 solutions as a JSON array into `OutputData`.

## Verification Flow (tig-verifier)

1. Parse track, set up CUDA context.
2. Call `Challenge::generate_multiple_instances()` (deterministic given seed/track).
3. Deserialize solution as `Vec<Solution>` (must be length 18).
4. For each `(challenge, solution)` pair:
   - Call `challenge.evaluate_solution()` → sub_score `f64`.
   - Any validation error → invalid submission.
5. Compute **arithmetic mean** of the 18 sub-scores and scale to `i32` via `QUALITY_PRECISION`.

## Key Files

| File | Purpose |
|------|---------|
| `tig-challenges/src/cur_decomposition/mod.rs` | Challenge generation, evaluation, structs |
| `tig-challenges/src/cur_decomposition/kernels.cu` | CUDA kernels (coherent Gaussian, column/row extraction) |
| `tig-algorithms/src/cur_decomposition/template.rs` | Solver template |
| `tig-runtime/src/main.rs` | Runtime execution (c008 dispatch block) |
| `tig-verifier/src/main.rs` | Solution verification (c008 dispatch block) |
