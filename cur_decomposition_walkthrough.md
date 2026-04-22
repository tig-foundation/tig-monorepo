# CUR Decomposition Challenge (c008) — End-to-End Walkthrough

How a `cur_decomposition` nonce flows from matrix generation to innovator solution to verifier score, grounded in the code in `tig-challenges/src/cur_decomposition/` and `tig-algorithms/src/cur_decomposition/`.

## 1. The problem

Given a dense matrix `A ∈ R^{m×n}`, produce a rank-`k` reconstruction of the form

```
A ≈ C · U · R
```

where `C` is `k` columns selected from `A`, `R` is `k` rows selected from `A`, and `U` is a `k × k` linking matrix the solver computes. Quality is graded against the best possible rank-`k` approximation (truncated SVD), which the challenge knows in closed form.

Unlike c001–c006, a single nonce spawns **18 sub-instances**, each with its own `A` and its own `k`. The solver is called 18 times with no state carried between calls, and the final score is the arithmetic mean of the 18 per-sub-instance scores.

## 2. Track and parameters

`Track` (see `tig-challenges/src/cur_decomposition/mod.rs:17`):

```rust
Track { n: i32, m: i32, poly: bool }
```

- `n`, `m`: matrix shape. The constraint is `τ = min(m, n) = 2^p + 1` for `p ≥ 5` (e.g. 1025, 2049, 4097). Enforced by `validate_max_rank` (`mod.rs:129`).
- `poly`: singular-value decay profile. `false` → exponential, `true` → polynomial. Both decay from `σ_0 = 1` down to roughly `σ_{τ-1} ≈ exp(-L)` with `L = 13.0`.

The 10 official tracks live in `TRACKS` (`mod.rs:42`).

## 3. Instance generation (per nonce)

Driven by `Challenge::generate_multiple_instances` (`mod.rs:400`). Every step runs on a non-fuel-checked CUDA stream — generation cost is not charged to the solver.

### 3a. Orthogonal bases via QR

`generate_uv` (`mod.rs:169`) does, once per nonce:

1. Launch `gaussian_matrix_kernel` (in `kernels.cu`) to fill two scratch buffers with iid `N(0,1)` entries pre-scaled by a coherence profile (`max_d = 10000.0`, linearly interpolated down to 1). Seeds for `U` and `V` come from disjoint halves of the 32-byte nonce seed.
2. Run cuSolver `Sgeqrf` + `Sorgqr` to produce orthogonal `U ∈ R^{m×τ}` and `V ∈ R^{n×τ}`.

This is the most expensive step and is amortized across all 18 sub-instances.

### 3b. Singular values

`generate_scalars` (`mod.rs:63`) builds a length-`τ` descending spectrum:

- Exponential (`poly = false`): `σ_j = exp(-L · √(j+1) / √(τ+1))`
- Polynomial (`poly = true`):  `σ_j = c / ((j/τ)² + c)`, with `c = exp(-L)/(1 - exp(-L))`

Then `U`'s columns are scaled in-place by `σ` via `scale_columns_kernel`. `A_full = U_scaled · V^T` would be the full matrix, but it's never materialized directly — only the dyadic slices below are.

### 3c. Five nested matrices via dyadic index sets

`STRIDES = [1, 2, 4, 8, 16]` (`mod.rs:87`). Repo matrix `i` uses columns of `U`, `V` at indices `{0, STRIDES[i], 2·STRIDES[i], …}`. Matrix 0 is the densest (stride 1, rank ≈ `2^p + 1`); matrix 4 is the coarsest (stride 16, rank ≈ `2^(p−4) + 1`).

They are built incrementally from smallest to largest (the `for level in (0..NUM_MATRICES).rev()` loop at `mod.rs:450`): copy the previous denser approximation, then `cublasSgemm` only the **new** index columns with `beta = 1.0`. This reuses work instead of doing five separate GEMMs.

### 3d. From 5 matrices to 18 sub-instances

For each matrix, target ranks are drawn from three families (`mod.rs:96`):

```
C_k = { ⌊2^(p-4+k) / 3⌋, ⌊2^(p-4+k) / 5⌋ }   for k ∈ {0, 1, 2}
```

`targets_for_repo_matrix` (`mod.rs:103`) assigns them pyramidally:

| repo matrix | stride | target-rank set         | count |
|-------------|--------|-------------------------|-------|
| 0           | 1      | C₂                      | 2     |
| 1           | 2      | C₁ ∪ C₂                 | 4     |
| 2           | 4      | C₀ ∪ C₁ ∪ C₂            | 6     |
| 3           | 8      | C₀ ∪ C₁                 | 4     |
| 4           | 16     | C₀                      | 2     |

Total: **18 `Challenge` structs**, each carrying its own `target_k` and a device copy of the relevant matrix (`mod.rs:604`).

### 3e. Optimal Frobenius norm (the grading target)

For each sub-instance, the best possible rank-`target_k` approximation error is the tail of the **restricted** spectrum, sorted descending, after dropping the top `target_k` entries *by position in the sorted list*.

Let `I_i = { j_0 < j_1 < … < j_{|I_i|-1} }` be the dyadic indices used by matrix `i`. Since the full spectrum `σ_0 ≥ σ_1 ≥ …` is monotonically decreasing, picking σ at those indices preserves the ordering, so the restricted sorted spectrum is just `σ_{j_0} ≥ σ_{j_1} ≥ …`. Then:

```
optimal_fnorm = sqrt( Σ_{ℓ = target_k}^{|I_i| - 1} σ²_{j_ℓ} )
```

i.e. drop the first `target_k` entries of the restricted spectrum and sum squares of what remains.

Concrete example: matrix 4 (stride 16, indices `{0, 16, 32, 48, 64, …}`) with `target_k = 4`. The optimal norm drops `σ_0, σ_16, σ_32, σ_48` and keeps `σ_64, σ_80, …`. It is **not** "keep every `σ_j` with `j > 4` whose index lies in `I_4`" — that would incorrectly retain `σ_16, σ_32, σ_48`.

Computed at `mod.rs:576` (`matrix_scalars[target_k as usize..]` after a descending sort) and stored privately. The solver never sees it.

## 4. What the solver sees

`Challenge` struct (`mod.rs:119`):

```rust
pub struct Challenge {
    pub seed:     [u8; 32],
    pub n: i32, pub m: i32,
    pub target_k: i32,
    pub d_a_mat:  CudaSlice<f32>,   // m × n, column-major, on device
    optimal_fnorm: f32,             // private
}
```

The solver entry point (see `template.rs`):

```rust
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Option<Solution>>
```

is called **18 times** per nonce, once per sub-instance. No state carries between calls.

### Solution shape

```rust
pub struct Solution {
    pub c_idxs: Vec<i32>,   // column indices into A, length target_k
    pub u_mat:  Vec<f32>,   // target_k × target_k, column-major
    pub r_idxs: Vec<i32>,   // row indices into A, length target_k
}
```

### Fuel

`tig-runtime`'s c008 dispatch splits the nonce's `max_fuel` as `fuel_per_instance = max_fuel / 18`. CPU and GPU fuel counters are reset between sub-instances. If a sub-instance runs out of fuel mid-solve, the runtime uses whatever the solver last passed to `save_solution(...)` — so solvers are expected to save a "good enough" answer early and refine it, rather than producing one final result at the end.

Matrix generation runs on a separate CUDA stream that is **not** fuel-checked; only the solver's work on the fuel-checked stream counts.

### What the sample algorithms do

The repo ships five reference solvers in `tig-algorithms/src/cur_decomposition/`:

- `simple_rand` — sample `target_k` rows and columns uniformly at random, solve for `U` via intersection inverse, try `num_trials` times.
- `leverage` — classical leverage-score sampling: probabilities proportional to `‖V_i‖²` of the top-k right singular vectors; falls back to the intersection-inverse shortcut when `cheap_u` is set.
- `sketchy` — sketching-based column/row selection.
- `fast_algo`, `fastest_algo` — stripped-down variants focused on fuel efficiency.

Common pattern: pick column and row indices cheaply, then compute `U = C⁺ · A · R⁺` (or a cheaper proxy like `W⁻¹` where `W` is the intersection block).

## 5. Verification and scoring

### Per-sub-instance evaluation

`evaluate_fnorm` (`mod.rs:625`) validates shapes/bounds, then on GPU:

1. `extract_columns_kernel` builds `C = A[:, c_idxs]`.
2. `extract_rows_kernel` builds `R = A[r_idxs, :]`.
3. Two GEMMs compute `CUR`.
4. `cublasSaxpy` subtracts to form `CUR − A`, then `cublasSnrm2` reads the Frobenius norm.

`evaluate_solution` (`mod.rs:806`) converts that to a score:

```
r = ‖A − CUR‖_F / optimal_fnorm          // always ≥ 1 by optimality of truncated SVD
T = target_k + 1

sub_score = 1          if r <  T
          = T / r      if r ≥  T
```

So a solver that gets within a factor `target_k + 1` of the optimal rank-`k` error is awarded a perfect `1.0` on that sub-instance; otherwise the score decays as `T/r`. `sub_score` is always in `(0, 1]`.

### Aggregation (verifier)

`tig-verifier/src/main.rs` at the `"c008"` arm (line 212):

1. Parse the track, set up CUDA, load the PTX.
2. Call `Challenge::generate_multiple_instances` → deterministic `Vec<Challenge>` of length 18 (the seed/track fully determine it).
3. Deserialize the submitted solution as `Vec<Solution>`; reject if length ≠ 18.
4. For each `(challenge, solution)` pair, call `evaluate_solution`. Any error (shape mismatch, out-of-bounds index, GPU fault) → invalid submission for the whole nonce.
5. Final integer quality:

```
quality = round( (Σ sub_score_i / 18) · 1_000_000 )
```

i.e. arithmetic mean of the 18 sub-scores, scaled by `QUALITY_PRECISION = 1_000_000`. There is **no rule that a single bad sub-instance invalidates the nonce** — it just drags the mean down. Only a malformed `Solution` does.

## 6. Why the design looks this way

- **Amortized generation**: one QR dominates cost; 5 dyadic matrices reuse it.
- **Same target rank against different true ranks**: each `C_k` value is paired with three different matrices, so an algorithm that only works well in a specific rank regime shows up in the aggregate score.
- **Arithmetic mean (not min, not geometric)**: gives partial credit — a solver that nails 17 of 18 sub-instances but tanks one still scores well, encouraging broad-spectrum solvers over specialists.
- **Per-sub-instance fuel reset**: one runaway sub-instance can't starve the others.
- **Early `save_solution`**: fuel-bounded; the solver's job is to monotonically improve a "good enough" baseline rather than bet on a single final answer.

## 7. File map

| File | Role |
|------|------|
| `tig-challenges/src/cur_decomposition/mod.rs` | Generation, per-sub-instance evaluation, `Challenge`/`Solution`/`Track` |
| `tig-challenges/src/cur_decomposition/kernels.cu` | `gaussian_matrix_kernel`, `scale_columns_kernel`, `extract_columns_kernel`, `extract_rows_kernel` |
| `tig-challenges/src/cur_decomposition/OVERVIEW.md` | Spec (paper ↔ repo index mapping, mathematical details) |
| `tig-algorithms/src/cur_decomposition/template.rs` | Solver skeleton innovators start from |
| `tig-algorithms/src/cur_decomposition/{simple_rand,leverage,sketchy,fast_algo,fastest_algo}/` | Reference solvers |
| `tig-runtime/src/main.rs` (c008 arm) | 18× solver loop, fuel split, solution aggregation |
| `tig-verifier/src/main.rs` (c008 arm, line 212) | Regeneration + arithmetic-mean scoring |
