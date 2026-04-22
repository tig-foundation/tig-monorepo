# CUR Decomposition Challenge — Redesign Specification

This document specifies the target design for the TIG CUR Decomposition challenge (c008). Use it to edit the current implementation in `tig-monorepo/`. The current repo implementation and this specification differ in several material ways — those differences are the work items.

## Files to edit

| File | What changes |
|------|--------------|
| `tig-challenges/src/cur_decomposition/mod.rs` | sub-instance count, `TARGET_RANK_MAP`, `TARGET_RANK_COUNTS`, singular-value formula, scoring function |
| `tig-challenges/src/cur_decomposition/OVERVIEW.md` | documentation of the above |
| `tig-runtime/src/main.rs` (c008 dispatch) | fuel split (13 → 18), solution aggregation |
| `tig-verifier/src/main.rs` (c008 dispatch) | aggregation (geometric mean → arithmetic mean of sub-scores); scoring rule |
| `tig-algorithms/examples/test_multi_instance.rs` | mirror the new `TARGET_RANK_MAP`, `TARGET_RANK_COUNTS`, and `NUM_SUB_INSTANCES` |

Do not touch `kernels.cu` — the CUDA kernels (`gaussian_matrix_kernel`, `scale_columns_kernel`, `extract_columns_kernel`, `extract_rows_kernel`) remain as-is. The coherence scaling already lives inside `gaussian_matrix_kernel` via the `max_d` parameter and is correct.

## Summary of changes vs current repo

| Aspect | Current repo | Target design |
|--------|--------------|---------------|
| Sub-instances per nonce | **13** | **18** |
| Target rank scheme | hardcoded denominator list `{3, 5, 9, 17, 20}` | `C_0, C_1, C_2` sets with `⌊2^(p-k)/3⌋` and `⌊2^(p-k)/5⌋` |
| Matrix ↔ target mapping | 5 independent lists | `C_k` sets shared across matrices (pyramidal) |
| σ exponential denominator | `sqrt(rank)` | `sqrt(τ+1)` |
| σ polynomial index | `((j+1)/rank)^2` | `(j/τ)^2` |
| Scoring function | linear `(baseline - fnorm)/(baseline - optimal)` with `baseline = 50·optimal` | threshold at `target_rank+1` on ratio `r = fnorm / optimal`; score `1` or `(target_rank+1)/r` |
| Aggregation | geometric mean; any sub ≤ 0 invalidates nonce | **arithmetic mean** of sub-scores |
| Coherence Δ | implicit in kernel, `max_d = 10000` | keep (document as Δ; `max_d = Δ`) |
| Index direction | matrix 0 = largest (stride 1) | Keep current convention in code; paper uses opposite. Document the mapping explicitly. |

---

## Problem

Given `A ∈ R^{m×n}`, find `C ∈ R^{m×c}` (actual columns of A), `R ∈ R^{r×n}` (actual rows of A), and a linking matrix `U ∈ R^{c×r}` such that `A ≈ C U R`. For this challenge `c = r = target_rank`.

## Track parameters

Track struct unchanged:
```rust
Track { n: i32, m: i32, poly: bool }
```
with `min(m, n) = 2^p + 1` and `p ≥ 5`. All 10 existing `(m, n, poly)` tracks stay.

## Instance generation

Constants: `τ = min(m, n) = 2^p + 1`, coherence `Δ = 10000.0`, exponential scale `a = 13` (keep the current `L = 13`; rename to `a` in docs for consistency with the paper, or leave).

1. **Gaussian + coherence**. Sample `G_U ∈ R^{m×τ}`, `G_V ∈ R^{n×τ}` with iid `N(0,1)` entries. Column-scale by `D = diag(d_0, …, d_{τ-1})` with `d_0 = Δ`, `d_{τ-1} = 1`, and `d_j` linearly interpolating — exactly what `gaussian_matrix_kernel` already does.

2. **QR**. `U, _ = qr(G_U · D)` and `V, _ = qr(G_V · D)`. Unchanged from the current implementation (cuSolver `Sgeqrf` + `Sorgqr`).

3. **Singular values**. For `j ∈ {0, …, τ-1}`:

   - **Exponential** (`poly = false`):
     ```
     σ_j = exp( -a · sqrt(j+1) / sqrt(τ+1) )
     ```
     Note `sqrt(τ+1)`, not `sqrt(rank)`. Replace current `(-L * ((j + 1) as f32).sqrt() / t.sqrt()).exp()` where `t = rank`, with denominator `((τ + 1) as f32).sqrt()`. In `generate_multiple_instances` these are equal since `rank = τ`, but the function should take `τ` not `rank`.

   - **Polynomial** (`poly = true`):
     ```
     σ_j = c / ( (j/τ)^2 + c ),   c = -e^{-a} / (e^{-a} - 1) = e^{-a} / (1 - e^{-a})
     ```
     Note `(j / τ)`, not `((j+1) / rank)`. This changes the σ_0 endpoint to `c / c = 1` (versus the current ≈ 1/(1/τ²+1)) and leaves σ_{τ-1} ≈ e^{-a}.

4. **Dyadic nested index sets**. For levels `i ∈ {0, 1, 2, 3, 4}`, with stride `s_i = 2^(4-i)`:
   ```
   I_i = { 0, s_i, 2·s_i, …, τ-1 }
   ```
   so `|I_i| = 2^(p-4+i) + 1`. Matches paper: `I_0` is coarsest (stride 16), `I_4` is finest (stride 1).

   **Repo convention note.** The current code indexes the other way: `STRIDES = [1, 2, 4, 8, 16]` with `matrix 0` being the largest. Keep the repo convention for internal variables, but make the mapping to the paper explicit in `OVERVIEW.md`:
   ```
   paper A_i  ↔  repo matrix index (4 - i)
   paper I_i stride 2^(4-i)  ↔  repo STRIDES[4 - i]
   ```

5. **Incremental matrix construction** (same idea as current repo; adjust direction comments if needed):
   ```
   A_0 = Σ_{j ∈ I_0} σ_j · u_j v_j^T
   A_i = A_{i-1} + Σ_{j ∈ I_i \ I_{i-1}} σ_j · u_j v_j^T     for i = 1, 2, 3, 4
   ```
   where `u_j` and `v_j` are columns of `U` and `V`. GEMM with `beta = 1` on the running accumulator, as now.

## Sub-instance mapping (18 sub-instances)

Let
```
C_0 = { ⌊2^(p-4) / 3⌋, ⌊2^(p-4) / 5⌋ }   // associated with matrices of rank ≈ 2^(p-4)
C_1 = { ⌊2^(p-3) / 3⌋, ⌊2^(p-3) / 5⌋ }
C_2 = { ⌊2^(p-2) / 3⌋, ⌊2^(p-2) / 5⌋ }
```

```
target_rank_map(0) = C_0                    // 2 targets
target_rank_map(1) = C_0 ∪ C_1              // 4 targets
target_rank_map(2) = C_0 ∪ C_1 ∪ C_2        // 6 targets
target_rank_map(3) = C_1 ∪ C_2              // 4 targets
target_rank_map(4) = C_2                    // 2 targets
```
(paper indexing: matrix `i` has rank `2^(p-4+i) + 1`)

Total = **18 sub-instances**. Each element of `C_k` is shared across exactly 3 matrices, so every target rank occurs 3 times paired with different true ranks.

In the repo (with its opposite indexing), this becomes:
```rust
pub const NUM_SUB_INSTANCES: usize = 18;

// Repo strides (matrix 0 = largest). Paper's A_i ↔ repo matrix (4 - i).
const STRIDES: [i32; NUM_MATRICES] = [1, 2, 4, 8, 16];

// TARGET_RANK_COUNTS for repo matrix 0..4 (= paper A_4..A_0):
// repo 0 (paper A_4) → C_2:           2
// repo 1 (paper A_3) → C_1 ∪ C_2:     4
// repo 2 (paper A_2) → C_0 ∪ C_1 ∪ C_2: 6
// repo 3 (paper A_1) → C_0 ∪ C_1:     4
// repo 4 (paper A_0) → C_0:           2
const TARGET_RANK_COUNTS: [usize; NUM_MATRICES] = [2, 4, 6, 4, 2];
```

Instead of a rank-denominator matrix, compute targets directly from `p`:
```rust
fn c_set(p: i32, k: i32) -> [i32; 2] {
    let base = 1i32 << (p - 4 + k); // 2^(p-4+k)
    [base / 3, base / 5]
}

fn targets_for_repo_matrix(p: i32, repo_idx: usize) -> Vec<i32> {
    // repo_idx 0..4 ↔ paper A_{4-repo_idx}
    let c0 = c_set(p, 0);
    let c1 = c_set(p, 1);
    let c2 = c_set(p, 2);
    match repo_idx {
        0 => c2.to_vec(),                                         // 2
        1 => [c1.as_slice(), c2.as_slice()].concat(),             // 4
        2 => [c0.as_slice(), c1.as_slice(), c2.as_slice()].concat(), // 6
        3 => [c0.as_slice(), c1.as_slice()].concat(),             // 4
        4 => c0.to_vec(),                                         // 2
        _ => unreachable!(),
    }
}
```

Deduplication: `C_0`, `C_1`, `C_2` are disjoint (since `⌊2^(p-4)/k⌋ < ⌊2^(p-3)/k⌋ < ⌊2^(p-2)/k⌋` for `p ≥ 5`), so no dedup needed inside a union. Still worth an `assert` on `target_k ≥ 1`.

## Optimal denominator (already closed-form)

For sub-instance `(A_i, target_rank)` (paper indexing: `A_i` has rank `|I_i|` supported on indices `I_i`):
```
||A_i - SVD_{i, target_rank}||_F = sqrt( Σ_{j ∈ I_i, rank(σ_j) ≥ target_rank} σ_j^2 )
```
i.e. restrict to the σ values at indices in `I_i`, sort them by magnitude descending, drop the top `target_rank`, and take the Frobenius norm of the tail. Current code already does this via `matrix_scalars` — keep the pattern but re-index for 18 instances.

## Scoring

Per sub-instance, with `fnorm = ||A_i - CUR||_F` and `optimal = ||A_i - SVD_{i, target_rank}||_F`:
```
r = fnorm / optimal
sub_score =
    1                                  if r <  target_rank + 1
    (target_rank + 1) / r              if r >= target_rank + 1
```

Sub-score is always in `(0, 1]`. `r < 1` is impossible (CUR cannot beat the best rank-`target_rank` approximation) so `r ≥ 1` always and the piecewise rule is well-defined.

**Aggregate score** (final instance score):
```
score = (1 / 18) · Σ sub_score_i
```
i.e. arithmetic mean of the 18 sub-scores, in `(0, 1]`.

Quality to integer: `round(score * QUALITY_PRECISION)` with `QUALITY_PRECISION = 1_000_000`.

### Removed from current implementation
- No baseline multiplier (`BASELINE_MULTIPLIER = 50.0` — delete).
- No "any negative quality invalidates the nonce" rule. A terrible sub-instance just contributes a near-zero sub-score; it does not poison the others.
- No geometric mean.

### `evaluate_solution` signature
```rust
pub fn evaluate_solution(&self, solution: &Solution, …) -> Result<f64> {
    let fnorm = self.evaluate_fnorm(solution, …)? as f64;
    let optimal = self.optimal_fnorm as f64;
    let r = fnorm / optimal;
    let threshold = (self.target_k as f64) + 1.0;
    let sub_score = if r < threshold { 1.0 } else { threshold / r };
    Ok(sub_score)
}
```
The verifier then averages across 18 calls.

## Fuel

Split evenly: `fuel_per_instance = max_fuel / 18`. Matrix generation still runs on the non-fuel-checked stream. Each sub-instance solver runs on the fuel-checked stream with its own budget; GPU and CPU counters reset between sub-instances. No change to this pattern.

## Solution shape (unchanged)

```rust
pub struct Solution {
    pub c_idxs: Vec<i32>,  // length = target_k, in [0, n)
    pub u_mat:  Vec<f32>,  // target_k × target_k, column-major
    pub r_idxs: Vec<i32>,  // length = target_k, in [0, m)
}
```
Serialized solution is a JSON array of 18 `Solution` entries.

## Asymmetry rationale (unchanged, restate in OVERVIEW)

- Generation amortizes one QR across 5 matrices and 18 sub-instances.
- Verification `optimal_fnorm` is closed-form from σ — no SVD at verify time.
- Solver time per sub-instance is expected to dominate both.

## Test scripts to update

`tig-algorithms/examples/test_multi_instance.rs` hardcodes the 13-sub-instance structure:
```rust
const STRIDES: [i32; 5] = [1, 2, 4, 8, 16];
const TARGET_RANK_COUNTS: [usize; 5] = [2, 3, 3, 3, 2];
const TARGET_RANK_MAP: [[i32; 3]; 5] = [ … ];
```
Replace with the 18-instance `C_k` logic (see snippet above); `TARGET_RANK_COUNTS = [2, 4, 6, 4, 2]`. Keep the rest of the benchmark harness. Also update the `sub_idx_to_true_rank` / `sub_instance_pairs` helpers to iterate the new 18-entry layout.

`tig-algorithms/examples/test_leverage.rs` uses `generate_single_instance` directly with explicit `(true_rank, target_rank)` and is independent of the sub-instance layout — but its local `compute_optimal_fnorm` uses `l5 = 2.0` which is inconsistent with the production `a = 13`. Fix the constant to match, and change the `sqrt(true_rank)` denominator to `sqrt(τ + 1)` where `τ = min(m, n)` to match the new exponential formula. (Or delete the local helper and call the `Challenge`-computed optimal.)

## Checklist

- [ ] `NUM_SUB_INSTANCES = 18` in `mod.rs`
- [ ] `TARGET_RANK_COUNTS = [2, 4, 6, 4, 2]`
- [ ] Replace `TARGET_RANK_MAP` with `C_k` logic driven by `p`
- [ ] `generate_scalars`: exponential denominator `sqrt(τ+1)`; polynomial index `j/τ`
- [ ] Assert `target_k ≥ 1` for every sub-instance
- [ ] Build 18 `Challenge` instances (loop adjusted) with correct `optimal_fnorm` per sub-instance
- [ ] `evaluate_solution` returns per-sub-instance score in `(0, 1]` using the threshold rule; drop `BASELINE_MULTIPLIER`, drop non-positive error
- [ ] Verifier aggregates via arithmetic mean of 18 sub-scores, then scales by `QUALITY_PRECISION`
- [ ] Runtime splits fuel as `max_fuel / 18`, serializes 18 solutions
- [ ] Update `OVERVIEW.md`: 18 sub-instances, new scoring rule, paper↔repo index mapping, formulas
- [ ] Update `test_multi_instance.rs` constants and helpers
- [ ] Fix `test_leverage.rs` optimal helper (constant + denominator) or replace with `Challenge::optimal_fnorm()`
