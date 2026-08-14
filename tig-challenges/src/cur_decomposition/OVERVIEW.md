# CUR decomposition challenge

This challenge asks an innovator to construct a CUR approximation

~~~
A ≈ C U R
~~~

for each of eight related matrix/target-rank sub-instances. For target rank
k, the algorithm must return exactly k distinct column indices, exactly k
distinct row indices, and a finite k × k float32 linking matrix. The selected
columns and rows define C and R; constructing U is part of the innovator's
work.

The authoritative mathematical specification is docs/cur.tex.

## Tracks

The initial tracks are:

| m | n | spectrum |
|---:|---:|---|
| 2000 | 3000 | exponential |
| 4000 | 4000 | exponential |
| 2000 | 2000 | polynomial |
| 2000 | 3000 | polynomial |
| 8000 | 8000 | polynomial |

The implementation represents the spectrum choice as poly: bool. The remaining
design constants are fixed:

- number of sub-instances: 8
- column-scaling parameter: delta = 10,000
- spectral-decay parameter: a = 13
- independent singular-value perturbation: Uniform(-0.15, 0.15)

## Instance generation

Let tau = min(m, n). A nonce seed deterministically generates Gaussian matrices
of shapes m × tau and n × tau. After the prescribed column scaling, one QR
factorization of each matrix produces shared orthonormal pools U and V. Those
two expensive QR factorizations are reused by all eight sub-instances.

One true-rank ratio is sampled from each stratum:

~~~
[0.030, 0.055]  [0.055, 0.080]  [0.080, 0.105]  [0.105, 0.130]
[0.130, 0.155]  [0.155, 0.180]  [0.180, 0.200]  [0.200, 0.220]
~~~

For sampled ratio alpha_s, the true rank is r_s = round(alpha_s * tau).

One target-rank ratio is independently sampled from each stratum:

~~~
[0.10, 0.19]  [0.19, 0.28]  [0.28, 0.37]  [0.37, 0.46]
[0.46, 0.55]  [0.55, 0.64]  [0.64, 0.72]  [0.72, 0.80]
~~~

These eight ratios are shuffled before they are paired with the eight true
ranks. The target rank is k_s = round(rho_s * r_s).

For each sub-instance independently:

1. sample r_s singular-vector indices without replacement from the shared
   pools;
2. generate the selected exponential or polynomial base spectrum;
3. perturb every singular value independently by up to 15%;
4. randomly re-pair the selected left and right singular vectors; and
5. form A_s = U[:, I_s] Sigma_s V[:, I_s^pi]^T.

The index sets may overlap between sub-instances. The matrices therefore share
some singular directions but have different ranks, spectra, index sets, and
left/right pairings.

## Challenge and solution API

Each generated sub-instance is exposed as:

~~~rust
pub struct Challenge {
    pub seed: [u8; 32],
    pub n: i32,
    pub m: i32,
    pub target_k: i32,
    pub d_a_mat: CudaSlice<f32>, // column-major m × n matrix
}
~~~

The hidden challenge data also stores the known optimal rank-k Frobenius error.

A solution is:

~~~rust
pub struct Solution {
    pub c_idxs: Vec<i32>, // length k, distinct and in [0, n)
    pub u_mat: Vec<f32>,  // length k², finite, column-major
    pub r_idxs: Vec<i32>, // length k, distinct and in [0, m)
}
~~~

The runtime calls the innovator once for each of the eight sub-instances and
collects the eight solutions into one JSON array. CPU and GPU fuel counters are
reset for every call, and the total nonce fuel budget is divided equally among
the eight calls.

## Verification and scoring

For every valid solution, the verifier extracts

~~~
C = A[:, c_idxs]
R = A[r_idxs, :]
~~~

and computes ||A - C U R||_F on the GPU. Because generation already knows the
perturbed singular values, the optimal rank-k denominator is evaluated without
running another SVD:

~~~
optimal_fnorm = sqrt(sum(sorted_singular_values[k..]²))
~~~

Define

~~~
q = ||A - C U R||_F / optimal_fnorm
z = ln(max(q, 1)) / ln(k + 1)
sub_score = 1 / (1 + z)
~~~

Thus SVD-quality performance scores 1, and the literature threshold q = k + 1
scores 0.5. Unlike the former clipped rule, this score continuously
distinguishes approximations on both sides of the threshold.

The final nonce quality is the arithmetic mean of all eight sub-scores,
multiplied by QUALITY_PRECISION and rounded to an integer. A malformed
sub-solution makes the submitted nonce invalid.

## Runtime integration

Challenge ID c008 has dedicated runtime and verifier dispatch because one nonce
contains eight solver calls and eight returned solutions.

Key files:

| File | Responsibility |
|---|---|
| tig-challenges/src/cur_decomposition/mod.rs | tracks, generation, solution validation, GPU residual |
| tig-challenges/src/cur_decomposition_scoring.rs | continuous sub-score and eight-score aggregation |
| tig-challenges/src/cur_decomposition/kernels.cu | deterministic generation and extraction kernels |
| tig-runtime/src/main.rs | eight-call execution and fuel accounting |
| tig-verifier/src/main.rs | regeneration, validation, and aggregate quality |
| tig-algorithms/src/cur_decomposition/template.rs | innovator-facing template |
