# CUR decomposition implementation walkthrough

The authoritative design is in docs/cur.tex. This walkthrough maps that design
onto challenge ID c008.

## Per-nonce flow

One nonce produces eight CUR sub-instances. The runtime calls the innovator once
per sub-instance, collects eight solutions, and the verifier returns the
arithmetic mean of eight continuous quality scores.

Each sub-instance asks for an approximation

~~~
A ≈ C U R
~~~

at target rank k. The innovator returns k distinct column indices, k distinct
row indices, and a finite k × k float32 linking matrix U.

## Shared-basis generation

For a track (m, n, poly), generation sets tau = min(m, n) and uses fixed design
constants delta = 10,000, spectral parameter a = 13, and singular-value noise
of plus or minus 15%.

The nonce seed generates Gaussian pools with shapes m × tau and n × tau. Two QR
factorizations produce orthonormal bases shared by all eight sub-instances.
The production Gaussian kernel uses bounded-grid Philox streams so large
matrices do not pay one random-state initialization per entry.

The true-rank ratios use eight strata spanning 0.03 through 0.22. Independently,
the target-rank ratios use eight strata spanning 0.10 through 0.80 and are
shuffled before pairing with true ranks.

Every sub-instance then independently samples its singular-vector index set,
perturbs its spectrum, and randomly re-pairs its selected left and right
singular vectors. The resulting matrix is formed by one GPU matrix
multiplication. The matrices can share some singular directions, but none is a
nested prefix of another.

The generator sorts the known perturbed singular values and stores

~~~
sqrt(sum(sigma[k..]²))
~~~

as hidden verification data. No verifier-side SVD is necessary.

## Solving

The innovator receives one Challenge at a time:

~~~rust
pub struct Challenge {
    pub seed: [u8; 32],
    pub n: i32,
    pub m: i32,
    pub target_k: i32,
    pub d_a_mat: CudaSlice<f32>,
}
~~~

and returns:

~~~rust
pub struct Solution {
    pub c_idxs: Vec<i32>,
    pub u_mat: Vec<f32>,
    pub r_idxs: Vec<i32>,
}
~~~

Constructing the linking matrix is explicitly part of the innovator's
algorithm. The verifier does not reconstruct U.

The runtime divides the total nonce fuel equally among eight calls. It resets
CPU fuel, GPU fuel, and the runtime signature before every sub-instance, while
retaining the last solution saved for that sub-instance.

## Verification

For each solution, the verifier checks:

- exactly k row and column indices;
- all indices in bounds and distinct;
- exactly k² linking-matrix values; and
- every linking-matrix value is finite.

It extracts C and R from A, forms C U R with GPU matrix multiplications, and
computes the Frobenius residual.

For q = residual / optimal_residual, verification computes

~~~
z = ln(max(q, 1)) / ln(k + 1)
score = 1 / (1 + z)
~~~

SVD quality has score 1, while q = k + 1 has score 0.5. Scores remain
continuous rather than clipping every result above a threshold to the same
value. Eight sub-scores are averaged and scaled by QUALITY_PRECISION.

## Integration points

- tig-challenges/src/cur_decomposition/mod.rs: tracks, generation, validation,
  and GPU residual
- tig-challenges/src/cur_decomposition_scoring.rs: scoring and aggregation
- tig-challenges/src/cur_decomposition/kernels.cu: generation and extraction
  kernels
- tig-runtime/src/main.rs: eight solver calls and fuel accounting
- tig-verifier/src/main.rs: regeneration and aggregate verification
- tig-algorithms/src/cur_decomposition/template.rs: innovator API
