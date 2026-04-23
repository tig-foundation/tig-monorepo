# ZK Challenge (c009) — R1CS Circuit Optimization

## Overview

The ZK Challenge asks participants to **optimize random R1CS circuits**. Given a baseline circuit C0 (randomly generated from a cryptographic seed), participants produce an optimized circuit C* with fewer constraints that computes the same function. Correctness is proven via Spartan ZK-SNARK proofs evaluated at a random point (Schwartz-Zippel lemma).

**Quality metric**: `epsilon = 1 - K*/K0` where K0 is the baseline constraint count and K* is the optimized constraint count. Higher epsilon = better optimization.

## How It Works

### The Full Pipeline

```
                        INSTANCE GENERATION
                        ===================
  seed + difficulty --> zk::dag (generate_dag) --> DAG --> R1CS matrices (A, B, C) = C0
                                                          ~1000 constraints at delta=1
                                                          ~50% are intentionally removable

                        PARTICIPANT (SOLVER)
                        ====================
  C0 (R1CS matrices) --> optimize() --> C* (fewer constraints, same function)

                        PROOF GENERATION
                        ================
  x_eval = Hash-to-Field(Blake3(C0) || Blake3(C*))    <-- anti-grinding: random eval point

  C0 witness:  compute_witness(DAG, x_eval)            <-- uses DAG (known to system)
  C* witness:  solve_witness_forward(C*, x_eval)       <-- single forward pass (rows must be in topological order)

  pi0    = Spartan.prove(C0, witness_C0)               <-- SNARK proof for baseline
  pi*    = Spartan.prove(C*, witness_C*)               <-- SNARK proof for optimized

                        VERIFICATION (verifier recomputes gens/commitments)
                        ============
  1. K* < K0                          constraint count reduced
  2. y0 == y*                         same outputs (functional equivalence at x_eval)
  3. Recompute gens + commitment      from C0 and C* matrices (not prover-supplied)
  4. Spartan.verify(pi0)              baseline proof valid
  5. Spartan.verify(pi*)              optimized proof valid
```

### Why This Is Sound

The verifier never re-executes either circuit. Instead, it relies on three cryptographic guarantees:

1. **Spartan SNARK proofs** (pi0, pi*): prove that C0 and C* were correctly evaluated at x_eval, producing outputs y0 and y*. The proofs are zero-knowledge: the verifier learns nothing about intermediate values.

2. **Schwartz-Zippel lemma**: if two degree-d polynomials agree at a random point, they are identical with probability >= 1 - d/|F|. Since the Curve25519 scalar field has order ~2^252, a single evaluation gives overwhelming confidence that C0 and C* compute the same function.

3. **Fiat-Shamir anti-grinding**: the evaluation point x_eval is derived from Blake3 hashes of both circuits. The participant cannot choose x_eval to make different functions appear equivalent -- changing C* changes x_eval.

### What the Participant Receives and Returns

**Input**: `SpartanInstance` (defined in `tig-challenges::zk::r1cs`), containing raw R1CS matrices (A, B, C) and dimensions.

```rust
pub struct SpartanInstance {
    pub num_cons: usize,     // number of constraints (rows)
    pub num_vars: usize,     // number of private variables
    pub num_inputs: usize,   // number of public I/O values
    pub A: R1CSMatrix,       // sparse matrix in COO format
    pub B: R1CSMatrix,       // Vec<(row, col, [u8; 32])>
    pub C: R1CSMatrix,       // each [u8; 32] is a Curve25519 scalar
}
```

Each constraint row i encodes: `(A[i] . z) * (B[i] . z) = C[i] . z` where z is the witness vector.

**Output**: An optimized `SpartanInstance` with fewer constraints (`num_cons`) that computes the same function, with rows in **topological evaluation order**.

```rust
pub type OptimizeCircuitFn = fn(&SpartanInstance) -> SpartanInstance;
```

The participant does NOT need to provide a witness generator. The system computes witnesses automatically using `solve_witness_forward`, a single forward-pass solver. **C* rows must be in topological order** — each row may have at most one unknown when reached in sequence. If violated, `solve_challenge` returns an error and the solution is rejected. See the `OptimizeCircuitFn` rustdoc for full details.

### What Optimizations Are Possible

The random circuit generator injects intentional "optimization traps":

| Pattern | R1CS Signature | Optimization | Constraint Savings |
|---------|---------------|--------------|-------------------|
| **Alias** (`out = src`) | A has out, B has 1, C has src | Substitute src for out everywhere, remove row | 1 per alias |
| **Scale** (`out = k * src`) | A has k*src, B has 1, C has out | Absorb k into neighboring constraints, remove row | 1 per scale |
| **Pow5** (`out = src^5`) | 3 rows: src*src=sq, sq*sq=qd, qd*src=out | Algebraic identity reduction | 2 per Pow5 (3 -> 1) |

With default configuration, ~50% of constraints are removable. Advanced optimizations beyond these traps (constraint merging, common subexpression elimination, algebraic simplifications) are also possible.

## Data Structures

### Challenge

```rust
pub struct Challenge {
    pub seed: [u8; 32],                // cryptographic seed for circuit generation
    pub delta: usize,         // delta (1 = ~1000 constraints)
    pub circuit_c0: SpartanInstance,   // baseline R1CS circuit
    pub num_circuit_inputs: usize,     // number of public input signals
    pub num_circuit_outputs: usize,    // number of public output signals
}
```

### Solution

```rust
pub struct Solution {
    pub circuit_star: SpartanInstance,  // optimized circuit C* (rows in topological order)
    pub y0_pub: Vec<Scalar>,            // C0 outputs at x_eval
    pub y_star_pub: Vec<Scalar>,        // C* outputs at x_eval
    pub proof0: SNARK,                  // Spartan proof for C0
    pub proof_star: SNARK,              // Spartan proof for C*
}
```

Generators and commitments are NOT part of the Solution — the verifier recomputes them independently from the circuit matrices.

The Solution does not include Spartan generators or commitments. The verifier recomputes these independently from the circuit matrices it already has (C0 from the Challenge, C* from the Solution). This ensures the verifier derives everything it can and only receives what it cannot compute: the proofs, the optimized circuit, and the outputs.

## Key Functions

### `Challenge::generate_instance(seed, difficulty)`

Generates a random challenge:
1. Converts seed to hex string
2. Calls `dag::generate_dag()` to build a random computation graph (SHA256 seed → ChaCha20 → backward BFS DAG)
3. Converts DAG to Spartan R1CS via `r1cs::dag_to_spartan()`
4. Returns Challenge with the baseline circuit C0

### `solve_challenge(challenge, optimize)`

Produces a Solution:
1. Calls the participant's `optimize()` function: C0 -> C*
2. Computes anti-grinding hash: `x_eval = Hash-to-Field(Blake3(C0) || Blake3(C*))`
3. Computes C0 witness via `compute_witness()` (regenerates DAG from seed)
4. Computes C* witness via `solve_witness_forward()` (single forward pass — **requires rows in topological order**)
5. Generates Spartan SNARK proofs for both circuits
6. Returns Solution with proofs and outputs (no commitments or generators — the verifier recomputes those)

### `Challenge::verify_solution(solution)`

Verifies a Solution. The verifier recomputes everything it can from the circuit matrices and only trusts the proofs, outputs, and optimized circuit from the prover:

1. Recomputes x_eval from circuit hashes
2. Checks K* < K0 (constraint count reduced)
3. Checks y0 == y* (output equivalence)
4. Recomputes Spartan instance, generators, and commitment for C0 from the Challenge
5. Verifies pi0 against the recomputed C0 commitment
6. Recomputes Spartan instance, generators, and commitment for C* from the Solution
7. Verifies pi* against the recomputed C* commitment

## Dependencies

| Library | Version | Role |
|---------|---------|------|
| libspartan | 0.9.0 | ZK-SNARK proving system for R1CS |
| curve25519-dalek | 4.1 | Scalar field arithmetic (Curve25519) |
| rand_chacha | 0.3 | Deterministic ChaCha20 PRNG for DAG generation |
| sha2 | 0.10 | SHA256 seed hashing for PRNG initialization |
| merlin | 3.0 | Fiat-Shamir transcript (non-interactive proofs) |
| blake3 | 1.5.4 | Circuit hashing (anti-grinding commitment) |
| bincode | 1.3 | Circuit serialization for hashing |

## R1CS Background

An R1CS (Rank-1 Constraint System) is a set of equations:

```
(A . z) * (B . z) = C . z
```

where A, B, C are sparse matrices and z is the **witness vector**:

```
z = [ private_vars | 1 | outputs... | inputs... ]
     num_vars        1   num_outputs  num_circuit_inputs
                         <------- num_inputs -------->
```

- **Private variables**: intermediate computation values (hidden from verifier)
- **1**: implicit constant (inserted by libspartan at index num_vars)
- **Outputs**: public circuit outputs (checked for equivalence between C0 and C*)
- **Inputs**: public circuit inputs (the evaluation point x_eval)

The matrices are stored in COO (Coordinate) sparse format: `Vec<(row, col, [u8; 32])>` where each `[u8; 32]` is a Curve25519 scalar in little-endian canonical form.

## Testing

```bash
# Run all tests (fast tests + full Spartan roundtrip)
cargo test --features c007 -p tig-challenges

# With verbose timing output
cargo test --features c007 -p tig-challenges -- --nocapture
```

### Test Descriptions

| Test | Time | What it validates |
|------|------|-------------------|
| `test_generate_instance` | <1s | Circuit generation produces valid dimensions (~1000 constraints at delta=1) |
| `test_deterministic_generation` | <1s | Same seed produces byte-identical R1CS matrices |
| `test_hash_and_xeval_derivation` | <1s | Blake3 hashing is deterministic, x_eval scalars are non-zero |
| `test_witness_satisfies_c0` | <1s | Witness passes libspartan `is_sat()` check: (A*z)*(B*z) = C*z |
| `test_full_identity_roundtrip` | ~66s | **Full pipeline (identity)**: generate challenge, compute witnesses (DAG for C0, forward pass for C*), generate two Spartan SNARK proofs, verify both proofs, check output equivalence |
| `test_alias_optimizer_roundtrip` | ~90s | **Full pipeline (with optimization)**: generate challenge, optimize C0 via `remove_aliases`, compute witnesses, generate proofs, verify K* < K0, verify output equivalence, verify both proofs |
| `test_wrong_order_rejected` | <1s | **Rejection test**: reversed circuit rows trigger `NotInEvaluationOrder` immediately at row 0 |

The identity roundtrip test uses C* = C0 (no optimization) to validate the entire pipeline end-to-end. It exercises every component: circuit generation, hash derivation, witness computation via both methods, Spartan proving, Spartan verification, and output equivalence checking. The only check skipped is K* < K0 (since the circuits are identical).

The alias optimizer roundtrip test exercises the **complete challenge flow with actual optimization**. It uses `remove_aliases` (from `baselines`) to produce a C* with fewer constraints than C0, then runs the full pipeline including the K* < K0 check. This validates that optimized circuits survive the entire proof/verify pipeline and that `solve_witness_forward` succeeds on correctly-ordered compacted circuits.

### Performance Profile (delta=1, ~1000 constraints)

| Operation | Time |
|-----------|------|
| Circuit generation (DAG + R1CS) | ~10ms |
| Alias optimization (remove_aliases) | <1ms |
| Hash derivation + x_eval | <1ms |
| Witness computation (DAG-based) | <1ms |
| Witness computation (R1CS solver) | <1ms |
| Spartan encode + prove (per circuit) | ~30s |
| Spartan verify (per circuit) | ~2s |
| **Total (2 proofs + 2 verifications)** | **~66-90s** |

The bottleneck is Spartan proving (~30s per proof). All other operations are negligible. Verification is ~15x faster than proving. The alias optimizer roundtrip takes slightly longer (~90s) because the verifier recomputes separate generators and commitments for C0 and C*.
