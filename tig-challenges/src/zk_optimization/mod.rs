mod baselines;
mod crypto;
mod dag;
mod r1cs;

use crate::QUALITY_PRECISION;
pub use crypto::CryptoHash;
pub use dag::{CircuitConfig, DAG};
pub use r1cs::{R1CSMatrix, SpartanInstance, WitnessError};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, SNARKGens, VarsAssignment, SNARK};
use merlin::Transcript;

use dag::generate_dag;
use r1cs::{compute_witness, dag_to_spartan, solve_witness_forward};

// =============================================================================
// Data Structures
// =============================================================================

impl_kv_string_serde! {
    Track {
        delta: usize,
    }
}

impl_base64_serde! {
    Solution {
        circuit_star: SpartanInstance,
        y0_pub: Vec<Scalar>,
        y_star_pub: Vec<Scalar>,
        proof0: SNARK,
        proof_star: SNARK,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub delta: usize,
    pub circuit_c0: SpartanInstance,
    pub num_circuit_inputs: usize,
    pub num_circuit_outputs: usize,
}

// =============================================================================
// Solver Interface
// =============================================================================

/// Callback signature for the participant's circuit optimizer.
///
/// Takes the baseline circuit C⁰ (as a [`SpartanInstance`]) and returns C* —
/// an optimized circuit with strictly fewer constraints that computes the same
/// function.
///
/// # Requirements
///
/// ## 1. Fewer constraints
/// `C*.num_cons` must be strictly less than `C⁰.num_cons`.
///
/// ## 2. Functional equivalence
/// C* must compute the same function as C⁰ at the evaluation point `x_eval`
/// derived from `H(C⁰) || H(C*)`.
///
/// ## 3. Topological evaluation order (CRITICAL)
/// Constraint rows must be ordered so that when the witness solver processes
/// row `i`, all variables in A and B are already known from previous rows or
/// from the circuit inputs. At most one variable may be unknown per row.
///
/// Violations are rejected immediately with `NotInEvaluationOrder { row }`.
/// C⁰ is already in topological order — optimizers that only remove or merge
/// constraints naturally preserve it.
pub type OptimizeCircuitFn = fn(&SpartanInstance) -> SpartanInstance;

// =============================================================================
// Helpers
// =============================================================================

fn seed_to_hex(seed: &[u8; 32]) -> String {
    seed.iter().map(|b| format!("{:02x}", b)).collect()
}

fn num_non_zero(si: &SpartanInstance) -> usize {
    si.A.len().max(si.B.len()).max(si.C.len())
}

// =============================================================================
// Challenge Implementation
// =============================================================================

impl Challenge {
    /// Generates a challenge instance from a seed and track.
    ///
    /// `seed → hex → SHA256 → ChaCha20 → backward-BFS DAG → R1CS (C⁰)`
    pub fn generate_instance(seed: &[u8; 32], track: &Track) -> Result<Challenge> {
        let seed_hex = seed_to_hex(seed);
        let config = CircuitConfig::from_delta(track.delta);
        let dag = generate_dag(&seed_hex, &config);
        let circuit_c0 = dag_to_spartan(&dag);

        Ok(Challenge {
            seed: *seed,
            delta: track.delta.clone(),
            num_circuit_inputs: dag.num_inputs,
            num_circuit_outputs: dag.num_outputs,
            circuit_c0,
        })
    }

    /// Verifies a solution against this challenge.
    ///
    /// 1. Recomputes `x_eval = H(H(C⁰) || H(C*))`
    /// 2. Checks `K* < K⁰`
    /// 3. Checks `y⁰_pub == y*_pub`
    /// 4. Recomputes Spartan parameters for C⁰; verifies π⁰
    /// 5. Recomputes Spartan parameters for C*; verifies π*
    conditional_pub!(
        fn evaluate_num_constraints(&self, solution: &Solution) -> Result<usize> {
            // 1. Derive evaluation point
            let h0 = CryptoHash::from_serializable(&self.circuit_c0)?;
            let h_star = CryptoHash::from_serializable(&solution.circuit_star)?;
            let x_eval = h0.combine(&h_star).to_scalars(self.num_circuit_inputs);

            // 2. Constraint reduction
            if solution.circuit_star.num_cons >= self.circuit_c0.num_cons {
                return Err(anyhow!(
                    "C* has {} constraints, must be < {} (C⁰)",
                    solution.circuit_star.num_cons,
                    self.circuit_c0.num_cons
                ));
            }

            // 3. Output equivalence
            if solution.y0_pub != solution.y_star_pub {
                return Err(anyhow!(
                    "Output mismatch: C⁰ and C* produced different outputs"
                ));
            }

            // 4. Verify π⁰
            let c0 = &self.circuit_c0;
            let inst0 = Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C)
                .map_err(|e| anyhow!("Spartan instance for C⁰: {:?}", e))?;
            let gens0 = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
            let (comm0, _) = SNARK::encode(&inst0, &gens0);

            let mut io0: Vec<Scalar> = solution.y0_pub.clone();
            io0.extend_from_slice(&x_eval);
            let io0_bytes: Vec<[u8; 32]> = io0.iter().map(|s| s.to_bytes()).collect();
            let assignment_io0 = InputsAssignment::new(&io0_bytes)
                .map_err(|e| anyhow!("InputsAssignment for C⁰: {:?}", e))?;

            solution
                .proof0
                .verify(
                    &comm0,
                    &assignment_io0,
                    &mut Transcript::new(b"ZKChallenge_C0"),
                    &gens0,
                )
                .map_err(|e| anyhow!("π⁰ verification failed: {:?}", e))?;

            // 5. Verify π*
            let cs = &solution.circuit_star;
            let inst_star =
                Instance::new(cs.num_cons, cs.num_vars, cs.num_inputs, &cs.A, &cs.B, &cs.C)
                    .map_err(|e| anyhow!("Spartan instance for C*: {:?}", e))?;
            let gens_star =
                SNARKGens::new(cs.num_cons, cs.num_vars, cs.num_inputs, num_non_zero(cs));
            let (comm_star, _) = SNARK::encode(&inst_star, &gens_star);

            let mut io_star: Vec<Scalar> = solution.y_star_pub.clone();
            io_star.extend_from_slice(&x_eval);
            let io_star_bytes: Vec<[u8; 32]> = io_star.iter().map(|s| s.to_bytes()).collect();
            let assignment_io_star = InputsAssignment::new(&io_star_bytes)
                .map_err(|e| anyhow!("InputsAssignment for C*: {:?}", e))?;

            solution
                .proof_star
                .verify(
                    &comm_star,
                    &assignment_io_star,
                    &mut Transcript::new(b"ZKChallenge_Cstar"),
                    &gens_star,
                )
                .map_err(|e| anyhow!("π* verification failed: {:?}", e))?;

            Ok(solution.circuit_star.num_cons)
        }
    );

    conditional_pub!(
        fn compute_baseline(&self) -> Result<usize> {
            Ok(baselines::remove_aliases::run(&self.circuit_c0).num_cons)
        }
    );

    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            let num_constraints = self.evaluate_num_constraints(solution)?;
            let baseline_num_constraints = self.compute_baseline()?;
            let quality = (num_constraints as f64 - baseline_num_constraints as f64)
                / (baseline_num_constraints as f64 + 1e-6);
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}

// =============================================================================
// Solver
// =============================================================================

/// Produces a Solution for the challenge using the participant's optimizer.
///
/// 1. Calls `optimize(C⁰)` → C*
/// 2. Derives `x_eval = H(H(C⁰) || H(C*))` (anti-grinding)
/// 3. Computes C⁰ witness via DAG (`compute_witness`)
/// 4. Computes C* witness via single forward pass (`solve_witness_forward`)
/// 5. Generates Spartan proofs π⁰ and π*
pub fn solve_challenge(challenge: &Challenge, optimize: OptimizeCircuitFn) -> Result<Solution> {
    // 1. Optimize
    let circuit_star = optimize(&challenge.circuit_c0);

    // 2. Derive x_eval
    let h0 = CryptoHash::from_serializable(&challenge.circuit_c0)?;
    let h_star = CryptoHash::from_serializable(&circuit_star)?;
    let x_eval = h0.combine(&h_star).to_scalars(challenge.num_circuit_inputs);

    // 3. C⁰ witness (regenerate DAG)
    let seed_hex = seed_to_hex(&challenge.seed);
    let config = CircuitConfig::from_delta(challenge.delta);
    let dag = generate_dag(&seed_hex, &config);
    let (vars0, public_io0) = compute_witness(&dag, &x_eval);

    // 4. C* witness (single forward pass — rows must be in topological order)
    let (vars_star, public_io_star) =
        solve_witness_forward(&circuit_star, challenge.num_circuit_outputs, &x_eval)
            .map_err(|e| anyhow!("C* witness solver failed: {:?}", e))?;

    // 5a. Prove π⁰
    let c0 = &challenge.circuit_c0;
    let inst0 = Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C)
        .map_err(|e| anyhow!("Spartan instance for C⁰: {:?}", e))?;
    let gens0 = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
    let (comm0, decomm0) = SNARK::encode(&inst0, &gens0);

    let av0 = VarsAssignment::new(&scalars_to_bytes(&vars0))
        .map_err(|e| anyhow!("VarsAssignment for C⁰: {:?}", e))?;
    let ai0 = InputsAssignment::new(&scalars_to_bytes(&public_io0))
        .map_err(|e| anyhow!("InputsAssignment for C⁰: {:?}", e))?;

    let proof0 = SNARK::prove(
        &inst0,
        &comm0,
        &decomm0,
        av0,
        &ai0,
        &gens0,
        &mut Transcript::new(b"ZKChallenge_C0"),
    );

    // 5b. Prove π*
    let inst_star = Instance::new(
        circuit_star.num_cons,
        circuit_star.num_vars,
        circuit_star.num_inputs,
        &circuit_star.A,
        &circuit_star.B,
        &circuit_star.C,
    )
    .map_err(|e| anyhow!("Spartan instance for C*: {:?}", e))?;
    let gens_star = SNARKGens::new(
        circuit_star.num_cons,
        circuit_star.num_vars,
        circuit_star.num_inputs,
        num_non_zero(&circuit_star),
    );
    let (comm_star, decomm_star) = SNARK::encode(&inst_star, &gens_star);

    let av_star = VarsAssignment::new(&scalars_to_bytes(&vars_star))
        .map_err(|e| anyhow!("VarsAssignment for C*: {:?}", e))?;
    let ai_star = InputsAssignment::new(&scalars_to_bytes(&public_io_star))
        .map_err(|e| anyhow!("InputsAssignment for C*: {:?}", e))?;

    let proof_star = SNARK::prove(
        &inst_star,
        &comm_star,
        &decomm_star,
        av_star,
        &ai_star,
        &gens_star,
        &mut Transcript::new(b"ZKChallenge_Cstar"),
    );

    let y0_pub = public_io0[..challenge.num_circuit_outputs].to_vec();
    let y_star_pub = public_io_star[..challenge.num_circuit_outputs].to_vec();

    Ok(Solution {
        circuit_star,
        y0_pub,
        y_star_pub,
        proof0,
        proof_star,
    })
}

fn scalars_to_bytes(scalars: &[Scalar]) -> Vec<[u8; 32]> {
    scalars.iter().map(|s| s.to_bytes()).collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn make_challenge(delta: usize) -> Challenge {
        let mut seed = [0u8; 32];
        seed[0] = 42;
        Challenge::generate_instance(&seed, &Track { delta }).unwrap()
    }

    #[test]
    fn test_generate_instance() {
        let t0 = Instant::now();
        let ch = make_challenge(1);
        eprintln!(
            "[generate_instance] delta=1 -> {} cons, {} vars, {} pub_io in {:.2?}",
            ch.circuit_c0.num_cons,
            ch.circuit_c0.num_vars,
            ch.circuit_c0.num_inputs,
            t0.elapsed()
        );
        assert!(ch.circuit_c0.num_cons >= 900);
        assert!(ch.circuit_c0.num_vars > 0);
        assert!(ch.circuit_c0.num_inputs > 0);
    }

    #[test]
    fn test_deterministic_generation() {
        let c1 = make_challenge(1);
        let c2 = make_challenge(1);
        assert_eq!(c1.circuit_c0.num_cons, c2.circuit_c0.num_cons);
        assert_eq!(c1.circuit_c0.A, c2.circuit_c0.A);
        eprintln!(
            "[deterministic] OK ({} constraints)",
            c1.circuit_c0.num_cons
        );
    }

    #[test]
    fn test_hash_and_xeval_derivation() {
        let ch = make_challenge(1);
        let h0 = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
        let h0_again = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
        assert_eq!(h0, h0_again);

        let x_eval = h0.combine(&h0_again).to_scalars(ch.num_circuit_inputs);
        assert_eq!(x_eval.len(), ch.num_circuit_inputs);
        for (i, s) in x_eval.iter().enumerate() {
            assert_ne!(*s, Scalar::ZERO, "x_eval[{}] should not be zero", i);
        }
    }

    #[test]
    fn test_witness_satisfies_c0() {
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;
        let h0 = CryptoHash::from_serializable(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let seed_hex = seed_to_hex(&ch.seed);
        let config = CircuitConfig::from_delta(ch.delta);
        let dag_val = generate_dag(&seed_hex, &config);
        let (vars, public_io) = compute_witness(&dag_val, &x_eval);

        let inst =
            Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C).unwrap();
        let av = VarsAssignment::new(&scalars_to_bytes(&vars)).unwrap();
        let ai = InputsAssignment::new(&scalars_to_bytes(&public_io)).unwrap();
        assert!(inst.is_sat(&av, &ai).unwrap(), "Witness must satisfy C⁰");
    }

    #[test]
    fn test_full_identity_roundtrip() {
        let total = Instant::now();
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;
        eprintln!("\n=== Identity Roundtrip (delta=1) ===");
        eprintln!(
            "[1] {} cons, {} vars, {} pub_io",
            c0.num_cons, c0.num_vars, c0.num_inputs
        );

        let h0 = CryptoHash::from_serializable(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let seed_hex = seed_to_hex(&ch.seed);
        let config = CircuitConfig::from_delta(ch.delta);
        let dag_val = generate_dag(&seed_hex, &config);
        let (vars0, io0) = compute_witness(&dag_val, &x_eval);

        let inst0 =
            Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C).unwrap();
        let gens0 = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
        let (comm0, decomm0) = SNARK::encode(&inst0, &gens0);
        let av0 = VarsAssignment::new(&scalars_to_bytes(&vars0)).unwrap();
        let ai0 = InputsAssignment::new(&scalars_to_bytes(&io0)).unwrap();

        let t0 = Instant::now();
        let proof0 = SNARK::prove(
            &inst0,
            &comm0,
            &decomm0,
            av0,
            &ai0,
            &gens0,
            &mut Transcript::new(b"ZKChallenge_C0"),
        );
        eprintln!("[2] prove pi0 in {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        proof0
            .verify(
                &comm0,
                &ai0,
                &mut Transcript::new(b"ZKChallenge_C0"),
                &gens0,
            )
            .expect("pi0 must verify");
        eprintln!("[3] verify pi0 in {:.2?}", t0.elapsed());

        // C* = C0 (identity), witness via forward pass
        let (vars_star, io_star) =
            solve_witness_forward(c0, ch.num_circuit_outputs, &x_eval).unwrap();
        let gens_star = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
        let (comm_star, decomm_star) = SNARK::encode(&inst0, &gens_star);
        let av_star = VarsAssignment::new(&scalars_to_bytes(&vars_star)).unwrap();
        let ai_star = InputsAssignment::new(&scalars_to_bytes(&io_star)).unwrap();

        let t0 = Instant::now();
        let proof_star = SNARK::prove(
            &inst0,
            &comm_star,
            &decomm_star,
            av_star,
            &ai_star,
            &gens_star,
            &mut Transcript::new(b"ZKChallenge_Cstar"),
        );
        eprintln!("[4] prove pi* in {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        proof_star
            .verify(
                &comm_star,
                &ai_star,
                &mut Transcript::new(b"ZKChallenge_Cstar"),
                &gens_star,
            )
            .expect("pi* must verify");
        eprintln!("[5] verify pi* in {:.2?}", t0.elapsed());

        assert_eq!(
            &io0[..ch.num_circuit_outputs],
            &io_star[..ch.num_circuit_outputs]
        );
        eprintln!("=== PASSED in {:.2?} ===\n", total.elapsed());
    }

    #[test]
    fn test_alias_optimizer_roundtrip() {
        let total = Instant::now();
        eprintln!("\n=== Alias Optimizer Roundtrip (delta=1) ===");

        let ch = make_challenge(1);
        eprintln!("[1] baseline: {} constraints", ch.circuit_c0.num_cons);

        fn alias_optimizer(c0: &SpartanInstance) -> SpartanInstance {
            baselines::remove_aliases::run(c0)
        }

        let t0 = Instant::now();
        let solution = solve_challenge(&ch, alias_optimizer).expect("solve_challenge must succeed");
        eprintln!(
            "[2] solve: {} -> {} constraints in {:.2?}",
            ch.circuit_c0.num_cons,
            solution.circuit_star.num_cons,
            t0.elapsed()
        );

        assert!(solution.circuit_star.num_cons < ch.circuit_c0.num_cons);

        let t0 = Instant::now();
        ch.evaluate_num_constraints(&solution)
            .expect("evaluate_num_constraints must succeed");
        eprintln!("[3] verify OK in {:.2?}", t0.elapsed());

        let eps = 1.0 - solution.circuit_star.num_cons as f64 / ch.circuit_c0.num_cons as f64;
        eprintln!("[4] epsilon = {:.4}", eps);
        eprintln!("=== PASSED in {:.2?} ===\n", total.elapsed());
    }

    #[test]
    fn test_wrong_order_rejected() {
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;
        let num_cons = c0.num_cons;

        let flip = |mat: &R1CSMatrix| -> R1CSMatrix {
            mat.iter()
                .map(|&(row, col, val)| (num_cons - 1 - row, col, val))
                .collect()
        };

        let reversed = SpartanInstance {
            num_cons: c0.num_cons,
            num_vars: c0.num_vars,
            num_inputs: c0.num_inputs,
            A: flip(&c0.A),
            B: flip(&c0.B),
            C: flip(&c0.C),
        };

        let h0 = CryptoHash::from_serializable(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let result = solve_witness_forward(&reversed, ch.num_circuit_outputs, &x_eval);
        assert!(
            matches!(result, Err(WitnessError::NotInEvaluationOrder { .. })),
            "Expected NotInEvaluationOrder, got: {:?}",
            result
        );
        eprintln!(
            "[wrong_order] correctly rejected: {:?}",
            result.unwrap_err()
        );
    }
}
