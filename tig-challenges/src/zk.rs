use anyhow::{anyhow, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap; 
use std::hash::Hash;

use curve25519_dalek::scalar::Scalar;

use libspartan::{
    Instance, SNARKGens, VarsAssignment, InputsAssignment, SNARK, 
    ComputationCommitment as Commitment, ComputationDecommitment as Decommitment,
};


use merlin::Transcript;
use blake3::{Hasher as Blake3};

// =============================================================================
// Utility: Cryptographic Hashing, Hash-to-Field, and Helpers
// =============================================================================

// Define a type alias for the R1CS Matrix format used by spartan
pub type R1CSMatrix = Vec<(usize, usize, [u8; 32])>;

// Helper function for modular exponentiation (Scalar^u64) using repeated squaring.
// This is needed because curve25519_dalek::Scalar does not expose a simple integer pow method.
fn scalar_pow(base: Scalar, exponent: u64) -> Scalar {
    if exponent == 0 {
        return Scalar::ONE;
    }
    let mut result = Scalar::ONE;
    let mut current_power = base;
    let mut exp = exponent;

    while exp > 0 {
        if exp % 2 == 1 {
            result *= current_power;
        }
        exp /= 2;
        if exp > 0 {
            current_power *= current_power; // Square
        }
    }
    result
}


// Concrete implementation of the cryptographic hash H().
// We use a struct to hold the hash value (512 bits).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct CryptoHash(#[serde(with = "serde_bytes")] pub [u8; 64]);

impl CryptoHash {
    // Hashes data that can be serialized (Used for H(C) and H(P)).
    pub fn from_serializable<T: Serialize>(data: &T) -> Result<Self> {
        let mut hasher = Blake3::new();
        // We use bincode for serialization; ensure this is consistent and deterministic.
        let bytes = bincode::serialize(data)?;
        hasher.update(&bytes);
        let mut result = hasher.finalize_xof();
        let mut hash: [u8; 64] = [0; 64];
        result.fill(&mut hash);
        Ok(CryptoHash(hash))
    }

    // Helper to hash the Circuit structure H(C)
    pub fn from_circuit(circuit: &Circuit) -> Result<Self> {
         Self::from_serializable(circuit)
    }

    // Simulates H(H1 || H2)
    pub fn combine(&self, other: &CryptoHash) -> Self {
        let mut hasher = Blake3::new();
        hasher.update(&self.0);
        hasher.update(&other.0);
        let mut result = hasher.finalize_xof();
        let mut hash: [u8; 64] = [0; 64];
        result.fill(&mut hash);
        CryptoHash(hash)
    }

    // Hash-to-Field: converting a hash digest into Scalars.
    // Implements: r_i = H( seed || i ) mod P (Section 2.2.2 of Overleaf)
    pub fn to_scalars(&self, count: usize) -> Vec<Scalar> {
        let mut scalars = Vec::with_capacity(count);
        for i in 0..count {
            let mut hasher = Blake3::new();
            hasher.update(&self.0); // seed (H(C) || H(P))
            hasher.update(&(i as u64).to_le_bytes()); // index i (domain separator)
            
            let mut hash: [u8; 64] = [0; 64];
            hasher.finalize_xof().fill(&mut hash);
            
            // Convert the 512-bit digest to a Scalar.
            // curve25519-dalek provides `from_bytes_mod_order_wide` for this purpose.
            //let mut digest_array = [0u8; 64];
            //digest_array.clone_from_slice(&digest);
            scalars.push(Scalar::from_bytes_mod_order_wide(&hash));
        }
        scalars
    }
}

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty {
    pub delta: usize,
    pub epsilon: f64,
    pub num_variables: usize,
    pub num_monomials: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Polynomial {
    pub coefficients: Vec<Scalar>,
    pub degrees_matrix: Vec<Vec<u32>>,
    pub num_variables: usize,
    pub hash: CryptoHash, // H(P)
}

impl Polynomial {
    fn new(coefficients: Vec<Scalar>, degrees_matrix: Vec<Vec<u32>>, num_variables: usize) -> Result<Self> {
        // Calculate H(P) based on the structure
        let hash_input = (&coefficients, &degrees_matrix, num_variables);
        let hash = CryptoHash::from_serializable(&hash_input)?;
        Ok(Polynomial {
            coefficients,
            degrees_matrix,
            num_variables,
            hash,
        })
    }

    // K_base(P) = Σ max(0, E_i - 1)
    pub fn calculate_baseline(&self) -> usize {
        self.degrees_matrix.iter().map(|exponents| {
            let total_degree: u64 = exponents.iter().map(|&e| e as u64).sum();
            (total_degree as usize).saturating_sub(1)
        }).sum()
    }

    // Evaluate P(x) at a given point.
    pub fn evaluate(&self, point: &[Scalar]) -> Scalar {
        if point.len() != self.num_variables {
            panic!("Evaluation point dimension mismatch.");
        }
        let mut result = Scalar::ZERO;
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            let mut term = coeff;
            for j in 0..self.num_variables {
                let exponent = self.degrees_matrix[i][j] as u64;
                if exponent > 0 {
                    // Use the helper function for exponentiation
                    term *= scalar_pow(point[j], exponent); 
                }
            }
            result += term;
        }
        result
    }
}

// The Circuit must be serializable to calculate H(C).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Circuit {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub A: R1CSMatrix,
    pub B: R1CSMatrix,
    pub C: R1CSMatrix,
}

impl Circuit {
    fn num_non_zero_entries(&self) -> usize {
        self.A.len() + self.B.len() + self.C.len()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub polynomial: Polynomial,
    pub difficulty: Difficulty,
}

// The solution format. With the 'serde' feature enabled in Cargo.toml, this will compile.
#[derive(Serialize, Deserialize)]
pub struct Solution {
    pub circuit: Circuit,
    pub comm: Commitment,
    pub assignment_inputs: Vec<Scalar>, // [Y, x1, x2...]
    pub public_output: Scalar,          // Y
    pub proof: SNARK, 
    pub gens: SNARKGens,
    pub circuit_hash: CryptoHash, // H(C)
}

// =============================================================================
// Challenge Implementation (Framework: tig-challenges)
// =============================================================================

impl Challenge {
    // Implementation of Random Instance Generation (Section 2.4)
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Challenge> {
        let mut rng = StdRng::from_seed(seed.clone());
        let n = difficulty.num_variables;
        let m = difficulty.num_monomials;
        let target_delta = difficulty.delta;

        if n == 0 || m == 0 {
            return Err(anyhow!("N and M must be > 0."));
        }

        let total_degree = target_delta.checked_add(m).ok_or(anyhow!("Total degree overflow"))?;

        // 1. Set row totals (E_i)
        // Explicitly define type as Vec<usize> to resolve ambiguity for saturating_add.
        let mut degrees: Vec<usize> = vec![1; m];
        let remaining_degree = total_degree - m;
        for _ in 0..remaining_degree {
            let index = rng.gen_range(0..m);
            degrees[index] = degrees[index].saturating_add(1);
        }

        // 2. Form monomials
        let mut coefficients = Vec::new();
        let mut degrees_matrix = Vec::new();

        for &e_i in &degrees {
            let mut exponents = vec![0u32; n];
            
            // Safety check: ensure the degree fits within u32 bounds used for exponents
            if e_i > u32::MAX as usize {
                return Err(anyhow!("Exponent overflow (total degree exceeds u32 capacity)."));
            }

            for _ in 0..e_i {
                let index = rng.gen_range(0..n);
                // This check is technically redundant now but safe to keep
                if exponents[index] == u32::MAX {
                     return Err(anyhow!("Exponent overflow during distribution."));
                }
                exponents[index] += 1;
            }
            // Generate random Scalar coefficients
            coefficients.push(Scalar::random(&mut rng));
            degrees_matrix.push(exponents);
        }

        let polynomial = Polynomial::new(coefficients, degrees_matrix, n)?;
        if polynomial.calculate_baseline() != target_delta {
            return Err(anyhow!("Internal Error: Baseline mismatch."));
        }

        Ok(Challenge { seed: seed.clone(), polynomial, difficulty: difficulty.clone() })
    }

    // Implementation of Solution Verification (Section 2.2)
    pub fn verify_solution(&self, solution: &Solution) -> Result<()> {
        let k_prime = solution.circuit.num_cons;
        let k_base = self.difficulty.delta;

        // 1. Scoring
        if k_base == 0 {
             if k_prime != 0 {
                 return Err(anyhow!("Circuit optimization failed. Baseline is 0, K' > 0."));
             }
        } else {
            let achieved_ratio = k_prime as f64 / k_base as f64;
            if achieved_ratio > self.difficulty.epsilon {
                return Err(anyhow!("Circuit optimization failed. Ratio ({:.4}) > Epsilon ({:.4})", achieved_ratio, self.difficulty.epsilon));
            }
        }

        // 2. Verify Circuit Hash H(C)
        let computed_circuit_hash = CryptoHash::from_circuit(&solution.circuit)?;
        if computed_circuit_hash != solution.circuit_hash {
            return Err(anyhow!("Circuit hash mismatch."));
        }

        // 3. Verify Evaluation Point (Anti-Grinding)
        // Check that x_pub = Hash-to-Field( H(C) || H(P) )
        let expected_x_pub_hash = solution.circuit_hash.combine(&self.polynomial.hash);
        let expected_x_pub = expected_x_pub_hash.to_scalars(self.polynomial.num_variables);


        // The solution's assignment_inputs should be [Y, x1, x2...]
        if solution.assignment_inputs.len() != self.polynomial.num_variables + 1 {
             return Err(anyhow!("Invalid public input length in solution."));
        }
        let actual_x_pub = &solution.assignment_inputs[1..];

        if actual_x_pub != expected_x_pub {
           return Err(anyhow!("Public inputs (x_pub) do not match the required evaluation point. Grinding detected."));
        }

        // 4. Schwartz-Zippel Check (Verify P(x_pub) = Y)
        let expected_Y = self.polynomial.evaluate(&expected_x_pub);
        if expected_Y != solution.public_output || expected_Y != solution.assignment_inputs[0] {
            return Err(anyhow!("Polynomial evaluation mismatch (Schwartz-Zippel check failed)."));
        }

        // 5. Verify the Spartan Proof (π)
        let mut verifier_transcript = Transcript::new(b"PolyR1CSChallenge");
        
        // Convert Vec<Scalar> to InputsAssignment. spartan expects byte arrays.
        let inputs_bytes: Vec<[u8; 32]> = solution.assignment_inputs.iter().map(|s| s.to_bytes()).collect();
        let inputs_assignment = InputsAssignment::new(&inputs_bytes)
             .map_err(|e| anyhow!("Failed to process InputsAssignment: {:?}", e))?;
        
        // Verify the proof using the commitment, inputs, transcript, and generators.
        solution.proof.verify(
            &solution.comm,
            &inputs_assignment,
            &mut verifier_transcript,
            &solution.gens,
        ).map_err(|e| anyhow!("Spartan proof verification failed: {:?}", e))?;

        Ok(())
    }
}

// =============================================================================
// R1CS Compiler and Optimizer Module (Example Algorithm)
// =============================================================================
// This module implements the translation from Polynomial to R1CS using 
// Common Subexpression Elimination (CSE) and Binary Exponentiation.
mod compiler {
    use super::*;
    
    // We must implement Ord/PartialOrd for CSE canonicalization (L < R).
    // curve25519-dalek Scalars do not implement Ord by default, so we must wrap it.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct OrdScalar(pub Scalar);

    impl PartialOrd for OrdScalar {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for OrdScalar {
        // Compare based on byte representation for a consistent (though not necessarily numerical) ordering.
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.to_bytes().cmp(&other.0.to_bytes())
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Operand {
        Input(usize),      // [Y, x1, x2...]
        Intermediate(usize),
        Constant(OrdScalar),
    }

    impl Operand {
        fn constant(s: Scalar) -> Self {
            Operand::Constant(OrdScalar(s))
        }
    }

    #[derive(Clone, Debug)]
    struct CalculationStep {
        L: Operand,
        R: Operand,
    }
    
    type CSEKey = (Operand, Operand);

    pub struct Compiler {
        num_inputs: usize,
        A: R1CSMatrix, B: R1CSMatrix, C: R1CSMatrix,
        steps: HashMap<usize, CalculationStep>,
        cse_map: HashMap<CSEKey, Operand>,
        next_var_idx: usize,
        current_row: usize,
    }

    impl Compiler {
        fn new(num_poly_vars: usize) -> Self {
            Compiler {
                num_inputs: num_poly_vars + 1,
                A: Vec::new(), B: Vec::new(), C: Vec::new(),
                steps: HashMap::new(),
                cse_map: HashMap::new(),
                next_var_idx: 0,
                current_row: 0,
            }
        }

        // Z = (vars, 1, inputs)
        fn operand_to_z_index(&self, operand: &Operand) -> (usize, Scalar) {
            match operand {
                Operand::Intermediate(idx) => (*idx, Scalar::ONE),
                Operand::Constant(os) => (self.idx_one(), os.0),
                Operand::Input(idx) => (self.idx_input(*idx), Scalar::ONE),
            }
        }

        fn idx_one(&self) -> usize { self.next_var_idx }
        fn idx_input(&self, input_idx: usize) -> usize { self.next_var_idx + 1 + input_idx }

        fn multiply(&mut self, L: Operand, R: Operand) -> Operand {
            // Optimization 1: Handle constants
            if L == Operand::constant(Scalar::ONE) { return R; }
            if R == Operand::constant(Scalar::ONE) { return L; }
            if L == Operand::constant(Scalar::ZERO) { return Operand::constant(Scalar::ZERO); }
            if R == Operand::constant(Scalar::ZERO) { return Operand::constant(Scalar::ZERO); }

            // Optimization 2: CSE (Canonicalize key using Ord)
            let key = if L < R { (L.clone(), R.clone()) } else { (R.clone(), L.clone()) };
            
            if let Some(result) = self.cse_map.get(&key) {
                return result.clone();
            }

            // New constraint
            let result_idx = self.next_var_idx;
            self.next_var_idx += 1;
            let result_operand = Operand::Intermediate(result_idx);

            let row = self.current_row;
            self.current_row += 1;

            // Add constraint (A*Z * B*Z = C*Z)
            let (idx_l, scalar_l) = self.operand_to_z_index(&L);
            self.A.push((row, idx_l, scalar_l.to_bytes()));

            let (idx_r, scalar_r) = self.operand_to_z_index(&R);
            self.B.push((row, idx_r, scalar_r.to_bytes()));

            self.C.push((row, result_idx, Scalar::ONE.to_bytes()));

            self.steps.insert(result_idx, CalculationStep { L, R });
            self.cse_map.insert(key, result_operand.clone());

            result_operand
        }
        
        // Final constraint: (P(x) - Y) * 1 = 0
        fn finalize_output_constraint(&mut self, linear_combination: &[(Scalar, Operand)]) {
            let row = self.current_row;
            self.current_row += 1;

            // A term (P(x) - Y)
            for (coeff, operand) in linear_combination {
                let (z_idx, op_scalar) = self.operand_to_z_index(operand);
                let total_coeff = *coeff * op_scalar;
                if total_coeff != Scalar::ZERO {
                    self.A.push((row, z_idx, total_coeff.to_bytes()));
                }
            }

            // B term (1)
            self.B.push((row, self.idx_one(), Scalar::ONE.to_bytes()));

            // C term (0) - Implicit.
        }

        fn build(self) -> (Circuit, ConcreteWitnessBuilder) {
            let circuit = Circuit {
                num_cons: self.current_row,
                num_vars: self.next_var_idx,
                num_inputs: self.num_inputs,
                A: self.A, B: self.B, C: self.C,
            };

            let witness_builder = ConcreteWitnessBuilder {
                steps: self.steps,
                num_vars: self.next_var_idx,
            };

            (circuit, witness_builder)
        }
    }

    // The concrete implementation of the WitnessBuilder.
    pub struct ConcreteWitnessBuilder {
        steps: HashMap<usize, CalculationStep>,
        num_vars: usize,
    }

    impl WitnessBuilder for ConcreteWitnessBuilder {
        // Calculates the intermediate variables based on the full inputs [Y, x1, x2...]
        fn build_witness(&self, inputs: &[Scalar]) -> Result<Vec<Scalar>> {
            let mut vars = vec![Scalar::ZERO; self.num_vars];
            
            let resolve = |op: &Operand, vars: &[Scalar], inputs: &[Scalar]| -> Scalar {
                match op {
                    Operand::Constant(os) => os.0,
                    Operand::Input(idx) => inputs[*idx],
                    Operand::Intermediate(idx) => vars[*idx],
                }
            };

            let mut sorted_indices: Vec<usize> = self.steps.keys().cloned().collect();
            sorted_indices.sort();

            for idx in sorted_indices {
                let step = &self.steps[&idx];
                let L_val = resolve(&step.L, &vars, inputs);
                let R_val = resolve(&step.R, &vars, inputs);
                vars[idx] = L_val * R_val;
            }

            Ok(vars)
        }
    }
    
    // The main compilation function implementing the optimization algorithm.
    pub fn compile_optimized(polynomial: &Polynomial) -> (Circuit, Box<dyn WitnessBuilder>) {
        let num_poly_vars = polynomial.num_variables;
        let mut builder = Compiler::new(num_poly_vars);

        // Input 0 is Y. Input 1 is x1, etc.
        let input_operands: Vec<Operand> = (0..num_poly_vars).map(|i| Operand::Input(i + 1)).collect();

        let mut monomial_results = Vec::new();

        for i in 0..polynomial.coefficients.len() {
            let coeff = polynomial.coefficients[i];
            let exponents = &polynomial.degrees_matrix[i];
            
            let mut current_operand = Operand::constant(Scalar::ONE);

            for j in 0..num_poly_vars {
                let exponent = exponents[j];
                if exponent == 0 { continue; }

                let base_operand = input_operands[j].clone();
                
                // Binary Exponentiation (Addition Chains) with CSE
                let mut power_operand = base_operand;
                let mut temp_exponent = exponent;
                let mut result_of_power = Operand::constant(Scalar::ONE);

                while temp_exponent > 0 {
                    if temp_exponent % 2 == 1 {
                        result_of_power = builder.multiply(result_of_power, power_operand.clone());
                    }
                    temp_exponent /= 2;
                    if temp_exponent > 0 {
                         power_operand = builder.multiply(power_operand.clone(), power_operand.clone());
                    }
                }
                
                current_operand = builder.multiply(current_operand, result_of_power);
            }
            
            monomial_results.push((coeff, current_operand));
        }
        
        // Final constraint: (c1*M1 + c2*M2 + ...) - Y = 0
        let mut linear_combination = monomial_results;
        
        // Add -Y term (Y is Input 0). We use unary negation (-Scalar::ONE).
        linear_combination.push((-Scalar::ONE, Operand::Input(0)));

        builder.finalize_output_constraint(&linear_combination);

        let (circuit, witness_builder) = builder.build();
        
        (circuit, Box::new(witness_builder))
    }
}


// =============================================================================
// Solver Interface (Framework side: tig-challenges)
// =============================================================================

pub trait WitnessBuilder: Send + Sync {
    fn build_witness(&self, inputs: &[Scalar]) -> Result<Vec<Scalar>>;
}

pub type BuildCircuitFn = fn(&Polynomial) -> (Circuit, Box<dyn WitnessBuilder>);

// This function orchestrates the solution generation process.
pub fn solve_challenge_inner(challenge: &Challenge, build_circuit: BuildCircuitFn) -> Result<Solution> {
    
    // 1. Get the optimized circuit C and witness builder
    let (circuit, witness_builder) = build_circuit(&challenge.polynomial);
    
    // 2. Calculate H(C)
    let circuit_hash = CryptoHash::from_circuit(&circuit)?;

    // 3. Determine the evaluation point (Anti-Grinding)
    // x_pub = Hash-to-Field( H(C) || H(P) )
    let x_pub_hash = circuit_hash.combine(&challenge.polynomial.hash);
    let x_pub = x_pub_hash.to_scalars(challenge.polynomial.num_variables);

    // 4. Calculate the output Y = P(x_pub)
    let y_pub = challenge.polynomial.evaluate(&x_pub);
    
    // 5. Construct the full public input vector: [Y, x1, x2...]
    let mut full_public_inputs = vec![y_pub];
    full_public_inputs.extend_from_slice(&x_pub);

    // 6. Generate the witness (intermediate vars)
    let intermediate_vars = witness_builder.build_witness(&full_public_inputs)?;

    // 7. Setup Spartan Instance and Inputs
    let num_cons = circuit.num_cons;
    let num_vars = circuit.num_vars;
    let num_inputs = circuit.num_inputs;

    if num_inputs != full_public_inputs.len() || num_vars != intermediate_vars.len() {
        return Err(anyhow!("Circuit/Witness dimension mismatch."));
    }

    // Initialize the Instance
    let inst = Instance::new(
        num_cons, num_vars, num_inputs,
        &circuit.A, &circuit.B, &circuit.C
    ).map_err(|e| anyhow!("Failed to create Spartan Instance: {:?}", e))?;

    // Prepare Assignments (spartan expects byte arrays).
    let vars_bytes: Vec<[u8; 32]> = intermediate_vars.iter().map(|s| s.to_bytes()).collect();
    let inputs_bytes: Vec<[u8; 32]> = full_public_inputs.iter().map(|s| s.to_bytes()).collect();

    let assignment_vars = VarsAssignment::new(&vars_bytes)
        .map_err(|e| anyhow!("Failed to create VarsAssignment: {:?}", e))?;
    let assignment_inputs = InputsAssignment::new(&inputs_bytes)
        .map_err(|e| anyhow!("Failed to create InputsAssignment: {:?}", e))?;


    // 8. Generate the Spartan Proof (following the pseudocode structure)
    let gens = SNARKGens::new(num_cons, num_vars, num_inputs, circuit.num_non_zero_entries());

    // Step 8a: Create a commitment to the R1CS instance (SNARK::encode)
    let (comm, decomm) = SNARK::encode(&inst, &gens);

    // Step 8b: Produce a proof of satisfiability (SNARK::prove)
    let mut prover_transcript = Transcript::new(b"PolyR1CSChallenge");
    
    let proof = SNARK::prove(
        &inst,
        &comm,
        &decomm,
        assignment_vars,
        &assignment_inputs,
        &gens,
        &mut prover_transcript,
    );

    Ok(Solution {
        circuit,
        comm,
        assignment_inputs: full_public_inputs,
        public_output: y_pub,
        proof,
        gens,
        circuit_hash,
    })
}


// =============================================================================
// Participant Implementation (tig-algorithms)
// =============================================================================

// Main entry point for the solver, using the optimized compiler.
pub fn solve_challenge(challenge: &Challenge, build_circuit: BuildCircuitFn) -> Result<Solution> {
    // The participant provides their optimization algorithm here.
    solve_challenge_inner(challenge, build_circuit)
}