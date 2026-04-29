use super::dag::{OpType, DAG};
use curve25519_dalek::scalar::Scalar;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Core types
// =============================================================================

/// Sparse R1CS matrix in COO (Coordinate) format.
/// Each entry is `(row_index, col_index, scalar_bytes_le)`.
pub type R1CSMatrix = Vec<(usize, usize, [u8; 32])>;

/// Errors from the R1CS witness solvers.
#[derive(Debug)]
pub enum WitnessError {
    /// `circuit_inputs` length doesn't match `num_inputs - num_outputs`.
    InvalidInputs { expected: usize, got: usize },
    /// Fixed-point iteration stalled — circuit is underconstrained or malformed.
    SolverStuck { solved: usize, total: usize },
    /// Row `row` has more than one unknown variable in the forward pass.
    /// The circuit rows are not in topological evaluation order.
    NotInEvaluationOrder { row: usize },
}

/// Sparse R1CS instance for libspartan.
///
/// z-vector layout: `[private_vars(0..num_vars-1) | 1 | outputs... | inputs...]`
///   - `z[0..num_vars-1]`  — private intermediate variables (hidden from verifier)
///   - `z[num_vars]`        — constant 1 (auto-inserted by libspartan)
///   - `z[num_vars+1..]`   — public I/O: `[outputs..., circuit_inputs...]`
///
/// Each constraint row `i`: `<A[i], z> * <B[i], z> = <C[i], z>`
#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(non_snake_case)]
pub struct SpartanInstance {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub A: R1CSMatrix,
    pub B: R1CSMatrix,
    pub C: R1CSMatrix,
}

// =============================================================================
// Column assignment (shared by dag_to_spartan and compute_witness)
// =============================================================================

pub(crate) struct ColumnAssignment {
    pub node_to_col: HashMap<usize, usize>,
    pub pow5_intermediates: HashMap<usize, (usize, usize, usize)>,
    pub num_private_vars: usize,
    pub num_public_inputs: usize,
    pub col_const_one: usize,
    pub output_node_order: Vec<usize>,
    pub input_node_order: Vec<usize>,
}

/// Assigns z-vector column indices to all DAG nodes.
///
/// Layout:
///   `0..num_private-1`  → private intermediate variables
///   `num_private`        → constant 1
///   `num_private+1..`   → public I/O [outputs..., inputs...]
pub(crate) fn assign_columns(dag: &DAG) -> ColumnAssignment {
    let output_node_order: Vec<usize> = (0..dag.num_outputs).collect();
    let output_set: HashSet<usize> = output_node_order.iter().cloned().collect();

    let input_node_order: Vec<usize> = dag
        .nodes
        .iter()
        .filter(|n| n.is_input() && !output_set.contains(&n.id))
        .map(|n| n.id)
        .collect();

    let public_set: HashSet<usize> = output_node_order
        .iter()
        .chain(input_node_order.iter())
        .cloned()
        .collect();

    let mut node_to_col: HashMap<usize, usize> = HashMap::new();
    let mut pow5_intermediates: HashMap<usize, (usize, usize, usize)> = HashMap::new();
    let mut next_private = 0usize;

    for node in &dag.nodes {
        if !public_set.contains(&node.id) {
            node_to_col.insert(node.id, next_private);
            next_private += 1;
        }
        if let OpType::Pow5(src) = node.op {
            let sq_col = next_private;
            next_private += 1;
            let qd_col = next_private;
            next_private += 1;
            pow5_intermediates.insert(node.id, (sq_col, qd_col, src));
        }
    }

    let num_private_vars = next_private;
    let col_const_one = num_private_vars;

    let num_public_inputs = output_node_order.len() + input_node_order.len();
    let mut public_offset = 0usize;
    for &node_id in output_node_order.iter().chain(input_node_order.iter()) {
        node_to_col.insert(node_id, num_private_vars + 1 + public_offset);
        public_offset += 1;
    }

    ColumnAssignment {
        node_to_col,
        pow5_intermediates,
        num_private_vars,
        num_public_inputs,
        col_const_one,
        output_node_order,
        input_node_order,
    }
}

#[inline]
fn push_entry(matrix: &mut R1CSMatrix, row: usize, col: usize, val: Scalar) {
    if val != Scalar::ZERO {
        matrix.push((row, col, val.to_bytes()));
    }
}

// =============================================================================
// dag_to_spartan
// =============================================================================

/// Converts a DAG to Spartan R1CS matrices.
///
/// Rows are emitted in **topological evaluation order** (reverse node-ID order):
/// deep dependencies first, output constraints last. This guarantees that a
/// single forward pass can compute the witness.
pub fn dag_to_spartan(dag: &DAG) -> SpartanInstance {
    let cols = assign_columns(dag);

    let mut a_mat: R1CSMatrix = Vec::new();
    let mut b_mat: R1CSMatrix = Vec::new();
    let mut c_mat: R1CSMatrix = Vec::new();
    let mut row = 0usize;

    for node in dag.nodes.iter().rev() {
        match node.op {
            OpType::Input => {}

            OpType::Alias(src) => {
                // node * 1 = src
                let node_col = cols.node_to_col[&node.id];
                let src_col = cols.node_to_col[&src];
                push_entry(&mut a_mat, row, node_col, Scalar::ONE);
                push_entry(&mut b_mat, row, cols.col_const_one, Scalar::ONE);
                push_entry(&mut c_mat, row, src_col, Scalar::ONE);
                row += 1;
            }

            OpType::Add(l, r) => {
                // (l + r) * 1 = node
                let l_col = cols.node_to_col[&l];
                let r_col = cols.node_to_col[&r];
                let out_col = cols.node_to_col[&node.id];
                push_entry(&mut a_mat, row, l_col, Scalar::ONE);
                push_entry(&mut a_mat, row, r_col, Scalar::ONE);
                push_entry(&mut b_mat, row, cols.col_const_one, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Mul(l, r) => {
                // l * r = node
                let l_col = cols.node_to_col[&l];
                let r_col = cols.node_to_col[&r];
                let out_col = cols.node_to_col[&node.id];
                push_entry(&mut a_mat, row, l_col, Scalar::ONE);
                push_entry(&mut b_mat, row, r_col, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Scale(src, k) => {
                // (k * src) * 1 = node
                let src_col = cols.node_to_col[&src];
                let out_col = cols.node_to_col[&node.id];
                push_entry(&mut a_mat, row, src_col, Scalar::from(k));
                push_entry(&mut b_mat, row, cols.col_const_one, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Pow5(_) => {
                // x^5 unrolled: sq = src*src, qd = sq*sq, out = qd*src
                let &(sq_col, qd_col, src_id) = cols.pow5_intermediates.get(&node.id).unwrap();
                let src_col = cols.node_to_col[&src_id];
                let out_col = cols.node_to_col[&node.id];

                // src * src = sq
                push_entry(&mut a_mat, row, src_col, Scalar::ONE);
                push_entry(&mut b_mat, row, src_col, Scalar::ONE);
                push_entry(&mut c_mat, row, sq_col, Scalar::ONE);
                row += 1;

                // sq * sq = qd
                push_entry(&mut a_mat, row, sq_col, Scalar::ONE);
                push_entry(&mut b_mat, row, sq_col, Scalar::ONE);
                push_entry(&mut c_mat, row, qd_col, Scalar::ONE);
                row += 1;

                // qd * src = out
                push_entry(&mut a_mat, row, qd_col, Scalar::ONE);
                push_entry(&mut b_mat, row, src_col, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Output | OpType::Undefined => {}
        }
    }

    SpartanInstance {
        num_cons: row,
        num_vars: cols.num_private_vars,
        num_inputs: cols.num_public_inputs,
        num_outputs: cols.output_node_order.len(),
        A: a_mat,
        B: b_mat,
        C: c_mat,
    }
}

// =============================================================================
// compute_witness (DAG-based, used for C0)
// =============================================================================

/// Computes the full witness from the DAG directly.
///
/// Returns `(vars, public_io)`:
/// - `vars`: private variable values, `len = num_vars`
/// - `public_io`: `[outputs..., circuit_inputs...]`, `len = num_inputs`
pub fn compute_witness(dag: &DAG, input_values: &[Scalar]) -> (Vec<Scalar>, Vec<Scalar>) {
    let cols = assign_columns(dag);

    assert_eq!(
        input_values.len(),
        cols.input_node_order.len(),
        "Expected {} input values, got {}",
        cols.input_node_order.len(),
        input_values.len()
    );

    let mut node_values: Vec<Option<Scalar>> = vec![None; dag.nodes.len()];

    for (i, &node_id) in cols.input_node_order.iter().enumerate() {
        node_values[node_id] = Some(input_values[i]);
    }

    for node in dag.nodes.iter().rev() {
        match node.op {
            OpType::Input => {}
            OpType::Add(l, r) => {
                node_values[node.id] = Some(node_values[l].unwrap() + node_values[r].unwrap());
            }
            OpType::Mul(l, r) => {
                node_values[node.id] = Some(node_values[l].unwrap() * node_values[r].unwrap());
            }
            OpType::Alias(src) => {
                node_values[node.id] = node_values[src];
            }
            OpType::Scale(src, k) => {
                node_values[node.id] = Some(Scalar::from(k) * node_values[src].unwrap());
            }
            OpType::Pow5(src) => {
                let x = node_values[src].unwrap();
                let sq = x * x;
                node_values[node.id] = Some(sq * sq * x);
            }
            _ => {}
        }
    }

    let mut vars = vec![Scalar::ZERO; cols.num_private_vars];
    for (&node_id, &col) in &cols.node_to_col {
        if col < cols.num_private_vars {
            vars[col] = node_values[node_id].expect("private node value not computed");
        }
    }
    for (_, &(sq_col, qd_col, src_id)) in &cols.pow5_intermediates {
        let x = node_values[src_id].unwrap();
        let sq = x * x;
        vars[sq_col] = sq;
        vars[qd_col] = sq * sq;
    }

    let mut public_io = Vec::with_capacity(cols.num_public_inputs);
    for &node_id in &cols.output_node_order {
        public_io.push(node_values[node_id].expect("output node value not computed"));
    }
    for &node_id in &cols.input_node_order {
        public_io.push(node_values[node_id].unwrap());
    }

    (vars, public_io)
}

// =============================================================================
// solve_witness_forward (single-pass, used for C*)
// =============================================================================

/// Computes the witness using a **single forward pass** over R1CS rows.
///
/// Requires rows to be in **topological evaluation order**: when row `i` is
/// reached, all variables in A and B must already be known except at most one.
/// Violations return `Err(WitnessError::NotInEvaluationOrder { row })`.
///
/// O(n) — no backtracking.
pub fn solve_witness_forward(
    instance: &SpartanInstance,
    num_outputs: usize,
    circuit_inputs: &[Scalar],
) -> Result<(Vec<Scalar>, Vec<Scalar>), WitnessError> {
    let expected = instance.num_inputs - num_outputs;
    if circuit_inputs.len() != expected {
        return Err(WitnessError::InvalidInputs {
            expected,
            got: circuit_inputs.len(),
        });
    }

    let (a_rows, b_rows, c_rows) = build_row_views(instance);

    let z_len = instance.num_vars + 1 + instance.num_inputs;
    let mut z = vec![Scalar::ZERO; z_len];
    let mut solved = vec![false; z_len];

    z[instance.num_vars] = Scalar::ONE;
    solved[instance.num_vars] = true;
    for (i, &val) in circuit_inputs.iter().enumerate() {
        let idx = instance.num_vars + 1 + num_outputs + i;
        z[idx] = val;
        solved[idx] = true;
    }

    for row in 0..instance.num_cons {
        let mut unsolved_col: Option<usize> = None;
        let mut multi = false;

        for &(col, _) in a_rows[row]
            .iter()
            .chain(b_rows[row].iter())
            .chain(c_rows[row].iter())
        {
            if !solved[col] {
                match unsolved_col {
                    None => unsolved_col = Some(col),
                    Some(prev) if prev == col => {}
                    Some(_) => {
                        multi = true;
                        break;
                    }
                }
            }
        }

        if multi {
            return Err(WitnessError::NotInEvaluationOrder { row });
        }

        let j = match unsolved_col {
            Some(j) => j,
            None => continue,
        };

        let (a_known, a_j) = accumulate(&a_rows[row], j, &z);
        let (b_known, b_j) = accumulate(&b_rows[row], j, &z);
        let (c_known, c_j) = accumulate(&c_rows[row], j, &z);

        let denom = a_j * b_known + b_j * a_known - c_j;
        if denom == Scalar::ZERO {
            continue;
        }
        z[j] = (c_known - a_known * b_known) * denom.invert();
        solved[j] = true;
    }

    check_convergence(&solved, instance.num_vars, num_outputs, instance.num_vars)?;
    Ok(extract_result(&z, instance))
}

// =============================================================================
// solve_witness_from_r1cs (fixed-point, for debugging)
// =============================================================================

/// Computes the witness via fixed-point iteration — works with rows in any order.
///
/// O(n²) worst case. Use for debugging or when row order is unknown.
pub fn solve_witness_from_r1cs(
    instance: &SpartanInstance,
    num_outputs: usize,
    circuit_inputs: &[Scalar],
) -> Result<(Vec<Scalar>, Vec<Scalar>), WitnessError> {
    let expected = instance.num_inputs - num_outputs;
    if circuit_inputs.len() != expected {
        return Err(WitnessError::InvalidInputs {
            expected,
            got: circuit_inputs.len(),
        });
    }

    let (a_rows, b_rows, c_rows) = build_row_views(instance);

    let z_len = instance.num_vars + 1 + instance.num_inputs;
    let mut z = vec![Scalar::ZERO; z_len];
    let mut solved = vec![false; z_len];

    z[instance.num_vars] = Scalar::ONE;
    solved[instance.num_vars] = true;
    for (i, &val) in circuit_inputs.iter().enumerate() {
        let idx = instance.num_vars + 1 + num_outputs + i;
        z[idx] = val;
        solved[idx] = true;
    }

    loop {
        let mut progress = false;

        for row in 0..instance.num_cons {
            let mut unsolved_col: Option<usize> = None;
            let mut multi = false;

            for &(col, _) in a_rows[row]
                .iter()
                .chain(b_rows[row].iter())
                .chain(c_rows[row].iter())
            {
                if !solved[col] {
                    match unsolved_col {
                        None => unsolved_col = Some(col),
                        Some(prev) if prev == col => {}
                        Some(_) => {
                            multi = true;
                            break;
                        }
                    }
                }
            }

            if multi || unsolved_col.is_none() {
                continue;
            }
            let j = unsolved_col.unwrap();

            let (a_known, a_j) = accumulate(&a_rows[row], j, &z);
            let (b_known, b_j) = accumulate(&b_rows[row], j, &z);
            let (c_known, c_j) = accumulate(&c_rows[row], j, &z);

            let denom = a_j * b_known + b_j * a_known - c_j;
            if denom == Scalar::ZERO {
                continue;
            }
            z[j] = (c_known - a_known * b_known) * denom.invert();
            solved[j] = true;
            progress = true;
        }

        if !progress {
            break;
        }
    }

    check_convergence(&solved, instance.num_vars, num_outputs, instance.num_vars)?;
    Ok(extract_result(&z, instance))
}

// =============================================================================
// Helpers
// =============================================================================

fn build_row_views(
    instance: &SpartanInstance,
) -> (
    Vec<Vec<(usize, Scalar)>>,
    Vec<Vec<(usize, Scalar)>>,
    Vec<Vec<(usize, Scalar)>>,
) {
    let mut a_rows = vec![Vec::new(); instance.num_cons];
    let mut b_rows = vec![Vec::new(); instance.num_cons];
    let mut c_rows = vec![Vec::new(); instance.num_cons];

    for &(row, col, bytes) in &instance.A {
        a_rows[row].push((col, Scalar::from_canonical_bytes(bytes).unwrap()));
    }
    for &(row, col, bytes) in &instance.B {
        b_rows[row].push((col, Scalar::from_canonical_bytes(bytes).unwrap()));
    }
    for &(row, col, bytes) in &instance.C {
        c_rows[row].push((col, Scalar::from_canonical_bytes(bytes).unwrap()));
    }

    (a_rows, b_rows, c_rows)
}

/// Splits a row into `(known_sum, coeff_of_j)`.
fn accumulate(row: &[(usize, Scalar)], j: usize, z: &[Scalar]) -> (Scalar, Scalar) {
    let mut known = Scalar::ZERO;
    let mut coeff_j = Scalar::ZERO;
    for &(col, val) in row {
        if col == j {
            coeff_j += val;
        } else {
            known += val * z[col];
        }
    }
    (known, coeff_j)
}

fn check_convergence(
    solved: &[bool],
    num_vars: usize,
    num_outputs: usize,
    const_col: usize,
) -> Result<(), WitnessError> {
    let total = num_vars + num_outputs;
    let mut count = 0;
    for i in 0..num_vars {
        if solved[i] {
            count += 1;
        }
    }
    for i in 0..num_outputs {
        if solved[const_col + 1 + i] {
            count += 1;
        }
    }
    if count < total {
        Err(WitnessError::SolverStuck {
            solved: count,
            total,
        })
    } else {
        Ok(())
    }
}

fn extract_result(z: &[Scalar], instance: &SpartanInstance) -> (Vec<Scalar>, Vec<Scalar>) {
    let vars = z[..instance.num_vars].to_vec();
    let public_io = z[instance.num_vars + 1..instance.num_vars + 1 + instance.num_inputs].to_vec();
    (vars, public_io)
}
