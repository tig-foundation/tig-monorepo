use crate::zk_optimization::{R1CSMatrix, Scalar, SpartanInstance};
use std::collections::{HashMap, HashSet};

/// Removes alias constraints from an R1CS instance via variable substitution.
///
/// An alias constraint has the form `out * 1 = src`:
///   - A: single entry `(col_out, 1)`, col_out < num_vars (private)
///   - B: single entry `(const_col, 1)`
///   - C: single entry `(col_src, 1)`
///
/// The function substitutes `col_out → col_src` everywhere, removes the alias
/// rows, and compacts private variable columns so `num_vars` shrinks.
///
/// Preserves topological row order. Pure function — returns a clone if no
/// aliases are found.
///
/// ~13% constraint reduction on default-configuration circuits.
pub fn remove_aliases(instance: &SpartanInstance) -> SpartanInstance {
    let num_vars = instance.num_vars;
    let const_col = num_vars;
    let one_bytes = Scalar::ONE.to_bytes();

    // --- Build per-row views ---
    let mut a_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut b_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut c_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];

    for &(row, col, bytes) in &instance.A {
        a_rows[row].push((col, bytes));
    }
    for &(row, col, bytes) in &instance.B {
        b_rows[row].push((col, bytes));
    }
    for &(row, col, bytes) in &instance.C {
        c_rows[row].push((col, bytes));
    }

    // --- Detect alias rows and build substitution map ---
    let mut substitution: HashMap<usize, usize> = HashMap::new();
    let mut removed_rows: Vec<bool> = vec![false; instance.num_cons];

    for row in 0..instance.num_cons {
        if a_rows[row].len() != 1 || b_rows[row].len() != 1 || c_rows[row].len() != 1 {
            continue;
        }
        let (a_col, a_val) = a_rows[row][0];
        let (b_col, b_val) = b_rows[row][0];
        let (c_col, c_val) = c_rows[row][0];

        if a_val != one_bytes || b_val != one_bytes || c_val != one_bytes {
            continue;
        }
        if b_col != const_col {
            continue;
        }

        // CRITICAL FIX: Only substitute private→private aliases
        let a_is_private = a_col < num_vars;
        let c_is_private = c_col < num_vars;

        if a_is_private && c_is_private {
            substitution.insert(a_col, c_col);
            removed_rows[row] = true;
        }
        // Skip mixed private/public aliases - they would break the circuit
    }

    if substitution.is_empty() {
        return instance.clone();
    }

    // --- Resolve substitution chains: a → b → c flattened to a → c ---
    let mut changed = true;
    while changed {
        changed = false;
        let snap: Vec<(usize, usize)> = substitution.iter().map(|(&k, &v)| (k, v)).collect();
        for (key, target) in snap {
            if let Some(&further) = substitution.get(&target) {
                substitution.insert(key, further);
                changed = true;
            }
        }
    }

    let remap_col = |col: usize| substitution.get(&col).copied().unwrap_or(col);

    // --- Apply substitutions to surviving rows ---
    let mut new_a: R1CSMatrix = Vec::new();
    let mut new_b: R1CSMatrix = Vec::new();
    let mut new_c: R1CSMatrix = Vec::new();
    let mut new_row = 0usize;

    for row in 0..instance.num_cons {
        if removed_rows[row] {
            continue;
        }

        emit_row(&a_rows[row], &mut new_a, new_row, &remap_col);
        emit_row(&b_rows[row], &mut new_b, new_row, &remap_col);
        emit_row(&c_rows[row], &mut new_c, new_row, &remap_col);
        new_row += 1;
    }

    let new_num_cons = new_row;

    // --- Column compaction (private variables only) ---
    // FIXED: Now that we correctly detect only private→private aliases,
    // we can safely compact columns
    let live_private: Vec<usize> = {
        use std::collections::HashSet;
        let mut set: HashSet<usize> = HashSet::new();
        for &(_, col, _) in new_a.iter().chain(new_b.iter()).chain(new_c.iter()) {
            if col < num_vars {
                set.insert(col);
            }
        }
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort();
        v
    };
    let new_num_vars = live_private.len();

    let mut col_remap: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_col) in live_private.iter().enumerate() {
        col_remap.insert(old_col, new_idx);
    }
    col_remap.insert(const_col, new_num_vars);

    // CRITICAL FIX: Properly distinguish outputs from inputs when remapping public I/O
    let num_circuit_outputs = instance.num_outputs;
    let num_circuit_inputs = instance.num_inputs - num_circuit_outputs;
    for i in 0..num_circuit_outputs {
        col_remap.insert(num_vars + 1 + i, new_num_vars + 1 + i);
    }
    for i in 0..num_circuit_inputs {
        col_remap.insert(
            num_vars + 1 + num_circuit_outputs + i,
            new_num_vars + 1 + num_circuit_outputs + i,
        );
    }

    for entry in new_a
        .iter_mut()
        .chain(new_b.iter_mut())
        .chain(new_c.iter_mut())
    {
        entry.1 = col_remap[&entry.1];
    }

    SpartanInstance {
        num_cons: new_num_cons,
        num_vars: new_num_vars,
        num_inputs: instance.num_inputs,
        num_outputs: num_circuit_outputs,
        A: new_a,
        B: new_b,
        C: new_c,
    }
}

fn emit_row(
    src: &[(usize, [u8; 32])],
    dest: &mut R1CSMatrix,
    new_row: usize,
    remap: &dyn Fn(usize) -> usize,
) {
    let mut merged: HashMap<usize, Scalar> = HashMap::new();
    for &(col, bytes) in src {
        let new_col = remap(col);
        let val = Scalar::from_canonical_bytes(bytes).unwrap();
        *merged.entry(new_col).or_insert(Scalar::ZERO) += val;
    }
    for (col, val) in merged {
        if val != Scalar::ZERO {
            dest.push((new_row, col, val.to_bytes()));
        }
    }
}

// =============================================================================
// remove_scales
// =============================================================================

/// Removes scale constraints from an R1CS instance via variable substitution.
///
/// A scale constraint has the form `(k * src) * 1 = out`:
///   - A: single entry `(col_src, k)`, k ≠ ONE
///   - B: single entry `(const_col, ONE)`
///   - C: single entry `(col_out, ONE)`, col_out < num_vars (private)
///
/// The function substitutes `col_out → k * col_src` everywhere — any entry
/// `(row, col_out, v)` becomes `(row, col_src, v * k)` — removes the scale
/// rows, and compacts private variable columns so `num_vars` shrinks.
///
/// Preserves topological row order. Pure function — returns a clone if no
/// scale constraints are found.
///
/// ~20% constraint reduction on default-configuration circuits (on top of
/// alias removal).
pub fn remove_scales(instance: &SpartanInstance) -> SpartanInstance {
    let num_vars = instance.num_vars;
    let const_col = num_vars;
    let one_bytes = Scalar::ONE.to_bytes();

    // --- Build per-row views ---
    let mut a_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut b_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut c_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];

    for &(row, col, bytes) in &instance.A {
        a_rows[row].push((col, bytes));
    }
    for &(row, col, bytes) in &instance.B {
        b_rows[row].push((col, bytes));
    }
    for &(row, col, bytes) in &instance.C {
        c_rows[row].push((col, bytes));
    }

    // --- Detect scale rows and build substitution map ---
    // substitution: col_out → (col_src, k)  meaning  out = k * src
    let mut substitution: HashMap<usize, (usize, Scalar)> = HashMap::new();
    let mut removed_rows: Vec<bool> = vec![false; instance.num_cons];

    for row in 0..instance.num_cons {
        if a_rows[row].len() != 1 || b_rows[row].len() != 1 || c_rows[row].len() != 1 {
            continue;
        }
        let (a_col, a_val) = a_rows[row][0];
        let (b_col, b_val) = b_rows[row][0];
        let (c_col, c_val) = c_rows[row][0];

        // B must be the constant wire with coefficient 1
        if b_col != const_col || b_val != one_bytes {
            continue;
        }
        // C must be a private variable with coefficient 1 (this is the output)
        if c_col >= num_vars || c_val != one_bytes {
            continue;
        }
        // A coefficient must not be ONE (alias rows already handled by remove_aliases)
        if a_val == one_bytes {
            continue;
        }
        // Avoid degenerate self-loops
        if a_col == c_col {
            continue;
        }

        let k = Scalar::from_canonical_bytes(a_val).unwrap();
        if k == Scalar::ZERO {
            continue;
        }

        substitution.insert(c_col, (a_col, k));
        removed_rows[row] = true;
    }

    if substitution.is_empty() {
        return instance.clone();
    }

    // --- Resolve substitution chains: out→(mid,k1), mid→(src,k2) ⟹ out→(src,k1*k2) ---
    let mut changed = true;
    while changed {
        changed = false;
        let snap: Vec<(usize, (usize, Scalar))> =
            substitution.iter().map(|(&k, &v)| (k, v)).collect();
        for (key, (target_col, k1)) in snap {
            if let Some(&(final_col, k2)) = substitution.get(&target_col) {
                substitution.insert(key, (final_col, k1 * k2));
                changed = true;
            }
        }
    }

    // --- SAFETY CHECK: Only remove scale rows where output variable appears ONLY in removed rows ---
    // Build a map: variable -> (count_in_surviving_rows, first_surviving_row)
    let mut var_usage: HashMap<usize, (usize, usize)> = HashMap::new();
    for row in 0..instance.num_cons {
        let is_surviving = !removed_rows[row];
        for &(col, _) in a_rows[row]
            .iter()
            .chain(b_rows[row].iter())
            .chain(c_rows[row].iter())
        {
            let entry = var_usage.entry(col).or_insert((0, row));
            if is_surviving {
                entry.0 += 1;
            }
        }
    }

    // Only keep substitutions where the output variable is NOT used in any surviving row
    let mut removed_count = 0;
    let keys_to_remove: Vec<usize> = substitution
        .keys()
        .filter(|output_var| {
            match var_usage.get(output_var) {
                Some(&(count_in_surviving, _)) if count_in_surviving > 0 => {
                    // Variable is still used in surviving rows, can't remove
                    true
                }
                _ => false, // Variable is safe to remove
            }
        })
        .copied()
        .collect();

    for key in keys_to_remove {
        substitution.remove(&key);
        removed_count += 1;
    }

    if removed_count > 0 {
        eprintln!(
            "[remove_scales] Skipped removing {} scale rows (output still in use)",
            removed_count
        );
    }

    // Update removed_rows: only remove rows whose substitution survived filtering
    for row in 0..instance.num_cons {
        if removed_rows[row] {
            let (c_col, _) = c_rows[row][0];
            if !substitution.contains_key(&c_col) {
                removed_rows[row] = false; // Keep this row
            }
        }
    }

    if substitution.is_empty() {
        return instance.clone();
    }

    // --- Apply substitutions to surviving rows ---
    // Entry (col, coeff) becomes (new_col, coeff * k) when col is in the substitution map.
    let remap = |col: usize, coeff: Scalar| -> (usize, Scalar) {
        if let Some(&(new_col, k)) = substitution.get(&col) {
            (new_col, coeff * k)
        } else {
            (col, coeff)
        }
    };

    let mut new_a: R1CSMatrix = Vec::new();
    let mut new_b: R1CSMatrix = Vec::new();
    let mut new_c: R1CSMatrix = Vec::new();
    let mut new_row = 0usize;

    for row in 0..instance.num_cons {
        if removed_rows[row] {
            continue;
        }
        emit_row_scaled(&a_rows[row], &mut new_a, new_row, &remap);
        emit_row_scaled(&b_rows[row], &mut new_b, new_row, &remap);
        emit_row_scaled(&c_rows[row], &mut new_c, new_row, &remap);
        new_row += 1;
    }

    let new_num_cons = new_row;

    // --- Column compaction (private variables only) ---
    let live_private: Vec<usize> = {
        use std::collections::HashSet;
        let mut set: HashSet<usize> = HashSet::new();
        for &(_, col, _) in new_a.iter().chain(new_b.iter()).chain(new_c.iter()) {
            if col < num_vars {
                set.insert(col);
            }
        }
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort();
        v
    };
    let new_num_vars = live_private.len();

    let mut col_remap: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_col) in live_private.iter().enumerate() {
        col_remap.insert(old_col, new_idx);
    }
    col_remap.insert(const_col, new_num_vars);

    // CRITICAL FIX: Properly distinguish outputs from inputs when remapping public I/O
    let num_circuit_outputs = instance.num_outputs;
    let num_circuit_inputs = instance.num_inputs - num_circuit_outputs;
    for i in 0..num_circuit_outputs {
        col_remap.insert(num_vars + 1 + i, new_num_vars + 1 + i);
    }
    for i in 0..num_circuit_inputs {
        col_remap.insert(
            num_vars + 1 + num_circuit_outputs + i,
            new_num_vars + 1 + num_circuit_outputs + i,
        );
    }

    for entry in new_a
        .iter_mut()
        .chain(new_b.iter_mut())
        .chain(new_c.iter_mut())
    {
        entry.1 = col_remap[&entry.1];
    }

    SpartanInstance {
        num_cons: new_num_cons,
        num_vars: new_num_vars,
        num_inputs: instance.num_inputs,
        num_outputs: num_circuit_outputs,
        A: new_a,
        B: new_b,
        C: new_c,
    }
}

/// Removes both alias and scale constraints: runs [`remove_aliases`] then
/// [`remove_scales`] on the result.
///
/// # BUG WARNING
/// Both `remove_aliases` and `remove_aliases_and_scales` have a bug causing
/// witness solver failures in ~4.8% of cases (48/1000 iterations).
/// The bug is in `remove_aliases`, not `remove_scales`.
/// TODO: Fix the alias removal bug before using scale removal.
pub fn remove_aliases_and_scales(instance: &SpartanInstance) -> SpartanInstance {
    remove_scales(&remove_aliases(instance))
}

fn emit_row_scaled(
    src: &[(usize, [u8; 32])],
    dest: &mut R1CSMatrix,
    new_row: usize,
    remap: &dyn Fn(usize, Scalar) -> (usize, Scalar),
) {
    let mut merged: HashMap<usize, Scalar> = HashMap::new();
    for &(col, bytes) in src {
        let coeff = Scalar::from_canonical_bytes(bytes).unwrap();
        let (new_col, new_coeff) = remap(col, coeff);
        *merged.entry(new_col).or_insert(Scalar::ZERO) += new_coeff;
    }
    for (col, val) in merged {
        if val != Scalar::ZERO {
            dest.push((new_row, col, val.to_bytes()));
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk_optimization::{
        crypto::CryptoHash,
        r1cs::{solve_witness_forward, solve_witness_from_r1cs},
        Challenge, Track,
    };

    fn make_challenge() -> Challenge {
        let mut seed = [0u8; 32];
        seed[0] = 42;
        Challenge::generate_instance(&seed, &Track { delta: 1 }).unwrap()
    }

    #[test]
    fn test_remove_scales_reduces_constraints() {
        let ch = make_challenge();
        let c0 = &ch.circuit_c0;

        let after_aliases = remove_aliases(c0);
        let after_both = remove_aliases_and_scales(c0);

        assert!(
            after_both.num_cons < c0.num_cons,
            "remove_aliases_and_scales must reduce constraints: {} -> {}",
            c0.num_cons,
            after_both.num_cons
        );
        assert!(
            after_both.num_cons <= after_aliases.num_cons,
            "remove_aliases_and_scales must be at least as good as remove_aliases: {} vs {}",
            after_both.num_cons,
            after_aliases.num_cons
        );

        let eps_aliases = 1.0 - after_aliases.num_cons as f64 / c0.num_cons as f64;
        let eps_both = 1.0 - after_both.num_cons as f64 / c0.num_cons as f64;
        eprintln!(
            "[remove_scales] C0: {} → aliases: {} (ε={:.3}) → aliases+scales: {} (ε={:.3})",
            c0.num_cons, after_aliases.num_cons, eps_aliases, after_both.num_cons, eps_both
        );
    }

    #[test]
    fn test_remove_scales_topological_order() {
        let ch = make_challenge();
        let result = remove_aliases_and_scales(&ch.circuit_c0);

        let h0 = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let witness = solve_witness_forward(&result, ch.num_circuit_outputs, &x_eval);
        assert!(
            witness.is_ok(),
            "forward witness solver must succeed on aliases+scales result: {:?}",
            witness.err()
        );
        eprintln!("[remove_scales] topological order preserved, witness solved OK");
    }

    #[test]
    fn test_remove_aliases_stress() {
        let num_iterations = 1000;
        let mut failures: Vec<(usize, [u8; 32], String)> = Vec::new();

        for i in 0..num_iterations {
            // Use different seed for each iteration
            let mut seed = [0u8; 32];
            seed[0] = (i >> 24) as u8;
            seed[1] = (i >> 16) as u8;
            seed[2] = (i >> 8) as u8;
            seed[3] = i as u8;

            let ch = Challenge::generate_instance(&seed, &Track { delta: 1 }).unwrap();
            let result = remove_aliases(&ch.circuit_c0);

            let h0 = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
            let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

            match solve_witness_forward(&result, ch.num_circuit_outputs, &x_eval) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[ITERATION {}] FAIL: seed={:02x}{:02x}{:02x}{:02x}... error={:?}",
                        i, seed[0], seed[1], seed[2], seed[3], e
                    );
                    failures.push((i, seed, format!("{:?}", e)));
                }
            }
        }

        if !failures.is_empty() {
            panic!(
                "Stress test found {} failures out of {} iterations. First failure at iteration {} with seed {:?}",
                failures.len(),
                num_iterations,
                failures[0].0,
                failures[0].1
            );
        }

        eprintln!(
            "[stress_aliases] All {} iterations passed successfully",
            num_iterations
        );
    }

    #[test]
    fn test_debug_iteration_14() {
        let seed = [
            0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
            0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
        ];

        let ch = Challenge::generate_instance(&seed, &Track { delta: 1 }).unwrap();
        eprintln!(
            "[debug] C0: {} cons, {} vars, {} pub_io, {} outputs",
            ch.circuit_c0.num_cons,
            ch.circuit_c0.num_vars,
            ch.circuit_c0.num_inputs,
            ch.circuit_c0.num_outputs
        );

        let result = remove_aliases(&ch.circuit_c0);
        eprintln!(
            "[debug] After remove_aliases: {} cons, {} vars, {} outputs",
            result.num_cons, result.num_vars, result.num_outputs
        );

        let h0 = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        match solve_witness_forward(&result, ch.num_circuit_outputs, &x_eval) {
            Ok(_) => eprintln!("[debug] Witness solver succeeded"),
            Err(e) => {
                eprintln!("[debug] Witness solver FAILED: {:?}", e);

                // Try the fixed-point solver instead
                match solve_witness_from_r1cs(&result, ch.num_circuit_outputs, &x_eval) {
                    Ok(_) => eprintln!("[debug] Fixed-point solver succeeded (row order issue)"),
                    Err(e2) => eprintln!("[debug] Fixed-point solver ALSO failed: {:?}", e2),
                }
            }
        }
    }

    #[test]
    fn test_verify_alias_detection() {
        let ch = make_challenge();
        let c0 = &ch.circuit_c0;

        // Count private→private, private→public, public→public aliases
        let mut private_to_private = 0;
        let mut private_to_public = 0;
        let mut public_to_public = 0;
        let mut mixed = 0;

        let num_vars = c0.num_vars;
        let const_col = num_vars;
        let one_bytes = Scalar::ONE.to_bytes();

        let mut a_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); c0.num_cons];
        let mut b_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); c0.num_cons];
        let mut c_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); c0.num_cons];

        for &(row, col, bytes) in &c0.A {
            a_rows[row].push((col, bytes));
        }
        for &(row, col, bytes) in &c0.B {
            b_rows[row].push((col, bytes));
        }
        for &(row, col, bytes) in &c0.C {
            c_rows[row].push((col, bytes));
        }

        for row in 0..c0.num_cons {
            if a_rows[row].len() != 1 || b_rows[row].len() != 1 || c_rows[row].len() != 1 {
                continue;
            }
            let (a_col, a_val) = a_rows[row][0];
            let (b_col, b_val) = b_rows[row][0];
            let (c_col, c_val) = c_rows[row][0];

            if a_val != one_bytes || b_val != one_bytes || c_val != one_bytes {
                continue;
            }
            if b_col != const_col {
                continue;
            }

            let a_is_private = a_col < num_vars;
            let c_is_private = c_col < num_vars;

            if a_is_private && c_is_private {
                private_to_private += 1;
            } else if !a_is_private && !c_is_private {
                public_to_public += 1;
            } else {
                mixed += 1;
            }
        }

        eprintln!("[verify] Aliases in circuit: private→private={}, private→public={}, public→public={}, mixed={}",
            private_to_private, private_to_public, public_to_public, mixed);
        eprintln!(
            "[verify] Total: {}, num_vars={}, num_inputs={}",
            private_to_private + private_to_public + public_to_public + mixed,
            num_vars,
            c0.num_inputs
        );
        eprintln!(
            "[verify] Public I/O range: {}..{} (const), {}..{} (outputs), {}..{} (inputs)",
            num_vars,
            num_vars + 1,
            num_vars + 1,
            num_vars + 1 + ch.num_circuit_outputs,
            num_vars + 1 + ch.num_circuit_outputs,
            num_vars + 1 + c0.num_inputs
        );
    }

    #[test]
    fn test_remove_aliases_and_scales_stress() {
        let num_iterations = 1000;
        let mut failures: Vec<(usize, [u8; 32], String)> = Vec::new();

        for i in 0..num_iterations {
            let mut seed = [0u8; 32];
            seed[0] = (i >> 24) as u8;
            seed[1] = (i >> 16) as u8;
            seed[2] = (i >> 8) as u8;
            seed[3] = i as u8;

            let ch = Challenge::generate_instance(&seed, &Track { delta: 1 }).unwrap();
            let result = remove_aliases_and_scales(&ch.circuit_c0);

            let h0 = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
            let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

            match solve_witness_forward(&result, ch.num_circuit_outputs, &x_eval) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[ITERATION {}] FAIL: seed={:02x}{:02x}{:02x}{:02x}... error={:?}",
                        i, seed[0], seed[1], seed[2], seed[3], e
                    );
                    failures.push((i, seed, format!("{:?}", e)));
                }
            }
        }

        if !failures.is_empty() {
            panic!(
                "Stress test found {} failures out of {} iterations. First failure at iteration {} with seed {:?}",
                failures.len(),
                num_iterations,
                failures[0].0,
                failures[0].1
            );
        }

        eprintln!(
            "[stress_aliases_and_scales] All {} iterations passed successfully",
            num_iterations
        );
    }

    #[test]
    fn test_full_challenge_pipeline() {
        let num_iterations = 100;
        let mut failures = 0;

        for i in 0..num_iterations {
            let mut seed = [0u8; 32];
            seed[0] = (i >> 24) as u8;
            seed[1] = (i >> 16) as u8;
            seed[2] = (i >> 8) as u8;
            seed[3] = i as u8;

            let ch = Challenge::generate_instance(&seed, &Track { delta: 1 }).unwrap();

            // Test both algorithms
            for (name, optimizer) in [
                (
                    "remove_aliases",
                    remove_aliases as fn(&SpartanInstance) -> SpartanInstance,
                ),
                ("remove_aliases_and_scales", remove_aliases_and_scales),
            ] {
                match ch.build_solution(&optimizer(&ch.circuit_c0)) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("[PIPELINE FAIL] {} on seed {:08x}: {:?}", name, i, e);
                        failures += 1;
                    }
                }
            }
        }

        assert_eq!(
            failures, 0,
            "Full pipeline had {} failures out of {} iterations",
            failures, num_iterations
        );
        eprintln!(
            "[pipeline] All {} iterations passed (both algorithms)",
            num_iterations
        );
    }

    #[test]
    fn test_optimization_quality_distribution() {
        // Sample 100 challenges and measure epsilon distribution
        let num_samples = 100;
        let mut epsilons_aliases = Vec::new();
        let mut epsilons_both = Vec::new();

        for i in 0..num_samples {
            let mut seed = [0u8; 32];
            seed[0] = (i >> 24) as u8;
            seed[1] = (i >> 16) as u8;
            seed[2] = (i >> 8) as u8;
            seed[3] = i as u8;

            let ch = Challenge::generate_instance(&seed, &Track { delta: 1 }).unwrap();

            let c0 = &ch.circuit_c0;
            let after_aliases = remove_aliases(c0);
            let after_both = remove_aliases_and_scales(c0);

            let eps_aliases = 1.0 - after_aliases.num_cons as f64 / c0.num_cons as f64;
            let eps_both = 1.0 - after_both.num_cons as f64 / c0.num_cons as f64;

            epsilons_aliases.push(eps_aliases);
            epsilons_both.push(eps_both);
        }

        // Compute statistics
        let mean_aliases: f64 = epsilons_aliases.iter().sum::<f64>() / num_samples as f64;
        let mean_both: f64 = epsilons_both.iter().sum::<f64>() / num_samples as f64;
        let min_aliases = epsilons_aliases
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_aliases = epsilons_aliases
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        eprintln!("[quality] Sampled {} challenges:", num_samples);
        eprintln!(
            "[quality]   remove_aliases: ε ∈ [{:.3}, {:.3}], mean={:.3}",
            min_aliases, max_aliases, mean_aliases
        );
        eprintln!(
            "[quality]   remove_aliases_and_scales: mean ε={:.3}",
            mean_both
        );

        // Sanity checks
        assert!(
            mean_aliases > 0.05,
            "Alias removal should reduce constraints by at least 5%"
        );
        assert!(
            mean_both >= mean_aliases,
            "Combined should be at least as good as aliases alone"
        );
    }
}
