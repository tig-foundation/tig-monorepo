use crate::zk::r1cs::{R1CSMatrix, SpartanInstance};
use curve25519_dalek::scalar::Scalar;
use std::collections::HashMap;

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

    for &(row, col, bytes) in &instance.A { a_rows[row].push((col, bytes)); }
    for &(row, col, bytes) in &instance.B { b_rows[row].push((col, bytes)); }
    for &(row, col, bytes) in &instance.C { c_rows[row].push((col, bytes)); }

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

        if a_val != one_bytes || b_val != one_bytes || c_val != one_bytes { continue; }
        if b_col != const_col { continue; }

        // a_col = c_col; substitute away whichever is private
        if a_col < num_vars {
            substitution.insert(a_col, c_col);
            removed_rows[row] = true;
        } else if c_col < num_vars {
            substitution.insert(c_col, a_col);
            removed_rows[row] = true;
        }
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
        if removed_rows[row] { continue; }

        emit_row(&a_rows[row], &mut new_a, new_row, &remap_col);
        emit_row(&b_rows[row], &mut new_b, new_row, &remap_col);
        emit_row(&c_rows[row], &mut new_c, new_row, &remap_col);
        new_row += 1;
    }

    let new_num_cons = new_row;

    // --- Column compaction (private variables only) ---
    let live_private: Vec<usize> = {
        use std::collections::HashSet;
        let mut set: HashSet<usize> = HashSet::new();
        for &(_, col, _) in new_a.iter().chain(new_b.iter()).chain(new_c.iter()) {
            if col < num_vars { set.insert(col); }
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
    for i in 0..instance.num_inputs {
        col_remap.insert(num_vars + 1 + i, new_num_vars + 1 + i);
    }

    for entry in new_a.iter_mut().chain(new_b.iter_mut()).chain(new_c.iter_mut()) {
        entry.1 = col_remap[&entry.1];
    }

    SpartanInstance {
        num_cons: new_num_cons,
        num_vars: new_num_vars,
        num_inputs: instance.num_inputs,
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
