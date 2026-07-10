use rand::{rngs::SmallRng, Rng};

use super::formula::{is_lit_sat, Formula};

const ABSENT: u32 = u32::MAX;

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[derive(Clone, Debug)]
pub(crate) struct State {
    pub(crate) num_good: Vec<u8>,
    pub(crate) break_score: Vec<u16>,
    pub(crate) make_score: Vec<u16>,
    clause_weight: Vec<u16>,
    break_score_w: Vec<u32>,
    make_score_w: Vec<u32>,
    sat_xor: Vec<u32>,
    unsat_since: Vec<u32>,
    step: u32,
    last_flip: Vec<u32>,
    unsat: Vec<u32>,
    unsat_pos: Vec<u32>,
}

impl State {
    #[inline(always)]
    pub(crate) fn unsat_len(&self) -> usize {
        self.unsat.len()
    }
}

pub(crate) fn init_state(formula: &Formula, vars: &[bool], track_unsat_age: bool) -> State {
    let mut state = State {
        num_good: vec![0u8; formula.nc],
        break_score: vec![0u16; formula.nv],
        make_score: vec![0u16; formula.nv],
        clause_weight: vec![1u16; formula.nc],
        break_score_w: vec![0u32; formula.nv],
        make_score_w: vec![0u32; formula.nv],
        sat_xor: vec![0u32; formula.nc],
        unsat_since: if track_unsat_age {
            vec![0u32; formula.nc]
        } else {
            Vec::new()
        },
        step: 1,
        last_flip: vec![0u32; formula.nv],
        unsat: Vec::with_capacity(formula.nc / 16 + 8),
        unsat_pos: vec![ABSENT; formula.nc],
    };

    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        for c in 0..formula.nc {
            let s = c * 3;
            let (cnt, xor) = clause3_sat_count_and_xor(&formula.cl, s, vars);
            state.num_good[c] = cnt;
            state.sat_xor[c] = xor;
            if cnt == 0 {
                add_unsat(&mut state, c);
                add_make_lit(&mut state.make_score, formula.cl[s]);
                add_make_lit(&mut state.make_score, formula.cl[s + 1]);
                add_make_lit(&mut state.make_score, formula.cl[s + 2]);
                add_make_weight_lit(&mut state.make_score_w, formula.cl[s], 1);
                add_make_weight_lit(&mut state.make_score_w, formula.cl[s + 1], 1);
                add_make_weight_lit(&mut state.make_score_w, formula.cl[s + 2], 1);
            } else if cnt == 1 {
                inc_break(&mut state.break_score, xor as usize);
                inc_break_w(&mut state.break_score_w, xor as usize, 1);
            }
        }
        return state;
    }

    for c in 0..formula.nc {
        let (cnt, xor) = clause_sat_count_and_xor(formula, c, vars);
        state.num_good[c] = cnt;
        state.sat_xor[c] = xor;
        if cnt == 0 {
            add_unsat(&mut state, c);
            add_make_for_clause(formula, &mut state.make_score, c);
            add_make_weight_for_clause(formula, &mut state.make_score_w, c, 1);
        } else if cnt == 1 {
            inc_break(&mut state.break_score, xor as usize);
            inc_break_w(&mut state.break_score_w, xor as usize, 1);
        }
    }

    state
}

#[inline(always)]
pub(crate) fn choose_unsat_clause(
    state: &State,
    rng: &mut SmallRng,
    clause_pick_samples: usize,
) -> usize {
    let id = rng.gen::<usize>() % state.unsat.len();
    if clause_pick_samples <= 1 || state.unsat_since.is_empty() {
        return state.unsat[id] as usize;
    }
    let mut best = state.unsat[id] as usize;
    let mut best_age = state.step.wrapping_sub(state.unsat_since[best]);
    let samples = clause_pick_samples.min(state.unsat.len());
    for _ in 1..samples {
        let cand = state.unsat[rng.gen::<usize>() % state.unsat.len()] as usize;
        let age = state.step.wrapping_sub(state.unsat_since[cand]);
        if age > best_age {
            best = cand;
            best_age = age;
        }
    }
    best
}

pub(crate) fn choose_var_from_clause(
    formula: &Formula,
    state: &State,
    c: usize,
    rng: &mut SmallRng,
    noise_threshold: u64,
    make_mult: i32,
    break_mult: i32,
    use_make_score: bool,
    use_clause_weights: bool,
    age_shift: u32,
    age_cap: i32,
) -> usize {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        let s = c * 3;

        if rng.gen::<u64>() < noise_threshold {
            let lit = formula.cl[s + (rng.gen::<usize>() % 3)];
            return lit_var_index(lit);
        }

        let mut best_var = lit_var_index(formula.cl[s]);
        let mut best_score = i64::MIN;
        let mut ties = 0usize;

        consider_clause_var(
            state,
            lit_var_index(formula.cl[s]),
            rng,
            &mut best_var,
            &mut best_score,
            &mut ties,
            make_mult,
            break_mult,
            use_make_score,
            use_clause_weights,
            age_shift,
            age_cap,
        );
        consider_clause_var(
            state,
            lit_var_index(formula.cl[s + 1]),
            rng,
            &mut best_var,
            &mut best_score,
            &mut ties,
            make_mult,
            break_mult,
            use_make_score,
            use_clause_weights,
            age_shift,
            age_cap,
        );
        consider_clause_var(
            state,
            lit_var_index(formula.cl[s + 2]),
            rng,
            &mut best_var,
            &mut best_score,
            &mut ties,
            make_mult,
            break_mult,
            use_make_score,
            use_clause_weights,
            age_shift,
            age_cap,
        );
        return best_var;
    }

    let (s, len) = solver_clause_bounds(formula, c);
    let e = s + len;
    debug_assert!(len > 0);
    if len == 0 {
        return 0;
    }

    if len > 1 && rng.gen::<u64>() < noise_threshold {
        let lit = formula.cl[s + (rng.gen::<usize>() % len)];
        return lit_var_index(lit);
    }

    let mut best_var = lit_var_index(formula.cl[s]);
    let mut best_score = i64::MIN;
    let mut ties = 0usize;

    for i in s..e {
        let v = lit_var_index(formula.cl[i]);
        consider_clause_var(
            state,
            v,
            rng,
            &mut best_var,
            &mut best_score,
            &mut ties,
            make_mult,
            break_mult,
            use_make_score,
            use_clause_weights,
            age_shift,
            age_cap,
        );
    }
    best_var
}

#[inline(always)]
fn consider_clause_var(
    state: &State,
    v: usize,
    rng: &mut SmallRng,
    best_var: &mut usize,
    best_score: &mut i64,
    ties: &mut usize,
    make_mult: i32,
    break_mult: i32,
    use_make_score: bool,
    use_clause_weights: bool,
    age_shift: u32,
    age_cap: i32,
) {
    let make = if use_make_score {
        if use_clause_weights {
            state.make_score_w[v]
        } else {
            u32::from(state.make_score[v])
        }
    } else {
        1
    };
    let br = if use_clause_weights {
        state.break_score_w[v]
    } else {
        u32::from(state.break_score[v])
    };
    let age = age_bonus(state, v, age_shift, age_cap);
    let score = i64::from(make_mult) * i64::from(make) - i64::from(break_mult) * i64::from(br)
        + i64::from(age);

    if score > *best_score {
        *best_score = score;
        *best_var = v;
        *ties = 1;
    } else if score == *best_score {
        *ties += 1;
        if rng.gen::<usize>() % *ties == 0 {
            *best_var = v;
        }
    }
}

pub(crate) fn flip_var(formula: &Formula, state: &mut State, vars: &mut [bool], v: usize) {
    let old_val = vars[v];
    let (inc_occ, dec_occ) = if old_val {
        (formula.neg_occ(v), formula.pos_occ(v))
    } else {
        (formula.pos_occ(v), formula.neg_occ(v))
    };

    for &cid in inc_occ {
        let c = cid as usize;
        let old = state.num_good[c];
        let w = u32::from(state.clause_weight[c]);
        debug_assert!(old <= 2);

        if old == 1 {
            let sole = state.sat_xor[c] as usize;
            dec_break(&mut state.break_score, sole);
            dec_break_w(&mut state.break_score_w, sole, w);
        }

        state.num_good[c] = old + 1;
        state.sat_xor[c] ^= v as u32;
        if old == 0 {
            remove_unsat(state, c);
            remove_make_for_clause(formula, &mut state.make_score, c);
            remove_make_weight_for_clause(formula, &mut state.make_score_w, c, w);
            inc_break(&mut state.break_score, v);
            inc_break_w(&mut state.break_score_w, v, w);
        }
    }

    for &cid in dec_occ {
        let c = cid as usize;
        let old = state.num_good[c];
        let w = u32::from(state.clause_weight[c]);
        debug_assert!(old >= 1);

        if old == 1 {
            dec_break(&mut state.break_score, v);
            dec_break_w(&mut state.break_score_w, v, w);
        }

        state.num_good[c] = old - 1;
        state.sat_xor[c] ^= v as u32;
        if old == 1 {
            add_unsat(state, c);
            add_make_for_clause(formula, &mut state.make_score, c);
            add_make_weight_for_clause(formula, &mut state.make_score_w, c, w);
        } else if old == 2 {
            let sole = state.sat_xor[c] as usize;
            inc_break(&mut state.break_score, sole);
            inc_break_w(&mut state.break_score_w, sole, w);
        }
    }

    vars[v] = !old_val;
    state.step = state.step.wrapping_add(1).max(1);
    state.last_flip[v] = state.step;
}

pub(crate) fn verify_invariants(formula: &Formula, state: &State, vars: &[bool]) {
    let mut expected_good = vec![0u8; formula.nc];
    let mut expected_break = vec![0u16; formula.nv];
    let mut expected_make = vec![0u16; formula.nv];
    let mut expected_break_w = vec![0u32; formula.nv];
    let mut expected_make_w = vec![0u32; formula.nv];
    let mut expected_xor = vec![0u32; formula.nc];
    let mut expected_unsat = vec![false; formula.nc];

    for c in 0..formula.nc {
        let (cnt, xor) = clause_sat_count_and_xor(formula, c, vars);
        let w = u32::from(state.clause_weight[c]);
        expected_good[c] = cnt;
        expected_xor[c] = xor;
        if cnt == 0 {
            expected_unsat[c] = true;
            add_make_for_clause(formula, &mut expected_make, c);
            add_make_weight_for_clause(formula, &mut expected_make_w, c, w);
        } else if cnt == 1 {
            inc_break(&mut expected_break, xor as usize);
            inc_break_w(&mut expected_break_w, xor as usize, w);
        }
    }

    assert_eq!(state.num_good, expected_good);
    assert_eq!(state.break_score, expected_break);
    assert_eq!(state.make_score, expected_make);
    assert_eq!(state.break_score_w, expected_break_w);
    assert_eq!(state.make_score_w, expected_make_w);
    assert_eq!(state.sat_xor, expected_xor);

    let mut actual_unsat = vec![false; formula.nc];
    for (pos, &cid) in state.unsat.iter().enumerate() {
        let c = cid as usize;
        assert!(c < formula.nc);
        assert_eq!(state.unsat_pos[c], pos as u32);
        actual_unsat[c] = true;
    }
    assert_eq!(actual_unsat, expected_unsat);
}

pub(crate) fn bump_unsat_weights(formula: &Formula, state: &mut State, max_weight: u16) -> bool {
    if max_weight <= 1 || state.unsat.is_empty() {
        return false;
    }

    let mut capped = 0usize;
    for i in 0..state.unsat.len() {
        let c = state.unsat[i] as usize;
        if state.clause_weight[c] < max_weight {
            state.clause_weight[c] += 1;
            add_make_weight_for_clause(formula, &mut state.make_score_w, c, 1);
        }
        if state.clause_weight[c] >= max_weight {
            capped += 1;
        }
    }

    capped * 4 >= state.unsat.len() * 3
}

pub(crate) fn rescale_clause_weights(formula: &Formula, state: &mut State, vars: &[bool]) {
    for w in &mut state.clause_weight {
        *w = ((*w as u32 + 1) / 2).max(1) as u16;
    }
    rebuild_weighted_scores(formula, state, vars);
}

fn rebuild_weighted_scores(formula: &Formula, state: &mut State, vars: &[bool]) {
    state.break_score_w.fill(0);
    state.make_score_w.fill(0);

    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        for c in 0..formula.nc {
            let s = c * 3;
            let (cnt, xor) = clause3_sat_count_and_xor(&formula.cl, s, vars);
            debug_assert_eq!(cnt, state.num_good[c]);
            debug_assert_eq!(xor, state.sat_xor[c]);
            let w = u32::from(state.clause_weight[c]);
            if cnt == 0 {
                add_make_weight_lit(&mut state.make_score_w, formula.cl[s], w);
                add_make_weight_lit(&mut state.make_score_w, formula.cl[s + 1], w);
                add_make_weight_lit(&mut state.make_score_w, formula.cl[s + 2], w);
            } else if cnt == 1 {
                inc_break_w(&mut state.break_score_w, xor as usize, w);
            }
        }
        return;
    }

    for c in 0..formula.nc {
        let (cnt, xor) = clause_sat_count_and_xor(formula, c, vars);
        debug_assert_eq!(cnt, state.num_good[c]);
        debug_assert_eq!(xor, state.sat_xor[c]);
        let w = u32::from(state.clause_weight[c]);
        if cnt == 0 {
            add_make_weight_for_clause(formula, &mut state.make_score_w, c, w);
        } else if cnt == 1 {
            inc_break_w(&mut state.break_score_w, xor as usize, w);
        }
    }
}

#[inline(always)]
fn age_bonus(state: &State, v: usize, age_shift: u32, age_cap: i32) -> i32 {
    let age = state.step.wrapping_sub(state.last_flip[v]) >> age_shift;
    (age.min(age_cap.max(0) as u32)) as i32
}

#[inline(always)]
fn solver_clause_bounds(formula: &Formula, c: usize) -> (usize, usize) {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        (c * 3, 3)
    } else {
        let s = formula.co[c] as usize;
        (s, formula.co[c + 1] as usize - s)
    }
}

fn clause_sat_count_and_xor(formula: &Formula, c: usize, vars: &[bool]) -> (u8, u32) {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        return clause3_sat_count_and_xor(&formula.cl, c * 3, vars);
    }

    let (s, len) = solver_clause_bounds(formula, c);
    let e = s + len;
    let mut cnt = 0u8;
    let mut xor = 0u32;
    for i in s..e {
        let lit = formula.cl[i];
        if is_lit_sat(lit, vars) {
            cnt += 1;
            xor ^= lit_var_index(lit) as u32;
        }
    }
    (cnt, xor)
}

#[inline(always)]
fn clause3_sat_count_and_xor(cl: &[i32], s: usize, vars: &[bool]) -> (u8, u32) {
    let mut cnt = 0u8;
    let mut xor = 0u32;
    let lit0 = cl[s];
    if is_lit_sat(lit0, vars) {
        cnt += 1;
        xor ^= lit_var_index(lit0) as u32;
    }
    let lit1 = cl[s + 1];
    if is_lit_sat(lit1, vars) {
        cnt += 1;
        xor ^= lit_var_index(lit1) as u32;
    }
    let lit2 = cl[s + 2];
    if is_lit_sat(lit2, vars) {
        cnt += 1;
        xor ^= lit_var_index(lit2) as u32;
    }
    (cnt, xor)
}

#[inline(always)]
fn add_unsat(state: &mut State, c: usize) {
    if state.unsat_pos[c] != ABSENT {
        return;
    }
    state.unsat_pos[c] = state.unsat.len() as u32;
    if !state.unsat_since.is_empty() {
        state.unsat_since[c] = state.step;
    }
    state.unsat.push(c as u32);
}

#[inline(always)]
fn remove_unsat(state: &mut State, c: usize) {
    let pos = state.unsat_pos[c];
    if pos == ABSENT {
        return;
    }
    let pos = pos as usize;
    let last = state.unsat.pop().expect("unsat set position without item");
    if pos < state.unsat.len() {
        state.unsat[pos] = last;
        state.unsat_pos[last as usize] = pos as u32;
    }
    state.unsat_pos[c] = ABSENT;
    if !state.unsat_since.is_empty() {
        state.unsat_since[c] = 0;
    }
}

#[inline(always)]
fn add_make_for_clause(formula: &Formula, make_score: &mut [u16], c: usize) {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        let s = c * 3;
        add_make_lit(make_score, formula.cl[s]);
        add_make_lit(make_score, formula.cl[s + 1]);
        add_make_lit(make_score, formula.cl[s + 2]);
        return;
    }

    let (s, len) = solver_clause_bounds(formula, c);
    let e = s + len;
    for i in s..e {
        add_make_lit(make_score, formula.cl[i]);
    }
}

#[inline(always)]
fn remove_make_for_clause(formula: &Formula, make_score: &mut [u16], c: usize) {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        let s = c * 3;
        remove_make_lit(make_score, formula.cl[s]);
        remove_make_lit(make_score, formula.cl[s + 1]);
        remove_make_lit(make_score, formula.cl[s + 2]);
        return;
    }

    let (s, len) = solver_clause_bounds(formula, c);
    let e = s + len;
    for i in s..e {
        remove_make_lit(make_score, formula.cl[i]);
    }
}

#[inline(always)]
fn add_make_weight_for_clause(formula: &Formula, make_score_w: &mut [u32], c: usize, delta: u32) {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        let s = c * 3;
        add_make_weight_lit(make_score_w, formula.cl[s], delta);
        add_make_weight_lit(make_score_w, formula.cl[s + 1], delta);
        add_make_weight_lit(make_score_w, formula.cl[s + 2], delta);
        return;
    }

    let (s, len) = solver_clause_bounds(formula, c);
    let e = s + len;
    for i in s..e {
        add_make_weight_lit(make_score_w, formula.cl[i], delta);
    }
}

#[inline(always)]
fn remove_make_weight_for_clause(
    formula: &Formula,
    make_score_w: &mut [u32],
    c: usize,
    delta: u32,
) {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        let s = c * 3;
        remove_make_weight_lit(make_score_w, formula.cl[s], delta);
        remove_make_weight_lit(make_score_w, formula.cl[s + 1], delta);
        remove_make_weight_lit(make_score_w, formula.cl[s + 2], delta);
        return;
    }

    let (s, len) = solver_clause_bounds(formula, c);
    let e = s + len;
    for i in s..e {
        remove_make_weight_lit(make_score_w, formula.cl[i], delta);
    }
}

#[inline(always)]
fn add_make_lit(make_score: &mut [u16], lit: i32) {
    let v = lit_var_index(lit);
    debug_assert!(make_score[v] < u16::MAX);
    make_score[v] += 1;
}

#[inline(always)]
fn remove_make_lit(make_score: &mut [u16], lit: i32) {
    let v = lit_var_index(lit);
    debug_assert!(make_score[v] > 0);
    make_score[v] -= 1;
}

#[inline(always)]
fn add_make_weight_lit(make_score_w: &mut [u32], lit: i32, delta: u32) {
    let v = lit_var_index(lit);
    debug_assert!(make_score_w[v] <= u32::MAX - delta);
    make_score_w[v] += delta;
}

#[inline(always)]
fn remove_make_weight_lit(make_score_w: &mut [u32], lit: i32, delta: u32) {
    let v = lit_var_index(lit);
    debug_assert!(make_score_w[v] >= delta);
    make_score_w[v] -= delta;
}

#[inline(always)]
fn inc_break(scores: &mut [u16], v: usize) {
    debug_assert!(scores[v] < u16::MAX);
    scores[v] += 1;
}

#[inline(always)]
fn dec_break(scores: &mut [u16], v: usize) {
    debug_assert!(scores[v] > 0);
    scores[v] -= 1;
}

#[inline(always)]
fn inc_break_w(scores: &mut [u32], v: usize, delta: u32) {
    debug_assert!(scores[v] <= u32::MAX - delta);
    scores[v] += delta;
}

#[inline(always)]
fn dec_break_w(scores: &mut [u32], v: usize, delta: u32) {
    debug_assert!(scores[v] >= delta);
    scores[v] -= delta;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn solver_lit_var_index_matches_unsigned_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), lit.unsigned_abs() as usize - 1);
        }
    }

    #[test]
    fn solver_clause_bounds_all_three_fast_path_matches_offsets() {
        let formula = Formula::from_raw(
            4,
            &[
                vec![1, -2, 3],
                vec![-1, 2, -3],
                vec![1, 3, -4],
                vec![-1, -3, 4],
            ],
        );
        assert!(formula.all_clauses_are_three);

        for c in 0..formula.nc {
            let (s, len) = solver_clause_bounds(&formula, c);
            assert_eq!(s, formula.co[c] as usize);
            assert_eq!(len, formula.co[c + 1] as usize - formula.co[c] as usize);
        }
    }

    #[test]
    fn solver_clause_bounds_mixed_lengths_use_offsets() {
        let formula = Formula::from_raw(3, &[vec![1, 2, 3], vec![1, 1, -2], vec![-1, 2]]);
        assert!(!formula.all_clauses_are_three);

        for c in 0..formula.nc {
            let (s, len) = solver_clause_bounds(&formula, c);
            assert_eq!(s, formula.co[c] as usize);
            assert_eq!(len, formula.co[c + 1] as usize - formula.co[c] as usize);
        }
    }

    #[test]
    fn solver_clause_count_and_make_updates_follow_bounds() {
        let formula = Formula::from_raw(
            4,
            &[
                vec![1, -2, 3],
                vec![-1, 2, -3],
                vec![1, 3, -4],
                vec![-1, -3, 4],
            ],
        );
        assert!(formula.all_clauses_are_three);
        let vars = vec![true, false, false, true];

        for c in 0..formula.nc {
            let (s, len) = solver_clause_bounds(&formula, c);
            let mut expected_cnt = 0u8;
            let mut expected_xor = 0u32;
            let mut expected_make = vec![0u16; formula.nv];
            let mut expected_make_w = vec![0u32; formula.nv];
            for &lit in &formula.cl[s..s + len] {
                let v = lit_var_index(lit);
                if is_lit_sat(lit, &vars) {
                    expected_cnt += 1;
                    expected_xor ^= v as u32;
                }
                expected_make[v] += 1;
                expected_make_w[v] += 7;
            }

            assert_eq!(
                clause_sat_count_and_xor(&formula, c, &vars),
                (expected_cnt, expected_xor)
            );

            let mut make = vec![0u16; formula.nv];
            let mut make_w = vec![0u32; formula.nv];
            add_make_for_clause(&formula, &mut make, c);
            add_make_weight_for_clause(&formula, &mut make_w, c, 7);
            assert_eq!(make, expected_make);
            assert_eq!(make_w, expected_make_w);

            remove_make_for_clause(&formula, &mut make, c);
            remove_make_weight_for_clause(&formula, &mut make_w, c, 7);
            assert_eq!(make, vec![0u16; formula.nv]);
            assert_eq!(make_w, vec![0u32; formula.nv]);
        }
    }

    #[test]
    fn solver_init_and_weight_rebuild_all_three_fast_paths_match_generic_reference() {
        let formula = Formula::from_raw(
            5,
            &[
                vec![1, -2, 3],
                vec![-1, 2, -3],
                vec![2, 4, -5],
                vec![-2, -4, 5],
                vec![1, 3, -5],
            ],
        );
        assert!(formula.all_clauses_are_three);
        let mut generic_formula = formula.clone();
        generic_formula.all_clauses_are_three = false;
        let vars = vec![true, false, false, true, false];

        let mut direct = init_state(&formula, &vars, true);
        let mut generic = init_state(&generic_formula, &vars, true);
        assert_eq!(direct.num_good, generic.num_good);
        assert_eq!(direct.break_score, generic.break_score);
        assert_eq!(direct.make_score, generic.make_score);
        assert_eq!(direct.clause_weight, generic.clause_weight);
        assert_eq!(direct.break_score_w, generic.break_score_w);
        assert_eq!(direct.make_score_w, generic.make_score_w);
        assert_eq!(direct.sat_xor, generic.sat_xor);
        assert_eq!(direct.unsat_since, generic.unsat_since);
        assert_eq!(direct.unsat, generic.unsat);
        assert_eq!(direct.unsat_pos, generic.unsat_pos);

        for (i, weight) in [7_u16, 3, 11, 5, 9].into_iter().enumerate() {
            direct.clause_weight[i] = weight;
            generic.clause_weight[i] = weight;
        }
        direct.break_score_w.fill(99);
        direct.make_score_w.fill(77);
        generic.break_score_w.fill(99);
        generic.make_score_w.fill(77);

        rebuild_weighted_scores(&formula, &mut direct, &vars);
        rebuild_weighted_scores(&generic_formula, &mut generic, &vars);
        assert_eq!(direct.break_score_w, generic.break_score_w);
        assert_eq!(direct.make_score_w, generic.make_score_w);
    }

    #[test]
    fn choose_var_all_three_direct_path_matches_generic_reference() {
        let formula = Formula::from_raw(
            5,
            &[
                vec![1, -2, 3],
                vec![-1, 2, -3],
                vec![2, 4, -5],
                vec![-2, -4, 5],
            ],
        );
        assert!(formula.all_clauses_are_three);
        let vars = vec![true, false, true, false, true];
        let state = init_state(&formula, &vars, true);

        let configs = [
            (0, 1, 1, true, false, 0, 16),
            (0, 3, 2, true, true, 1, 8),
            (u64::MAX, 2, 5, true, false, 0, 16),
            (0, 1, 4, false, true, 2, 4),
        ];

        for &(noise, make_mult, break_mult, use_make, use_weights, age_shift, age_cap) in &configs {
            for c in 0..formula.nc {
                for seed in [
                    0x0123_4567_89ab_cdef,
                    0x1111_2222_3333_4444,
                    0xfeed_face_cafe_beef,
                ] {
                    let mut direct_rng = SmallRng::seed_from_u64(seed);
                    let mut generic_rng = SmallRng::seed_from_u64(seed);

                    let direct = choose_var_from_clause(
                        &formula,
                        &state,
                        c,
                        &mut direct_rng,
                        noise,
                        make_mult,
                        break_mult,
                        use_make,
                        use_weights,
                        age_shift,
                        age_cap,
                    );
                    let generic = choose_var_from_clause_generic_reference(
                        &formula,
                        &state,
                        c,
                        &mut generic_rng,
                        noise,
                        make_mult,
                        break_mult,
                        use_make,
                        use_weights,
                        age_shift,
                        age_cap,
                    );

                    assert_eq!(direct, generic);
                }
            }
        }
    }

    fn choose_var_from_clause_generic_reference(
        formula: &Formula,
        state: &State,
        c: usize,
        rng: &mut SmallRng,
        noise_threshold: u64,
        make_mult: i32,
        break_mult: i32,
        use_make_score: bool,
        use_clause_weights: bool,
        age_shift: u32,
        age_cap: i32,
    ) -> usize {
        let s = formula.co[c] as usize;
        let len = formula.co[c + 1] as usize - s;
        let e = s + len;
        if len == 0 {
            return 0;
        }

        if len > 1 && rng.gen::<u64>() < noise_threshold {
            let lit = formula.cl[s + (rng.gen::<usize>() % len)];
            return lit_var_index(lit);
        }

        let mut best_var = lit_var_index(formula.cl[s]);
        let mut best_score = i64::MIN;
        let mut ties = 0usize;

        for i in s..e {
            consider_clause_var(
                state,
                lit_var_index(formula.cl[i]),
                rng,
                &mut best_var,
                &mut best_score,
                &mut ties,
                make_mult,
                break_mult,
                use_make_score,
                use_clause_weights,
                age_shift,
                age_cap,
            );
        }
        best_var
    }
}
