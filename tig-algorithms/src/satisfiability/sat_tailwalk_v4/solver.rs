use rand::{rngs::SmallRng, Rng};

use super::formula::{is_lit_sat, Formula};

const ABSENT: u32 = u32::MAX;

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
    let s = formula.co[c] as usize;
    let e = formula.co[c + 1] as usize;
    let len = e - s;
    debug_assert!(len > 0);
    if len == 0 {
        return 0;
    }

    if len > 1 && rng.gen::<u64>() < noise_threshold {
        let lit = formula.cl[s + (rng.gen::<usize>() % len)];
        return lit.unsigned_abs() as usize - 1;
    }

    let mut best_var = formula.cl[s].unsigned_abs() as usize - 1;
    let mut best_score = i64::MIN;
    let mut ties = 0usize;

    for i in s..e {
        let v = formula.cl[i].unsigned_abs() as usize - 1;
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

        if score > best_score {
            best_score = score;
            best_var = v;
            ties = 1;
        } else if score == best_score {
            ties += 1;
            if rng.gen::<usize>() % ties == 0 {
                best_var = v;
            }
        }
    }
    best_var
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

fn clause_sat_count_and_xor(formula: &Formula, c: usize, vars: &[bool]) -> (u8, u32) {
    let s = formula.co[c] as usize;
    let e = formula.co[c + 1] as usize;
    let mut cnt = 0u8;
    let mut xor = 0u32;
    for i in s..e {
        let lit = formula.cl[i];
        if is_lit_sat(lit, vars) {
            cnt += 1;
            xor ^= lit.unsigned_abs() - 1;
        }
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
    let s = formula.co[c] as usize;
    let e = formula.co[c + 1] as usize;
    for i in s..e {
        let v = formula.cl[i].unsigned_abs() as usize - 1;
        make_score[v] = make_score[v].saturating_add(1);
    }
}

#[inline(always)]
fn remove_make_for_clause(formula: &Formula, make_score: &mut [u16], c: usize) {
    let s = formula.co[c] as usize;
    let e = formula.co[c + 1] as usize;
    for i in s..e {
        let v = formula.cl[i].unsigned_abs() as usize - 1;
        debug_assert!(make_score[v] > 0);
        make_score[v] -= 1;
    }
}

#[inline(always)]
fn add_make_weight_for_clause(formula: &Formula, make_score_w: &mut [u32], c: usize, delta: u32) {
    let s = formula.co[c] as usize;
    let e = formula.co[c + 1] as usize;
    for i in s..e {
        let v = formula.cl[i].unsigned_abs() as usize - 1;
        make_score_w[v] = make_score_w[v].saturating_add(delta);
    }
}

#[inline(always)]
fn remove_make_weight_for_clause(
    formula: &Formula,
    make_score_w: &mut [u32],
    c: usize,
    delta: u32,
) {
    let s = formula.co[c] as usize;
    let e = formula.co[c + 1] as usize;
    for i in s..e {
        let v = formula.cl[i].unsigned_abs() as usize - 1;
        debug_assert!(make_score_w[v] >= delta);
        make_score_w[v] -= delta;
    }
}

#[inline(always)]
fn inc_break(scores: &mut [u16], v: usize) {
    debug_assert!(scores[v] < u16::MAX);
    scores[v] = scores[v].saturating_add(1);
}

#[inline(always)]
fn dec_break(scores: &mut [u16], v: usize) {
    debug_assert!(scores[v] > 0);
    scores[v] -= 1;
}

#[inline(always)]
fn inc_break_w(scores: &mut [u32], v: usize, delta: u32) {
    scores[v] = scores[v].saturating_add(delta);
}

#[inline(always)]
fn dec_break_w(scores: &mut [u32], v: usize, delta: u32) {
    debug_assert!(scores[v] >= delta);
    scores[v] -= delta;
}
