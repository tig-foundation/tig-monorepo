use super::Hyperparameters;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[inline(always)]
fn select_weighted3_f64_sub_le(
    mut threshold: f64,
    count: usize,
    w0: f64,
    v0: usize,
    w1: f64,
    v1: usize,
    w2: f64,
    v2: usize,
) -> usize {
    threshold -= w0;
    if threshold <= 0.0 {
        return v0;
    }
    if count > 1 {
        threshold -= w1;
        if threshold <= 0.0 {
            return v1;
        }
        if count > 2 {
            threshold -= w2;
            if threshold <= 0.0 {
                return v2;
            }
        }
    }
    v0
}

fn n10000_best_tail_enabled(hp: &Hyperparameters, nv: usize) -> bool {
    nv == 10_000
        && hp
            .target_n10000_best_tail_fuel
            .is_some_and(|fuel| fuel.is_finite() && fuel > 0.0)
}

fn n10000_best_tail_extension_flips(
    hp: &Hyperparameters,
    nv: usize,
    best_unsat: usize,
    flip_fuel: f64,
    already_extended: bool,
) -> usize {
    if already_extended
        || !n10000_best_tail_enabled(hp, nv)
        || flip_fuel <= 0.0
        || best_unsat == 0
        || best_unsat > hp.target_n10000_best_tail_max_unsat.unwrap_or(8)
    {
        return 0;
    }

    (hp.target_n10000_best_tail_fuel.unwrap_or(0.0) / flip_fuel) as usize
}

#[derive(Debug)]
struct N10000TraceStats {
    initial_unsat: usize,
    best_unsat: usize,
    final_unsat: usize,
    last_improvement_round: usize,
    rounds: usize,
    final_residual_len: usize,
    max_residual_len: usize,
    stale_pops: usize,
    kick_flips: usize,
    budget_growths: usize,
    base_max_flips: usize,
    final_max_flips: usize,
    best_tail_started: bool,
    best_tail_flips: usize,
}

impl N10000TraceStats {
    fn new(initial_unsat: usize, base_max_flips: usize) -> Self {
        Self {
            initial_unsat,
            best_unsat: initial_unsat,
            final_unsat: initial_unsat,
            last_improvement_round: 0,
            rounds: 0,
            final_residual_len: initial_unsat,
            max_residual_len: initial_unsat,
            stale_pops: 0,
            kick_flips: 0,
            budget_growths: 0,
            base_max_flips,
            final_max_flips: base_max_flips,
            best_tail_started: false,
            best_tail_flips: 0,
        }
    }

    #[inline(always)]
    fn observe(&mut self, round: usize, true_unsat: usize, residual_len: usize) {
        if true_unsat < self.best_unsat {
            self.best_unsat = true_unsat;
            self.last_improvement_round = round;
        }
        self.max_residual_len = self.max_residual_len.max(residual_len);
    }

    #[inline(always)]
    fn record_stale_pop(&mut self) {
        self.stale_pops += 1;
    }

    #[inline(always)]
    fn record_kick(&mut self) {
        self.kick_flips += 1;
    }

    #[inline(always)]
    fn record_budget_growth(&mut self) {
        self.budget_growths += 1;
    }

    fn record_best_tail(&mut self, extension_flips: usize) {
        self.best_tail_started = true;
        self.best_tail_flips = extension_flips;
    }

    fn finish(
        &mut self,
        rounds: usize,
        true_unsat: usize,
        residual_len: usize,
        final_max_flips: usize,
    ) {
        self.observe(rounds, true_unsat, residual_len);
        self.rounds = rounds;
        self.final_unsat = true_unsat;
        self.final_residual_len = residual_len;
        self.final_max_flips = final_max_flips;
    }

    fn render(&self) -> String {
        format!(
            "c001_n10000_trace_diag r={} bmf={} mf={} i={} f={} b={} lr={} rl={} mrl={} sp={} k={} g={} bt={} btf={} sol={}",
            self.rounds,
            self.base_max_flips,
            self.final_max_flips,
            self.initial_unsat,
            self.final_unsat,
            self.best_unsat,
            self.last_improvement_round,
            self.final_residual_len,
            self.max_residual_len,
            self.stale_pops,
            self.kick_flips,
            self.budget_growths,
            self.best_tail_started,
            self.best_tail_flips,
            self.final_unsat == 0,
        )
    }
}

pub(crate) fn solve(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
    nv: usize,
    nc: usize,
    density: f64,
    p_cnt: Vec<u32>,
    n_cnt: Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>,
    co: &[u32],
    all_three_clauses: bool,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let default_fuel = if nv == 10_000 {
        160_000_000_000.0
    } else if nv >= 10000 {
        125_000_000_000.0
    } else {
        250_000_000_000.0
    };
    let max_fuel = hp.target_max_fuel.unwrap_or(default_fuel);

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let mut max_flips = if flip_fuel > 0.0 {
        (remaining / flip_fuel) as usize
    } else {
        0
    };
    let base_max_flips = max_flips;
    let best_tail_enabled = n10000_best_tail_enabled(hp, nv);
    let best_tail_max_unsat = hp.target_n10000_best_tail_max_unsat.unwrap_or(8);

    debug_assert_eq!(
        all_three_clauses,
        track3_clause_offsets_are_three(nc, co, cl.len())
    );
    debug_assert!(!all_three_clauses || co.len() > nc);
    debug_assert!(!all_three_clauses || co[0] == 0);
    debug_assert!(!all_three_clauses || co[nc] as usize == cl.len());
    debug_assert!(!all_three_clauses || (0..=nc).all(|i| co[i] as usize == i * 3));

    let mut vars = vec![false; nv];
    greedy_initial_assignment(
        nc,
        co,
        cl,
        &mut vars,
        &p_cnt,
        &n_cnt,
        rng,
        all_three_clauses,
    );

    // Build num_good and residual from final assignment
    let mut num_good = vec![0u8; nc];
    let mut residual: Vec<u32> = Vec::with_capacity(super::initial_residual_capacity(nc));
    let mut true_unsat = rebuild_initial_state(
        nc,
        co,
        cl,
        &vars,
        &mut num_good,
        &mut residual,
        all_three_clauses,
    );
    let mut best_tail_unsat = best_tail_max_unsat.saturating_add(1);
    let mut best_tail_vars = Vec::<bool>::new();
    let mut best_tail_started = false;
    if best_tail_enabled && true_unsat > 0 && true_unsat <= best_tail_max_unsat {
        best_tail_unsat = true_unsat;
        best_tail_vars.clone_from(&vars);
    }
    let trace_enabled =
        hp.target_trace_10000.unwrap_or(false) && nv == 10_000 && (4.25..4.28).contains(&density);
    let mut trace = trace_enabled.then(|| N10000TraceStats::new(true_unsat, base_max_flips));

    if true_unsat == 0 {
        if let Some(trace) = trace.as_mut() {
            trace.finish(0, 0, residual.len(), max_flips);
            eprintln!("{}", trace.render());
        }
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 {
        15.0
    } else {
        25.0
    };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.check_interval.unwrap_or(
        (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0)))
            .max(min_interval) as usize,
    );

    let probsat_weights = build_probsat_weights(avg_clause_size, nc, &p_cnt, &n_cnt);

    let mut last_check_unsat = true_unsat;
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let mut check_countdown = check_interval;
    let mut check_due = false;

    unsafe {
        loop {
            if rounds >= max_flips {
                let extension_flips = n10000_best_tail_extension_flips(
                    hp,
                    nv,
                    best_tail_unsat,
                    flip_fuel,
                    best_tail_started,
                );
                if extension_flips == 0 || best_tail_vars.len() != nv {
                    break;
                }

                best_tail_started = true;
                max_flips = max_flips.saturating_add(extension_flips);
                vars.copy_from_slice(&best_tail_vars);
                true_unsat = rebuild_initial_state(
                    nc,
                    co,
                    cl,
                    &vars,
                    &mut num_good,
                    &mut residual,
                    all_three_clauses,
                );
                debug_assert_eq!(true_unsat, best_tail_unsat);
                last_check_unsat = true_unsat;
                stagnation = 0;
                check_countdown = check_interval;
                check_due = false;
                if let Some(trace) = trace.as_mut() {
                    trace.record_best_tail(extension_flips);
                    trace.observe(rounds, true_unsat, residual.len());
                }
                continue;
            }
            if true_unsat == 0 {
                break;
            }

            if check_due {
                check_due = false;
                let progress = last_check_unsat as i64 - true_unsat as i64;
                let mut grow_budget = false;

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= 4 {
                        for _ in 0..3 {
                            if true_unsat == 0 {
                                break;
                            }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            if *num_good.get_unchecked(pcid) > 0 {
                                remove_residual_unordered(&mut residual, rid);
                                if let Some(trace) = trace.as_mut() {
                                    trace.record_stale_pop();
                                }
                                continue;
                            }
                            let (pcs, pce) =
                                track3_clause_bounds_unchecked(pcid, co, all_three_clauses);
                            if pcs == pce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = lit_var_index(lit);

                            let was_true = *vars.get_unchecked(v);
                            let (is, ie) = if was_true {
                                (
                                    *p_bound.get_unchecked(v) as usize,
                                    *all_off.get_unchecked(v + 1) as usize,
                                )
                            } else {
                                (
                                    *all_off.get_unchecked(v) as usize,
                                    *p_bound.get_unchecked(v) as usize,
                                )
                            };
                            let (ds, de) = if was_true {
                                (
                                    *all_off.get_unchecked(v) as usize,
                                    *p_bound.get_unchecked(v) as usize,
                                )
                            } else {
                                (
                                    *p_bound.get_unchecked(v) as usize,
                                    *all_off.get_unchecked(v + 1) as usize,
                                )
                            };

                            for k in is..ie {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = num_good.get_unchecked_mut(c);
                                if *ng == 0 {
                                    true_unsat = true_unsat.saturating_sub(1);
                                }
                                *ng = ng.saturating_add(1);
                            }
                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = num_good.get_unchecked_mut(c);
                                let new_val = ng.saturating_sub(1);
                                *ng = new_val;
                                if new_val == 0 {
                                    residual.push(c as u32);
                                    true_unsat += 1;
                                }
                            }
                            *vars.get_unchecked_mut(v) = !was_true;
                            if let Some(trace) = trace.as_mut() {
                                trace.record_kick();
                            }
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                    let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;
                    grow_budget = progress_ratio > 0.2;
                }

                last_check_unsat = true_unsat;

                // Dynamic budget adjustment
                if grow_budget && !best_tail_enabled {
                    let increase = (base_max_flips / 100).max(1);
                    max_flips = (max_flips + increase).min(2 * base_max_flips);
                    if let Some(trace) = trace.as_mut() {
                        trace.record_budget_growth();
                    }
                }
            }

            if true_unsat == 0 {
                break;
            }

            let mut cid = usize::MAX;
            let mut min_len = usize::MAX;
            for _ in 0..3 {
                while !residual.is_empty() {
                    let id = rng.gen::<usize>() % residual.len();
                    let cand = *residual.get_unchecked(id) as usize;
                    if *num_good.get_unchecked(cand) > 0 {
                        remove_residual_unordered(&mut residual, id);
                        if let Some(trace) = trace.as_mut() {
                            trace.record_stale_pop();
                        }
                    } else {
                        let (c_s, c_e) =
                            track3_clause_bounds_unchecked(cand, co, all_three_clauses);
                        let clen = c_e - c_s;
                        if clen < min_len {
                            min_len = clen;
                            cid = cand;
                        }
                        break;
                    }
                }
                if residual.is_empty() {
                    break;
                }
            }
            if cid == usize::MAX {
                break;
            }

            let (cs, ce) = track3_clause_bounds_unchecked(cid, co, all_three_clauses);
            let clen = ce - cs;

            if clen > 1 {
                let ri = rng.gen::<usize>() % clen;
                if ri != 0 {
                    cl.swap(cs, cs + ri);
                }
            }

            let v_idx = if all_three_clauses {
                choose_var_small_clause3(
                    rng,
                    cs,
                    cl,
                    &vars,
                    &num_good,
                    &probsat_weights,
                    all_off,
                    p_bound,
                    all_data,
                )
            } else if clen <= 3 {
                choose_var_small_clause(
                    rng,
                    cs,
                    clen,
                    cl,
                    &vars,
                    &num_good,
                    &probsat_weights,
                    all_off,
                    p_bound,
                    all_data,
                )
            } else {
                choose_var_wide_clause(
                    rng,
                    cs,
                    clen.min(256),
                    cl,
                    &vars,
                    &num_good,
                    &probsat_weights,
                    all_off,
                    p_bound,
                    all_data,
                )
            };

            let was_true = *vars.get_unchecked(v_idx);
            let (is, ie) = if was_true {
                (
                    *p_bound.get_unchecked(v_idx) as usize,
                    *all_off.get_unchecked(v_idx + 1) as usize,
                )
            } else {
                (
                    *all_off.get_unchecked(v_idx) as usize,
                    *p_bound.get_unchecked(v_idx) as usize,
                )
            };
            let (ds, de) = if was_true {
                (
                    *all_off.get_unchecked(v_idx) as usize,
                    *p_bound.get_unchecked(v_idx) as usize,
                )
            } else {
                (
                    *p_bound.get_unchecked(v_idx) as usize,
                    *all_off.get_unchecked(v_idx + 1) as usize,
                )
            };

            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = num_good.get_unchecked_mut(c);
                if *ng == 0 {
                    true_unsat = true_unsat.saturating_sub(1);
                }
                *ng = ng.saturating_add(1);
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = num_good.get_unchecked_mut(c);
                let new_val = ng.saturating_sub(1);
                *ng = new_val;
                if new_val == 0 {
                    residual.push(c as u32);
                    true_unsat += 1;
                }
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;

            if best_tail_enabled
                && true_unsat > 0
                && true_unsat <= best_tail_max_unsat
                && true_unsat < best_tail_unsat
            {
                best_tail_unsat = true_unsat;
                best_tail_vars.clone_from(&vars);
            }

            rounds += 1;
            if let Some(trace) = trace.as_mut() {
                trace.observe(rounds, true_unsat, residual.len());
            }
            check_due = super::advance_interval_due(&mut check_countdown, check_interval);
        }
    }

    if let Some(trace) = trace.as_mut() {
        trace.finish(rounds, true_unsat, residual.len(), max_flips);
        eprintln!("{}", trace.render());
    }
    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

#[inline(always)]
unsafe fn track3_flip_sad(
    lit: i32,
    vars: &[bool],
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> (usize, usize) {
    let abs_l = lit_var_index(lit);
    let (os, oe) = if *vars.get_unchecked(abs_l) {
        (
            *all_off.get_unchecked(abs_l) as usize,
            *p_bound.get_unchecked(abs_l) as usize,
        )
    } else {
        (
            *p_bound.get_unchecked(abs_l) as usize,
            *all_off.get_unchecked(abs_l + 1) as usize,
        )
    };

    let mut sad = 0usize;
    for k in os..oe {
        let c = *all_data.get_unchecked(k) as usize;
        if *num_good.get_unchecked(c) == 1 {
            sad += 1;
        }
    }
    (abs_l, sad)
}

#[inline(always)]
unsafe fn choose_var_small_clause3(
    rng: &mut SmallRng,
    cs: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &[u8],
    probsat_weights: &[f64],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    let weight_limit = probsat_weights.len() - 1;

    let (v0, sad0) = track3_flip_sad(
        *cl.get_unchecked(cs),
        vars,
        num_good,
        all_off,
        p_bound,
        all_data,
    );
    if sad0 == 0 {
        return v0;
    }
    debug_assert!(sad0 <= weight_limit);
    let w0 = *probsat_weights.get_unchecked(sad0.min(weight_limit));

    let (v1, sad1) = track3_flip_sad(
        *cl.get_unchecked(cs + 1),
        vars,
        num_good,
        all_off,
        p_bound,
        all_data,
    );
    if sad1 == 0 {
        return v1;
    }
    debug_assert!(sad1 <= weight_limit);
    let w1 = *probsat_weights.get_unchecked(sad1.min(weight_limit));

    let (v2, sad2) = track3_flip_sad(
        *cl.get_unchecked(cs + 2),
        vars,
        num_good,
        all_off,
        p_bound,
        all_data,
    );
    if sad2 == 0 {
        return v2;
    }
    debug_assert!(sad2 <= weight_limit);
    let w2 = *probsat_weights.get_unchecked(sad2.min(weight_limit));

    select_weighted3_f64_sub_le(rng.gen::<f64>() * (w0 + w1 + w2), 3, w0, v0, w1, v1, w2, v2)
}

#[inline(always)]
unsafe fn choose_var_small_clause(
    rng: &mut SmallRng,
    cs: usize,
    clen_actual: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &[u8],
    probsat_weights: &[f64],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    debug_assert!(clen_actual > 0);
    debug_assert!(clen_actual <= 3);

    let mut total_weight = 0.0;
    let mut w0 = 0.0f64;
    let mut w1 = 0.0f64;
    let mut w2 = 0.0f64;
    let mut v0 = 0usize;
    let mut v1 = 0usize;
    let mut v2 = 0usize;

    for idx in 0..clen_actual {
        let (abs_l, sad) = track3_flip_sad(
            *cl.get_unchecked(cs + idx),
            vars,
            num_good,
            all_off,
            p_bound,
            all_data,
        );

        if sad == 0 {
            return abs_l;
        }

        let weight_limit = probsat_weights.len() - 1;
        debug_assert!(sad <= weight_limit);
        let w = *probsat_weights.get_unchecked(sad.min(weight_limit));
        match idx {
            0 => {
                w0 = w;
                v0 = abs_l;
            }
            1 => {
                w1 = w;
                v1 = abs_l;
            }
            _ => {
                w2 = w;
                v2 = abs_l;
            }
        }
        total_weight += w;
    }

    let r = rng.gen::<f64>() * total_weight;
    select_weighted3_f64_sub_le(r, clen_actual, w0, v0, w1, v1, w2, v2)
}

#[inline(always)]
unsafe fn choose_var_wide_clause(
    rng: &mut SmallRng,
    cs: usize,
    clen_actual: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &[u8],
    probsat_weights: &[f64],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    choose_var_with_weight_scratch::<256>(
        rng,
        cs,
        clen_actual,
        cl,
        vars,
        num_good,
        probsat_weights,
        all_off,
        p_bound,
        all_data,
    )
}

#[inline(always)]
unsafe fn choose_var_with_weight_scratch<const SCRATCH: usize>(
    rng: &mut SmallRng,
    cs: usize,
    clen_actual: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &[u8],
    probsat_weights: &[f64],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    debug_assert!(clen_actual > 0);
    debug_assert!(clen_actual <= SCRATCH);

    let mut total_weight = 0.0;
    let mut weights = [0.0f64; SCRATCH];

    for idx in 0..clen_actual {
        let j = cs + idx;
        let l = *cl.get_unchecked(j);
        let abs_l = lit_var_index(l);
        let (os, oe) = if *vars.get_unchecked(abs_l) {
            (
                *all_off.get_unchecked(abs_l) as usize,
                *p_bound.get_unchecked(abs_l) as usize,
            )
        } else {
            (
                *p_bound.get_unchecked(abs_l) as usize,
                *all_off.get_unchecked(abs_l + 1) as usize,
            )
        };

        let mut sad = 0usize;
        for k in os..oe {
            let c = *all_data.get_unchecked(k) as usize;
            if *num_good.get_unchecked(c) == 1 {
                sad += 1;
            }
        }

        if sad == 0 {
            return abs_l;
        }

        let weight_limit = probsat_weights.len() - 1;
        debug_assert!(sad <= weight_limit);
        let w = *probsat_weights.get_unchecked(sad.min(weight_limit));
        *weights.get_unchecked_mut(idx) = w;
        total_weight += w;
    }

    let mut v_idx = lit_var_index(*cl.get_unchecked(cs));
    let mut r = rng.gen::<f64>() * total_weight;
    for idx in 0..clen_actual {
        r -= *weights.get_unchecked(idx);
        if r <= 0.0 {
            v_idx = lit_var_index(*cl.get_unchecked(cs + idx));
            break;
        }
    }
    v_idx
}

fn build_greedy_clause_order(nc: usize, co: &[u32]) -> Vec<u32> {
    debug_assert!(nc <= u32::MAX as usize);
    let mut counts = [0usize; 4];
    let mut total = 0usize;
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len == 0 {
            continue;
        }
        if len > 3 {
            return build_greedy_clause_order_generic(nc, co);
        }
        counts[len] += 1;
        total += 1;
    }
    if total == 0 {
        return Vec::new();
    }
    if counts[1] == 0 && counts[2] == 0 && total == nc {
        return (0..nc as u32).collect();
    }

    let mut cursors = [0usize, 0, counts[1], counts[1] + counts[2]];
    let mut order = vec![0u32; total];
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len == 0 {
            continue;
        }
        let pos = cursors[len];
        order[pos] = cid as u32;
        cursors[len] = pos + 1;
    }
    order
}

fn greedy_initial_assignment(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &mut [bool],
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    all_three_clauses: bool,
) {
    if all_three_clauses {
        for cid in 0..nc {
            let s = cid * 3;
            greedy_assign_clause(cl, vars, p_cnt, n_cnt, rng, s, s + 3);
        }
        return;
    }

    let clause_order = build_greedy_clause_order(nc, co);
    for &cid in &clause_order {
        let cid = cid as usize;
        let s = co[cid] as usize;
        let e = co[cid + 1] as usize;
        greedy_assign_clause(cl, vars, p_cnt, n_cnt, rng, s, e);
    }
}

#[inline(always)]
fn greedy_assign_clause(
    cl: &[i32],
    vars: &mut [bool],
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    s: usize,
    e: usize,
) {
    let mut already = false;
    for &lit in &cl[s..e] {
        let v = lit_var_index(lit);
        if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
            already = true;
            break;
        }
    }
    if already {
        return;
    }

    let mut best_score: u32 = 0;
    let mut best_v = 0usize;
    let mut best_target = false;
    let mut count = 0usize;
    for &lit in &cl[s..e] {
        let v = lit_var_index(lit);
        let target_val = lit > 0;
        if vars[v] == target_val {
            continue;
        }
        let score = if target_val { p_cnt[v] } else { n_cnt[v] };
        if score > best_score {
            best_score = score;
            best_v = v;
            best_target = target_val;
            count = 1;
        } else if score == best_score {
            count += 1;
            if rng.gen::<usize>() % count == 0 {
                best_v = v;
                best_target = target_val;
            }
        }
    }
    if best_score == 0 {
        let idx = rng.gen::<usize>() % (e - s);
        let lit = cl[s + idx];
        best_v = lit_var_index(lit);
        best_target = lit > 0;
    }
    vars[best_v] = best_target;
}

fn build_greedy_clause_order_generic(nc: usize, co: &[u32]) -> Vec<u32> {
    debug_assert!(nc <= u32::MAX as usize);
    let mut max_len = 0usize;
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        max_len = max_len.max(len);
    }
    if max_len == 0 {
        return Vec::new();
    }

    let mut counts = vec![0usize; max_len + 1];
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len > 0 {
            counts[len] += 1;
        }
    }

    let mut starts = vec![0usize; max_len + 1];
    let mut total = 0usize;
    for len in 1..=max_len {
        starts[len] = total;
        total += counts[len];
    }

    let mut cursors = starts;
    let mut order = vec![0u32; total];
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len == 0 {
            continue;
        }
        let pos = cursors[len];
        order[pos] = cid as u32;
        cursors[len] = pos + 1;
    }
    order
}

fn build_probsat_weights(
    avg_clause_size: f64,
    nc: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
) -> Vec<f64> {
    let max_occ = max_occurrence_count(p_cnt, n_cnt) as usize;
    let limit = max_occ.min(nc);
    let mut weights = vec![0.0f64; limit + 1];
    if avg_clause_size <= 3.2 {
        let cb: f64 = 2.06;
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = cb.powf(-(i as f64));
        }
    } else {
        let cb: f64 = if avg_clause_size <= 4.2 {
            2.85
        } else if avg_clause_size <= 5.2 {
            3.7
        } else if avg_clause_size <= 6.2 {
            5.1
        } else {
            5.4
        };
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = (i as f64 + 1.0).powf(-cb);
        }
    }
    weights
}

fn max_occurrence_count(p_cnt: &[u32], n_cnt: &[u32]) -> u32 {
    let common = p_cnt.len().min(n_cnt.len());
    let mut max_occ = 0u32;
    for i in 0..common {
        max_occ = max_occ.max(p_cnt[i]).max(n_cnt[i]);
    }
    for &cnt in &p_cnt[common..] {
        max_occ = max_occ.max(cnt);
    }
    for &cnt in &n_cnt[common..] {
        max_occ = max_occ.max(cnt);
    }
    max_occ
}

fn rebuild_initial_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    all_three_clauses: bool,
) -> usize {
    residual.clear();
    if all_three_clauses {
        debug_assert!(co.len() > nc);
        debug_assert_eq!(co[0], 0);
        debug_assert_eq!(co[nc] as usize, cl.len());
        debug_assert!((0..=nc).all(|i| co[i] as usize == i * 3));
        for i in 0..nc {
            let good = unsafe { count_satisfied_clause3_unchecked(vars, cl, i * 3) };
            num_good[i] = good;
            if good == 0 {
                residual.push(i as u32);
            }
        }
        return residual.len();
    }
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let good = unsafe { count_satisfied_clause_unchecked(vars, cl, s, e) };
        num_good[i] = good;
        if good == 0 {
            residual.push(i as u32);
        }
    }
    residual.len()
}

#[inline(always)]
fn track3_clause_offsets_are_three(nc: usize, co: &[u32], cl_len: usize) -> bool {
    if co.len() <= nc || cl_len != nc.saturating_mul(3) {
        return false;
    }
    for (i, &off) in co.iter().take(nc + 1).enumerate() {
        if off as usize != i * 3 {
            return false;
        }
    }
    true
}

#[inline(always)]
unsafe fn track3_clause_bounds_unchecked(
    cid: usize,
    co: &[u32],
    all_three_clauses: bool,
) -> (usize, usize) {
    if all_three_clauses {
        let s = cid * 3;
        (s, s + 3)
    } else {
        (
            *co.get_unchecked(cid) as usize,
            *co.get_unchecked(cid + 1) as usize,
        )
    }
}

#[inline(always)]
unsafe fn count_satisfied_clause3_unchecked(vars: &[bool], cl: &[i32], s: usize) -> u8 {
    is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8
        + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 1)) as u8
        + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 2)) as u8
}

#[inline(always)]
fn is_lit_satisfied(vars: &[bool], lit: i32) -> bool {
    let v = lit_var_index(lit);
    (lit > 0 && vars[v]) || (lit < 0 && !vars[v])
}

#[inline(always)]
fn count_satisfied_clause(vars: &[bool], cl: &[i32], s: usize, e: usize) -> u8 {
    match e - s {
        1 => is_lit_satisfied(vars, cl[s]) as u8,
        2 => is_lit_satisfied(vars, cl[s]) as u8 + is_lit_satisfied(vars, cl[s + 1]) as u8,
        3 => {
            is_lit_satisfied(vars, cl[s]) as u8
                + is_lit_satisfied(vars, cl[s + 1]) as u8
                + is_lit_satisfied(vars, cl[s + 2]) as u8
        }
        _ => {
            let mut good = 0u8;
            for &lit in &cl[s..e] {
                if is_lit_satisfied(vars, lit) {
                    good = good.saturating_add(1);
                }
            }
            good
        }
    }
}

#[inline(always)]
unsafe fn is_lit_satisfied_unchecked(vars: &[bool], lit: i32) -> bool {
    let v = lit_var_index(lit);
    (lit > 0 && *vars.get_unchecked(v)) || (lit < 0 && !*vars.get_unchecked(v))
}

#[inline(always)]
unsafe fn count_satisfied_clause_unchecked(vars: &[bool], cl: &[i32], s: usize, e: usize) -> u8 {
    match e - s {
        1 => is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8,
        2 => {
            is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8
                + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 1)) as u8
        }
        3 => {
            is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8
                + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 1)) as u8
                + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 2)) as u8
        }
        _ => {
            let mut good = 0u8;
            for j in s..e {
                let lit = *cl.get_unchecked(j);
                if is_lit_satisfied_unchecked(vars, lit) {
                    good = good.saturating_add(1);
                }
            }
            good
        }
    }
}

#[inline(always)]
fn remove_residual_unordered(residual: &mut Vec<u32>, rid: usize) {
    if rid + 1 == residual.len() {
        residual.pop();
    } else {
        residual.swap_remove(rid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn n10000_trace_stats_capture_progress_and_queue_pressure() {
        let mut trace = N10000TraceStats::new(120, 1_000);
        trace.observe(25, 90, 140);
        trace.observe(40, 80, 155);
        trace.record_stale_pop();
        trace.record_kick();
        trace.record_budget_growth();
        trace.record_best_tail(200);
        trace.finish(100, 80, 150, 1_100);

        assert_eq!(trace.initial_unsat, 120);
        assert_eq!(trace.best_unsat, 80);
        assert_eq!(trace.last_improvement_round, 40);
        assert_eq!(trace.final_unsat, 80);
        assert_eq!(trace.max_residual_len, 155);
        assert_eq!(trace.stale_pops, 1);
        assert_eq!(trace.kick_flips, 1);
        assert_eq!(trace.budget_growths, 1);
        assert!(trace.best_tail_started);
        assert_eq!(trace.best_tail_flips, 200);
        assert!(trace.render().len() < 500);
    }

    #[test]
    fn n10000_best_tail_replaces_legacy_growth_only_when_explicitly_enabled() {
        let legacy = Hyperparameters::default();
        assert!(!n10000_best_tail_enabled(&legacy, 10_000));
        assert_eq!(
            n10000_best_tail_extension_flips(&legacy, 10_000, 1, 250.0, false),
            0
        );

        let enabled = Hyperparameters {
            target_n10000_best_tail_fuel: Some(40_000_000_000.0),
            target_n10000_best_tail_max_unsat: Some(8),
            ..Hyperparameters::default()
        };
        assert!(n10000_best_tail_enabled(&enabled, 10_000));
        assert_eq!(
            n10000_best_tail_extension_flips(&enabled, 10_000, 8, 250.0, false),
            160_000_000
        );
        assert_eq!(
            n10000_best_tail_extension_flips(&enabled, 10_000, 9, 250.0, false),
            0
        );
        assert_eq!(
            n10000_best_tail_extension_flips(&enabled, 10_000, 1, 250.0, true),
            0
        );
        assert_eq!(
            n10000_best_tail_extension_flips(&enabled, 7_500, 1, 250.0, false),
            0
        );
    }

    #[test]
    fn imp_v4_track3_lit_var_index_matches_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), (lit.abs() - 1) as usize);
        }
    }

    #[test]
    fn imp_v4_track3_remove_residual_unordered_handles_moved_and_last_entries() {
        let mut residual = vec![2_u32, 5, 7];

        remove_residual_unordered(&mut residual, 1);
        assert_eq!(residual, vec![2_u32, 7]);

        remove_residual_unordered(&mut residual, 1);
        assert_eq!(residual, vec![2_u32]);
    }

    #[test]
    fn greedy_clause_order_is_stable_length_ascending() {
        let co = [0_u32, 3, 3, 4, 6, 7, 7, 8];

        let order: Vec<u32> = build_greedy_clause_order(7, &co);
        assert_eq!(order, vec![2_u32, 4, 6, 3, 0]);
    }

    #[test]
    fn greedy_clause_order_falls_back_for_long_clauses() {
        let co = [0_u32, 4, 5, 8, 13, 13, 15];

        let order: Vec<u32> = build_greedy_clause_order(6, &co);

        assert_eq!(order, vec![1_u32, 5, 2, 0, 3]);
    }

    #[test]
    fn greedy_clause_order_ignores_empty_clauses() {
        let co = [0_u32, 0, 0, 0];

        assert!(build_greedy_clause_order(3, &co).is_empty());
    }

    #[test]
    fn greedy_clause_order_all_three_literal_returns_natural_order() {
        let co = [0_u32, 3, 6, 9, 12];

        assert_eq!(build_greedy_clause_order(4, &co), vec![0_u32, 1, 2, 3]);
    }

    #[test]
    fn greedy_initial_assignment_all_three_direct_path_matches_ordered_reference() {
        let nc = 5;
        let co = [0_u32, 3, 6, 9, 12, 15];
        let cl = [1, -2, 3, -1, 2, -3, 1, 4, -5, -1, -4, 5, 2, -3, -5];
        let p_cnt = [3_u32, 5, 2, 4, 1];
        let n_cnt = [4_u32, 2, 6, 1, 7];
        let mut ordered_vars = vec![false; 5];
        let mut direct_vars = vec![false; 5];
        let mut ordered_rng = SmallRng::seed_from_u64(91);
        let mut direct_rng = SmallRng::seed_from_u64(91);

        greedy_initial_assignment(
            nc,
            &co,
            &cl,
            &mut ordered_vars,
            &p_cnt,
            &n_cnt,
            &mut ordered_rng,
            false,
        );
        greedy_initial_assignment(
            nc,
            &co,
            &cl,
            &mut direct_vars,
            &p_cnt,
            &n_cnt,
            &mut direct_rng,
            true,
        );

        assert_eq!(direct_vars, ordered_vars);
        assert_eq!(direct_rng.gen::<u64>(), ordered_rng.gen::<u64>());
    }

    #[test]
    fn probsat_weights_are_capped_by_occurrence_bound() {
        let p_cnt = [2_u32, 5, 1];
        let n_cnt = [0_u32, 3, 4];
        let weights = build_probsat_weights(3.0, 100, &p_cnt, &n_cnt);
        assert_eq!(weights.len(), 6);
        assert!((weights[5] - 2.06_f64.powf(-5.0)).abs() < 1e-12);

        let capped = build_probsat_weights(4.0, 3, &[9_u32], &[7_u32]);
        assert_eq!(capped.len(), 4);
        assert!((capped[3] - 4.0_f64.powf(-2.85)).abs() < 1e-12);
    }

    #[test]
    fn probsat_occurrence_bound_matches_two_array_max_reference() {
        assert_eq!(max_occurrence_count(&[2_u32, 5, 1], &[0_u32, 3, 4]), 5);
        assert_eq!(max_occurrence_count(&[2_u32, 1], &[0_u32, 7, 4]), 7);
        assert_eq!(max_occurrence_count(&[2_u32, 9, 1], &[0_u32]), 9);
        assert_eq!(max_occurrence_count(&[], &[]), 0);
    }

    #[test]
    fn adapted_track3_weighted3_selector_matches_array_reference() {
        fn reference(
            mut threshold: f64,
            count: usize,
            weights: [f64; 3],
            vars: [usize; 3],
        ) -> usize {
            let mut selected = vars[0];
            for idx in 0..count {
                threshold -= weights[idx];
                if threshold <= 0.0 {
                    selected = vars[idx];
                    break;
                }
            }
            selected
        }

        let weights = [0.20, 0.30, 0.50];
        let vars = [3, 9, 17];
        for &(count, threshold) in &[
            (1, 0.0),
            (1, 0.21),
            (2, 0.20),
            (2, 0.50),
            (3, 0.51),
            (3, 1.01),
        ] {
            assert_eq!(
                select_weighted3_f64_sub_le(
                    threshold, count, weights[0], vars[0], weights[1], vars[1], weights[2],
                    vars[2],
                ),
                reference(threshold, count, weights, vars)
            );
        }
    }

    #[test]
    fn small_clause_local_slots_match_wide_path() {
        let cl = vec![1, 2, 3];
        let vars = vec![false, false, false];
        let all_off = vec![0_u32, 1, 2, 3];
        let p_bound = vec![0_u32, 1, 2];
        let all_data = vec![0_u32, 1, 2];
        let probsat_weights = vec![1.0_f64, 0.5, 0.25, 0.125];

        let weighted_num_good = vec![1_u8, 1, 1];
        let mut small_rng = SmallRng::seed_from_u64(0xface_cafe_dead_beef);
        let mut wide_rng = SmallRng::seed_from_u64(0xface_cafe_dead_beef);
        let small = unsafe {
            choose_var_small_clause(
                &mut small_rng,
                0,
                3,
                &cl,
                &vars,
                &weighted_num_good,
                &probsat_weights,
                &all_off,
                &p_bound,
                &all_data,
            )
        };
        let wide = unsafe {
            choose_var_wide_clause(
                &mut wide_rng,
                0,
                3,
                &cl,
                &vars,
                &weighted_num_good,
                &probsat_weights,
                &all_off,
                &p_bound,
                &all_data,
            )
        };
        assert_eq!(small, wide);
        assert_eq!(small_rng.gen::<u64>(), wide_rng.gen::<u64>());

        let zero_break_num_good = vec![2_u8, 2, 2];
        let mut small_rng = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
        let mut wide_rng = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
        let small = unsafe {
            choose_var_small_clause(
                &mut small_rng,
                0,
                3,
                &cl,
                &vars,
                &zero_break_num_good,
                &probsat_weights,
                &all_off,
                &p_bound,
                &all_data,
            )
        };
        let wide = unsafe {
            choose_var_wide_clause(
                &mut wide_rng,
                0,
                3,
                &cl,
                &vars,
                &zero_break_num_good,
                &probsat_weights,
                &all_off,
                &p_bound,
                &all_data,
            )
        };
        assert_eq!(small, wide);
        assert_eq!(small_rng.gen::<u64>(), wide_rng.gen::<u64>());
    }

    #[test]
    fn small_clause3_fast_path_matches_small_clause_path() {
        let cl = vec![1, -2, 3];
        let vars = vec![false, false, false];
        let all_off = vec![0_u32, 1, 2, 3];
        let p_bound = vec![0_u32, 1, 2];
        let all_data = vec![0_u32, 1, 2];
        let probsat_weights = vec![1.0_f64, 0.5, 0.25, 0.125];
        let cases = [
            vec![1_u8, 1, 1],
            vec![2_u8, 1, 1],
            vec![1_u8, 2, 1],
            vec![1_u8, 1, 2],
            vec![2_u8, 2, 2],
        ];

        for num_good in cases {
            for seed in 0_u64..16 {
                let mut generic_rng = SmallRng::seed_from_u64(seed);
                let mut fast_rng = SmallRng::seed_from_u64(seed);
                let generic = unsafe {
                    choose_var_small_clause(
                        &mut generic_rng,
                        0,
                        3,
                        &cl,
                        &vars,
                        &num_good,
                        &probsat_weights,
                        &all_off,
                        &p_bound,
                        &all_data,
                    )
                };
                let fast = unsafe {
                    choose_var_small_clause3(
                        &mut fast_rng,
                        0,
                        &cl,
                        &vars,
                        &num_good,
                        &probsat_weights,
                        &all_off,
                        &p_bound,
                        &all_data,
                    )
                };

                assert_eq!(fast, generic, "num_good={num_good:?} seed={seed}");
                assert_eq!(
                    fast_rng.gen::<u64>(),
                    generic_rng.gen::<u64>(),
                    "rng mismatch for num_good={num_good:?} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn count_satisfied_clause_matches_generic_reference() {
        let vars = vec![true, false, true, false, true];
        let cl = vec![1, -2, 3, -1, 2, 4, -5, 5, -4, 3];

        for (s, e) in [(0, 0), (0, 1), (1, 3), (0, 3), (3, 7), (7, 10)] {
            let mut expected = 0u8;
            for &lit in &cl[s..e] {
                let v = lit_var_index(lit);
                if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                    expected = expected.saturating_add(1);
                }
            }

            assert_eq!(count_satisfied_clause(&vars, &cl, s, e), expected);
            assert_eq!(
                unsafe { count_satisfied_clause_unchecked(&vars, &cl, s, e) },
                expected
            );
        }
    }

    #[test]
    fn rebuild_initial_state_collects_residual_in_clause_order() {
        let vars = vec![true, true, false];
        let co = vec![0_u32, 2, 4, 7, 10];
        let cl = vec![1, -2, -1, 2, 1, 2, -3, -1, -2, 3];
        let mut num_good = vec![9_u8; 4];
        let mut residual = vec![99_u32];

        let true_unsat =
            rebuild_initial_state(4, &co, &cl, &vars, &mut num_good, &mut residual, false);

        assert_eq!(num_good, vec![1, 1, 3, 0]);
        assert_eq!(residual, vec![3]);
        assert_eq!(true_unsat, 1);
    }

    #[test]
    fn rebuild_initial_state_all_three_fast_path_matches_generic_reference() {
        let nc = 6;
        let co = [0_u32, 3, 6, 9, 12, 15, 18];
        let cl = [
            1, -2, 3, -1, 2, -3, 1, 2, 3, -1, -2, -3, 2, -3, 4, -2, 3, -4,
        ];
        let vars = [true, false, true, false];

        let mut expected_good = vec![0_u8; nc];
        let mut expected_residual = Vec::new();
        let expected_unsat = rebuild_initial_state(
            nc,
            &co,
            &cl,
            &vars,
            &mut expected_good,
            &mut expected_residual,
            false,
        );

        let mut actual_good = vec![9_u8; nc];
        let mut actual_residual = vec![99_u32];
        let actual_unsat = rebuild_initial_state(
            nc,
            &co,
            &cl,
            &vars,
            &mut actual_good,
            &mut actual_residual,
            true,
        );

        assert_eq!(actual_good, expected_good);
        assert_eq!(actual_residual, expected_residual);
        assert_eq!(actual_unsat, expected_unsat);
    }

    #[test]
    fn track3_clause_bounds_all_three_fast_path_matches_offsets() {
        let co = [0_u32, 3, 6, 9, 12];
        assert!(track3_clause_offsets_are_three(4, &co, 12));
        for cid in 0..4 {
            assert_eq!(
                unsafe { track3_clause_bounds_unchecked(cid, &co, true) },
                (co[cid] as usize, co[cid + 1] as usize)
            );
        }
    }

    #[test]
    fn track3_clause_offsets_reject_mixed_lengths_with_average_three() {
        let co = [0_u32, 2, 5, 9];
        assert!(!track3_clause_offsets_are_three(3, &co, 9));
        assert_eq!(
            unsafe { track3_clause_bounds_unchecked(1, &co, false) },
            (2, 5)
        );
    }
}
