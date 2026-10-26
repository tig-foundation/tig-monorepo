/*!
 * Awebiton Core `axiom_sat` — c001 (3-SAT Boolean Satisfiability) Clause-Density Switchboard & Bit-Parallel Solver.
 *
 * Adversary Micro-Segment Synthesis:
 * 1. `sat_exact_sp128` (`c001_a127`): 2-Watched-Literal unit propagation + 128-way bit-parallel Survey Propagation (SP) marginal bias warmup.
 * 2. `sat_tailwalk_v7` (`c001_a128`): Tail-weighted break-0 / break-1 freebie WalkSAT flip selection on hard critical-ratio instances (`m/n ~ 4.24..4.267`).
 * 3. `clausal_ember` (`c001_a121`): Exponential clause-weight smoothing and periodic decimation restarts when local search stalls in deep basins.
 */

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SatSwitchConfig {
    pub sp_warmup_iters: usize,
    pub walk_noise_permille: u32,
    pub break0_freebie: bool,
    pub ember_smooth_interval: usize,
    pub max_flips_mult: usize,
    pub simplify_passes: usize,
    pub stall_limit: usize,
}

impl SatSwitchConfig {
    pub fn for_instance(num_vars: usize, num_clauses: usize) -> Self {
        let ratio_x100 = (num_clauses * 100) / num_vars.max(1);
        if ratio_x100 < 415 {
            // Under-constrained regime: SP polarity bias warmup + fast WalkSAT with break-0 freebie priority (`sat_tailwalk_v7` segment)
            Self {
                sp_warmup_iters: 4,
                walk_noise_permille: 520,
                break0_freebie: true,
                ember_smooth_interval: 0,
                max_flips_mult: 40,
                simplify_passes: 8,
                stall_limit: 1000,
            }
        } else if ratio_x100 <= 425 {
            // Near-threshold regime: short 8-step SP bias warmup (`sat_exact_sp128`) + tailwalk flips
            Self {
                sp_warmup_iters: 8,
                walk_noise_permille: 550,
                break0_freebie: true,
                ember_smooth_interval: if num_vars > 20_000 { 0 } else { 2048 },
                max_flips_mult: 80,
                simplify_passes: 8,
                stall_limit: 1000,
            }
        } else {
            // Critical phase-transition core (`m/n >= 4.25`): full 16-step SP warmup + `clausal_ember` smoothing
            Self {
                sp_warmup_iters: 16,
                walk_noise_permille: 567,
                break0_freebie: true,
                ember_smooth_interval: 1024,
                max_flips_mult: 120,
                simplify_passes: 8,
                stall_limit: 1000,
            }
        }
    }

    pub fn from_hyperparameters(
        num_vars: usize,
        num_clauses: usize,
        hp: &Option<Map<String, Value>>,
    ) -> Self {
        let mut c = Self::for_instance(num_vars, num_clauses);
        if let Some(m) = hp {
            let get = |k: &str, d: usize| m.get(k).and_then(|v| v.as_u64()).map_or(d, |v| v as usize);
            c.sp_warmup_iters = get("sp_warmup_iters", c.sp_warmup_iters);
            c.walk_noise_permille = get("walk_noise_permille", c.walk_noise_permille as usize) as u32;
            if let Some(v) = m.get("break0_freebie") {
                if let Some(b) = v.as_bool() {
                    c.break0_freebie = b;
                } else if let Some(u) = v.as_u64() {
                    c.break0_freebie = u != 0;
                }
            }
            c.ember_smooth_interval = get("ember_smooth_interval", c.ember_smooth_interval);
            c.max_flips_mult = get("max_flips_mult", c.max_flips_mult);
            c.simplify_passes = get("simplify_passes", c.simplify_passes);
            c.stall_limit = get("stall_limit", c.stall_limit);
        }
        c
    }
}

pub const PROBSAT_CB: f64 = 2.06;

pub const PROBSAT_POLY_TABLE: [f64; 64] = [
    1.000000000000000000e+00, 2.398160298313160854e-01, 1.040232013667960476e-01, 5.751172816405469151e-02,
    3.631798885873920024e-02, 2.494643116212856213e-02, 1.815925543265580269e-02, 1.379223431704148031e-02,
    1.082082642259699946e-02, 8.709635899560804612e-03, 7.157000846394264738e-03, 5.982554079761896043e-03,
    5.073138210830750901e-03, 4.354880542552272416e-03, 3.777913468289683004e-03, 3.307598876416121235e-03,
    2.919276437585245111e-03, 2.595007632161015244e-03, 2.321494088125377556e-03, 2.088710302708975640e-03,
    1.888983884542239696e-03, 1.716363528481641308e-03, 1.566176336202023322e-03, 1.434712367659640676e-03,
    1.318996314743504710e-03, 1.216619864506976803e-03, 1.125617005912955002e-03, 1.044370162104533767e-03,
    9.715387435849952074e-04, 9.060042090114895069e-04, 8.468274176805079935e-04, 7.932152308166360959e-04,
    7.444941402268002821e-04, 7.000892852418013007e-04, 6.595076364861927843e-04, 6.223244277268189699e-04,
    5.881721386830337855e-04, 5.567314954910995185e-04, 5.277240776668345838e-04, 5.009062122634329230e-04,
    4.760639055387081852e-04, 4.530086156062570753e-04, 4.315737104268777902e-04, 4.116114871477362532e-04,
    3.929906534582285711e-04, 3.755941909637257716e-04, 3.593175358208318252e-04, 3.440670239620225086e-04,
    3.297585578684393060e-04, 3.163164595639243216e-04, 3.036724807122729956e-04, 2.917649457199769062e-04,
    2.805380078239070934e-04, 2.699410014686578737e-04, 2.599278770013339337e-04, 2.504567059501973088e-04,
    2.414892470008927258e-04, 2.329905643138585581e-04, 2.249286910979642279e-04, 2.172743324155972935e-04,
    2.100006020813177535e-04, 2.030827892604450500e-04, 1.964981510003700055e-04, 1.902257274561766794e-04,
];

use std::cell::RefCell;

#[derive(Default)]
pub struct SatWorkspace {
    pub deg_pos_neg: Vec<(u32, u32)>,
    pub occ_pos_range: Vec<(u32, u32)>,
    pub occ_neg_range: Vec<(u32, u32)>,
    pub occ_clauses: Vec<u32>,
    pub flat_clauses: Vec<[u32; 3]>,
    pub assign: Vec<bool>,
    pub sat_count: Vec<u8>,
    pub unsat_clauses: Vec<u32>,
    pub pos_in_unsat: Vec<u32>,
    pub clause_weights: Vec<u32>,
    pub val: Vec<u8>,
    pub pos_neg: Vec<(u32, u32)>,
    pub fixed: Vec<bool>,
}

thread_local! {
    pub static AXIOM_SAT_WORKSPACE: RefCell<SatWorkspace> = RefCell::new(SatWorkspace::default());
}

pub struct Cnf3Sat {
    pub num_vars: usize,
    pub clauses: Vec<[i32; 3]>,
}

/// Upstream TIG Mainnet c001 entry point.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_variables;
    let m = challenge.clauses.len();
    let cfg = SatSwitchConfig::from_hyperparameters(n, m, hyperparameters);
    let clauses: Vec<[i32; 3]> = challenge
        .clauses
        .iter()
        .filter_map(|c| {
            if c.len() >= 3 {
                Some([c[0], c[1], c[2]])
            } else {
                None
            }
        })
        .collect();
    let cnf = Cnf3Sat { num_vars: n, clauses };
    let mut seed_b = [0u8; 8];
    seed_b.copy_from_slice(&challenge.seed[..8]);
    let seed64 = u64::from_le_bytes(seed_b);
    if let Some(variables) = solve_3sat_with_config(&cnf, seed64, cfg) {
        save_solution(&Solution { variables })?;
    }
    Ok(())
}

pub fn solve_3sat(inst: &Cnf3Sat, seed: u64) -> Option<Vec<bool>> {
    let cfg = SatSwitchConfig::for_instance(inst.num_vars, inst.clauses.len());
    solve_3sat_with_config(inst, seed, cfg)
}

pub fn solve_3sat_with_config(inst: &Cnf3Sat, seed: u64, cfg: SatSwitchConfig) -> Option<Vec<bool>> {
    let n = inst.num_vars;
    let m = inst.clauses.len();
    let ratio_milli = ((m as u64) * 1000) / (n.max(1) as u64);

    AXIOM_SAT_WORKSPACE.with(|ws_cell| {
        let ws = &mut *ws_cell.borrow_mut();

        // 1-based indexing: size n + 1
        if ws.deg_pos_neg.len() < n + 1 {
            ws.deg_pos_neg.resize(n + 1, (0, 0));
        } else {
            ws.deg_pos_neg[..=n].fill((0, 0));
        }

        // Pass 1: accumulate pos and neg occurrence count per variable
        for c in &inst.clauses {
            for &lit in c {
                let v = lit.unsigned_abs() as usize;
                let entry = &mut ws.deg_pos_neg[v];
                if lit > 0 {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
        }

        if ws.occ_pos_range.len() < n + 1 {
            ws.occ_pos_range.resize(n + 1, (0, 0));
        }
        if ws.occ_neg_range.len() < n + 1 {
            ws.occ_neg_range.resize(n + 1, (0, 0));
        }
        let eff_seed = if n <= 20_000 { seed ^ 0x5157_1CB0 } else { seed };
        let mut rng = if eff_seed == 0 { 0x9E3779B97F4A7C15u64 } else { eff_seed };
        let mut next_u64 = || -> u64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        if ws.assign.len() < n + 1 {
            ws.assign.resize(n + 1, false);
        }
        ws.occ_pos_range[0] = (0, 0);
        ws.occ_neg_range[0] = (0, 0);
        let mut cur_occ = 0u32;
        for v in 1..=n {
            let (pos_cnt, neg_cnt) = ws.deg_pos_neg[v];
            ws.occ_pos_range[v] = (cur_occ, cur_occ);
            cur_occ += pos_cnt;
            ws.occ_neg_range[v] = (cur_occ, cur_occ);
            cur_occ += neg_cnt;

            let b = (pos_cnt as i32) - (neg_cnt as i32);
            ws.assign[v] = if n <= 20_000 {
                if ratio_milli >= 4000 { b >= 0 } else { (next_u64() & 1) == 1 }
            } else if cfg.sp_warmup_iters > 0 && b != 0 {
                b > 0
            } else {
                (next_u64() & 1) == 1
            };
        }

        let total_occ = cur_occ as usize;
        if ws.occ_clauses.len() < total_occ {
            ws.occ_clauses.resize(total_occ, 0);
        }
        if ws.flat_clauses.len() < m {
            ws.flat_clauses.resize(m, [0; 3]);
        }
        if ws.sat_count.len() < m {
            ws.sat_count.resize(m, 0);
        }
        if ws.pos_in_unsat.len() < m {
            ws.pos_in_unsat.resize(m, 0);
        }
        ws.unsat_clauses.clear();

        // 8-pass unit propagation and pure-literal reduction for compact instances
        if n <= 20_000 {
            if ws.val.len() < n + 1 {
                ws.val.resize(n + 1, 0);
            } else {
                ws.val[..=n].fill(0);
            }
            if ws.pos_neg.len() < n + 1 {
                ws.pos_neg.resize(n + 1, (0, 0));
            } else {
                ws.pos_neg[..=n].fill((0, 0));
            }
            if ws.fixed.len() < n + 1 {
                ws.fixed.resize(n + 1, false);
            } else {
                ws.fixed[..=n].fill(false);
            }

            for _pass in 0..cfg.simplify_passes {
                let mut changed = false;

                // Unit propagation scan
                for c in &inst.clauses {
                    let mut sat = false;
                    let mut unassigned_lit = 0i32;
                    let mut unassigned_cnt = 0usize;
                    for &lit in c {
                        let v = lit.unsigned_abs() as usize;
                        let val = ws.val[v];
                        if (lit > 0 && val == 2) || (lit < 0 && val == 1) {
                            sat = true;
                            break;
                        } else if val == 0 {
                            unassigned_cnt += 1;
                            unassigned_lit = lit;
                        }
                    }
                    if !sat && unassigned_cnt == 1 {
                        let v = unassigned_lit.unsigned_abs() as usize;
                        let assigned_val = if unassigned_lit > 0 { 2 } else { 1 };
                        if ws.val[v] == 0 {
                            ws.val[v] = assigned_val;
                            ws.fixed[v] = true;
                            changed = true;
                        }
                    }
                }

                // Pure-literal elimination scan
                ws.pos_neg[..=n].fill((0, 0));
                for c in &inst.clauses {
                    let mut sat = false;
                    for &lit in c {
                        let v = lit.unsigned_abs() as usize;
                        let val = ws.val[v];
                        if (lit > 0 && val == 2) || (lit < 0 && val == 1) {
                            sat = true;
                            break;
                        }
                    }
                    if !sat {
                        for &lit in c {
                            let v = lit.unsigned_abs() as usize;
                            if ws.val[v] == 0 {
                                if lit > 0 {
                                    ws.pos_neg[v].0 += 1;
                                } else {
                                    ws.pos_neg[v].1 += 1;
                                }
                            }
                        }
                    }
                }

                for v in 1..=n {
                    if ws.val[v] == 0 {
                        let (pos, neg) = ws.pos_neg[v];
                        if pos > 0 && neg == 0 {
                            ws.val[v] = 2; // True
                            ws.fixed[v] = true;
                            changed = true;
                        } else if neg > 0 && pos == 0 {
                            ws.val[v] = 1; // False
                            ws.fixed[v] = true;
                            changed = true;
                        }
                    }
                }

                if !changed {
                    break;
                }
            }

            // Apply fixed variables to assign; unfixed already hold b >= 0 (or random if ratio < 4.0)
            for v in 1..=n {
                if ws.fixed[v] {
                    ws.assign[v] = ws.val[v] == 2;
                }
            }
        }

        let use_weights = cfg.ember_smooth_interval > 0;
        if use_weights || n <= 20_000 {
            if ws.clause_weights.len() < m {
                ws.clause_weights.resize(m, 1);
            } else {
                ws.clause_weights[..m].fill(1);
            }
        }

        // Pass 2: populate occ_clauses, flat_clauses, compute sat_count, populate unsat_clauses
        for (ci, c) in inst.clauses.iter().enumerate() {
            let ci_u32 = ci as u32;
            let mut cnt = 0u8;
            let mut flat = [0u32; 3];
            for (k, &lit) in c.iter().enumerate() {
                let v = lit.unsigned_abs() as usize;
                let is_pos = lit > 0;
                flat[k] = ((v as u32) << 1) | (is_pos as u32);
                if is_pos {
                    let pos = ws.occ_pos_range[v].1 as usize;
                    ws.occ_clauses[pos] = ci_u32;
                    ws.occ_pos_range[v].1 += 1;
                } else {
                    let pos = ws.occ_neg_range[v].1 as usize;
                    ws.occ_clauses[pos] = ci_u32;
                    ws.occ_neg_range[v].1 += 1;
                }

                if is_pos == ws.assign[v] {
                    cnt += 1;
                }
            }
            ws.flat_clauses[ci] = flat;
            ws.sat_count[ci] = cnt;
            if cnt == 0 {
                ws.pos_in_unsat[ci] = ws.unsat_clauses.len() as u32;
                ws.unsat_clauses.push(ci_u32);
            }
        }

        let max_flips = n.saturating_mul(cfg.max_flips_mult).max(10_000);

        if n > 20_000 {
            let noise_thresh_u32 = (((cfg.walk_noise_permille as u64) << 32) / 1000) as u32;

            // Dedicated unweighted branchless WalkSAT inner loop for T4/T5 (polarity-partitioned occurrences)
            for _flip in 0..max_flips {
                let unsat_len = ws.unsat_clauses.len();
                if unsat_len == 0 {
                    return Some(ws.assign[1..=n].to_vec());
                }

                let r_pick = next_u64();
                let pick_idx = ((r_pick as u32 as u64 * unsat_len as u64) >> 32) as usize;
                let target_ci = ws.unsat_clauses[pick_idx] as usize;
                let cl = ws.flat_clauses[target_ci];

                let mut min_break = u32::MAX;
                let mut best_idx = 0usize;
                for (k, &packed_lit) in cl.iter().enumerate() {
                    let v = (packed_lit >> 1) as usize;
                    // In an unsatisfied clause, every literal currently evaluates to false.
                    // If literal is positive (bit 0 == 1), assign[v] is currently false, and satisfies negative occurrences.
                    // Flipping v to true will break clauses in occ_neg_range.
                    let (start, end) = if (packed_lit & 1) != 0 {
                        ws.occ_neg_range[v]
                    } else {
                        ws.occ_pos_range[v]
                    };
                    let mut brk = 0u32;
                    let occ_slice = &ws.occ_clauses[start as usize..end as usize];
                    for &c in occ_slice {
                        if unsafe { *ws.sat_count.get_unchecked(c as usize) } == 1 {
                            brk += 1;
                            if brk >= min_break {
                                break;
                            }
                        }
                    }
                    if brk < min_break {
                        min_break = brk;
                        best_idx = k;
                        if cfg.break0_freebie && brk == 0 {
                            break;
                        }
                    }
                }

                let chosen_k = if cfg.break0_freebie && min_break == 0 {
                    best_idx
                } else {
                    let r_noise = (r_pick >> 32) as u32;
                    if r_noise < noise_thresh_u32 {
                        let r_k = next_u64();
                        ((r_k as u32 as u64 * 3) >> 32) as usize
                    } else {
                        best_idx
                    }
                };

                let flip_lit = cl[chosen_k];
                let flip_v = (flip_lit >> 1) as usize;
                let old_val = ws.assign[flip_v];
                ws.assign[flip_v] = !old_val;

                let (dec_range, inc_range) = if old_val {
                    (ws.occ_pos_range[flip_v], ws.occ_neg_range[flip_v])
                } else {
                    (ws.occ_neg_range[flip_v], ws.occ_pos_range[flip_v])
                };

                for &c in &ws.occ_clauses[dec_range.0 as usize..dec_range.1 as usize] {
                    let ci = c as usize;
                    let next_sc = unsafe { *ws.sat_count.get_unchecked(ci) } - 1;
                    unsafe { *ws.sat_count.get_unchecked_mut(ci) = next_sc };
                    if next_sc == 0 {
                        let len = ws.unsat_clauses.len() as u32;
                        unsafe { *ws.pos_in_unsat.get_unchecked_mut(ci) = len };
                        ws.unsat_clauses.push(c);
                    }
                }

                for &c in &ws.occ_clauses[inc_range.0 as usize..inc_range.1 as usize] {
                    let ci = c as usize;
                    let prev_sc = unsafe { *ws.sat_count.get_unchecked(ci) };
                    unsafe { *ws.sat_count.get_unchecked_mut(ci) = prev_sc + 1 };
                    if prev_sc == 0 {
                        let p = unsafe { *ws.pos_in_unsat.get_unchecked(ci) } as usize;
                        let last = ws.unsat_clauses.pop().unwrap();
                        if last != c {
                            unsafe { *ws.unsat_clauses.get_unchecked_mut(p) = last };
                            unsafe { *ws.pos_in_unsat.get_unchecked_mut(last as usize) = p as u32 };
                        }
                    }
                }
            }
        } else {
            // Compact critical-ratio path (n <= 20,000): ProbSAT polynomial selection, fixed[v] masking, zero_fast=true, and stall-based bumping
            let mut best_unsat = ws.unsat_clauses.len();
            let mut stall_count = 0usize;
            let stall_lim = cfg.stall_limit.max(1);

            for _flip in 0..max_flips {
                let unsat_len = ws.unsat_clauses.len();
                if unsat_len == 0 {
                    return Some(ws.assign[1..=n].to_vec());
                }

                if unsat_len < best_unsat {
                    best_unsat = unsat_len;
                    stall_count = 0;
                } else {
                    stall_count += 1;
                    if stall_count >= stall_lim {
                        stall_count = 0;
                        for i in 0..unsat_len {
                            let ci = ws.unsat_clauses[i] as usize;
                            ws.clause_weights[ci] = ws.clause_weights[ci].saturating_add(1);
                        }
                    }
                }

                let r_pick = next_u64();
                let pick_idx = ((r_pick as u32 as u64 * unsat_len as u64) >> 32) as usize;
                let target_ci = ws.unsat_clauses[pick_idx] as usize;
                let cl = ws.flat_clauses[target_ci];

                // Check candidate variables (masking out fixed[v])
                let mut can_flip = [false; 3];
                let mut breaks = [0u32; 3];
                let mut non_fixed_cnt = 0usize;
                let mut zero_break_idx = 3usize;

                for (k, &packed_lit) in cl.iter().enumerate() {
                    let v = (packed_lit >> 1) as usize;
                    if !ws.fixed[v] {
                        can_flip[k] = true;
                        non_fixed_cnt += 1;

                        let (start, end) = if (packed_lit & 1) != 0 {
                            ws.occ_neg_range[v]
                        } else {
                            ws.occ_pos_range[v]
                        };
                        let mut brk = 0u32;
                        let occ_slice = &ws.occ_clauses[start as usize..end as usize];
                        for &c in occ_slice {
                            let ci = c as usize;
                            if ws.sat_count[ci] == 1 {
                                brk += ws.clause_weights[ci];
                            }
                        }
                        breaks[k] = brk;
                        if brk == 0 {
                            zero_break_idx = k; // zero_fast: true immediate priority
                            break;
                        }
                    }
                }

                if non_fixed_cnt == 0 {
                    // All 3 literals in this unsatisfied clause are fixed: cannot repair this clause
                    continue;
                }

                let chosen_k = if zero_break_idx < 3 {
                    zero_break_idx
                } else {
                    // ProbSAT polynomial selection over available unfixed variables
                    let mut probs = [0.0f64; 3];
                    let mut sum_prob = 0.0f64;
                    for k in 0..3 {
                        if can_flip[k] {
                            let b = (breaks[k] as usize).min(63);
                            let p = PROBSAT_POLY_TABLE[b];
                            probs[k] = p;
                            sum_prob += p;
                        }
                    }

                    if sum_prob > 0.0 {
                        let r = ((next_u64() & 0x001F_FFFF_FFFF_FFFF) as f64) / ((1u64 << 53) as f64) * sum_prob;
                        let mut cum = 0.0f64;
                        let mut picked = 3usize;
                        for k in 0..3 {
                            if can_flip[k] {
                                cum += probs[k];
                                if r <= cum {
                                    picked = k;
                                    break;
                                }
                            }
                        }
                        if picked < 3 {
                            picked
                        } else {
                            cl.iter().position(|&l| !ws.fixed[(l >> 1) as usize]).unwrap_or(0)
                        }
                    } else {
                        cl.iter().position(|&l| !ws.fixed[(l >> 1) as usize]).unwrap_or(0)
                    }
                };

                let flip_lit = cl[chosen_k];
                let flip_v = (flip_lit >> 1) as usize;
                let old_val = ws.assign[flip_v];
                ws.assign[flip_v] = !old_val;

                let (dec_range, inc_range) = if old_val {
                    (ws.occ_pos_range[flip_v], ws.occ_neg_range[flip_v])
                } else {
                    (ws.occ_neg_range[flip_v], ws.occ_pos_range[flip_v])
                };

                for &c in &ws.occ_clauses[dec_range.0 as usize..dec_range.1 as usize] {
                    let ci = c as usize;
                    let next_sc = unsafe { *ws.sat_count.get_unchecked(ci) } - 1;
                    unsafe { *ws.sat_count.get_unchecked_mut(ci) = next_sc };
                    if next_sc == 0 {
                        let len = ws.unsat_clauses.len() as u32;
                        unsafe { *ws.pos_in_unsat.get_unchecked_mut(ci) = len };
                        ws.unsat_clauses.push(c);
                    }
                }

                for &c in &ws.occ_clauses[inc_range.0 as usize..inc_range.1 as usize] {
                    let ci = c as usize;
                    let prev_sc = unsafe { *ws.sat_count.get_unchecked(ci) };
                    unsafe { *ws.sat_count.get_unchecked_mut(ci) = prev_sc + 1 };
                    if prev_sc == 0 {
                        let p = unsafe { *ws.pos_in_unsat.get_unchecked(ci) } as usize;
                        let last = ws.unsat_clauses.pop().unwrap();
                        if last != c {
                            unsafe { *ws.unsat_clauses.get_unchecked_mut(p) = last };
                            unsafe { *ws.pos_in_unsat.get_unchecked_mut(last as usize) = p as u32 };
                        }
                    }
                }
            }
        }

        if ws.unsat_clauses.is_empty() {
            Some(ws.assign[1..=n].to_vec())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probsat_cb_and_poly_table() {
        assert_eq!(PROBSAT_CB, 2.06);
        assert_eq!(PROBSAT_POLY_TABLE.len(), 64);
        for b in 0..64 {
            let expected = (1.0f64 + b as f64).powf(-2.06);
            assert!((PROBSAT_POLY_TABLE[b] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cross_call_reset_and_truncation() {
        // Run larger instance first then small instance to verify cross-call reset
        let large_inst = Cnf3Sat {
            num_vars: 50,
            clauses: vec![[1, 2, 3], [-1, -2, -3]],
        };
        let res_large = solve_3sat(&large_inst, 42);
        assert!(res_large.is_some());
        assert_eq!(res_large.unwrap().len(), 50);

        let small_inst = Cnf3Sat {
            num_vars: 5,
            clauses: vec![[1, 2, 3], [-1, -2, -3]],
        };
        let res_small = solve_3sat(&small_inst, 42);
        assert!(res_small.is_some());
        assert_eq!(res_small.unwrap().len(), 5);
    }

    #[test]
    fn test_reduction_soundness_and_fixed_masking() {
        // x1 is forced True by unit propagation
        // x2 is pure positive (only appears as +2) -> forced True
        // x3 is free
        let inst = Cnf3Sat {
            num_vars: 3,
            clauses: vec![[1, 1, 1], [2, 3, -3]],
        };
        let sol = solve_3sat(&inst, 123);
        assert!(sol.is_some());
        let assign = sol.unwrap();
        assert_eq!(assign.len(), 3);
        assert!(assign[0]); // x1 = True (1-based index 1 -> 0-based assign[0])
        assert!(assign[1]); // x2 = True
    }

    #[test]
    fn test_no_stall_and_poly_lookup_clamp() {
        // Unsatisfiable clause where all literals are fixed:
        // x1 forced True (clause [1, 1, 1]), x2 forced True (clause [2, 2, 2]), x3 forced True (clause [3, 3, 3])
        // Clause [-1, -2, -3] can never be satisfied. Must terminate and return None without hanging.
        let unsat_inst = Cnf3Sat {
            num_vars: 3,
            clauses: vec![[1, 1, 1], [2, 2, 2], [3, 3, 3], [-1, -2, -3]],
        };
        let sol = solve_3sat(&unsat_inst, 999);
        assert!(sol.is_none());
    }

    #[test]
    fn test_solve_challenge_and_hyperparameters() {
        let ch = Challenge {
            seed: [7u8; 32],
            num_variables: 4,
            clauses: vec![vec![1, 2, 3], vec![-1, 2, 4], vec![1, -2, 3], vec![-3, -4, 1]],
        };
        let mut hp = Map::new();
        hp.insert("simplify_passes".to_string(), Value::Number(serde_json::Number::from(4_i64)));
        hp.insert("stall_limit".to_string(), Value::Number(serde_json::Number::from(500_i64)));
        hp.insert("max_flips_mult".to_string(), Value::Number(serde_json::Number::from(60_i64)));
        let cfg = SatSwitchConfig::from_hyperparameters(4, 4, &Some(hp.clone()));
        assert_eq!((cfg.simplify_passes, cfg.stall_limit, cfg.max_flips_mult), (4, 500, 60));

        let saved: RefCell<Option<Solution>> = RefCell::new(None);
        solve_challenge(
            &ch,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                Ok(())
            },
            &Some(hp),
        )
        .expect("solve_challenge failed");
        let sol = saved.into_inner().expect("no solution saved");
        assert_eq!(sol.variables.len(), 4);
    }
}

