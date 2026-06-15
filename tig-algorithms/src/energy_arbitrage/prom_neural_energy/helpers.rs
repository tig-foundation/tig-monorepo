use anyhow::Result;
use serde_json::{Map, Value};
use std::cell::RefCell;
use super::*;
#[derive(Clone, Debug)]
pub struct TrackHp {
    pub soc_levels: usize,
    pub action_grid: usize,
    pub asca_iters: usize,
    pub ternary_iters: usize,
    pub convergence_tol: f64,
    pub anticipate_lmp: bool,
    pub lmp_threshold: f64,
    pub lmp_premium_scale: f64,
    pub jump_premium: f64,
    pub prune_ratio: f64,
    pub deflator_iters: usize,
    pub flow_margin: f64,
    pub network_derating: f64,
}

impl TrackHp {
    pub fn override_from_map(&mut self, h: &Option<Map<String, Value>>) {
        let Some(m) = h else { return };
        if let Some(v) = m.get("soc_levels").and_then(|v| v.as_u64()) { self.soc_levels = (v as usize).max(3); }
        if let Some(v) = m.get("action_grid").and_then(|v| v.as_u64()) { self.action_grid = (v as usize).max(4); }
        if let Some(v) = m.get("asca_iters").and_then(|v| v.as_u64()) { self.asca_iters = v as usize; }
        if let Some(v) = m.get("ternary_iters").and_then(|v| v.as_u64()) { self.ternary_iters = v as usize; }
        if let Some(v) = m.get("convergence_tol").and_then(|v| v.as_f64()) { self.convergence_tol = v; }
        if let Some(v) = m.get("anticipate_lmp").and_then(|v| v.as_bool()) { self.anticipate_lmp = v; }
        if let Some(v) = m.get("lmp_threshold").and_then(|v| v.as_f64()) { self.lmp_threshold = v; }
        if let Some(v) = m.get("lmp_premium_scale").and_then(|v| v.as_f64()) { self.lmp_premium_scale = v; }
        if let Some(v) = m.get("jump_premium").and_then(|v| v.as_f64()) { self.jump_premium = v; }
        if let Some(v) = m.get("prune_ratio").and_then(|v| v.as_f64()) { self.prune_ratio = v.clamp(0.0, 0.9); }
        if let Some(v) = m.get("deflator_iters").and_then(|v| v.as_u64()) { self.deflator_iters = v as usize; }
        if let Some(v) = m.get("flow_margin").and_then(|v| v.as_f64()) { self.flow_margin = v.max(0.0); }
        if let Some(v) = m.get("network_derating").and_then(|v| v.as_f64()) { self.network_derating = v.clamp(0.01, 1.0); }
    }
}

pub struct TitanCache {
    pub dp: Vec<Vec<Vec<f64>>>,
    pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
    pub b_to_lines: Vec<Vec<(usize, f64)>>,
    pub batt_nodes: Vec<usize>,
}

struct Inner {
    hp: TrackHp,
    cache: Option<TitanCache>,
}

thread_local! {
    static STATE: RefCell<Option<Inner>> = RefCell::new(None);
}

pub fn solve_with_hp(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: TrackHp,
) -> Result<()> {
    STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
    let out = challenge.grid_optimize(&policy_entry);
    STATE.with(|s| *s.borrow_mut() = None);
    let solution = out?;
    save_solution(&solution)?;
    Ok(())
}

fn policy_entry(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    STATE.with(|s| -> Result<Vec<f64>> {
        let mut guard = s.borrow_mut();
        let inner = guard.as_mut().expect("titan: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;

        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);

        let mut actions = vec![0.0; challenge.num_batteries];

        if challenge.num_batteries <= 12 {
            run_joint_enumeration(challenge, state, cache, hp, &flows_base, &mut actions);
            run_baseline_refine(challenge, state, cache, hp, &flows_base, &mut actions, 3);
        }
        {
            let mut refine_hp = hp.clone();
            refine_hp.asca_iters = if challenge.num_batteries <= 15 {
                refine_hp.asca_iters.min(8)
            } else {
                refine_hp.asca_iters
            };
            if challenge.num_batteries <= 12 {
                run_greedy_flow_room(challenge, state, cache, &refine_hp, &flows_base, &mut actions);
            } else {
                run_greedy_flow_room(challenge, state, cache, hp, &flows_base, &mut actions);
            }
            run_asca(challenge, state, cache, &refine_hp, &flows_base, &mut actions);
        }
        run_deflator(challenge, state, cache, hp, &flows_base, &mut actions);
        run_local_search(challenge, state, cache, hp, &flows_base, &mut actions);
        Ok(actions)
    })
}

fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> TitanCache {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    let num_t = challenge.num_steps;
    let num_n = challenge.network.num_nodes;

    let zero_action = vec![0.0_f64; num_b];
    let inj_base = challenge.compute_total_injections(state, &zero_action);
    let flows0 = challenge.network.compute_flows(&inj_base);

    let mut batt_nodes = vec![0usize; num_b];
    let mut ptdf_sparse: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_l];
    let mut b_to_lines: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_b];
    let mut dummy = zero_action.clone();
    for b in 0..num_b {
        dummy[b] = 1.0;
        let inj1 = challenge.compute_total_injections(state, &dummy);
        let flows1 = challenge.network.compute_flows(&inj1);
        for k in 0..num_n {
            if (inj1[k] - inj_base[k]).abs() > 0.5 && k != challenge.network.slack_bus {
                batt_nodes[b] = k;
                break;
            }
        }
        for l in 0..num_l {
            let impact = flows1[l] - flows0[l];
            if impact.abs() > 1e-8 {
                ptdf_sparse[l].push((b, impact));
                b_to_lines[b].push((l, impact));
            }
        }
        dummy[b] = 0.0;
    }

    let mut expected_premiums = vec![vec![0.0_f64; num_b]; num_t];
    if hp.anticipate_lmp && num_l > 0 {
        let base_premium = 20.0 * hp.lmp_premium_scale;
        for t in 0..num_t {
            let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let ratio = f_exo[l].abs() / limit;
                if ratio > hp.lmp_threshold {
                    let proba = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6))
                        .clamp(0.0, 1.0);
                    let premium = base_premium * proba;
                    let sign_f = f_exo[l].signum();
                    for &(b, impact) in &ptdf_sparse[l] {
                        if impact.abs() > 1e-6 {
                            let nodal_shift = -impact * sign_f * premium;
                            expected_premiums[t][b] += nodal_shift;
                        }
                    }
                }
            }
        }
    }

    let soc_levels = hp.soc_levels;
    let action_grid = hp.action_grid;
    let dt = 0.25_f64;
    let mut dp = vec![vec![vec![0.0_f64; soc_levels]; num_t + 1]; num_b];

    for b in 0..num_b {
        let bat = &challenge.batteries[b];
        let node = batt_nodes[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);

        for t in (0..num_t).rev() {
            let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                challenge.market.day_ahead_prices[t][node]
            } else {
                challenge.market.day_ahead_prices[t][0]
            };
            let extra = expected_premiums[t][b];
            let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
            let p_buy = p_da + extra;

            for i in 0..soc_levels {
                let soc = bat.soc_min_mwh + soc_span * (i as f64) / ((soc_levels - 1) as f64);

                let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                    (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                } else { 0.0 };
                let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                    (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                } else { 0.0 };

                let max_pwr_c = bat.power_charge_mw * hp.network_derating;
                let max_pwr_d = bat.power_discharge_mw * hp.network_derating;

                let u_min = -(max_pwr_c.min(charge_soc_limit.max(0.0)));
                let u_max = max_pwr_d.min(discharge_soc_limit.max(0.0));
                let u_max = u_max.max(u_min);

                let mut max_val = f64::NEG_INFINITY;
                let span = u_max - u_min;
                for j in 0..=action_grid {
                    let u = if span > 0.0 {
                        u_min + span * (j as f64) / (action_grid as f64)
                    } else { u_min };
                    let price = if u > 0.0 { p_sell } else { p_buy };
                    let abs_u = u.abs();
                    let revenue = u * price * dt;
                    let tx = 0.25 * abs_u * dt;
                    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                    let deg = deg_base * deg_base;
                    let profit = revenue - tx - deg;

                    let next_soc_raw = if u < 0.0 {
                        soc + bat.efficiency_charge * (-u) * dt
                    } else {
                        soc - u / bat.efficiency_discharge.max(1e-9) * dt
                    };
                    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

                    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let idx0 = (idx_f.floor() as isize).max(0) as usize;
                    let idx0c = idx0.min(soc_levels - 1);
                    let idx1c = (idx0 + 1).min(soc_levels - 1);
                    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                    let v_next = dp[b][t + 1][idx0c] * (1.0 - frac)
                        + dp[b][t + 1][idx1c] * frac;

                    let val = profit + v_next;
                    if val > max_val { max_val = val; }
                }
                dp[b][t][i] = max_val;
            }
        }
    }

    TitanCache { dp, ptdf_sparse, b_to_lines, batt_nodes }
}

#[inline]
fn eval_profit(challenge: &Challenge, state: &State, ca: &TitanCache, b: usize, u: f64) -> f64 {
    let bat = &challenge.batteries[b];
    let node = ca.batt_nodes[b];
    let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
    let dt = 0.25_f64;
    let abs_u = u.abs();
    let revenue = u * rt_price * dt;
    let tx = 0.25 * abs_u * dt;
    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
    let deg = deg_base * deg_base;
    let profit = revenue - tx - deg;

    let soc = state.socs[b];
    let next_soc_raw = if u < 0.0 {
        soc + bat.efficiency_charge * (-u) * dt
    } else {
        soc - u / bat.efficiency_discharge.max(1e-9) * dt
    };
    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

    let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
    let soc_levels = ca.dp[b][0].len();
    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
    let idx0 = (idx_f.floor() as isize).max(0) as usize;
    let idx0c = idx0.min(soc_levels - 1);
    let idx1c = (idx0 + 1).min(soc_levels - 1);
    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
    let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
    profit + ca.dp[b][t_next][idx0c] * (1.0 - frac) + ca.dp[b][t_next][idx1c] * frac
}

fn run_joint_enumeration(
    challenge: &Challenge,
    state: &State,
    ca: &TitanCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let m = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();

    let levels: Vec<Vec<f64>> = (0..m)
        .map(|b| vec![state.action_bounds[b].0, 0.0, state.action_bounds[b].1])
        .collect();
    let level_values: Vec<Vec<f64>> = (0..m)
        .map(|b| levels[b].iter().map(|&u| eval_profit(challenge, state, ca, b, u)).collect())
        .collect();
    let zero_idx: Vec<usize> = (0..m)
        .map(|b| levels[b].iter().position(|&v| v.abs() < 1e-9).unwrap_or(1))
        .collect();

    let mut best_val: f64 = (0..m).map(|b| level_values[b][zero_idx[b]]).sum();
    let mut best = vec![0.0_f64; m];

    let total = 3usize.pow(m as u32);
    let mut idx = vec![0usize; m];
    for _ in 0..total {
        let is_zero = (0..m).all(|b| idx[b] == zero_idx[b]);
        if !is_zero {
            let mut val = 0.0_f64;
            for b in 0..m { val += level_values[b][idx[b]]; }
            if val > best_val {
                let mut cand = vec![0.0_f64; m];
                let mut cf = flows_base.to_vec();
                let mut ok = true;
                for b in 0..m {
                    let u = levels[b][idx[b]];
                    cand[b] = u;
                    if u.abs() < 1e-12 { continue; }
                    for &(l, p) in &ca.b_to_lines[b] {
                        if l < num_l { cf[l] += p * u; }
                    }
                }
                for l in 0..num_l {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if cf[l].abs() > limit { ok = false; break; }
                }
                if ok {
                    best_val = val;
                    best = cand;
                }
            }
        }
        let mut carry = true;
        for b in 0..m {
            if carry {
                idx[b] += 1;
                if idx[b] >= 3 { idx[b] = 0; } else { carry = false; }
            }
        }
        if carry { break; }
    }
    for b in 0..m { actions[b] = best[b]; }
}

fn run_baseline_refine(
    challenge: &Challenge,
    state: &State,
    ca: &TitanCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
    sweeps: usize,
) {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();

    let mut flows: Vec<f64> = flows_base.to_vec();
    for b in 0..num_b {
        let u = actions[b];
        if u.abs() < 1e-12 { continue; }
        for &(l, p) in &ca.b_to_lines[b] { if l < num_l { flows[l] += p * u; } }
    }

    for _ in 0..sweeps {
        for b in 0..num_b {
            let (mut u_min, mut u_max) = state.action_bounds[b];

            for &(l, p) in &ca.b_to_lines[b] {
                if p.abs() < 1e-9 { continue; }
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let f_other = flows[l] - p * actions[b];
                let b1 = (-limit - f_other) / p;
                let b2 = (limit - f_other) / p;
                let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
                if lo > u_min { u_min = lo; }
                if hi < u_max { u_max = hi; }
            }

            if u_min > u_max { continue; }

            let mut best_u = actions[b];
            let mut best_v = eval_profit(challenge, state, ca, b, best_u);

            let v0 = eval_profit(challenge, state, ca, b, 0.0);
            if u_min <= 0.0 && 0.0 <= u_max && v0 > best_v { best_v = v0; best_u = 0.0; }

            if u_min < 0.0 {
                let lo = u_min; let hi = 0.0_f64.min(u_max);
                if lo < hi {
                    let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
                    if v > best_v { best_v = v; best_u = u; }
                }
            }

            if u_max > 0.0 {
                let lo = 0.0_f64.max(u_min); let hi = u_max;
                if lo < hi {
                    let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
                    if v > best_v { best_v = v; best_u = u; }
                }
            }

            let delta = best_u - actions[b];
            if delta.abs() > 1e-6 {
                actions[b] = best_u;
                for &(l, p) in &ca.b_to_lines[b] { if l < num_l { flows[l] += p * delta; } }
            }
        }
    }
}

fn run_greedy_flow_room(
    challenge: &Challenge,
    state: &State,
    ca: &TitanCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();

    let mut flows: Vec<f64> = flows_base.to_vec();
    for b in 0..num_b {
        let u = actions[b];
        if u.abs() < 1e-12 { continue; }
        for &(l, p) in &ca.b_to_lines[b] { if l < num_l { flows[l] += p * u; } }
    }

    let mut items: Vec<(usize, f64, f64)> = Vec::with_capacity(num_b);
    for b in 0..num_b {
        let (u_min, u_max) = state.action_bounds[b];
        let v0 = eval_profit(challenge, state, ca, b, 0.0);

        let (u_c, v_c) = if u_min < 0.0 {
            ternary_search(|u| eval_profit(challenge, state, ca, b, u), u_min, 0.0_f64.min(u_max), hp.ternary_iters)
        } else { (u_min, eval_profit(challenge, state, ca, b, u_min)) };

        let (u_d, v_d) = if u_max > 0.0 {
            ternary_search(|u| eval_profit(challenge, state, ca, b, u), 0.0_f64.max(u_min), u_max, hp.ternary_iters)
        } else { (u_max, eval_profit(challenge, state, ca, b, u_max)) };

        let (u_opt, v_opt) = if v_c >= v_d { (u_c, v_c) } else { (u_d, v_d) };
        let density = if u_opt.abs() > 1e-12 { (v_opt - v0).max(0.0) / u_opt.abs() } else { 0.0 };
        items.push((b, u_opt, density));
    }

    items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    for &(b, u_opt, density) in &items {
        if density < 1e-12 || u_opt.abs() < 1e-12 { continue; }

        let mut u = u_opt;
        for &(l, p) in &ca.b_to_lines[b] {
            if p.abs() < 1e-9 { continue; }
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            let f_other = flows[l] - p * actions[b];
            if u > 0.0 {
                let bound = if p > 0.0 { (limit - f_other) / p } else { (-limit - f_other) / p };
                u = u.min(bound);
            } else {
                let bound = if p > 0.0 { (-limit - f_other) / p } else { (limit - f_other) / p };
                u = u.max(bound);
            }
        }

        let (u_min_soc, u_max_soc) = state.action_bounds[b];
        u = u.clamp(u_min_soc, u_max_soc);
        if (u > 0.0) != (u_opt > 0.0) && u.abs() > 1e-12 {
            let v_u = eval_profit(challenge, state, ca, b, u);
            let v_z = eval_profit(challenge, state, ca, b, 0.0);
            if v_u < v_z { u = 0.0; }
        }

        let delta = u - actions[b];
        if delta.abs() > 1e-12 {
            actions[b] = u;
            for &(l, p) in &ca.b_to_lines[b] { if l < num_l { flows[l] += p * delta; } }
        }
    }
}

fn run_asca(
    challenge: &Challenge,
    state: &State,
    ca: &TitanCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    let mut flows: Vec<f64> = flows_base.to_vec();
    for b in 0..num_b {
        let u = actions[b];
        if u.abs() < 1e-12 { continue; }
        for &(l, p) in &ca.b_to_lines[b] { if l < num_l { flows[l] += p * u; } }
    }

    let mut active = vec![true; num_b];
    if hp.prune_ratio > 0.0 && num_b >= 2 {
        let cutoff = ((num_b as f64) * hp.prune_ratio) as usize;
        if cutoff > 0 {
            let mut caps: Vec<(usize, f64)> = challenge.batteries.iter().enumerate().map(|(i, b)| (i, b.capacity_mwh)).collect();
            caps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for i in 0..cutoff.min(num_b) { active[caps[i].0] = false; }
        }
    }

    let footprint = |batt: usize| -> f64 {
        let mut fp = 1e-4;
        for &(l, p) in &ca.b_to_lines[batt] {
            let limit = challenge.network.flow_limits[l];
            if limit > 1e-6 {
                let utilization = flows_base[l].abs() / limit;
                fp += p.abs() * utilization.powi(2) * 10.0;
            }
        }
        fp
    };

    let mut order: Vec<usize> = (0..num_b).filter(|&b| active[b]).collect();
    order.sort_by(|&a, &b| {
        let va = potential(challenge, state, ca, a);
        let vb = potential(challenge, state, ca, b);
        let sa = va / footprint(a);
        let sb = vb / footprint(b);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    for _sweep in 0..hp.asca_iters {
        let mut max_change = 0.0_f64;

        for &b in &order {
            let (mut u_min, mut u_max) = state.action_bounds[b];

            for &(l, p) in &ca.b_to_lines[b] {
                if p.abs() < 1e-9 { continue; }
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let f_other = flows[l] - p * actions[b];
                let b1 = (-limit - f_other) / p;
                let b2 = (limit - f_other) / p;
                let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
                if lo > u_min { u_min = lo; }
                if hi < u_max { u_max = hi; }
            }

            if u_min > u_max { u_min = actions[b]; u_max = actions[b]; }
            u_min = u_min.min(actions[b]); u_max = u_max.max(actions[b]);

            let mut best_u = actions[b];
            let mut best_v = eval_profit(challenge, state, ca, b, best_u);

            let v0 = eval_profit(challenge, state, ca, b, 0.0);
            if u_min <= 0.0 && 0.0 <= u_max && v0 > best_v { best_v = v0; best_u = 0.0; }

            if u_min < 0.0 {
                let lo = u_min; let hi = 0.0_f64.min(u_max);
                if lo < hi {
                    let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
                    if v > best_v { best_v = v; best_u = u; }
                }
            }

            if u_max > 0.0 {
                let lo = 0.0_f64.max(u_min); let hi = u_max;
                if lo < hi {
                    let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
                    if v > best_v { best_v = v; best_u = u; }
                }
            }

            let delta = best_u - actions[b];
            if delta.abs() > 1e-6 {
                actions[b] = best_u;
                for &(l, p) in &ca.b_to_lines[b] { if l < num_l { flows[l] += p * delta; } }
                if delta.abs() > max_change { max_change = delta.abs(); }
            }
        }
        if max_change < hp.convergence_tol { break; }
    }
}

#[inline]
fn potential(challenge: &Challenge, state: &State, ca: &TitanCache, b: usize) -> f64 {
    let (u_lo, u_hi) = state.action_bounds[b];
    let v_lo = eval_profit(challenge, state, ca, b, u_lo);
    let v_hi = eval_profit(challenge, state, ca, b, u_hi);
    let v0 = eval_profit(challenge, state, ca, b, 0.0);
    (v_lo.max(v_hi) - v0).max(0.0)
}

fn ternary_search<F: Fn(f64) -> f64>(f: F, mut l: f64, mut r: f64, iters: usize) -> (f64, f64) {
    if l >= r { return (l, f(l)); }
    for _ in 0..iters {
        let m1 = l + (r - l) / 3.0;
        let m2 = r - (r - l) / 3.0;
        if f(m1) < f(m2) { l = m1; } else { r = m2; }
    }
    let u = 0.5 * (l + r);
    (u, f(u))
}

fn run_deflator(
    challenge: &Challenge,
    state: &State,
    ca: &TitanCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_l = challenge.network.flow_limits.len();
    let num_b = challenge.num_batteries;

    let mut flows = vec![0.0_f64; num_l];
    for l in 0..num_l {
        let mut f = 0.0_f64;
        for &(b, imp) in &ca.ptdf_sparse[l] { f += imp * actions[b]; }
        flows[l] = flows_base[l] + f;
    }

    let mut is_safe = true;
    for _ in 0..hp.deflator_iters {
        is_safe = true;
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if flows[l].abs() <= limit { continue; }
            is_safe = false;
            let overflow = flows[l].abs() - limit;
            let sign = flows[l].signum();

            let mut culprits: Vec<(usize, f64, f64)> = Vec::new();
            for &(b, impact) in &ca.ptdf_sparse[l] {
                let contrib = impact * actions[b];
                if contrib * sign > 1e-9 {
                    let val_curr = eval_profit(challenge, state, ca, b, actions[b]);
                    let val_zero = eval_profit(challenge, state, ca, b, 0.0);
                    let denom = actions[b].abs().max(1.0);
                    let roi = ((val_curr - val_zero).max(0.0)) / denom;
                    culprits.push((b, contrib, roi));
                }
            }
            culprits.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            let mut remaining = overflow;
            for (b, contrib, _) in culprits {
                if remaining <= 1e-9 { break; }
                let contrib_abs = contrib.abs();
                if contrib_abs < 1e-12 { continue; }
                let reduction = contrib_abs.min(remaining);
                let ratio = 1.0 - (reduction / contrib_abs);
                let new_action = actions[b] * ratio;
                let delta = new_action - actions[b];
                actions[b] = new_action;
                for &(ll, pp) in &ca.b_to_lines[b] { if ll < num_l { flows[ll] += pp * delta; } }
                remaining -= reduction;
            }
        }
        if is_safe { break; }
    }

    if is_safe { return; }

    let f_act: Vec<f64> = (0..num_l).map(|l| {
        let mut s = 0.0;
        for &(b, imp) in &ca.ptdf_sparse[l] { s += imp * actions[b]; }
        s
    }).collect();

    let mut beta = 1.0_f64;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        let total = flows_base[l] + f_act[l];
        if total.abs() <= limit { continue; }
        if f_act[l].abs() < 1e-9 { continue; }
        let target = if total > 0.0 { limit } else { -limit };
        let candidate = (target - flows_base[l]) / f_act[l];
        if candidate < beta { beta = candidate; }
    }
    let beta = beta.clamp(0.0, 1.0);
    for b in 0..num_b { actions[b] *= beta; }

    for b in 0..num_b {
        let (lo, hi) = state.action_bounds[b];
        if actions[b] < lo { actions[b] = lo; }
        if actions[b] > hi { actions[b] = hi; }
    }
}

fn run_local_search(
    challenge: &Challenge,
    state: &State,
    ca: &TitanCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();

    let mut base_flows: Vec<f64> = flows_base.to_vec();
    for b in 0..num_b {
        let u = actions[b];
        if u.abs() < 1e-12 { continue; }
        for &(l, p) in &ca.b_to_lines[b] { if l < num_l { base_flows[l] += p * u; } }
    }

    let mut best_actions: Vec<f64> = actions.to_vec();
    let mut best_val: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, best_actions[b])).sum();

    let seed = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) ^ (state.time_step as u64);
    let mut rng = SmallRng::seed_from_u64(seed);

    let iters = (num_b * 30).max(120);
    let mut scratch = vec![0.0_f64; num_l];
    let mut cand = vec![0.0_f64; num_b];
    let mut indices: Vec<usize> = (0..num_b).collect();

    for _ in 0..iters {
        cand.copy_from_slice(&best_actions);

        for i in (1..num_b).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let nmut = if num_b >= 3 { 3 } else { num_b };
        let mut changed = false;
        for &b in indices.iter().take(nmut) {
            let (u_lo, u_hi) = state.action_bounds[b];
            let span = u_hi - u_lo;
            if span < 1e-9 { continue; }
            let cur = best_actions[b];
            let delta = (rng.gen::<f64>() - 0.5) * span * 0.7;
            let nu = (cur + delta).clamp(u_lo, u_hi);
            if (nu - cur).abs() > 1e-12 { cand[b] = nu; changed = true; }
        }
        if !changed { continue; }

        scratch.copy_from_slice(&base_flows);
        let mut feasible = true;
        for b in 0..num_b {
            let delta = cand[b] - best_actions[b];
            if delta.abs() < 1e-12 { continue; }
            for &(l, p) in &ca.b_to_lines[b] {
                if l < num_l { scratch[l] += p * delta; }
            }
        }
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if scratch[l].abs() > limit { feasible = false; break; }
        }
        if !feasible { continue; }

        let val: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, cand[b])).sum();
        if val > best_val + 1e-8 {
            best_val = val;
            best_actions.copy_from_slice(&cand);
            base_flows.copy_from_slice(&scratch);
        }
    }

    actions.copy_from_slice(&best_actions);
}
