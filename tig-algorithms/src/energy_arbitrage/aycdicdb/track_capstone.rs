
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod helpers {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use std::cell::RefCell;
    use std::collections::BinaryHeap;
    use tig_challenges::energy_arbitrage::*;

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
        pub dual_iters: usize,
        pub da_step_size: f64,
        pub ldd_iters: usize,
        pub ldd_step_size: f64,        
        pub use_kkt: bool,
        pub kkt_cong_threshold: f64,
        pub kkt_price_scale: f64,
        pub max_admm_iters: usize,
        pub admm_rho: f64,
        pub admm_primal_tol: f64,
        pub use_lp: bool,
        pub lp_soft_lambda: f64,  
        pub lp_per_call_pivots: usize,  
        pub lp_total_pivots: usize,     
        pub lp_horizon: usize,  
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
            if let Some(v) = m.get("dual_iters").and_then(|v| v.as_u64()) { self.dual_iters = v as usize; }
            if let Some(v) = m.get("da_step_size").and_then(|v| v.as_f64()) { self.da_step_size = v.max(0.0); }
            if let Some(v) = m.get("ldd_iters").and_then(|v| v.as_u64()) { self.ldd_iters = v as usize; }
            if let Some(v) = m.get("ldd_step_size").and_then(|v| v.as_f64()) { self.ldd_step_size = v.max(0.0); }
            if let Some(v) = m.get("use_kkt").and_then(|v| v.as_bool()) { self.use_kkt = v; }
            if let Some(v) = m.get("kkt_cong_threshold").and_then(|v| v.as_f64()) { self.kkt_cong_threshold = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("kkt_price_scale").and_then(|v| v.as_f64()) { self.kkt_price_scale = v.max(0.1); }
            if let Some(v) = m.get("max_admm_iters").and_then(|v| v.as_u64()) { self.max_admm_iters = v as usize; }
            if let Some(v) = m.get("admm_rho").and_then(|v| v.as_f64()) { self.admm_rho = v; }
            if let Some(v) = m.get("admm_primal_tol").and_then(|v| v.as_f64()) { self.admm_primal_tol = v; }
            if let Some(v) = m.get("use_lp").and_then(|v| v.as_bool()) { self.use_lp = v; }
            if let Some(v) = m.get("lp_soft_lambda").and_then(|v| v.as_f64()) { self.lp_soft_lambda = v.max(0.0); }
            if let Some(v) = m.get("lp_per_call_pivots").and_then(|v| v.as_u64()) { self.lp_per_call_pivots = v as usize; }
            if let Some(v) = m.get("lp_total_pivots").and_then(|v| v.as_u64()) { self.lp_total_pivots = v as usize; }
            if let Some(v) = m.get("lp_horizon").and_then(|v| v.as_u64()) { self.lp_horizon = v as usize; }
        }
    }

    pub struct AycdicdbCache {
        pub dp: Vec<Vec<Vec<f64>>>,
        pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
        pub b_to_lines: Vec<Vec<(usize, f64)>>,
        pub batt_nodes: Vec<usize>,
    }

    struct Inner {
        hp: TrackHp,
        cache: Option<AycdicdbCache>,
        lp_pivots_consumed: usize,
    }

    thread_local! {
        static STATE: RefCell<Option<Inner>> = RefCell::new(None);
    }

    pub fn solve_with_hp(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hp: TrackHp,
    ) -> Result<()> {
        STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None, lp_pivots_consumed: 0 }));
        let out = challenge.grid_optimize(&policy_entry);
        STATE.with(|s| *s.borrow_mut() = None);
        let solution = out?;
        save_solution(&solution)?;
        Ok(())
    }

    fn policy_entry(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        STATE.with(|s| -> Result<Vec<f64>> {
            let mut guard = s.borrow_mut();
            let inner = guard.as_mut().expect("aycdicdb: STATE not initialised");
            if inner.cache.is_none() {
                inner.cache = Some(build_cache(challenge, state, &inner.hp));
            }
            let cache = inner.cache.as_ref().unwrap();
            let hp = &inner.hp;

            let zero_action = vec![0.0_f64; challenge.num_batteries];
            let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base_cur);

            let base_actions = if hp.use_kkt {
                kkt_policy(challenge, state, cache, hp, &flows_base)
            } else {
                let mut a = vec![0.0; challenge.num_batteries];
                run_asca(challenge, state, cache, hp, &flows_base, &mut a);
                if hp.dual_iters > 0 {
                    let asca_actions = a.clone();
                    let dual_actions = run_dual_ascent(challenge, state, cache, hp, &flows_base, &asca_actions);
                    let profit_asca: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, asca_actions[b]))
                        .sum();
                    let profit_dual: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, dual_actions[b]))
                        .sum();
                    if profit_dual >= profit_asca { a = dual_actions; }
                } else if hp.max_admm_iters > 0 {
                    if !run_admm_dispatch(challenge, state, cache, hp, &flows_base, &mut a) {
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut a);
                    }
                } else {
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut a);
                }
                a
            };

            let actions = if hp.use_lp {
                let max_util = (0..flows_base.len())
                    .map(|l| {
                        let limit = challenge.network.flow_limits[l];
                        if limit > 1e-6 { flows_base[l].abs() / limit } else { 0.0 }
                    })
                    .fold(0.0_f64, f64::max);

                let per_call_limit = if hp.lp_per_call_pivots > 0 {
                    hp.lp_per_call_pivots
                } else { 3000 };

                let budget_ok = hp.lp_total_pivots == 0
                    || inner.lp_pivots_consumed.saturating_add(per_call_limit) <= hp.lp_total_pivots;

                let multi_lp_result = if hp.lp_horizon > 0 && max_util >= hp.kkt_cong_threshold && budget_ok {
                    multi_lp_dispatch(challenge, state, cache, hp, hp.lp_horizon)
                } else { None };

                if let Some(mut lp_act) = multi_lp_result {
                    inner.lp_pivots_consumed += per_call_limit;
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut lp_act);
                    let lp_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, lp_act[b])).sum();
                    let base_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, base_actions[b])).sum();
                    if lp_p >= base_p { lp_act } else { base_actions }
                } else if max_util >= hp.kkt_cong_threshold && budget_ok {
                    if let Some(mut lp_act) = joint_lp_dispatch(challenge, state, cache, &flows_base, hp.lp_soft_lambda, per_call_limit) {
                        inner.lp_pivots_consumed += per_call_limit;
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut lp_act);
                        let lp_p: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, lp_act[b])).sum();
                        let base_p: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, base_actions[b])).sum();
                        if lp_p >= base_p { lp_act } else { base_actions }
                    } else { base_actions }
                } else {
                    base_actions
                }
            } else { base_actions };

            let baseline_actions = actions;
            let is_last_step = state.time_step + 1 >= challenge.num_steps;
            if is_last_step {
                let num_l = challenge.network.flow_limits.len();
                let baseline_score: f64 = (0..challenge.num_batteries)
                    .map(|b| eval_profit(challenge, state, cache, b, baseline_actions[b]))
                    .sum();
                let fuel_budget: u64 = 5_000_000_000;
                let refined = refine_solution(challenge, state, cache, hp, &flows_base, baseline_actions.clone(), fuel_budget);
                let refined_score: f64 = (0..challenge.num_batteries)
                    .map(|b| eval_profit(challenge, state, cache, b, refined[b]))
                    .sum();
                let mut refined_flows = vec![0.0_f64; num_l];
                for l in 0..num_l {
                    let mut f = flows_base[l];
                    for &(bb, imp) in &cache.ptdf_sparse[l] { f += imp * refined[bb]; }
                    refined_flows[l] = f;
                }
                let feasible = refined_flows.iter().enumerate().all(|(l, f)| {
                    let limit = challenge.network.flow_limits[l];
                    f.abs() <= limit * 1.001
                });
                Ok(if refined_score > baseline_score && feasible {
                    refined
                } else {
                    baseline_actions
                })
            } else {
                Ok(baseline_actions)
            }
        })
    }

    fn build_dp_with_mu(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
    ) -> Vec<Vec<Vec<f64>>> {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
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

                let mu_adjust: f64 = b_to_lines[b].iter()
                    .map(|&(l, impact)| mu[t][l] * impact)
                    .sum();

                let extra = expected_premiums[t][b] - mu_adjust;
                let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
                let p_buy = p_da + extra;

                let max_pwr_c = bat.power_charge_mw * hp.network_derating;
                let max_pwr_d = bat.power_discharge_mw * hp.network_derating;

                for i in 0..soc_levels {
                    let soc = bat.soc_min_mwh + soc_span * (i as f64) / ((soc_levels - 1) as f64);

                    let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                        (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                    } else { 0.0 };
                    let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                        (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                    } else { 0.0 };

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
        dp
    }

    fn ldd_simulate_flows(
        challenge: &Challenge,
        state: &State,
        dp: &[Vec<Vec<f64>>],
        batt_nodes: &[usize],
        ptdf_sparse: &[Vec<(usize, f64)>],
    ) -> Vec<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let sim_pts = 20usize;

        let mut socs: Vec<f64> = state.socs.clone();
        let mut flows_all = vec![vec![0.0_f64; num_l]; num_t];

        for t in 0..num_t {
            let exo_flows = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let mut bat_actions = vec![0.0_f64; num_b];

            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc = socs[b];
                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let soc_levels = dp[b][0].len();
                let node = batt_nodes[b];

                let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                    challenge.market.day_ahead_prices[t][node]
                } else {
                    challenge.market.day_ahead_prices[t][0]
                };

                let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                    (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                } else { 0.0 };
                let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                    (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                } else { 0.0 };

                let u_min = -(bat.power_charge_mw.min(charge_soc_limit.max(0.0)));
                let u_max = bat.power_discharge_mw.min(discharge_soc_limit.max(0.0));
                let u_max = u_max.max(u_min);

                let mut best_u = 0.0_f64;
                let mut best_val = f64::NEG_INFINITY;
                let span = u_max - u_min;

                for j in 0..=sim_pts {
                    let u = if span > 0.0 { u_min + span * (j as f64) / (sim_pts as f64) } else { u_min };
                    let abs_u = u.abs();
                    let revenue = u * p_da * dt;
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
                    let t_next = (t + 1).min(num_t);
                    let v_next = dp[b][t_next][idx0c] * (1.0 - frac) + dp[b][t_next][idx1c] * frac;

                    let val = profit + v_next;
                    if val > best_val { best_val = val; best_u = u; }
                }

                bat_actions[b] = best_u;
                let next_soc_raw = if best_u < 0.0 {
                    soc + bat.efficiency_charge * (-best_u) * dt
                } else {
                    soc - best_u / bat.efficiency_discharge.max(1e-9) * dt
                };
                socs[b] = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            }

            let mut bat_flows = vec![0.0_f64; num_l];
            for l in 0..num_l {
                for &(b, impact) in &ptdf_sparse[l] {
                    bat_flows[l] += impact * bat_actions[b];
                }
            }
            for l in 0..num_l {
                flows_all[t][l] = exo_flows[l] + bat_flows[l];
            }
        }
        flows_all
    }

    fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> AycdicdbCache {
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

        if hp.use_kkt {
            for row in expected_premiums.iter_mut() {
                for v in row.iter_mut() { *v = 0.0; }
            }
        }

        let dp = if hp.ldd_iters > 0 && num_l > 0 {
            let mut mu = vec![vec![0.0_f64; num_l]; num_t];
            let mut dp = build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

            for ldd_iter in 0..hp.ldd_iters {
                let flows_all = ldd_simulate_flows(challenge, state, &dp, &batt_nodes, &ptdf_sparse);

                let alpha = hp.ldd_step_size / ((ldd_iter + 1) as f64);
                let mut max_viol = 0.0_f64;
                for t in 0..num_t {
                    for l in 0..num_l {
                        let limit = challenge.network.flow_limits[l];
                        if limit <= 1e-6 { continue; }
                        let f = flows_all[t][l];
                        if f > limit {
                            let v = f - limit;
                            mu[t][l] += alpha * v;
                            if v > max_viol { max_viol = v; }
                        } else if f < -limit {
                            let v = -f - limit;
                            mu[t][l] -= alpha * v;
                            if v > max_viol { max_viol = v; }
                        }
                    }
                }

                dp = build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

                if max_viol < 1e-4 { break; }
            }
            dp
        } else if hp.max_admm_iters > 0 && hp.anticipate_lmp && num_l > 0 {
            let mut hp_cheap = hp.clone();
            hp_cheap.soc_levels = 31;
            hp_cheap.action_grid = 15;
            let mu_zero = vec![vec![0.0_f64; num_l]; num_t];
            let dp_low = build_dp_with_mu(challenge, &hp_cheap, &batt_nodes, &expected_premiums, &b_to_lines, &mu_zero);
            let flows_all = ldd_simulate_flows(challenge, state, &dp_low, &batt_nodes, &ptdf_sparse);
            for t in 0..num_t {
                for l in 0..num_l {
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6 { continue; }
                    let f = flows_all[t][l];
                    let viol = f.abs() - limit;
                    if viol > 0.0 {
                        let sign = f.signum();
                        let scale = (viol / limit) * 0.2;
                        for &(b, imp) in &ptdf_sparse[l] {
                            expected_premiums[t][b] -= imp * sign * scale;
                        }
                    }
                }
            }
            let mu = vec![vec![0.0_f64; num_l]; num_t];
            build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu)
        } else {
            let mu = vec![vec![0.0_f64; num_l]; num_t];
            build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu)
        };

        AycdicdbCache { dp, ptdf_sparse, b_to_lines, batt_nodes }
    }

    #[inline]
    fn eval_profit(challenge: &Challenge, state: &State, ca: &AycdicdbCache, b: usize, u: f64) -> f64 {
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

    fn run_asca(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut [f64],
    ) {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let mut flows: Vec<f64> = if actions.iter().any(|u| u.abs() > 1e-12) {
            let mut f = flows_base.to_vec();
            for l in 0..num_l {
                for &(b, p) in &ca.ptdf_sparse[l] {
                    f[l] += p * actions[b];
                }
            }
            f
        } else {
            flows_base.to_vec()
        };

        let mut active = vec![true; num_b];
        if hp.prune_ratio > 0.0 && num_b >= 2 {
            let cutoff = ((num_b as f64) * hp.prune_ratio) as usize;
            if cutoff > 0 {
                let mut caps: Vec<(usize, f64)> = challenge.batteries.iter().enumerate().map(|(i, b)| (i, b.capacity_mwh)).collect();
                caps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..cutoff.min(num_b) { active[caps[i].0] = false; }
            }
        }

        let active_list: Vec<usize> = (0..num_b).filter(|&b| active[b]).collect();
        if active_list.is_empty() { return; }

        let score_key = |score: f64| -> i64 {
            if score.is_finite() && score > 0.0 {
                (score.min(1.0e12) * 1.0e6) as i64
            } else {
                0
            }
        };

        let rescore_battery = |b: usize,
                               flows: &[f64],
                               line_pressure: &[f64],
                               actions: &[f64],
                               cur_vals: &mut [f64],
                               zero_vals: &mut [f64]| -> f64 {
            let cur_u = actions[b];
            let v_cur = eval_profit(challenge, state, ca, b, cur_u);
            cur_vals[b] = v_cur;
            zero_vals[b] = f64::NAN;

            let (mut score_lo, mut score_hi) = state.action_bounds[b];
            for &(l, p) in &ca.b_to_lines[b] {
                if p.abs() < 1e-9 { continue; }
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let f_other = flows[l] - p * cur_u;
                let b1 = (-limit - f_other) / p;
                let b2 = (limit - f_other) / p;
                let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
                if lo > score_lo { score_lo = lo; }
                if hi < score_hi { score_hi = hi; }
            }
            if score_lo > score_hi {
                score_lo = cur_u;
                score_hi = cur_u;
            }
            score_lo = score_lo.min(cur_u);
            score_hi = score_hi.max(cur_u);

            let v_lo = if (score_lo - cur_u).abs() <= 1e-12 {
                v_cur
            } else {
                eval_profit(challenge, state, ca, b, score_lo)
            };
            let v_hi = if (score_hi - cur_u).abs() <= 1e-12 {
                v_cur
            } else {
                eval_profit(challenge, state, ca, b, score_hi)
            };

            let mut target_u = cur_u;
            let mut base = 0.0_f64;

            let gain_lo = v_lo - v_cur;
            if gain_lo > base {
                base = gain_lo;
                target_u = score_lo;
            }

            let gain_hi = v_hi - v_cur;
            if gain_hi > base {
                base = gain_hi;
                target_u = score_hi;
            }

            if score_lo <= 0.0 && 0.0 <= score_hi {
                let v0 = if cur_u.abs() <= 1e-12 { v_cur } else { eval_profit(challenge, state, ca, b, 0.0) };
                zero_vals[b] = v0;
                let gain0 = v0 - v_cur;
                if gain0 > base {
                    base = gain0;
                    target_u = 0.0;
                }
            }

            let mut relief = 0.0_f64;
            let move_u = target_u - cur_u;
            if base > 1e-12 && move_u.abs() > 1e-9 {
                for &(l, p) in &ca.b_to_lines[b] {
                    let pressure = line_pressure[l];
                    if pressure <= 1e-9 { continue; }
                    let desired = -flows[l].signum();
                    let effect = move_u * p * desired;
                    if effect > 0.0 {
                        relief += pressure * effect.abs();
                    } else if effect < 0.0 {
                        relief -= 0.20 * pressure * effect.abs();
                    }
                }
            }

            if base > 0.0 {
                base * (1.0 + 0.40 * relief).max(0.1)
            } else {
                0.0
            }
        };

        let optimize_battery = |b: usize,
                                flows: &mut [f64],
                                actions: &mut [f64],
                                cur_vals: &[f64],
                                zero_vals: &[f64]| -> f64 {
            let cur_u = actions[b];
            let (mut u_min, mut u_max) = state.action_bounds[b];

            for &(l, p) in &ca.b_to_lines[b] {
                if p.abs() < 1e-9 { continue; }
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let f_other = flows[l] - p * cur_u;
                let b1 = (-limit - f_other) / p;
                let b2 = (limit - f_other) / p;
                let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
                if lo > u_min { u_min = lo; }
                if hi < u_max { u_max = hi; }
            }

            if u_min > u_max {
                u_min = cur_u;
                u_max = cur_u;
            }
            u_min = u_min.min(cur_u);
            u_max = u_max.max(cur_u);

            let mut best_u = cur_u;
            let mut best_v = cur_vals[b];

            if u_min <= 0.0 && 0.0 <= u_max {
                let v0 = if zero_vals[b].is_finite() {
                    zero_vals[b]
                } else if cur_u.abs() <= 1e-12 {
                    cur_vals[b]
                } else {
                    eval_profit(challenge, state, ca, b, 0.0)
                };
                if v0 > best_v {
                    best_v = v0;
                    best_u = 0.0;
                }
            }

            let base_v = best_v;

            if u_min < 0.0 {
                let lo = u_min;
                let hi = 0.0_f64.min(u_max);
                if lo < hi {
                    let edge_u = lo;
                    let edge_v = if (edge_u - cur_u).abs() <= 1e-12 {
                        cur_vals[b]
                    } else {
                        eval_profit(challenge, state, ca, b, edge_u)
                    };
                    if edge_v > best_v {
                        best_v = edge_v;
                        best_u = edge_u;
                    }

                    let probe_u = 0.5 * (lo + hi);
                    let probe_v = if (probe_u - cur_u).abs() <= 1e-12 {
                        cur_vals[b]
                    } else if (probe_u - edge_u).abs() <= 1e-12 {
                        edge_v
                    } else {
                        eval_profit(challenge, state, ca, b, probe_u)
                    };
                    if probe_v > best_v {
                        best_v = probe_v;
                        best_u = probe_u;
                    }

                    let neg_signal = edge_v.max(probe_v) - base_v;
                    let search_neg = u_max <= 0.0 || cur_u < -1e-9 || neg_signal > 1e-8;
                    if search_neg {
                        let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
                        if v > best_v {
                            best_v = v;
                            best_u = u;
                        }
                    }
                }
            }

            if u_max > 0.0 {
                let lo = 0.0_f64.max(u_min);
                let hi = u_max;
                if lo < hi {
                    let edge_u = hi;
                    let edge_v = if (edge_u - cur_u).abs() <= 1e-12 {
                        cur_vals[b]
                    } else {
                        eval_profit(challenge, state, ca, b, edge_u)
                    };
                    if edge_v > best_v {
                        best_v = edge_v;
                        best_u = edge_u;
                    }

                    let probe_u = 0.5 * (lo + hi);
                    let probe_v = if (probe_u - cur_u).abs() <= 1e-12 {
                        cur_vals[b]
                    } else if (probe_u - edge_u).abs() <= 1e-12 {
                        edge_v
                    } else {
                        eval_profit(challenge, state, ca, b, probe_u)
                    };
                    if probe_v > best_v {
                        best_v = probe_v;
                        best_u = probe_u;
                    }

                    let pos_signal = edge_v.max(probe_v) - base_v;
                    let search_pos = u_min >= 0.0 || cur_u > 1e-9 || pos_signal > 1e-8;
                    if search_pos {
                        let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
                        if v > best_v {
                            best_u = u;
                        }
                    }
                }
            }

            let delta = best_u - cur_u;
            if delta.abs() > 1e-6 {
                actions[b] = best_u;
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows[l] += p * delta; }
                }
                delta
            } else {
                0.0
            }
        };

        let mut cur_vals = vec![0.0_f64; num_b];
        let mut zero_vals = vec![f64::NAN; num_b];
        let mut score_cache = vec![0.0_f64; num_b];
        let mut version = vec![0_u32; num_b];
        let mut line_pressure = vec![0.0_f64; num_l];
        let mut heap: BinaryHeap<(i64, usize, u32)> = BinaryHeap::new();
        let mut marked = vec![0_u32; num_b];
        let rebuild_every = (active_list.len() / 3).clamp(6, 24);

        for _sweep in 0..hp.asca_iters {
            heap.clear();
            for l in 0..num_l {
                line_pressure[l] = asca_line_pressure(flows[l], challenge.network.flow_limits[l]);
            }
            for &b in &active_list {
                let score = rescore_battery(b, &flows, &line_pressure, actions, &mut cur_vals, &mut zero_vals);
                score_cache[b] = score;
                version[b] = version[b].wrapping_add(1);
                heap.push((score_key(score), b, version[b]));
            }

            let mut max_change = 0.0_f64;
            let mut processed = 0usize;
            let mut changes_since_rebuild = 0usize;
            let pop_budget = active_list.len();

            while processed < pop_budget {
                let Some((_, b, ver)) = heap.pop() else { break; };
                if ver != version[b] { continue; }

                let score_now = rescore_battery(b, &flows, &line_pressure, actions, &mut cur_vals, &mut zero_vals);
                if (score_now - score_cache[b]).abs() > 1e-9 * (1.0 + score_now.abs().max(score_cache[b].abs())) {
                    score_cache[b] = score_now;
                    version[b] = version[b].wrapping_add(1);
                    heap.push((score_key(score_now), b, version[b]));
                    continue;
                }
                score_cache[b] = score_now;
                processed += 1;
                if score_now <= 1e-12 { continue; }

                let delta = optimize_battery(b, &mut flows, actions, &cur_vals, &zero_vals);
                if delta.abs() <= 1e-6 { continue; }

                if delta.abs() > max_change { max_change = delta.abs(); }
                changes_since_rebuild += 1;

                let mut token = version[b].wrapping_add(processed as u32).wrapping_add(1);
                if token == 0 {
                    marked.fill(0);
                    token = 1;
                }
                let mut impacted = vec![b];
                marked[b] = token;

                for &(l, p) in &ca.b_to_lines[b] {
                    if l >= num_l { continue; }
                    line_pressure[l] = asca_line_pressure(flows[l], challenge.network.flow_limits[l]);
                    if (p * delta).abs() <= 1e-10 { continue; }
                    for &(bb, _) in &ca.ptdf_sparse[l] {
                        if active[bb] && marked[bb] != token {
                            marked[bb] = token;
                            impacted.push(bb);
                        }
                    }
                }

                for bb in impacted {
                    let score = rescore_battery(bb, &flows, &line_pressure, actions, &mut cur_vals, &mut zero_vals);
                    score_cache[bb] = score;
                    version[bb] = version[bb].wrapping_add(1);
                    heap.push((score_key(score), bb, version[bb]));
                }

                if changes_since_rebuild >= rebuild_every {
                    heap.clear();
                    for l in 0..num_l {
                        line_pressure[l] = asca_line_pressure(flows[l], challenge.network.flow_limits[l]);
                    }
                    for &bb in &active_list {
                        let score = rescore_battery(bb, &flows, &line_pressure, actions, &mut cur_vals, &mut zero_vals);
                        score_cache[bb] = score;
                        version[bb] = version[bb].wrapping_add(1);
                        heap.push((score_key(score), bb, version[bb]));
                    }
                    changes_since_rebuild = 0;
                }
            }

            if max_change < hp.convergence_tol { break; }
        }
    }

    #[inline]
    fn asca_line_pressure(flow: f64, limit: f64) -> f64 {
        if limit <= 1e-6 { return 0.0; }
        let util = flow.abs() / limit;
        if util > 1.0 {
            2.5 + 6.0 * (util - 1.0)
        } else if util > 0.92 {
            1.0 + 4.0 * (util - 0.92) / 0.08
        } else {
            util * util * util
        }
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

    fn run_dual_ascent(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        warm_start: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let mut actions = warm_start.to_vec();

        let mut flows = vec![0.0_f64; num_l];
        for l in 0..num_l {
            let mut sum = 0.0;
            for &(b, imp) in &ca.ptdf_sparse[l] { sum += imp * actions[b]; }
            flows[l] = flows_base[l] + sum;
        }

        let mut nu = vec![0.0_f64; num_l];

        for k in 1..=hp.dual_iters {
            let alpha = hp.da_step_size / (k as f64);

            for b in 0..num_b {
                let nu_dot_ptdf: f64 = ca.b_to_lines[b].iter()
                    .map(|&(l, impact)| nu[l] * impact)
                    .sum();

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

                let mut best_u = actions[b];
                let mut best_v = eval_profit(challenge, state, ca, b, best_u) - nu_dot_ptdf * best_u;

                if u_min <= 0.0 && 0.0 <= u_max {
                    let v0 = eval_profit(challenge, state, ca, b, 0.0);
                    if v0 > best_v { best_v = v0; best_u = 0.0; }
                }
                if u_min < 0.0 {
                    let lo = u_min; let hi = 0.0_f64.min(u_max);
                    if lo < hi {
                        let (u, v) = ternary_search(
                            |u| eval_profit(challenge, state, ca, b, u) - nu_dot_ptdf * u,
                            lo, hi, hp.ternary_iters,
                        );
                        if v > best_v { best_v = v; best_u = u; }
                    }
                }
                if u_max > 0.0 {
                    let lo = 0.0_f64.max(u_min); let hi = u_max;
                    if lo < hi {
                        let (u, v) = ternary_search(
                            |u| eval_profit(challenge, state, ca, b, u) - nu_dot_ptdf * u,
                            lo, hi, hp.ternary_iters,
                        );
                        if v > best_v { best_u = u; }
                    }
                }

                let delta = best_u - actions[b];
                if delta.abs() > 1e-10 {
                    actions[b] = best_u;
                    for &(l, p) in &ca.b_to_lines[b] {
                        if l < num_l { flows[l] += p * delta; }
                    }
                }
            }

            let mut max_viol = 0.0_f64;
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let v = flows[l].abs() - limit;
                if v > max_viol { max_viol = v; }
            }
            if max_viol < 1e-6 { break; }

            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if flows[l] > limit {
                    nu[l] += alpha * (flows[l] - limit);
                } else if flows[l] < -limit {
                    nu[l] -= alpha * (-flows[l] - limit);
                }
            }
        }

        actions
    }

    fn run_deflator(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut [f64],
    ) {

        let num_l = challenge.network.flow_limits.len();
        let num_b = challenge.num_batteries;

        let mut limits = vec![0.0_f64; num_l];
        for l in 0..num_l {
            limits[l] = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        }

        let mut flows = vec![0.0_f64; num_l];
        for l in 0..num_l {
            let mut f = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { f += imp * actions[b]; }
            flows[l] = flows_base[l] + f;
        }

        for _ in 0..hp.deflator_iters {
            let mut total_over = 0.0_f64;
            for l in 0..num_l {
                total_over += (flows[l].abs() - limits[l]).max(0.0);
            }
            if total_over <= 1e-9 { break; }

            let mut best_move: Option<(usize, f64, f64)> = None;

            for b in 0..num_b {
                let u = actions[b];
                if u.abs() <= 1e-10 { continue; }
                let u_sign = u.signum();
                let current_v = eval_profit(challenge, state, ca, b, u);

                let mut numer = 0.0_f64;
                let mut denom = 0.0_f64;
                for &(l, impact) in &ca.b_to_lines[b] {
                    let overflow = (flows[l].abs() - limits[l]).max(0.0);
                    if overflow <= 1e-9 { continue; }
                    let rel_slope = flows[l].signum() * impact * u_sign;
                    if rel_slope > 1e-12 {
                        let w = 1.0 + 4.0 * overflow / limits[l].max(1.0);
                        numer += w * overflow * rel_slope;
                        denom += w * rel_slope * rel_slope;
                    }
                }
                if denom <= 1e-12 { continue; }

                let step_ls = (numer / denom).max(0.0).min(u.abs());
                if step_ls <= 1e-10 { continue; }
                let partial_u = u - u_sign * step_ls;

                let mut local_best_u = u;
                let mut local_best_score = f64::INFINITY;

                for cand_u in [partial_u, 0.0_f64] {
                    let delta = cand_u - u;
                    if delta.abs() <= 1e-10 { continue; }

                    let cand_v = eval_profit(challenge, state, ca, b, cand_u);
                    let loss = current_v - cand_v;

                    let mut good = 0.0_f64;
                    let mut bad = 0.0_f64;
                    for &(l, impact) in &ca.b_to_lines[b] {
                        let overflow_before = (flows[l].abs() - limits[l]).max(0.0);
                        let overflow_after = ((flows[l] + impact * delta).abs() - limits[l]).max(0.0);
                        let w = 1.0 + 4.0 * overflow_before.max(overflow_after) / limits[l].max(1.0);
                        if overflow_after + 1e-12 < overflow_before {
                            good += w * (overflow_before - overflow_after);
                        } else if overflow_after > overflow_before + 1e-12 {
                            bad += w * (overflow_after - overflow_before);
                        }
                    }
                    let relief = good - 1.5 * bad;
                    if relief <= 1e-12 { continue; }

                    let score = loss / relief;
                    if score < local_best_score {
                        local_best_score = score;
                        local_best_u = cand_u;
                    }
                }

                if local_best_score.is_finite() {
                    match best_move {
                        Some((_, _, best_score)) if best_score <= local_best_score => {}
                        _ => best_move = Some((b, local_best_u, local_best_score)),
                    }
                }
            }

            let Some((b, new_u, _)) = best_move else { break; };
            let delta = new_u - actions[b];
            if delta.abs() <= 1e-10 { break; }
            actions[b] = new_u;
            for &(l, p) in &ca.b_to_lines[b] {
                if l < num_l { flows[l] += p * delta; }
            }
        }

        let max_emergency = hp.deflator_iters.max(1) + num_l.min(num_b.max(1));
        for _ in 0..max_emergency {
            let mut worst_l: Option<usize> = None;
            let mut worst_over = 0.0_f64;
            for l in 0..num_l {
                let overflow = (flows[l].abs() - limits[l]).max(0.0);
                if overflow > worst_over {
                    worst_over = overflow;
                    worst_l = Some(l);
                }
            }
            let Some(l_star) = worst_l else { break; };
            if worst_over <= 1e-9 { break; }

            let line_sign = flows[l_star].signum();
            let mut best_move: Option<(usize, f64, f64)> = None;

            for &(b, impact_star) in &ca.ptdf_sparse[l_star] {
                let u = actions[b];
                if u.abs() <= 1e-10 { continue; }

                let rel_slope_star = line_sign * impact_star * u.signum();
                if rel_slope_star <= 1e-12 { continue; }

                let step = (worst_over / rel_slope_star).max(0.0).min(u.abs());
                if step <= 1e-10 { continue; }

                let cand_u = u - u.signum() * step;
                let delta = cand_u - u;
                let current_v = eval_profit(challenge, state, ca, b, u);
                let cand_v = eval_profit(challenge, state, ca, b, cand_u);
                let loss = current_v - cand_v;

                let mut good = 0.0_f64;
                let mut bad = 0.0_f64;
                for &(ll, pp) in &ca.b_to_lines[b] {
                    let overflow_before = (flows[ll].abs() - limits[ll]).max(0.0);
                    let overflow_after = ((flows[ll] + pp * delta).abs() - limits[ll]).max(0.0);
                    let w = if ll == l_star {
                        1.5 + 6.0 * overflow_before.max(overflow_after) / limits[ll].max(1.0)
                    } else {
                        0.75 + 2.0 * overflow_before.max(overflow_after) / limits[ll].max(1.0)
                    };
                    if overflow_after + 1e-12 < overflow_before {
                        good += w * (overflow_before - overflow_after);
                    } else if overflow_after > overflow_before + 1e-12 {
                        bad += w * (overflow_after - overflow_before);
                    }
                }
                let relief = good - 2.0 * bad;
                if relief <= 1e-12 { continue; }

                let score = loss / relief;
                match best_move {
                    Some((_, _, best_score)) if best_score <= score => {}
                    _ => best_move = Some((b, cand_u, score)),
                }
            }

            let Some((b, new_u, _)) = best_move else { break; };
            let delta = new_u - actions[b];
            if delta.abs() <= 1e-10 { break; }
            actions[b] = new_u;
            for &(l, p) in &ca.b_to_lines[b] {
                if l < num_l { flows[l] += p * delta; }
            }
        }

        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            actions[b] = actions[b].clamp(lo, hi);
        }
    }

    #[inline]
    fn eval_profit_with_price(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        b: usize,
        u: f64,
        price: f64,
    ) -> f64 {
        let bat = &challenge.batteries[b];
        let dt = 0.25_f64;
        let abs_u = u.abs();
        let revenue = u * price * dt;
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

    fn kkt_best_action(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        b: usize,
        price_override: Option<f64>,
        ternary_iters: usize,
    ) -> f64 {
        let (u_min, u_max) = state.action_bounds[b];
        if u_min >= u_max { return u_min; }
        let eval = |u: f64| -> f64 {
            match price_override {
                Some(p) => eval_profit_with_price(challenge, state, ca, b, u, p),
                None => eval_profit(challenge, state, ca, b, u),
            }
        };
        let mut best_u = 0.0_f64.clamp(u_min, u_max);
        let mut best_v = eval(best_u);
        if u_min < 0.0 {
            let lo = u_min; let hi = 0.0_f64.min(u_max);
            if lo < hi {
                let (u, v) = ternary_search(|x| eval(x), lo, hi, ternary_iters);
                if v > best_v { best_v = v; best_u = u; }
            }
        }
        if u_max > 0.0 {
            let lo = 0.0_f64.max(u_min); let hi = u_max;
            if lo < hi {
                let (u, v) = ternary_search(|x| eval(x), lo, hi, ternary_iters);
                if v > best_v { best_u = u; }
            }
        }
        best_u
    }

    fn kkt_pass(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        mu_override: &[f64],
        price_scale: f64,
    ) -> (Vec<f64>, f64) {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let mu: Vec<f64> = if mu_override.is_empty() {
            let mut m = vec![0.0_f64; num_l];
            for l in 0..num_l {
                let lim = challenge.network.flow_limits[l];
                if lim < 1e-6 { continue; }
                let f_exo = flows_base[l];
                let util = f_exo.abs() / lim;
                if util > hp.kkt_cong_threshold {
                    let excess_frac = ((util - hp.kkt_cong_threshold)
                        / (1.0 - hp.kkt_cong_threshold).max(1e-6)).min(1.0);
                    m[l] = excess_frac * price_scale * f_exo.signum();
                }
            }
            m
        } else {
            mu_override.to_vec()
        };

        let mut p_eff = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let cong_adj: f64 = ca.b_to_lines[b].iter().map(|&(l, imp)| mu[l] * imp).sum();
            p_eff[b] = rt - cong_adj;
        }

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            actions[b] = kkt_best_action(challenge, state, ca, b, Some(p_eff[b]), hp.ternary_iters);
        }

        run_deflator(challenge, state, ca, hp, flows_base, &mut actions);

        let profit: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions[b])).sum();
        (actions, profit)
    }

    pub fn kkt_policy(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let avg_price: f64 = if num_b > 0 {
            (0..num_b)
                .map(|b| {
                    let n = ca.batt_nodes[b];
                    if n < state.rt_prices.len() { state.rt_prices[n].abs() } else { 0.0 }
                })
                .sum::<f64>() / num_b as f64
        } else { 0.0 };
        let price_scale = avg_price.max(10.0) * hp.kkt_price_scale;

        let (actions1, profit1) = kkt_pass(challenge, state, ca, hp, flows_base, &[], price_scale);

        let mut total_flow1 = flows_base.to_vec();
        for l in 0..num_l {
            for &(b, imp) in &ca.ptdf_sparse[l] {
                total_flow1[l] += imp * actions1[b];
            }
        }

        let cong_threshold2 = 0.85_f64;
        let mut mu2 = vec![0.0_f64; num_l];
        for l in 0..num_l {
            let lim = challenge.network.flow_limits[l];
            if lim < 1e-6 { continue; }
            let util = total_flow1[l].abs() / lim;
            if util > cong_threshold2 {
                let excess_frac = ((util - cong_threshold2)
                    / (1.0 - cong_threshold2).max(1e-6)).min(1.0);
                mu2[l] = excess_frac * price_scale * total_flow1[l].signum();
            }
        }

        let (actions2, profit2) = kkt_pass(challenge, state, ca, hp, flows_base, &mu2, price_scale);

        let best_kkt = if profit2 >= profit1 * 0.98 && profit2 >= profit1 {
            actions2
        } else {
            actions1
        };
        let profit_kkt = profit1.max(profit2);

        let mut actions_asca = vec![0.0_f64; num_b];
        run_asca(challenge, state, ca, hp, flows_base, &mut actions_asca);
        if hp.dual_iters > 0 {
            let dual = run_dual_ascent(challenge, state, ca, hp, flows_base, &actions_asca);
            let pa: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions_asca[b])).sum();
            let pd: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, dual[b])).sum();
            if pd > pa { actions_asca = dual; }
        } else {
            run_deflator(challenge, state, ca, hp, flows_base, &mut actions_asca);
        }

        let profit_asca: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions_asca[b])).sum();

        if profit_kkt >= profit_asca {
            let mut out = best_kkt;
            for b in 0..num_b {
                let (lo, hi) = state.action_bounds[b];
                out[b] = out[b].clamp(lo, hi);
            }
            out
        } else {
            actions_asca
        }
    }

    fn run_admm_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut [f64],
    ) -> bool {
        let num_l = challenge.network.flow_limits.len();
        let num_b = challenge.num_batteries;
        let rho = hp.admm_rho;
        let tol = hp.admm_primal_tol;

        let mut any_violated = false;
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            let mut bat_f = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { bat_f += imp * actions[b]; }
            if (flows_base[l] + bat_f).abs() > limit { any_violated = true; break; }
        }
        if !any_violated { return true; }

        let mut y = vec![0.0_f64; num_l];
        let mut s: Vec<f64> = (0..num_l).map(|l| {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            let mut bat_f = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { bat_f += imp * actions[b]; }
            (flows_base[l] + bat_f).clamp(-limit, limit)
        }).collect();

        for _iter in 0..hp.max_admm_iters {
            let prev_actions = actions.to_vec();

            let mut bat_flow = vec![0.0_f64; num_l];
            for l in 0..num_l {
                for &(b, imp) in &ca.ptdf_sparse[l] { bat_flow[l] += imp * actions[b]; }
            }

            for b in 0..num_b {
                let (lo, hi) = state.action_bounds[b];
                if (hi - lo).abs() < 1e-12 { continue; }

                let lines_b = &ca.b_to_lines[b];
                let offsets: Vec<(f64, f64)> = lines_b.iter().map(|&(l, a_lb)| {
                    let off = s[l] - flows_base[l] + y[l] / rho - (bat_flow[l] - a_lb * actions[b]);
                    (off, a_lb)
                }).collect();

                const GRID: usize = 200;
                let step = (hi - lo) / GRID as f64;
                let mut best_u = actions[b];
                let mut best_val = f64::NEG_INFINITY;
                for k in 0..=GRID {
                    let u = (lo + k as f64 * step).clamp(lo, hi);
                    let profit = eval_profit(challenge, state, ca, b, u);
                    let penalty: f64 = offsets.iter().map(|&(off, a_lb)| {
                        let err = off - a_lb * u;
                        (rho / 2.0) * err * err
                    }).sum();
                    let val = profit - penalty;
                    if val > best_val { best_val = val; best_u = u; }
                }

                let delta = best_u - actions[b];
                for &(l, a_lb) in lines_b { bat_flow[l] += a_lb * delta; }
                actions[b] = best_u;
            }

            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                s[l] = (bat_flow[l] + flows_base[l] - y[l] / rho).clamp(-limit, limit);
            }

            let mut max_resid = 0.0_f64;
            for l in 0..num_l {
                let resid = s[l] - bat_flow[l] - flows_base[l];
                y[l] += rho * resid;
                max_resid = max_resid.max(resid.abs());
            }
            let max_du = (0..num_b).map(|b| (actions[b] - prev_actions[b]).abs()).fold(0.0_f64, f64::max);
            if max_resid < tol && max_du < tol { return true; }
        }

        let mut bat_flow_final = vec![0.0_f64; num_l];
        for l in 0..num_l {
            for &(b, imp) in &ca.ptdf_sparse[l] { bat_flow_final[l] += imp * actions[b]; }
        }
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if (bat_flow_final[l] + flows_base[l]).abs() > limit + 1.0 { return false; }
        }
        true
    }

    fn joint_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        flows_base: &[f64],
        lp_lambda: f64,
        max_pivots: usize,
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;

        let n = 2 * num_b + num_l;
        let m = 4 * num_b + 2 * num_l;

        let mut c_vec = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        let t_next = (state.time_step + 1).min(ca.dp[0].len() - 1);

        for l in 0..num_l {
            c_vec[2 * num_b + l] = -lp_lambda;
        }

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let (u_min, u_max) = state.action_bounds[b];
            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let soc = state.socs[b];

            let soc_levels = ca.dp[b][0].len();
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1) as f64;
            let idx_f = (soc - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo_idx = (idx_f.floor() as isize).max(0) as usize;
            let lo_idx = lo_idx.min(soc_levels - 1);
            let hi_idx = (lo_idx + 1).min(soc_levels - 1);
            let dv = (ca.dp[b][t_next][hi_idx] - ca.dp[b][t_next][lo_idx]) / delta_s;

            c_vec[b]           = (rt - 0.25) * dt - dv * dt / eta_d;  
            c_vec[num_b + b]   = -(rt + 0.25) * dt + dv * eta_c * dt; 

            let r = 4 * b;
            a_mat[r][b] = 1.0;
            b_vec[r] = u_max.max(0.0);

            a_mat[r + 1][num_b + b] = 1.0;
            b_vec[r + 1] = (-u_min).max(0.0);

            a_mat[r + 2][b]         =  dt / eta_d;
            a_mat[r + 2][num_b + b] = -eta_c * dt;
            b_vec[r + 2] = (soc - bat.soc_min_mwh).max(0.0);

            a_mat[r + 3][b]         = -dt / eta_d;
            a_mat[r + 3][num_b + b] =  eta_c * dt;
            b_vec[r + 3] = (bat.soc_max_mwh - soc).max(0.0);
        }

        let row_f = 4 * num_b;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            let exo = flows_base[l];
            let viol_idx = 2 * num_b + l;
            let rp = row_f + 2 * l;     
            let rn = rp + 1;             

            for &(b, impact) in &ca.ptdf_sparse[l] {
                a_mat[rp][b]         += impact;
                a_mat[rp][num_b + b] -= impact;
                a_mat[rn][b]         -= impact;
                a_mat[rn][num_b + b] += impact;
            }
            a_mat[rp][viol_idx] = -1.0;
            a_mat[rn][viol_idx] = -1.0;

            b_vec[rp] = (limit - exo).max(0.0);
            b_vec[rn] = (limit + exo).max(0.0);
        }

        let (opt_x, _) = super::lp::lp_solve_with_budget(n, m, &c_vec, &a_mat, &b_vec, max_pivots);
        let opt_x = opt_x?;

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let u = opt_x[b] - opt_x[num_b + b];
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }
        Some(actions)
    }

    fn multi_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        horizon: usize,
    ) -> Option<Vec<f64>> {
        let t0 = state.time_step;
        let nb = challenge.num_batteries;
        let nl = challenge.network.num_lines;
        let h = horizon.min(challenge.num_steps - t0);
        if h == 0 { return Some(vec![0.0; nb]); }

        let da = &challenge.market.day_ahead_prices;
        let mut p_sell = vec![vec![0.0f64; nb]; h];
        let mut p_buy = vec![vec![0.0f64; nb]; h];
        for b in 0..nb {
            let node = ca.batt_nodes[b];
            for k in 0..h {
                let da_price = if t0 + k < da.len() && node < da[t0 + k].len() {
                    da[t0 + k][node]
                } else {
                    da[t0].get(0).copied().unwrap_or(0.0)
                };
                p_sell[k][b] = da_price * (1.0 + hp.jump_premium);
                p_buy[k][b] = da_price;
            }
        }

        let n = 2 * nb * h + h * nl + nb;
        let terminal_knots = 11;

        let mut active_lines = Vec::new();
        for l in 0..nl {
            if challenge.network.flow_limits[l] > 1e-6 { active_lines.push(l); }
        }

        let mut m_count = 0;
        m_count += 2 * nb * h; 
        m_count += 2 * nb * h; 
        m_count += 2 * active_lines.len() * h; 
        m_count += (terminal_knots - 1) * nb; 
        let m = m_count;

        let mut c = vec![0.0f64; n];
        let mut a = vec![vec![0.0f64; n]; m];
        let mut b_rhs = vec![0.0f64; m];

        let dt = 0.25;
        let kappa = 0.25;

        for b in 0..nb {
            for k in 0..h {
                let idx_d = k * 2 * nb + b;
                let idx_c = k * 2 * nb + nb + b;
                c[idx_d] = (p_sell[k][b] - kappa) * dt;
                c[idx_c] = -(p_buy[k][b] + kappa) * dt;
            }
        }
        let slack_penalty = hp.lp_soft_lambda;
        for k in 0..h {
            for &l in &active_lines {
                c[2 * nb * h + k * nl + l] = -slack_penalty;
            }
        }
        for b in 0..nb {
            c[2 * nb * h + h * nl + b] = 1.0;
        }

        let max_d: Vec<f64> = challenge.batteries.iter()
            .map(|batt| batt.power_discharge_mw * hp.network_derating).collect();
        let max_c: Vec<f64> = challenge.batteries.iter()
            .map(|batt| batt.power_charge_mw * hp.network_derating).collect();
        let soc_min: Vec<f64> = challenge.batteries.iter().map(|batt| batt.soc_min_mwh).collect();
        let soc_max: Vec<f64> = challenge.batteries.iter().map(|batt| batt.soc_max_mwh).collect();
        let soc_init = state.socs.clone();
        let eta_c: Vec<f64> = challenge.batteries.iter().map(|batt| batt.efficiency_charge).collect();
        let eta_d: Vec<f64> = challenge.batteries.iter().map(|batt| batt.efficiency_discharge).collect();

        let mut row = 0usize;

        for b in 0..nb {
            for k in 0..h {
                a[row][k * 2 * nb + b] = 1.0;
                b_rhs[row] = max_d[b];
                row += 1;
            }
        }
        for b in 0..nb {
            for k in 0..h {
                a[row][k * 2 * nb + nb + b] = 1.0;
                b_rhs[row] = max_c[b];
                row += 1;
            }
        }

        for b in 0..nb {
            for k in 1..=h {
                for tau in 0..k {
                    a[row][tau * 2 * nb + b] = -dt / eta_d[b].max(1e-9);
                    a[row][tau * 2 * nb + nb + b] = eta_c[b] * dt;
                }
                b_rhs[row] = soc_max[b] - soc_init[b];
                row += 1;
                for tau in 0..k {
                    a[row][tau * 2 * nb + b] = dt / eta_d[b].max(1e-9);
                    a[row][tau * 2 * nb + nb + b] = -eta_c[b] * dt;
                }
                b_rhs[row] = soc_init[b] - soc_min[b];
                row += 1;
            }
        }

        let mut exo_flows: Vec<Vec<f64>> = Vec::with_capacity(h);
        for k in 0..h {
            exo_flows.push(challenge.network.compute_flows(&challenge.exogenous_injections[t0 + k]));
        }
        for k in 0..h {
            let ef = &exo_flows[k];
            for &l in &active_lines {
                let limit = challenge.network.flow_limits[l];
                let idx_s = 2 * nb * h + k * nl + l;
                for &(b, ptdf_val) in &ca.ptdf_sparse[l] {
                    a[row][k * 2 * nb + b] += ptdf_val;
                    a[row][k * 2 * nb + nb + b] -= ptdf_val;
                }
                a[row][idx_s] = -1.0;
                b_rhs[row] = (limit - ef[l]).max(0.0);
                row += 1;
                for &(b, ptdf_val) in &ca.ptdf_sparse[l] {
                    a[row][k * 2 * nb + b] -= ptdf_val;
                    a[row][k * 2 * nb + nb + b] += ptdf_val;
                }
                a[row][idx_s] = -1.0;
                b_rhs[row] = (limit + ef[l]).max(0.0);
                row += 1;
            }
        }

        let t_term = t0 + h;
        for b in 0..nb {
            let bat = &challenge.batteries[b];
            let soc_sp = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let step_f = (soc_levels - 1) as f64 / (terminal_knots - 1) as f64;
            let dp_term_idx = t_term.min(ca.dp[b].len() - 1);
            let dp_term = &ca.dp[b][dp_term_idx];
            for i in 0..(terminal_knots - 1) {
                let idx_lo = (i as f64 * step_f) as usize;
                let idx_hi = (idx_lo + 1).min(soc_levels - 1);
                let v_lo = dp_term[idx_lo];
                let v_hi = dp_term[idx_hi];
                let soc_lo = bat.soc_min_mwh + soc_sp * idx_lo as f64 / (soc_levels - 1) as f64;
                let soc_hi = bat.soc_min_mwh + soc_sp * idx_hi as f64 / (soc_levels - 1) as f64;
                let delta_soc = soc_hi - soc_lo;
                if delta_soc < 1e-9 { continue; }
                let slope = (v_hi - v_lo) / delta_soc;
                let intercept = v_lo - slope * soc_lo;
                let idx_v = 2 * nb * h + h * nl + b;
                a[row][idx_v] = 1.0;
                for k in 0..h {
                    a[row][k * 2 * nb + b] += slope * dt / eta_d[b].max(1e-9);
                    a[row][k * 2 * nb + nb + b] -= slope * eta_c[b] * dt;
                }
                b_rhs[row] = intercept + slope * soc_init[b];
                row += 1;
            }
        }

        let max_pivots = hp.lp_per_call_pivots.max(8000);
        let (opt_x, _) = super::lp::lp_solve_with_budget(n, m, &c, &a, &b_rhs, max_pivots);
        match opt_x {
            Some(x) => {
                let mut actions = vec![0.0_f64; nb];
                for b in 0..nb {
                    let d = x[b];
                    let c_val = x[nb + b];
                    let u = d - c_val;
                    let (lo, hi) = state.action_bounds[b];
                    actions[b] = u.clamp(lo, hi);
                }
                Some(actions)
            }
            None => None,
        }
    }

    fn refine_solution(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: Vec<f64>,
        fuel_budget: u64,
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let fuel: u64 = fuel_budget;

        let mut res = actions.clone();
        let mut flows = vec![0.0_f64; num_l];
        for l in 0..num_l {
            let mut f = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { f += imp * res[b]; }
            flows[l] = flows_base[l] + f;
        }

        let baseline_profit: f64 = (0..num_b)
            .map(|b| eval_profit(challenge, state, ca, b, res[b]))
            .sum();

        let mut order: Vec<usize> = (0..num_b).collect();
        order.sort_by(|&a, &b| {
            let va = eval_profit(challenge, state, ca, a, res[a]).abs()
                - eval_profit(challenge, state, ca, a, 0.0).abs();
            let vb = eval_profit(challenge, state, ca, b, res[b]).abs()
                - eval_profit(challenge, state, ca, b, 0.0).abs();
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut steps: u64 = 0;
        let pass1_fuel = fuel * 2 / 3;
        for &b in &order {
            if steps >= pass1_fuel { break; }

            let (u_min, u_max) = state.action_bounds[b];

            let (mut lo, mut hi) = (u_min, u_max);
            for &(l, p) in &ca.b_to_lines[b] {
                if p.abs() < 1e-9 { continue; }
                let limit = (challenge.network.flow_limits[l] * 0.999).max(0.0);
                let f_other = flows[l] - p * res[b];
                let b1 = (-limit - f_other) / p;
                let b2 = (limit - f_other) / p;
                let (c_lo, c_hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
                if c_lo > lo { lo = c_lo; }
                if c_hi < hi { hi = c_hi; }
            }
            if lo > hi { continue; }
            lo = lo.max(u_min);
            hi = hi.min(u_max);

            let current = res[b];
            let perturb_range = (hi - lo) * 0.15;
            let search_lo = (current - perturb_range).max(lo);
            let search_hi = (current + perturb_range).min(hi);
            if search_hi - search_lo < 1e-10 { continue; }

            let iters = if pass1_fuel - steps > 500_000_000 { 20 } else { 12 };
            let (best_u, best_val) = ternary_search(
                |u| eval_profit(challenge, state, ca, b, u),
                search_lo, search_hi,
                iters.min(hp.ternary_iters.max(10)),
            );

            let old_val = eval_profit(challenge, state, ca, b, current);
            if best_val > old_val + 1e-10 {
                let delta = best_u - current;
                res[b] = best_u;
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows[l] += p * delta; }
                }
            }
            steps += 1;
        }

        for &b in &order {
            let current = res[b];
            if current.abs() < 1e-6 { continue; }
            let profit_curr = eval_profit(challenge, state, ca, b, current);
            if profit_curr <= 0.0 { continue; }
            let (u_min, u_max) = state.action_bounds[b];
            let scaled = (current * 1.08).clamp(u_min, u_max);
            let delta = scaled - current;
            if delta.abs() < 1e-8 { continue; }
            let mut feasible = true;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = challenge.network.flow_limits[l];
                if (flows[l] + p * delta).abs() > limit * 0.999 {
                    feasible = false;
                    break;
                }
            }
            if feasible {
                let profit_scaled = eval_profit(challenge, state, ca, b, scaled);
                if profit_scaled > profit_curr + 1e-10 {
                    res[b] = scaled;
                    for &(l, p) in &ca.b_to_lines[b] {
                        if l < num_l { flows[l] += p * delta; }
                    }
                }
            }
        }

        let pass2_fuel = fuel / 3;
        if pass2_fuel > 0 {
            let mut mu = vec![0.0_f64; num_l];
            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                if limit < 1e-6 { continue; }
                let util = flows[l].abs() / limit;
                if util > 0.7 {
                    let excess_frac = ((util - 0.7) / 0.3).min(1.0);
                    mu[l] = excess_frac * flows[l].signum();
                }
            }

            let mut p_eff = vec![0.0_f64; num_b];
            for b in 0..num_b {
                let node = ca.batt_nodes[b];
                let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                let cong_adj: f64 = ca.b_to_lines[b].iter().map(|&(l, imp)| mu[l] * imp).sum();
                p_eff[b] = rt - cong_adj;
            }

            let dual_magnitude: f64 = mu.iter().map(|x| x.abs()).sum();
            if dual_magnitude > 1e-6 {
                let mut dual_res = res.clone();
                for b in 0..num_b {
                    dual_res[b] = kkt_best_action(
                        challenge, state, ca, b,
                        Some(p_eff[b]),
                        hp.ternary_iters,
                    );
                }

                run_deflator(challenge, state, ca, hp, flows_base, &mut dual_res);

                let dual_profit: f64 = (0..num_b)
                    .map(|b| eval_profit(challenge, state, ca, b, dual_res[b]))
                    .sum();

                let current_profit: f64 = (0..num_b)
                    .map(|b| eval_profit(challenge, state, ca, b, res[b]))
                    .sum();

                if dual_profit > current_profit + 1e-8 {
                    let mut dual_feasible = true;
                    for l in 0..num_l {
                        let limit = challenge.network.flow_limits[l];
                        let mut f = flows_base[l];
                        for &(b, imp) in &ca.ptdf_sparse[l] { f += imp * dual_res[b]; }
                        if f.abs() > limit * 1.001 {
                            dual_feasible = false;
                            break;
                        }
                    }
                    if dual_feasible {
                        res = dual_res;
                        for l in 0..num_l {
                            let mut f = 0.0_f64;
                            for &(b, imp) in &ca.ptdf_sparse[l] { f += imp * res[b]; }
                            flows[l] = flows_base[l] + f;
                        }
                    }
                }
            }
        }

        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            res[b] = res[b].clamp(lo, hi);
        }

        let mut feasible = true;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            if flows[l].abs() > limit * 1.001 {
                feasible = false;
                break;
            }
        }

        let refined_profit: f64 = (0..num_b)
            .map(|b| eval_profit(challenge, state, ca, b, res[b]))
            .sum();

        if feasible && refined_profit > baseline_profit {
            res
        } else {
            actions
        }
    }    
}
mod lp {
    const LP_EPS: f64 = 1e-9;

    pub fn lp_solve_with_budget(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, usize) {
        if b.iter().any(|&x| x < -1e-6) {
            return (None, 0);
        }

        let n_vars = n + m;
        let rhs_col = n_vars;
        let n_cols = n_vars + 1;

        let mut tab = vec![vec![0.0_f64; n_cols]; m + 1];

        for i in 0..m {
            for j in 0..n {
                tab[i][j] = a[i][j];
            }
            tab[i][n + i] = 1.0;
            tab[i][rhs_col] = b[i].max(0.0);
        }

        for j in 0..n {
            tab[m][j] = -c[j];
        }

        let mut basis: Vec<usize> = (n..n + m).collect();

        for _ in 0..max_pivots {
            let entering = match (0..n_vars).find(|&j| tab[m][j] < -LP_EPS) {
                Some(j) => j,
                None => break, 
            };

            let leaving_row = (0..m)
                .filter(|&i| tab[i][entering] > LP_EPS)
                .min_by(|&i1, &i2| {
                    let r1 = tab[i1][rhs_col] / tab[i1][entering];
                    let r2 = tab[i2][rhs_col] / tab[i2][entering];
                    r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
                });

            let leaving_row = match leaving_row {
                Some(r) => r,
                None => return (None, 0), 
            };

            let pivot_val = tab[leaving_row][entering];
            if pivot_val.abs() < LP_EPS {
                return (None, 0); 
            }
            for j in 0..n_cols {
                tab[leaving_row][j] /= pivot_val;
            }

            for i in 0..=m {
                if i != leaving_row {
                    let factor = tab[i][entering];
                    if factor.abs() > 1e-15 {
                        for j in 0..n_cols {
                            tab[i][j] -= factor * tab[leaving_row][j];
                        }
                    }
                }
            }

            basis[leaving_row] = entering;
        }

        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i][rhs_col].max(0.0);
            }
        }
        (Some(x), max_pivots) 
    }

}
mod track_baseline {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 201,
            action_grid: 30,
            asca_iters: 15,
            ternary_iters: 15,
            convergence_tol: 1e-3,
            anticipate_lmp: false,
            lmp_threshold: 0.85,
            lmp_premium_scale: 0.45,
            jump_premium: 0.00,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            network_derating: 1.00,
            dual_iters: 0,
            da_step_size: 0.0,
            ldd_iters: 0,
            ldd_step_size: 0.0,
            use_kkt: false,
            kkt_cong_threshold: 0.70,
            kkt_price_scale: 0.8,
            max_admm_iters: 0,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: false,
            lp_soft_lambda: 1e5,
            lp_per_call_pivots: 500,
            lp_total_pivots: 8000,
            lp_horizon: 0,
        }
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let mut hp = defaults();
        hp.override_from_map(hyperparameters);
        solve_with_hp(challenge, save_solution, hp)
    }
}
mod track_capstone {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 201,
            action_grid: 30,
            asca_iters: 66,
            ternary_iters: 15,
            convergence_tol: 1e-3,
            anticipate_lmp: true,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.00,
            jump_premium: 0.00,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            network_derating: 1.00,
            dual_iters: 0,
            da_step_size: 0.0,
            ldd_iters: 0,
            ldd_step_size: 0.0,
            use_kkt: false,
            kkt_cong_threshold: 0.70,
            kkt_price_scale: 0.8,
            max_admm_iters: 0,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: false,
            lp_soft_lambda: 1e5,
            lp_per_call_pivots: 500,
            lp_total_pivots: 8000,
            lp_horizon: 0,
        }
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let mut hp = defaults();
        hp.override_from_map(hyperparameters);
        solve_with_hp(challenge, save_solution, hp)
    }
}
mod track_congested {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 101,
            action_grid: 25,
            asca_iters: 4,
            ternary_iters: 20,
            convergence_tol: 1e-3,
            anticipate_lmp: true,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.00,
            jump_premium: 0.00,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            network_derating: 0.22,
            dual_iters: 0,
            da_step_size: 0.01,
            ldd_iters: 0,
            ldd_step_size: 0.1,
            use_kkt: true,
            kkt_cong_threshold: 0.70,
            kkt_price_scale: 0.8,
            max_admm_iters: 10,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: true,
            lp_soft_lambda: 1e6,
            lp_per_call_pivots: 1000,
            lp_total_pivots: 16000,
            lp_horizon: 12,
        }
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let mut hp = defaults();
        hp.override_from_map(hyperparameters);
        solve_with_hp(challenge, save_solution, hp)
    }
}
mod track_dense {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 201,
            action_grid: 30,
            asca_iters: 45,
            ternary_iters: 15,
            convergence_tol: 1e-3,
            anticipate_lmp: true,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.00,
            jump_premium: 0.00,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            network_derating: 0.10,
            dual_iters: 0,
            da_step_size: 0.0,
            ldd_iters: 0,
            ldd_step_size: 0.0,
            use_kkt: false,
            kkt_cong_threshold: 0.70,
            kkt_price_scale: 0.8,
            max_admm_iters: 0,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: false,
            lp_soft_lambda: 1e5,
            lp_per_call_pivots: 500,
            lp_total_pivots: 8000,
            lp_horizon: 0,
        }
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let mut hp = defaults();
        hp.override_from_map(hyperparameters);
        solve_with_hp(challenge, save_solution, hp)
    }
}
mod track_multiday {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 201,
            action_grid: 30,
            asca_iters: 35,
            ternary_iters: 15,
            convergence_tol: 1e-3,
            anticipate_lmp: true,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.00,
            jump_premium: 0.00,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            network_derating: 0.10,
            dual_iters: 0,
            da_step_size: 0.0,
            ldd_iters: 0,
            ldd_step_size: 0.0,
            use_kkt: false,
            kkt_cong_threshold: 0.70,
            kkt_price_scale: 0.8,
            max_admm_iters: 0,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: false,
            lp_soft_lambda: 1e5,
            lp_per_call_pivots: 500,
            lp_total_pivots: 8000,
            lp_horizon: 0,
        }
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let mut hp = defaults();
        hp.override_from_map(hyperparameters);
        solve_with_hp(challenge, save_solution, hp)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub soc_levels: Option<usize>,
    pub action_grid: Option<usize>,
    pub asca_iters: Option<usize>,
    pub ternary_iters: Option<usize>,
    pub convergence_tol: Option<f64>,
    pub anticipate_lmp: Option<bool>,
    pub lmp_threshold: Option<f64>,
    pub lmp_premium_scale: Option<f64>,
    pub jump_premium: Option<f64>,
    pub prune_ratio: Option<f64>,
    pub deflator_iters: Option<usize>,
    pub flow_margin: Option<f64>,
    pub network_derating: Option<f64>,
    pub ldd_iters: Option<usize>,
    pub ldd_step_size: Option<f64>,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if n <= 15 => track_baseline::solve(challenge, save_solution, hyperparameters),
        n if n <= 30 => track_congested::solve(challenge, save_solution, hyperparameters),
        n if n <= 50 => track_multiday::solve(challenge, save_solution, hyperparameters),
        n if n <= 80 => track_dense::solve(challenge, save_solution, hyperparameters),
        n if n <= 150 => track_capstone::solve(challenge, save_solution, hyperparameters),
        n => Err(anyhow!(
            "aycdicdb: unsupported num_batteries={} (expected one of 10/20/40/60/100)", n
        )),
    }
}