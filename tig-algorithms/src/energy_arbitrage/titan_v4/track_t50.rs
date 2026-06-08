
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod helpers {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use std::cell::RefCell;
    use std::sync::Arc;
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
        
        pub use_policy: bool,
        
        pub use_warmstart: bool,
        
        pub use_mpc: bool,
        pub mpc_horizon: usize,
        pub mpc_pivot_budget: usize,
        
        pub use_dw: bool,
        pub dw_iters: usize,
        pub dw_max_lines: usize,
        pub dw_max_cols_per_batt: usize,
        pub dw_pivot_budget_per_solve: usize,
        pub dw_total_pivot_budget: usize,
        
        pub use_dw_prescreen: bool,
        
        pub use_lns: bool,
        pub lns_cg_iters: usize,
        pub lns_cg_column_limit: usize,
        pub lns_max_lines: usize,
        pub lns_lp_pivots_total: usize,
        
        pub use_pivot_reserve: bool,
        
        pub lp_max_lines: usize,
        
        pub use_parallel_dp: bool,
        
        pub use_sdp: bool,
        pub sdp_k: usize,
        
        pub use_ldd_proximal: bool,
        pub ldd_momentum: f64,
        pub ldd_clip_fraction: f64,
        
        pub use_tail_quadrature: bool,
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
            if let Some(v) = m.get("use_policy").and_then(|v| v.as_bool()) { self.use_policy = v; }
            if let Some(v) = m.get("use_warmstart").and_then(|v| v.as_bool()) { self.use_warmstart = v; }
            if let Some(v) = m.get("use_mpc").and_then(|v| v.as_bool()) { self.use_mpc = v; }
            if let Some(v) = m.get("mpc_horizon").and_then(|v| v.as_u64()) { self.mpc_horizon = v as usize; }
            if let Some(v) = m.get("mpc_pivot_budget").and_then(|v| v.as_u64()) { self.mpc_pivot_budget = v as usize; }
            if let Some(v) = m.get("use_dw").and_then(|v| v.as_bool()) { self.use_dw = v; }
            if let Some(v) = m.get("dw_iters").and_then(|v| v.as_u64()) { self.dw_iters = v as usize; }
            if let Some(v) = m.get("dw_max_lines").and_then(|v| v.as_u64()) { self.dw_max_lines = v as usize; }
            if let Some(v) = m.get("dw_max_cols_per_batt").and_then(|v| v.as_u64()) { self.dw_max_cols_per_batt = v as usize; }
            if let Some(v) = m.get("dw_pivot_budget_per_solve").and_then(|v| v.as_u64()) { self.dw_pivot_budget_per_solve = v as usize; }
            if let Some(v) = m.get("dw_total_pivot_budget").and_then(|v| v.as_u64()) { self.dw_total_pivot_budget = v as usize; }
            if let Some(v) = m.get("use_dw_prescreen").and_then(|v| v.as_bool()) { self.use_dw_prescreen = v; }
            if let Some(v) = m.get("use_lns").and_then(|v| v.as_bool()) { self.use_lns = v; }
            if let Some(v) = m.get("lns_cg_iters").and_then(|v| v.as_u64()) { self.lns_cg_iters = v as usize; }
            if let Some(v) = m.get("lns_cg_column_limit").and_then(|v| v.as_u64()) { self.lns_cg_column_limit = (v as usize).max(2); }
            if let Some(v) = m.get("lns_max_lines").and_then(|v| v.as_u64()) { self.lns_max_lines = (v as usize).max(1); }
            if let Some(v) = m.get("lns_lp_pivots_total").and_then(|v| v.as_u64()) { self.lns_lp_pivots_total = v as usize; }
            if let Some(v) = m.get("use_pivot_reserve").and_then(|v| v.as_bool()) { self.use_pivot_reserve = v; }
            if let Some(v) = m.get("lp_max_lines").and_then(|v| v.as_u64()) { self.lp_max_lines = (v as usize).max(1); }
            if let Some(v) = m.get("use_parallel_dp").and_then(|v| v.as_bool()) { self.use_parallel_dp = v; }
            if let Some(v) = m.get("use_sdp").and_then(|v| v.as_bool()) { self.use_sdp = v; }
            if let Some(v) = m.get("sdp_k").and_then(|v| v.as_u64()) { self.sdp_k = (v as usize).max(2); }
            if let Some(v) = m.get("use_ldd_proximal").and_then(|v| v.as_bool()) { self.use_ldd_proximal = v; }
            if let Some(v) = m.get("ldd_momentum").and_then(|v| v.as_f64()) { self.ldd_momentum = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("ldd_clip_fraction").and_then(|v| v.as_f64()) { self.ldd_clip_fraction = v.clamp(0.01, 1.0); }
            if let Some(v) = m.get("use_tail_quadrature").and_then(|v| v.as_bool()) { self.use_tail_quadrature = v; }
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
        lp_pivots_consumed: usize,
        lp_pivot_reserve: isize,
        policy_weights: Option<Vec<Vec<f64>>>,
    }

    thread_local! {
        static STATE: RefCell<Option<Inner>> = RefCell::new(None);
    }

    pub fn solve_with_hp(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hp: TrackHp,
    ) -> Result<()> {
        STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None, lp_pivots_consumed: 0, lp_pivot_reserve: 0, policy_weights: None }));
        let out = challenge.grid_optimize(&policy_entry);
        STATE.with(|s| *s.borrow_mut() = None);
        let solution = out?;
        save_solution(&solution)?;
        Ok(())
    }

    fn policy_entry(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        STATE.with(|s| -> Result<Vec<f64>> {
            let mut guard = s.borrow_mut();
            let inner = guard.as_mut().expect("STATE not initialised");
            if inner.cache.is_none() {
                inner.cache = Some(build_cache(challenge, state, &inner.hp));
            }
            let cache = inner.cache.as_ref().unwrap();
            let hp = &inner.hp;

            let zero_action = vec![0.0_f64; challenge.num_batteries];
            let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base_cur);

            // always compute uncons_actions (needed for targeted_tight_line_polish seeding)
            let uncons_actions: Vec<f64> = (0..challenge.num_batteries)
                .map(|b| optimal_unconstrained_action(challenge, state, cache, b))
                .collect();
            let warm_init: Vec<f64> = if hp.use_warmstart {
                uncons_actions.clone()
            } else {
                vec![0.0; challenge.num_batteries]
            };

            
            let base_actions = if hp.use_kkt {
                kkt_policy(challenge, state, cache, hp, &flows_base, &warm_init)
            } else {
                let mut a = warm_init.clone();
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

                
                let budget = if hp.use_pivot_reserve {
                    compute_lp_budget_for_step(hp, inner.lp_pivots_consumed, &mut inner.lp_pivot_reserve)
                } else {
                    per_call_limit
                };

                
                let budget_ok = if hp.use_pivot_reserve {
                    budget > 0
                } else {
                    hp.lp_total_pivots == 0
                        || inner.lp_pivots_consumed.saturating_add(per_call_limit) <= hp.lp_total_pivots
                };

                if max_util >= hp.kkt_cong_threshold && budget_ok {
                    let lp_result = if hp.use_pivot_reserve {
                        joint_lp_dispatch_with_used(challenge, state, cache, &flows_base, hp.lp_soft_lambda, budget, hp)
                    } else {
                        let sol = joint_lp_dispatch(challenge, state, cache, &flows_base, hp.lp_soft_lambda, budget, hp);
                        (sol, budget)
                    };

                    if let Some(mut lp_act) = lp_result.0 {
                        let lp_used = lp_result.1;
                        if hp.use_pivot_reserve {
                            inner.lp_pivots_consumed = inner.lp_pivots_consumed.saturating_add(lp_used).min(hp.lp_total_pivots.max(1));
                            let allocated = budget;
                            let unused = allocated.saturating_sub(lp_used);
                            inner.lp_pivot_reserve = (inner.lp_pivot_reserve + unused as isize).min(hp.lp_per_call_pivots as isize);
                        } else {
                            inner.lp_pivots_consumed += per_call_limit;
                        }
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut lp_act);
                        let lp_p: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, lp_act[b])).sum();
                        let base_p: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, base_actions[b])).sum();
                        if lp_p >= base_p { lp_act } else { base_actions }
                    } else {
                        if hp.use_pivot_reserve {
                            inner.lp_pivot_reserve = (inner.lp_pivot_reserve + budget as isize).min(hp.lp_per_call_pivots as isize);
                        }
                        base_actions
                    }
                } else {
                    if hp.use_pivot_reserve {
                        inner.lp_pivot_reserve = (inner.lp_pivot_reserve + budget as isize).min(hp.lp_per_call_pivots as isize);
                    }
                    base_actions
                }
            } else { base_actions };

            
            
            let actions = if hp.use_mpc {
                if let Some(mut mpc_act) = mpc_dispatch_2step(challenge, state, cache, hp, &flows_base) {
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut mpc_act);
                    let mpc_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, mpc_act[b])).sum();
                    let base_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, actions[b])).sum();
                    if mpc_p >= base_p { mpc_act } else { actions }
                } else {
                    actions
                }
            } else {
                actions
            };

            
            let actions = if hp.use_policy {
                if let Some(ref weights) = inner.policy_weights {
                    let mut policy_act = policy_dispatch(challenge, state, weights);
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut policy_act);
                    policy_act
                } else {
                    
                    if state.time_step == 0 {
                        let trained = train_policy_cmaes(challenge, state, cache, hp);
                        inner.policy_weights = Some(trained);
                        let weights = inner.policy_weights.as_ref().unwrap();
                        let mut policy_act = policy_dispatch(challenge, state, weights);
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut policy_act);
                        policy_act
                    } else {
                        actions
                    }
                }
            } else {
                actions
            };



            // track used_lns to conditionally apply targeted_tight_line_polish
            let (actions, used_lns) = if hp.use_lns {
                if let Some(mut lns_act) = lns_dw_per_step(challenge, state, cache, hp, &flows_base, &actions) {
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut lns_act);
                    let lns_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, lns_act[b])).sum();
                    let base_p: f64 = (0..challenge.num_batteries)
                        .map(|b| eval_profit(challenge, state, cache, b, actions[b])).sum();
                    if lns_p > base_p { (lns_act, true) } else { (actions, false) }
                } else {
                    (actions, false)
                }
            } else {
                (actions, false)
            };

            // targeted_tight_line_polish — generic refinement
            // Runs when LNS was not applied: targeted LP on batteries near tight lines
            let actions = if !used_lns {
                if let Some(polish) = targeted_tight_line_polish(challenge, state, cache, hp, &flows_base, &actions) {
                    polish
                } else {
                    actions
                }
            } else {
                actions
            };
            let _ = uncons_actions; // used for warm_init seeding above
            let baseline_actions = actions;
            let is_last_step = state.time_step + 1 >= challenge.num_steps;
            Ok(if is_last_step {
                let baseline_profit = challenge.compute_profit(state, &baseline_actions);
                let fuel_budget: u64 = 5_000_000_000;
                let refined = refine_solution(challenge, state, baseline_actions.clone(), fuel_budget);
                let refined_profit = challenge.compute_profit(state, &refined);
                
                
                
                let inj = challenge.compute_total_injections(state, &refined);
                let flows = challenge.network.compute_flows(&inj);
                let feasible = (0..flows.len()).all(|l| flows[l].abs() <= challenge.network.flow_limits[l] * 1.001);
                if refined_profit > baseline_profit && feasible {
                    refined
                } else {
                    baseline_actions
                }
            } else {
                baseline_actions
            })
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

    
    
    #[inline]
    fn dp_lambda(dp_b: &[Vec<f64>], t_next: usize, soc: f64, soc_min: f64, soc_span: f64, soc_levels: usize) -> f64 {
        if soc_levels < 2 || t_next >= dp_b.len() {
            return 0.0;
        }
        let delta_s = soc_span / (soc_levels - 1) as f64;
        let idx_f = (soc - soc_min) / soc_span * (soc_levels - 1) as f64;
        let lo = ((idx_f.floor() as isize).max(0) as usize).min(soc_levels - 2);
        let hi = (lo + 1).min(soc_levels - 1);
        let v_lo = dp_b[t_next][lo];
        let v_hi = dp_b[t_next][hi];
        if !v_lo.is_finite() || !v_hi.is_finite() {
            return 0.0;
        }
        (v_hi - v_lo) / delta_s
    }

    
    
    
    #[inline]
    fn optimal_unconstrained_action(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        b: usize,
    ) -> f64 {
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        if !rt.is_finite() { return 0.0; }

        let soc = state.socs[b];
        let (u_min, u_max) = state.action_bounds[b];
        if u_min >= u_max { return u_min; }

        let dt = 0.25_f64;
        let deg_coeff = (dt / bat.capacity_mwh.max(1e-9)).powi(2);
        let two_deg = 2.0 * deg_coeff;
        let soc_levels = ca.dp[b][0].len();
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
        let lambda = dp_lambda(&ca.dp[b], t_next, soc, bat.soc_min_mwh, soc_span, soc_levels);
        if !lambda.is_finite() { return 0.0; }

        
        let mut best_u = 0.0_f64;
        let mut best_val = eval_profit(challenge, state, ca, b, 0.0);
        if !best_val.is_finite() { best_val = f64::NEG_INFINITY; }

        
        if u_min < -1e-12 {
            let hi = 0.0_f64.min(u_max);
            if u_min < hi {
                
                
                let b_c = dt * (lambda * bat.efficiency_charge - rt - 0.25);
                if two_deg > 1e-30 {
                    let raw_charge = (-b_c) / two_deg; 
                    let cand = (-raw_charge.clamp(0.0, (-u_min).max(0.0))).clamp(u_min, hi);
                    if cand >= u_min && cand <= hi {
                        let v = eval_profit(challenge, state, ca, b, cand);
                        if v.is_finite() && v > best_val { best_val = v; best_u = cand; }
                    }
                }
                
                {
                    let v_lo = eval_profit(challenge, state, ca, b, u_min);
                    if v_lo.is_finite() && v_lo > best_val { best_val = v_lo; best_u = u_min; }
                }
            }
        }

        
        if u_max > 1e-12 {
            let lo = 0.0_f64.max(u_min);
            if lo < u_max {
                let eff_d = bat.efficiency_discharge.max(1e-9);
                let b_d = dt * (rt - 0.25 - lambda / eff_d);
                if two_deg > 1e-30 {
                    let raw_disch = b_d / two_deg;
                    let cand = raw_disch.clamp(lo, u_max);
                    if cand >= lo && cand <= u_max {
                        let v = eval_profit(challenge, state, ca, b, cand);
                        if v.is_finite() && v > best_val { best_val = v; best_u = cand; }
                    }
                }
                
                {
                    let v_hi = eval_profit(challenge, state, ca, b, u_max);
                    if v_hi.is_finite() && v_hi > best_val { best_val = v_hi; best_u = u_max; }
                }
            }
        }

        if best_u.is_finite() { best_u } else { 0.0 }
    }

    
    
    
    
    
    
    fn prescreen_binding_lines(
        challenge: &Challenge,
        flows_all: &[Vec<f64>],
        max_lines: usize,
    ) -> Vec<usize> {
        let num_lines = challenge.network.flow_limits.len();
        if num_lines == 0 || flows_all.is_empty() || max_lines == 0 {
            return Vec::new();
        }

        let mut max_viol = vec![0.0_f64; num_lines];
        for t_flows in flows_all.iter() {
            for l in 0..num_lines {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let absflow = t_flows.get(l).copied().unwrap_or(0.0).abs();
                if absflow > limit {
                    let v = absflow - limit;
                    if v > max_viol[l] {
                        max_viol[l] = v;
                    }
                }
            }
        }

        let threshold = 1e-6;
        let mut ranked: Vec<(usize, f64)> = (0..num_lines)
            .filter(|&l| max_viol[l] > threshold * challenge.network.flow_limits[l].max(threshold))
            .map(|l| (l, max_viol[l]))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(max_lines);
        ranked.into_iter().map(|(l, _)| l).collect()
    }

    
    
    
    
    
    
    
    
    fn build_dw_dual_prices(
        challenge: &Challenge,
        state: &State,
        hp: &TrackHp,
        batt_nodes: &[usize],
        b_to_lines: &[Vec<(usize, f64)>],
        preselected_lines: Option<&[usize]>,
    ) -> Vec<Vec<f64>> {
        let num_l = challenge.network.flow_limits.len();
        let num_t = challenge.num_steps;
        let num_b = challenge.num_batteries;
        if num_l == 0 || num_b == 0 || num_t == 0 {
            return vec![vec![0.0_f64; num_l]; num_t];
        }

        
        
        let active_lines: Vec<usize> = if let Some(preselected) = preselected_lines {
            
            let k_active = hp.dw_max_lines.min(preselected.len());
            preselected.iter().take(k_active).copied().collect()
        } else {
            
            let mut line_cong: Vec<(usize, f64)> = Vec::new();
            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let mut sum_abs = 0.0_f64;
                for t in 0..num_t {
                    let exo_flows = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                    sum_abs += exo_flows[l].abs();
                }
                let avg = sum_abs / num_t as f64;
                let util = avg / limit;
                line_cong.push((l, util));
            }
            line_cong.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let k_active = hp.dw_max_lines.min(line_cong.len());
            line_cong.iter().take(k_active).map(|(l, _)| *l).collect()
        };
        let n_active = active_lines.len();
        if n_active == 0 {
            return vec![vec![0.0_f64; num_l]; num_t];
        }

        let _line_idx_map: Vec<usize> = {
            let mut map = vec![usize::MAX; num_l];
            for (i, &l) in active_lines.iter().enumerate() {
                map[l] = i;
            }
            map
        };

        
        let ptdf_batt: Vec<Vec<f64>> = (0..num_b).map(|b| {
            (0..n_active).map(|ai| {
                let l = active_lines[ai];
                let val = b_to_lines[b].iter()
                    .find(|&&(ll, _)| ll == l)
                    .map(|&(_, coef)| coef)
                    .unwrap_or(0.0);
                val
            }).collect()
        }).collect();

        
        let exo_flows_all: Vec<Vec<f64>> = (0..num_t).map(|t| {
            challenge.network.compute_flows(&challenge.exogenous_injections[t])
        }).collect();

        
        let mut columns: Vec<Vec<f64>> = Vec::new();
        let mut col_batt_idx: Vec<usize> = Vec::new();

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = batt_nodes[b];
            let mut greedy = vec![0.0_f64; num_t];
            let mut soc = state.socs[b];
            let dt = 0.25_f64;

            for t in 0..num_t {
                let da_price = if node < challenge.market.day_ahead_prices[t].len() {
                    challenge.market.day_ahead_prices[t][node]
                } else {
                    challenge.market.day_ahead_prices[t][0]
                };
                if !da_price.is_finite() { continue; }

                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let eta_c = bat.efficiency_charge;
                let eta_d = bat.efficiency_discharge.max(1e-9);

                let charge_lim = if eta_c > 0.0 {
                    (bat.soc_max_mwh - soc) / (eta_c * dt)
                } else { 0.0 };
                let disch_lim = if eta_d > 0.0 {
                    (soc - bat.soc_min_mwh) * eta_d / dt
                } else { 0.0 };

                let u_min = -(bat.power_charge_mw.min(charge_lim.max(0.0)));
                let u_max = bat.power_discharge_mw.min(disch_lim.max(0.0));
                if u_min >= u_max { continue; }

                
                let median_price = state.rt_prices.get(node).copied().unwrap_or(da_price);
                let u: f64;
                if median_price > da_price + 5.0 && u_max > 0.0 {
                    u = u_max;
                } else if median_price < da_price - 5.0 && u_min < 0.0 {
                    u = u_min;
                } else {
                    u = 0.0;
                }

                greedy[t] = u;
                let next_soc_raw = if u < 0.0 {
                    soc + eta_c * (-u) * dt
                } else {
                    soc - u / eta_d * dt
                };
                soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            }

            columns.push(greedy);
            col_batt_idx.push(b);
        }

        
        let n_constraints = n_active * num_t;
        let M_penalty: f64 = 1e7; 

        let mut total_pivots_used = 0usize;

        
        let mut mu = vec![vec![0.0_f64; num_l]; num_t];

        'dw_outer: for _dw_iter in 0..hp.dw_iters {
            let n_cols = columns.len();
            if n_cols == 0 { break 'dw_outer; }

            
            let n_slacks = n_constraints;
            let total_vars = n_cols + n_slacks;

            
            let mut c_lp = vec![0.0_f64; total_vars];
            for (j, col) in columns.iter().enumerate() {
                let b_idx = col_batt_idx[j];
                let profit = compute_schedule_profit(challenge, b_idx, col);
                c_lp[j] = if profit.is_finite() { profit } else { 0.0 };
            }
            
            for s in 0..n_slacks {
                c_lp[n_cols + s] = -M_penalty;
            }

            
            let mut a_lp: Vec<Vec<f64>> = vec![vec![0.0_f64; total_vars]; n_constraints];
            let mut b_lp = vec![0.0_f64; n_constraints];

            for ai in 0..n_active {
                let l = active_lines[ai];
                let limit = challenge.network.flow_limits[l];
                for t in 0..num_t {
                    let row = ai * num_t + t;
                    let exo = exo_flows_all[t][l];
                    b_lp[row] = (limit - exo).max(0.0);

                    
                    for c in 0..n_cols {
                        let b = col_batt_idx[c];
                        let action_val = columns[c].get(t).copied().unwrap_or(0.0);
                        let ptdf_val = ptdf_batt[b][ai];
                        a_lp[row][c] += action_val * ptdf_val;
                    }

                    
                    
                    
                    a_lp[row][n_cols + row] = -1.0;
                }
            }

            
            let pivot_budget = hp.dw_pivot_budget_per_solve;
            total_pivots_used += pivot_budget;
            if total_pivots_used > hp.dw_total_pivot_budget {
                break 'dw_outer;
            }

            let (sol, duals_opt, _) = super::lp::lp_solve_with_duals(
                total_vars, n_constraints, &c_lp, &a_lp, &b_lp, pivot_budget,
            );

            
            let Some(duals) = duals_opt else { break 'dw_outer; };
            let Some(_x_sol) = sol else { break 'dw_outer; };

            let mut mu_new = vec![vec![0.0_f64; num_l]; num_t];
            let mut dual_valid = true;

            for ai in 0..n_active {
                let l = active_lines[ai];
                for t in 0..num_t {
                    let row = ai * num_t + t;
                    if row < duals.len() {
                        let d = duals[row];
                        
                        if d.abs() > 1e6 {
                            dual_valid = false;
                            continue;
                        }
                        mu_new[t][l] = d;
                    }
                }
            }

            if !dual_valid { break 'dw_outer; }

            
            let mut converged = true;
            for t in 0..num_t {
                for ai in 0..n_active {
                    let l = active_lines[ai];
                    if (mu[t][l] - mu_new[t][l]).abs() > 1e-3 {
                        converged = true; 
                        converged = false;
                        break;
                    }
                }
                if !converged { break; }
            }
            mu = mu_new;
            if converged { break 'dw_outer; }

            
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let node = batt_nodes[b];
                let eta_c = bat.efficiency_charge;
                let eta_d = bat.efficiency_discharge.max(1e-9);
                let dt = 0.25_f64;

                
                let mut eff_prices = vec![0.0_f64; num_t];
                for t in 0..num_t {
                    let da = if node < challenge.market.day_ahead_prices[t].len() {
                        challenge.market.day_ahead_prices[t][node]
                    } else {
                        challenge.market.day_ahead_prices[t][0]
                    };
                    let cong_adj: f64 = (0..n_active).map(|ai| {
                        mu[t][active_lines[ai]] * ptdf_batt[b][ai]
                    }).sum();
                    eff_prices[t] = da - cong_adj;
                }

                
                
                let median_eff = eff_prices.iter().fold(0.0_f64, |a, &b| a + b) / num_t as f64;
                let mut cand_soc = state.socs[b];
                let mut candidate_col = vec![0.0_f64; num_t];

                for t in 0..num_t {
                    let eff = eff_prices[t];
                    let charge_lim = if eta_c > 0.0 {
                        (bat.soc_max_mwh - cand_soc) / (eta_c * dt)
                    } else { 0.0 };
                    let disch_lim = if eta_d > 0.0 {
                        (cand_soc - bat.soc_min_mwh) * eta_d / dt
                    } else { 0.0 };

                    let u_min = -(bat.power_charge_mw.min(charge_lim.max(0.0)));
                    let u_max = bat.power_discharge_mw.min(disch_lim.max(0.0));
                    if u_min >= u_max { continue; }

                    let u: f64;
                    if eff > median_eff + 3.0 && u_max > 0.0 {
                        u = u_max;
                    } else if eff < median_eff - 3.0 && u_min < 0.0 {
                        u = u_min;
                    } else {
                        u = 0.0;
                    }

                    candidate_col[t] = u;
                    let next_soc_raw = if u < 0.0 {
                        cand_soc + eta_c * (-u) * dt
                    } else {
                        cand_soc - u / eta_d * dt
                    };
                    cand_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
                }

                
                let is_duplicate = columns.iter().zip(col_batt_idx.iter())
                    .filter(|(_, &idx)| idx == b)
                    .any(|(col, _)| {
                        col.iter().zip(candidate_col.iter())
                            .all(|(a, c)| (a - c).abs() < 1e-4)
                    });

                if !is_duplicate {
                    
                    columns.push(candidate_col);
                    col_batt_idx.push(b);

                    let new_count = col_batt_idx.iter().filter(|&&idx| idx == b).count();
                    if new_count > hp.dw_max_cols_per_batt {
                        
                        if let Some(pos) = col_batt_idx.iter().position(|&idx| idx == b) {
                            columns.remove(pos);
                            col_batt_idx.remove(pos);
                        }
                    }
                }
            }

            
            let cols_added = columns.len() > n_cols;
            if !cols_added { break 'dw_outer; }
        }

        
        for t in 0..num_t {
            for l in 0..num_l {
                if !mu[t][l].is_finite() || mu[t][l].abs() > 1e6 {
                    mu[t][l] = 0.0;
                }
            }
        }

        mu
    }

    
    fn compute_schedule_profit(challenge: &Challenge, b: usize, schedule: &[f64]) -> f64 {
        let bat = &challenge.batteries[b];
        let node = b; 
        let dt = 0.25_f64;
        let mut profit = 0.0_f64;

        for t in 0..schedule.len().min(challenge.num_steps) {
            let u = schedule[t];
            if !u.is_finite() { continue; }

            let da_price = if challenge.batteries[b].node < challenge.market.day_ahead_prices[t].len() {
                challenge.market.day_ahead_prices[t][challenge.batteries[b].node]
            } else {
                challenge.market.day_ahead_prices[t].get(0).copied().unwrap_or(0.0)
            };

            let abs_u = u.abs();
            let revenue = u * da_price * dt;
            let tx = 0.25 * abs_u * dt;
            let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
            let deg = deg_base * deg_base;
            profit += revenue - tx - deg;
        }

        if profit.is_finite() { profit } else { 0.0 }
    }

    
    #[allow(dead_code)]
    fn col_dual_contribution(
        challenge: &Challenge,
        b: usize,
        col: &[f64],
        mu: &[Vec<f64>],
        ptdf_batt_line: &[f64],
        active_lines: &[usize],
    ) -> f64 {
        let num_t = col.len().min(challenge.num_steps);
        let mut contrib = 0.0_f64;
        for t in 0..num_t {
            let action_t = col[t];
            for ai in 0..active_lines.len() {
                let l = active_lines[ai];
                contrib += mu[t][l] * ptdf_batt_line[ai] * action_t;
            }
        }
        if contrib.is_finite() { contrib } else { 0.0 }
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    fn lns_dw_per_step(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
        base_actions: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let _t = state.time_step;

        
        let mut max_util = 0.0_f64;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            if limit > 1e-6 {
                let util = flows_base[l].abs() / limit;
                if util > max_util { max_util = util; }
            }
        }
        if max_util < hp.kkt_cong_threshold {
            return None; 
        }

        
        let mut violations: Vec<(usize, f64)> = Vec::new();
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            if limit < 1e-6 { continue; }
            
            let mut bat_flow = 0.0_f64;
            for &(b_idx, imp) in &ca.ptdf_sparse[l] {
                if b_idx < base_actions.len() {
                    bat_flow += imp * base_actions[b_idx];
                }
            }
            let total = flows_base[l] + bat_flow;
            let viol = total.abs() - limit;
            if viol > 1e-6 {
                violations.push((l, viol));
            }
        }

        
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            if limit < 1e-6 { continue; }
            let already = violations.iter().any(|&(idx, _)| idx == l);
            if already { continue; }
            let mut bat_flow = 0.0_f64;
            for &(b_idx, imp) in &ca.ptdf_sparse[l] {
                if b_idx < base_actions.len() {
                    bat_flow += imp * base_actions[b_idx];
                }
            }
            let total = flows_base[l] + bat_flow;
            let util = total.abs() / limit;
            if util > (hp.kkt_cong_threshold - 0.05).max(0.3) {
                violations.push((l, 0.0));
            }
        }

        if violations.is_empty() {
            return None;
        }

        violations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k_lines = hp.lns_max_lines.min(violations.len());
        let line_set: Vec<usize> = violations.into_iter().take(k_lines).map(|(l, _)| l).collect();
        let n_lines = line_set.len();

        
        let mut col_pools: Vec<Vec<f64>> = vec![Vec::new(); num_b];
        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            let cur = base_actions[b];
            let mut seed_actions: Vec<f64> = vec![cur, 0.0, lo, hi];

            
            if hi - lo > 1e-12 {
                for &frac in &[0.25, 0.5, 0.75] {
                    seed_actions.push(lo + frac * (hi - lo));
                }
            }

            
            let mut unique: Vec<f64> = Vec::new();
            for &a in &seed_actions {
                if a.is_finite() && !unique.iter().any(|&u| (u - a).abs() < 1e-6) {
                    unique.push(a);
                }
            }
            
            if unique.len() > hp.lns_cg_column_limit {
                
                let mut keep = Vec::new();
                let mut has_zero = false;
                let mut has_lo = false;
                let mut has_hi = false;
                for &a in &unique {
                    if keep.is_empty() || (!has_zero && a.abs() < 1e-6) || (!has_lo && (a - lo).abs() < 1e-6) || (!has_hi && (a - hi).abs() < 1e-6) {
                        if !has_zero && a.abs() < 1e-6 { has_zero = true; }
                        if !has_lo && (a - lo).abs() < 1e-6 { has_lo = true; }
                        if !has_hi && (a - hi).abs() < 1e-6 { has_hi = true; }
                        keep.push(a);
                    }
                    if keep.len() >= hp.lns_cg_column_limit { break; }
                }
                unique = keep;
            }
            col_pools[b] = unique;
        }

        
        let mut best_actions: Option<Vec<f64>> = None;
        let mut best_obj = f64::NEG_INFINITY;
        let mut pivot_budget = hp.lns_lp_pivots_total;

        for _cg_iter in 0..hp.lns_cg_iters {
            if pivot_budget < 50 { break; } 

            
            let n_cols_total: usize = col_pools.iter().map(|v| v.len()).sum();
            if n_cols_total == 0 { break; }

            
            
            let m = 2 * num_b + 2 * n_lines;
            let mut c_vec = vec![0.0_f64; n_cols_total];
            let mut a_mat = vec![vec![0.0_f64; n_cols_total]; m];
            let mut b_vec = vec![0.0_f64; m];

            
            let mut col_offset = 0usize;
            for b in 0..num_b {
                
                for j in 0..col_pools[b].len() {
                    a_mat[2 * b][col_offset + j] = 1.0;
                }
                b_vec[2 * b] = 1.0;

                
                for j in 0..col_pools[b].len() {
                    a_mat[2 * b + 1][col_offset + j] = -1.0;
                }
                
                
                
                
                
                
                
                
                a_mat[2 * b + 1] = vec![0.0_f64; n_cols_total]; 
                b_vec[2 * b + 1] = 0.0;

                
                for j in 0..col_pools[b].len() {
                    let u = col_pools[b][j];
                    
                    let node = ca.batt_nodes[b];
                    let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                    let mut profit = eval_profit_with_price(challenge, state, ca, b, u, rt);
                    if !profit.is_finite() { profit = 0.0; }
                    c_vec[col_offset + j] = profit;
                }

                col_offset += col_pools[b].len();
            }

            
            for (i, &l) in line_set.iter().enumerate() {
                let limit = challenge.network.flow_limits[l];
                let exo = flows_base[l];

                
                let row_p = 2 * num_b + 2 * i;
                let mut col_idx = 0usize;
                for b in 0..num_b {
                    let ptdf_coef = ca.b_to_lines[b].iter()
                        .find(|&&(ll, _)| ll == l)
                        .map(|&(_, coef)| coef)
                        .unwrap_or(0.0);
                    for j in 0..col_pools[b].len() {
                        a_mat[row_p][col_idx] = ptdf_coef * col_pools[b][j];
                        col_idx += 1;
                    }
                }
                b_vec[row_p] = (limit - exo).max(0.0);

                
                let row_n = 2 * num_b + 2 * i + 1;
                let mut col_idx = 0usize;
                for b in 0..num_b {
                    let ptdf_coef = ca.b_to_lines[b].iter()
                        .find(|&&(ll, _)| ll == l)
                        .map(|&(_, coef)| coef)
                        .unwrap_or(0.0);
                    for j in 0..col_pools[b].len() {
                        a_mat[row_n][col_idx] = -ptdf_coef * col_pools[b][j];
                        col_idx += 1;
                    }
                }
                b_vec[row_n] = (limit + exo).max(0.0);
            }

            
            let per_solve = (pivot_budget / (hp.lns_cg_iters.max(1))).min(pivot_budget);
            let per_solve = per_solve.max(50);
            let (primal, duals_opt, pivots_used) = super::lp::lp_solve_with_duals(
                n_cols_total, m, &c_vec, &a_mat, &b_vec, per_solve,
            );
            pivot_budget = pivot_budget.saturating_sub(pivots_used);

            let Some(primal_x) = primal else { break; };

            
            let mut actions_t = vec![0.0_f64; num_b];
            let mut obj_val = 0.0_f64;
            let mut col_idx = 0usize;
            for b in 0..num_b {
                let n_local = col_pools[b].len();
                let mut best_j = 0usize;
                let mut best_w = 0.0_f64;
                for j in 0..n_local {
                    let w = primal_x[col_idx + j];
                    if w > best_w {
                        best_w = w;
                        best_j = j;
                    }
                }
                actions_t[b] = col_pools[b][best_j];
                obj_val += c_vec[col_idx + best_j] * best_w;
                col_idx += n_local;
            }

            if obj_val.is_finite() && obj_val > best_obj {
                best_obj = obj_val;
                best_actions = Some(actions_t.clone());
            }

            
            let Some(duals) = duals_opt else { break; };

            
            let line_dual_offset = 2 * num_b;
            if line_dual_offset >= duals.len() { break; }

            let mut added_any = false;
            for b in 0..num_b {
                if col_pools[b].len() >= hp.lns_cg_column_limit { continue; }

                let node = ca.batt_nodes[b];
                let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };

                
                
                let mut cong_adj = 0.0_f64;
                for (i, &l) in line_set.iter().enumerate() {
                    let dual_row = line_dual_offset + 2 * i; 
                    if dual_row >= duals.len() { continue; }
                    let d = duals[dual_row];
                    if !d.is_finite() { continue; }
                    let ptdf_val = ca.b_to_lines[b].iter()
                        .find(|&&(ll, _)| ll == l)
                        .map(|&(_, coef)| coef)
                        .unwrap_or(0.0);
                    cong_adj += d * ptdf_val;
                }

                let eff_price = rt - cong_adj;
                if !eff_price.is_finite() { continue; }

                
                let (lo, hi) = state.action_bounds[b];
                if hi - lo < 1e-12 { continue; }

                let eval_action = |u: f64| -> f64 {
                    eval_profit_with_price(challenge, state, ca, b, u, eff_price)
                };

                let mut candidate_u = 0.0_f64;
                let mut best_cand_val = eval_action(0.0);

                if lo < 0.0 {
                    let (u, v) = ternary_search(&eval_action, lo, 0.0_f64.min(hi), hp.ternary_iters);
                    if v > best_cand_val { best_cand_val = v; candidate_u = u; }
                }
                if hi > 0.0 {
                    let (u, v) = ternary_search(&eval_action, 0.0_f64.max(lo), hi, hp.ternary_iters);
                    if v > best_cand_val { candidate_u = u; }
                }

                
                let mut exists = false;
                for &existing in &col_pools[b] {
                    if (existing - candidate_u).abs() < 1e-4 {
                        exists = true;
                        break;
                    }
                }

                if !exists && candidate_u.is_finite() {
                    col_pools[b].push(candidate_u);
                    added_any = true;
                }
            }

            if !added_any { break; } 
        }

        
        let mut actions = best_actions?;

        
        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            actions[b] = actions[b].clamp(lo, hi);
            if !actions[b].is_finite() {
                actions[b] = 0.0;
            }
        }

        Some(actions)
    }

    
    
    
    // targeted_tight_line_polish — generic refinement
    // Runs a targeted LP on the subset of batteries with highest PTDF impact on tight lines.
    // Called when LNS was not applied (used_lns=false). Max 500 pivots.
    fn targeted_tight_line_polish(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        if num_b == 0 || num_l == 0 || num_b > 60 || actions.len() != num_b {
            return None;
        }

        let limits = &challenge.network.flow_limits;
        let mut flows = flows_base.to_vec();
        for l in 0..num_l {
            for &(b, imp) in &ca.ptdf_sparse[l] {
                if b < actions.len() { flows[l] += imp * actions[b]; }
            }
        }

        let base_profit: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, actions[b])).sum();

        let gate = (hp.kkt_cong_threshold - 0.05).max(0.55);
        let relaxed_gate = (gate - 0.10).max(0.45);

        let mut tight_lines: Vec<(f64, usize)> = (0..num_l)
            .filter_map(|l| {
                let limit = limits[l];
                if limit <= 1e-6 { None }
                else {
                    let util = flows[l].abs() / limit;
                    if util >= gate { Some((util, l)) } else { None }
                }
            })
            .collect();

        if tight_lines.is_empty() {
            tight_lines = (0..num_l)
                .filter_map(|l| {
                    let limit = limits[l];
                    if limit <= 1e-6 { None }
                    else {
                        let util = flows[l].abs() / limit;
                        if util >= relaxed_gate { Some((util, l)) } else { None }
                    }
                })
                .collect();
        }

        if tight_lines.is_empty() { return None; }

        tight_lines.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        tight_lines.truncate(if num_b <= 30 { 6 } else { 4 });

        let subset_cap = if num_b <= 24 { 8.min(num_b) } else { 6.min(num_b) };
        let mut ranked: Vec<(f64, usize)> = Vec::new();
        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            let room = (hi - actions[b]).abs() + (actions[b] - lo).abs();
            let mut score = 0.0_f64;
            for &(_, l) in &tight_lines {
                if let Some(&(_, p)) = ca.b_to_lines[b].iter().find(|&&(ll, _)| ll == l) {
                    score += p.abs();
                }
            }
            if score > 1e-8 {
                ranked.push((score * (1.0 + 0.10 * room), b));
            }
        }

        if ranked.len() < 2 { return None; }

        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(subset_cap);
        let subset: Vec<usize> = ranked.iter().map(|&(_, b)| b).collect();

        let n_active = tight_lines.len();
        let n_vars = 2 * subset.len() + n_active;
        let m = 2 * subset.len() + 2 * n_active;

        let mut c_vec = vec![0.0_f64; n_vars];
        let mut a_mat = vec![vec![0.0_f64; n_vars]; m];
        let mut b_vec = vec![0.0_f64; m];

        let dt = 0.25_f64;
        let lp_lambda = hp.lp_soft_lambda;

        for ai in 0..n_active {
            c_vec[2 * subset.len() + ai] = -lp_lambda;
        }

        for (i, &b) in subset.iter().enumerate() {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let (lo, hi) = state.action_bounds[b];
            let soc = state.socs[b];

            let soc_levels = ca.dp[b][0].len();
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1).max(1) as f64;
            let idx_f = (soc - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo_idx = ((idx_f.floor() as isize).max(0) as usize).min(soc_levels.saturating_sub(1));
            let hi_idx = (lo_idx + 1).min(soc_levels.saturating_sub(1));
            let t_next = (state.time_step + 1).min(ca.dp[b].len().saturating_sub(1));
            let dv = if soc_levels >= 2 && delta_s > 1e-12 {
                (ca.dp[b][t_next][hi_idx] - ca.dp[b][t_next][lo_idx]) / delta_s
            } else { 0.0 };

            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);

            c_vec[i]                = (rt - 0.25) * dt - dv * dt / eta_d;
            c_vec[subset.len() + i] = -(rt + 0.25) * dt + dv * eta_c * dt;

            let r = 2 * i;
            a_mat[r][i] = 1.0;
            b_vec[r] = hi.max(0.0);
            a_mat[r + 1][subset.len() + i] = 1.0;
            b_vec[r + 1] = (-lo).max(0.0);
        }

        let row_flow = 2 * subset.len();
        for (ai, &(_, li)) in tight_lines.iter().enumerate() {
            let limit = limits[li];
            let mut flow_others = flows[li];
            for &b in &subset {
                if let Some(&(_, p)) = ca.b_to_lines[b].iter().find(|&&(ll, _)| ll == li) {
                    flow_others -= p * actions[b];
                }
            }
            let rp = row_flow + 2 * ai;
            let rn = rp + 1;

            for (i, &b) in subset.iter().enumerate() {
                if let Some(&(_, p)) = ca.b_to_lines[b].iter().find(|&&(ll, _)| ll == li) {
                    a_mat[rp][i]                += p;
                    a_mat[rp][subset.len() + i] -= p;
                    a_mat[rn][i]                -= p;
                    a_mat[rn][subset.len() + i] += p;
                }
            }

            let viol_idx = 2 * subset.len() + ai;
            a_mat[rp][viol_idx] = -1.0;
            a_mat[rn][viol_idx] = -1.0;

            b_vec[rp] = (limit - flow_others).max(0.0);
            b_vec[rn] = (limit + flow_others).max(0.0);
        }

        let max_pivots = 500;
        let (opt_x, _) = super::lp::lp_solve_with_budget(n_vars, m, &c_vec, &a_mat, &b_vec, max_pivots);

        if let Some(opt_x) = opt_x {
            let mut new_actions = actions.to_vec();
            for (i, &b) in subset.iter().enumerate() {
                let u = opt_x[i] - opt_x[subset.len() + i];
                let (lo, hi) = state.action_bounds[b];
                new_actions[b] = u.clamp(lo, hi);
            }

            let new_profit: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, new_actions[b])).sum();
            if new_profit > base_profit + 1e-9 {
                let mut check_flows = flows_base.to_vec();
                for l in 0..num_l {
                    for &(b, imp) in &ca.ptdf_sparse[l] {
                        check_flows[l] += imp * new_actions[b];
                    }
                }
                let feasible = (0..num_l).all(|l| check_flows[l].abs() <= limits[l] + 1e-6 * limits[l].max(1.0));
                if feasible {
                    return Some(new_actions);
                }
            }
        }

        None
    }

    fn compute_lp_budget_for_step(hp: &TrackHp, consumed: usize, reserve: &mut isize) -> usize {
        let base = hp.lp_per_call_pivots;
        if base == 0 { return 0; }
        let total = hp.lp_total_pivots;
        let remaining = if total > 0 { total.saturating_sub(consumed) } else { base };
        if remaining == 0 { return 0; }

        let reserve_usable = (*reserve).min(base as isize).max(0) as usize;
        let budget = base + reserve_usable;
        budget.min(remaining).min(2 * base)
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

        
        
        if hp.use_kkt {
            for row in expected_premiums.iter_mut() {
                for v in row.iter_mut() { *v = 0.0; }
            }
        }

        
        
        
        
        
        
        let dp = if hp.use_dw && num_l > 0 {
            let prescreen_lines: Vec<usize> = if hp.use_dw_prescreen {
                
                let zero_mu = vec![vec![0.0_f64; num_l]; num_t];
                let dp0 = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &zero_mu);
                let flows_all = ldd_simulate_flows(challenge, state, &dp0, &batt_nodes, &ptdf_sparse);
                prescreen_binding_lines(challenge, &flows_all, hp.dw_max_lines)
            } else {
                Vec::new()
            };
            let ps: Option<&[usize]> = if prescreen_lines.is_empty() { None } else { Some(&prescreen_lines) };
            let mu_dw = build_dw_dual_prices(challenge, state, hp, &batt_nodes, &b_to_lines, ps);
            build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu_dw)
        } else if hp.ldd_iters > 0 && num_l > 0 {
            let mut mu = vec![vec![0.0_f64; num_l]; num_t];
            let mut dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

            let active_lines_ldd: Vec<usize> = (0..num_l).collect();

            if hp.use_ldd_proximal {
                
                let mut prev_average = vec![vec![0.0_f64; num_l]; num_t];
                let mut fallback = false;

                for ldd_iter in 0..hp.ldd_iters {
                    let flows_all = ldd_simulate_flows(challenge, state, &dp, &batt_nodes, &ptdf_sparse);
                    let step_base = hp.ldd_step_size;

                    
                    let mut l2_per_t = vec![0.0_f64; num_t];
                    let mut max_viol = 0.0_f64;
                    for t in 0..num_t {
                        let mut l2_sq = 0.0_f64;
                        for l in 0..num_l {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let abs_f = f.abs();
                            if abs_f > limit {
                                let v = abs_f - limit;
                                l2_sq += v * v;
                                if v > max_viol { max_viol = v; }
                            }
                        }
                        l2_per_t[t] = l2_sq.sqrt();
                    }

                    
                    for t in 0..num_t {
                        let raw_step = step_base / (1.0 + l2_per_t[t]);

                        for &l in &active_lines_ldd {
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let abs_f = f.abs();

                            if abs_f > limit {
                                let v = abs_f - limit;
                                let sign = f.signum();
                                let delta = raw_step * v;
                                let max_delta = limit * hp.ldd_clip_fraction;
                                let clamped_delta = delta.min(max_delta);

                                let eff_delta = if ldd_iter == 0 {
                                    prev_average[t][l] = clamped_delta;
                                    clamped_delta
                                } else {
                                    let avg_delta = hp.ldd_momentum * prev_average[t][l] + (1.0 - hp.ldd_momentum) * clamped_delta;
                                    prev_average[t][l] = avg_delta;
                                    avg_delta
                                };

                                mu[t][l] = (mu[t][l] + sign * eff_delta).abs();
                            }
                        }
                    }

                    dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

                    
                    if ldd_iter == 0 {
                        let mut any_positive = 0usize;
                        for b in 0..num_b.min(20) {
                            let node = batt_nodes[b];
                            let da = if node < challenge.market.day_ahead_prices.len() && 0 < challenge.market.day_ahead_prices[node].len() {
                                challenge.market.day_ahead_prices[node][0]
                            } else { 0.0 };
                            let cong_adj: f64 = b_to_lines[b].iter()
                                .map(|&(l, impact)| if l < mu[0].len() { mu[0][l] * impact.abs() } else { 0.0 })
                                .sum();
                            let p_eff = da - cong_adj;
                            if p_eff > 0.0 { any_positive += 1; }
                        }
                        if any_positive < (num_b.min(20)).max(1) / 2 {
                            fallback = true;
                        }
                    }

                    if max_viol < 1e-4 { break; }
                }

                
                if fallback {
                    mu = vec![vec![0.0_f64; num_l]; num_t];
                    dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);
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
                        dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);
                        if max_viol < 1e-4 { break; }
                    }
                }
            } else {
                
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

                    dp = build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu);

                    if max_viol < 1e-4 { break; }
                }
            }
            dp
        } else if hp.max_admm_iters > 0 && hp.anticipate_lmp && num_l > 0 {
            
            
            
            let mut hp_cheap = hp.clone();
            hp_cheap.soc_levels = 31;
            hp_cheap.action_grid = 15;
            let mu_zero = vec![vec![0.0_f64; num_l]; num_t];
            let dp_low = build_dp_parallel_or_serial(challenge, &hp_cheap, &batt_nodes, &expected_premiums, &b_to_lines, &mu_zero);
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
            build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu)
        } else {
            let mu = vec![vec![0.0_f64; num_l]; num_t];
            build_dp_parallel_or_serial(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu)
        };

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

    fn run_dual_ascent(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
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

                        
                        
                        let cong_spread: f64 = ca.b_to_lines[b].iter()
                            .map(|&(ll, imp2)| {
                                let lim2 = (challenge.network.flow_limits[ll] - hp.flow_margin).max(0.0);
                                if flows[ll].abs() > lim2 { imp2.abs() } else { 0.0 }
                            })
                            .sum();

                        
                        let score = roi / (1.0 + cong_spread);
                        culprits.push((b, contrib, score));
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    #[inline]
    fn eval_profit_with_price(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
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
        ca: &TitanCache,
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
        ca: &TitanCache,
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
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
        warm_init: &[f64],
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

        
        let mut actions_asca = if warm_init.len() == num_b { warm_init.to_vec() } else { vec![0.0_f64; num_b] };
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
        ca: &TitanCache,
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

    
    
    
    fn select_active_lines(challenge: &Challenge, flows_base: &[f64], k: usize) -> Vec<usize> {
        let num_l = challenge.network.flow_limits.len();
        if k >= num_l {
            return (0..num_l).collect();
        }
        let mut scores: Vec<(f64, usize)> = (0..num_l)
            .filter_map(|l| {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { None }
                else {
                    let ratio = flows_base.get(l).copied().unwrap_or(0.0).abs() / limit;
                    Some((ratio, l))
                }
            })
            .collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores.into_iter().map(|(_, l)| l).collect()
    }

    
    
    
    fn joint_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        flows_base: &[f64],
        lp_lambda: f64,
        max_pivots: usize,
        lp_hp: &TrackHp,
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let dt = 0.25_f64;

        
        let active_lines = select_active_lines(challenge, flows_base, lp_hp.lp_max_lines);
        let n_active = active_lines.len();

        
        let n = 2 * num_b + n_active;
        
        let m = 4 * num_b + 2 * n_active;

        let mut c_vec = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        let t_next = (state.time_step + 1).min(ca.dp[0].len() - 1);

        
        for ai in 0..n_active {
            c_vec[2 * num_b + ai] = -lp_lambda;
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
        for (ai, &l) in active_lines.iter().enumerate() {
            let limit = challenge.network.flow_limits[l];
            let exo = flows_base[l];
            let viol_idx = 2 * num_b + ai;
            let rp = row_f + 2 * ai;     
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

    
    
    fn joint_lp_dispatch_with_used(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        flows_base: &[f64],
        lp_lambda: f64,
        max_pivots: usize,
        lp_hp: &TrackHp,
    ) -> (Option<Vec<f64>>, usize) {
        let num_b = challenge.num_batteries;
        let dt = 0.25_f64;

        
        let active_lines = select_active_lines(challenge, flows_base, lp_hp.lp_max_lines);
        let n_active = active_lines.len();

        let n = 2 * num_b + n_active;
        let m = 4 * num_b + 2 * n_active;

        let mut c_vec = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        let t_next = (state.time_step + 1).min(ca.dp[0].len() - 1);

        for ai in 0..n_active {
            c_vec[2 * num_b + ai] = -lp_lambda;
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
        for (ai, &l) in active_lines.iter().enumerate() {
            let limit = challenge.network.flow_limits[l];
            let exo = flows_base[l];
            let viol_idx = 2 * num_b + ai;
            let rp = row_f + 2 * ai;
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

        let (opt_x, pivots_used) = super::lp::lp_solve_with_budget(n, m, &c_vec, &a_mat, &b_vec, max_pivots);
        let Some(opt_x) = opt_x else { return (None, 0); };

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let u = opt_x[b] - opt_x[num_b + b];
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }
        (Some(actions), pivots_used)
    }

    
    
    
    
    
    
    fn mpc_dispatch_2step(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let t = state.time_step;
        let hz = hp.mpc_horizon.min(2); 
        let remaining = challenge.num_steps.saturating_sub(t);
        if remaining < 2 || hz < 2 {
            return None; 
        }
        let dt = 0.25_f64;
        let lambda_soft = hp.lp_soft_lambda;
        let t2 = (t + 2).min(challenge.num_steps);

        
        
        
        
        
        let n = 4 * num_b + 4 * num_l;

        
        
        
        
        let m = 4 * num_b + 4 * num_l;

        let mut c_vec = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        
        let mut lambdas = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let soc = state.socs[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let t2_idx = t2.min(ca.dp[b].len() - 1);
            lambdas[b] = dp_lambda(&ca.dp[b], t2_idx, soc, bat.soc_min_mwh, soc_span, soc_levels);
            if !lambdas[b].is_finite() {
                lambdas[b] = 0.0;
            }
        }

        
        let da_t = if t < challenge.market.day_ahead_prices.len() {
            &challenge.market.day_ahead_prices[t]
        } else {
            challenge.market.day_ahead_prices.last().unwrap_or(&challenge.market.day_ahead_prices[0])
        };
        let da_t1 = if t + 1 < challenge.market.day_ahead_prices.len() {
            &challenge.market.day_ahead_prices[t + 1]
        } else {
            da_t
        };

        
        let exo_flows_t = flows_base;
        let exo_flows_t1;
        {
            let exo_inj = if t + 1 < challenge.exogenous_injections.len() {
                &challenge.exogenous_injections[t + 1]
            } else {
                challenge.exogenous_injections.last().unwrap_or(&challenge.exogenous_injections[t])
            };
            exo_flows_t1 = challenge.network.compute_flows(exo_inj);
        }
        let exo_for_h = [exo_flows_t, &exo_flows_t1];

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let (u_min, u_max) = state.action_bounds[b];
            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let ub_d = u_max.max(0.0);
            let ub_c = (-u_min).max(0.0);

            let lambda_b = lambdas[b];

            for h in 0..2 {
                let da_p = if node < da_t.len() {
                    if h == 0 { da_t[node] } else { da_t1[node] }
                } else {
                    if h == 0 { da_t[0] } else { da_t1[0] }
                };

                let h_base = 2 * num_b * h;
                let d_idx = h_base + b;
                let c_idx = h_base + num_b + b;

                
                
                
                c_vec[d_idx] = (da_p - 0.25) * dt - lambda_b * dt / eta_d;
                c_vec[c_idx] = (-da_p - 0.25) * dt + lambda_b * eta_c * dt;

                
                
                
                let r_d = 2 * h * num_b + b;
                let r_c = 2 * h * num_b + num_b + b;

                
                a_mat[r_d][d_idx] = 1.0;
                b_vec[r_d] = ub_d;

                
                a_mat[r_c][c_idx] = 1.0;
                b_vec[r_c] = ub_c;
            }
        }

        
        for h in 0..2 {
            let exo_h = exo_for_h[h];
            let constraint_base = 4 * num_b + 2 * h * num_l;

            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                let exo = exo_h[l];

                let v_plus_idx = 4 * num_b + 2 * h * num_l + l;
                let v_minus_idx = 4 * num_b + (2 * h + 1) * num_l + l;

                let rp = constraint_base + 2 * l;
                let rn = constraint_base + 2 * l + 1;

                let h_base = 2 * num_b * h;
                for &(b, impact) in &ca.ptdf_sparse[l] {
                    let d_idx = h_base + b;
                    let c_idx = h_base + num_b + b;
                    a_mat[rp][d_idx] += impact;
                    a_mat[rp][c_idx] -= impact;
                    a_mat[rn][d_idx] -= impact;
                    a_mat[rn][c_idx] += impact;
                }

                
                a_mat[rp][v_plus_idx] = -1.0;
                a_mat[rn][v_minus_idx] = -1.0;

                
                c_vec[v_plus_idx] = -lambda_soft;
                c_vec[v_minus_idx] = -lambda_soft;

                
                b_vec[rp] = (limit - exo).max(0.0);
                b_vec[rn] = (limit + exo).max(0.0);
            }
        }

        let pivots = hp.mpc_pivot_budget.max(100);
        let (opt_x, _) = super::lp::lp_solve_with_budget(n, m, &c_vec, &a_mat, &b_vec, pivots);
        let opt_x = opt_x?;

        
        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let d0 = opt_x[b];
            let c0 = opt_x[num_b + b];
            let u = d0 - c0;
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }

        Some(actions)
    }

    fn lp_dispatch_with_dp(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        flows_base: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let n = 2 * num_b;
        let m = 2 * num_b + 2 * num_l;

        let mut a_mat: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
        let mut b_vec: Vec<f64> = vec![0.0; m];
        let mut c_vec: Vec<f64> = vec![0.0; n];

        let t_next = (state.time_step + 1).min(ca.dp[0].len() - 1);

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let (u_min, u_max) = state.action_bounds[b];
            let u_max_pos = u_max.max(0.0);
            let u_max_neg = (-u_min).max(0.0);

            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1) as f64;
            let idx_f = (state.socs[b] - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo = (idx_f.floor() as isize).max(0) as usize;
            let hi = (lo + 1).min(soc_levels - 1);
            let v_lo = ca.dp[b][t_next][lo];
            let v_hi = ca.dp[b][t_next][hi];
            let dv = (v_hi - v_lo) / delta_s;

            
            
            
            c_vec[b] = (rt - 0.25) * dt - dv * dt / eta_d;
            c_vec[num_b + b] = (-rt - 0.25) * dt + dv * eta_c * dt;

            
            a_mat[2 * b][b] = 1.0;
            b_vec[2 * b] = u_max_pos;
            a_mat[2 * b + 1][num_b + b] = 1.0;
            b_vec[2 * b + 1] = u_max_neg;
        }

        let offset = 2 * num_b;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            let f_exo = flows_base[l];
            let r1 = offset + 2 * l;
            let r2 = offset + 2 * l + 1;
            for &(b, impact) in &ca.ptdf_sparse[l] {
                a_mat[r1][b] += impact;
                a_mat[r1][num_b + b] -= impact;
                a_mat[r2][b] -= impact;
                a_mat[r2][num_b + b] += impact;
            }
            
            b_vec[r1] = limit - f_exo;
            b_vec[r2] = limit + f_exo;
        }

        let opt_x = super::lp::lp_solve(n, m, &c_vec, &a_mat, &b_vec)?;

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let u = opt_x[b] - opt_x[num_b + b];
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }

        
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            let mut flow = flows_base[l];
            for &(b, imp) in &ca.ptdf_sparse[l] { flow += imp * actions[b]; }
            if flow.abs() > limit + 1e-3 { return None; }
        }
        Some(actions)
    }

    
    
    
    
    
    
    
    
    
    

    const POLICY_NUM_SEGMENTS: usize = 4;
    const POLICY_NUM_FEATURES: usize = 6;
    const POLICY_PARAMS_PER_BATT: usize = POLICY_NUM_SEGMENTS * (POLICY_NUM_FEATURES + 1);

    
    fn da_price_norm(p: f64) -> f64 {
        (p - 100.0) / 50.0
    }

    
    fn rt_da_delta_norm(d: f64) -> f64 {
        d / 30.0
    }

    
    
    
    fn eval_policy_for_battery(weights: &[f64], features: &[f64], norm_soc: f64, action_lo: f64, action_hi: f64) -> f64 {
        let seg_f = (norm_soc * POLICY_NUM_SEGMENTS as f64).floor();
        let seg = (seg_f as usize).min(POLICY_NUM_SEGMENTS - 1);
        let base = seg * (POLICY_NUM_FEATURES + 1);

        let mut action = weights[base + POLICY_NUM_FEATURES]; 
        for f in 0..POLICY_NUM_FEATURES {
            action += weights[base + f] * features[f];
        }

        action.clamp(action_lo, action_hi)
    }

    
    pub fn policy_dispatch(
        challenge: &Challenge,
        state: &State,
        weights: &[Vec<f64>],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let mut actions = vec![0.0_f64; num_b];

        
        let exo_inj = &challenge.exogenous_injections[state.time_step.min(challenge.exogenous_injections.len().saturating_sub(1))];
        let exo_flows = challenge.network.compute_flows(exo_inj);
        let mut max_util = 0.0_f64;
        for l in 0..challenge.network.flow_limits.len() {
            let limit = challenge.network.flow_limits[l];
            if limit > 1e-6 {
                let u = exo_flows[l].abs() / limit;
                if u > max_util { max_util = u; }
            }
        }
        let congestion_bin = if max_util < 0.5 { 0.0 } else if max_util < 0.8 { 0.5 } else { 1.0 };

        for b in 0..num_b {
            if b >= weights.len() || weights[b].len() < POLICY_PARAMS_PER_BATT {
                continue;
            }

            let bat = &challenge.batteries[b];
            let soc = state.socs[b];
            let soc_range = bat.soc_max_mwh - bat.soc_min_mwh;
            let norm_soc = if soc_range > 1e-9 { (soc - bat.soc_min_mwh) / soc_range } else { 0.5 };

            let time_fraction = if challenge.num_steps > 1 {
                state.time_step as f64 / challenge.num_steps as f64
            } else { 0.0 };

            let node = challenge.batteries[b].node;
            let node_idx = if node < state.rt_prices.len() { node } else { state.rt_prices.len() - 1 };
            let rt_price = state.rt_prices[node_idx];
            let da_node = challenge.market.day_ahead_prices.len().saturating_sub(1);
            let da_node_idx = if node < da_node || node == 0 { node.min(da_node) } else { da_node };
            let da_vec = &challenge.market.day_ahead_prices[da_node_idx];
            let t_idx = if state.time_step < da_vec.len() { state.time_step } else { da_vec.len() - 1 };
            let da_price = da_vec[t_idx];
            let rt_da_delta = rt_price - da_price;

            let (lo, hi) = state.action_bounds[b];
            let action_range = hi - lo;

            let features: [f64; POLICY_NUM_FEATURES] =
                [norm_soc, time_fraction, da_price_norm(da_price), rt_da_delta_norm(rt_da_delta), congestion_bin, action_range];

            actions[b] = eval_policy_for_battery(&weights[b], &features, norm_soc, lo, hi);
        }

        actions
    }

    
    
    

    
    fn policy_rand_f64() -> f64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEED: AtomicU64 = AtomicU64::new(12345);
        let mut s = SEED.fetch_add(7, Ordering::Relaxed).wrapping_add(1);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f64) / (u32::MAX as f64)
    }

    
    fn policy_randn() -> f64 {
        thread_local! { static SPARE: RefCell<Option<f64>> = RefCell::new(None); }
        SPARE.with(|sp| {
            let mut guard = sp.borrow_mut();
            if let Some(s) = guard.take() { return s; }
            let u1 = policy_rand_f64().max(1e-15);
            let u2 = policy_rand_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let n1 = r * (2.0 * std::f64::consts::PI * u2).cos();
            let n2 = r * (2.0 * std::f64::consts::PI * u2).sin();
            *guard = Some(n2);
            n1
        })
    }

    
    fn simulate_policy(
        challenge: &Challenge,
        initial_state: &State,
        cache: &TitanCache,
        hp: &TrackHp,
        weights: &[Vec<f64>],
    ) -> f64 {
        let num_t = challenge.num_steps;
        let num_b = challenge.num_batteries;
        let mut socs = initial_state.socs.clone();
        let mut total_profit = 0.0_f64;

        for t in 0..num_t {
            let rt_prices: Vec<f64> = challenge.market.day_ahead_prices.iter().map(|np| {
                if t < np.len() { np[t] } else { *np.last().unwrap_or(&0.0) }
            }).collect();

            let mut actions = vec![0.0_f64; num_b];
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc = socs[b];
                let max_ch = bat.power_charge_mw * 0.25;
                let max_dis = bat.power_discharge_mw * 0.25;
                let charge_limit = -(soc - bat.soc_min_mwh) / bat.efficiency_charge.max(1e-9);
                let disch_limit = (bat.soc_max_mwh - soc) * bat.efficiency_discharge.max(1e-9) / 0.25;
                let lo = (-max_ch).max(charge_limit);
                let hi = max_dis.min(disch_limit);
                let (lo, hi) = (lo, hi);

                
                let state_t = State {
                    time_step: t, socs: socs.clone(), rt_prices: rt_prices.clone(),
                    exogenous_injections: challenge.exogenous_injections[t].clone(),
                    action_bounds: vec![(lo, hi); num_b], total_profit,
                };
                actions[b] = 0.0; 
            }

            
            actions = policy_dispatch(challenge, &State {
                time_step: t, socs: socs.clone(), rt_prices: rt_prices.clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: (0..num_b).map(|b| {
                    let bat = &challenge.batteries[b];
                    let soc = socs[b];
                    let lo = (-(bat.power_charge_mw * 0.25)).max(-(soc - bat.soc_min_mwh) / bat.efficiency_charge.max(1e-9));
                    let hi = (bat.power_discharge_mw * 0.25).min((bat.soc_max_mwh - soc) * bat.efficiency_discharge.max(1e-9) / 0.25);
                    (lo, hi)
                }).collect(),
                total_profit,
            }, weights);

            
            let zero_action = vec![0.0_f64; num_b];
            let inj_base = challenge.compute_total_injections(&State {
                time_step: t, socs: socs.clone(), rt_prices: rt_prices.clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: actions.iter().map(|&a| (a, a)).collect(),
                total_profit,
            }, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base);
            run_deflator(challenge, &State {
                time_step: t, socs: socs.clone(), rt_prices: rt_prices.clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: (0..num_b).map(|_| (f64::NEG_INFINITY, f64::INFINITY)).collect(),
                total_profit,
            }, cache, hp, &flows_base, &mut actions);

            
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let node = cache.batt_nodes[b].min(rt_prices.len() - 1);
                let rt_price = rt_prices[node];
                let u = actions[b];
                let dt = 0.25_f64;
                let revenue = u * rt_price * dt;
                let tx = 0.25 * u.abs() * dt;
                let deg_base = (u.abs() * dt) / bat.capacity_mwh.max(1e-9);
                total_profit += revenue - tx - deg_base * deg_base;

                let next_soc = if u < 0.0 {
                    socs[b] + bat.efficiency_charge * (-u) * dt
                } else {
                    socs[b] - u / bat.efficiency_discharge.max(1e-9) * dt
                };
                socs[b] = next_soc.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            }
        }

        total_profit
    }

    
    pub fn train_policy_cmaes(
        challenge: &Challenge,
        initial_state: &State,
        cache: &TitanCache,
        hp: &TrackHp,
    ) -> Vec<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let dpb = POLICY_PARAMS_PER_BATT;
        let dim = num_b * dpb;

        let mut mean = vec![0.0; dim];
        let mut sigma = 1.0;
        let popsize = (20 + (3.0 * dim as f64).sqrt() as usize).min(60);
        let mu = popsize / 2;
        let max_gens = 15;

        
        let mut w: Vec<f64> = vec![0.0; mu];
        let mut w_sum = 0.0;
        for i in 0..mu {
            w[i] = (mu as f64 + 1.0 - i as f64).ln() / (mu as f64 + 1.0 - (i + 1) as f64).ln();
            w[i] = 1.0 / (mu as f64 / 2.0 + w[i].exp());
            w_sum += w[i];
        }
        for i in 0..mu { w[i] /= w_sum; }
        let eff_popsize: f64 = w_sum * w_sum / (0..mu as usize).map(|i| w[i] * w[i]).sum::<f64>();

        let cc = (4.0 + 2.0 / (dim as f64 + 4.0)) / (dim as f64 + 4.0 + 2.0 * (1.0 - 2.0 / (dim as f64 + 4.0)).sqrt());
        let cs = (eff_popsize + 2.0) / (dim as f64 + eff_popsize + 2.0);
        let ca = cs / (cs + 2.0);
        let dc = (1.0 + 2.0 * ((eff_popsize - 1.0) / (dim as f64 + 1.0)).sqrt() + cs) / (dim as f64 + cs);
        let ds = (1.0 + 2.0 * ((eff_popsize - 1.0) / (dim as f64 + 1.0)).sqrt() + cs) / (dim as f64 + cs);

        let mut dc_acc = 1.0_f64;
        let mut ds_acc = 1.0_f64;
        let mut pc = vec![0.0; dim];

        let mut best_weights = mean.clone();
        let mut best_fitness = f64::NEG_INFINITY;
        let ec = 2.0 + 2.0 / (dim as f64 + 1.0);

        for _gen in 0..max_gens {
            
            let mut fitnesses = Vec::with_capacity(popsize);
            for _ in 0..popsize {
                let mut ind = mean.clone();
                for d in 0..dim {
                    ind[d] += sigma * policy_randn();
                }
                let weights_batt: Vec<Vec<f64>> = (0..num_b)
                    .map(|b| ind[b * dpb..(b + 1) * dpb].to_vec())
                    .collect();
                let fit = simulate_policy(challenge, initial_state, cache, hp, &weights_batt);
                fitnesses.push((fit, ind));
            }

            
            fitnesses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).reverse());

            if fitnesses[0].0 > best_fitness {
                best_fitness = fitnesses[0].0;
                best_weights = fitnesses[0].1.clone();
            }

            
            let mut m_new = mean.clone();
            for i in 0..mu {
                for d in 0..dim {
                    m_new[d] += w[i] * fitnesses[i].1[d];
                }
            }

            
            let chi_n = (dim as f64).sqrt() * (1.0 - 1.0 / dim as f64 + (dim as f64) * (1.0 - 2.0 / (dim as f64 + 4.0)).sqrt() / (dim as f64 - 1.0)).sqrt();
            for d in 0..dim {
                pc[d] = (1.0 - dc) * pc[d] + (dc * (eff_popsize / ec).sqrt()).min(1.0) * (m_new[d] - mean[d]) / sigma;
            }

            
            let pc_norm: f64 = pc.iter().map(|&x| x * x).sum::<f64>().sqrt();
            ds_acc = (1.0 - ds) * ds_acc + ds * (pc_norm / chi_n);
            sigma *= (ds_acc - 1.0) * cs * ca;
            sigma = sigma.max(1e-10).min(100.0);

            mean = m_new;
        }

        (0..num_b)
            .map(|b| best_weights[b * dpb..(b + 1) * dpb].to_vec())
            .collect()
    }

    
    
    
    
    
    
    

    fn build_dp_parallel_or_serial(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
    ) -> Vec<Vec<Vec<f64>>> {
        let num_b = challenge.num_batteries;
        let mut dp: Vec<Vec<Vec<f64>>>;

        if num_b <= 1 || !hp.use_parallel_dp {
            dp = build_dp_with_mu(challenge, hp, batt_nodes, expected_premiums, b_to_lines, mu);
        } else {
            let num_workers = num_b.min(32);
            let chunk_size = (num_b + num_workers - 1) / num_workers;

            let arc_challenge: Arc<Challenge> = Arc::new(challenge.clone());
            let arc_hp: Arc<TrackHp> = Arc::new(hp.clone());
            let arc_batt_nodes: Arc<Vec<usize>> = Arc::new(batt_nodes.to_vec());
            let arc_expected_premiums: Arc<Vec<Vec<f64>>> = Arc::new(expected_premiums.to_vec());
            let arc_b_to_lines: Arc<Vec<Vec<(usize, f64)>>> = Arc::new(b_to_lines.to_vec());
            let arc_mu: Arc<Vec<Vec<f64>>> = Arc::new(mu.to_vec());
            let _ = chunk_size;

            dp = vec![vec![vec![0.0_f64; hp.soc_levels]; challenge.num_steps + 1]; num_b];
            for b in 0..num_b {
                let tbl = build_dp_with_mu_for_battery(
                    &*arc_challenge, &*arc_hp, &*arc_batt_nodes,
                    &*arc_expected_premiums, &*arc_b_to_lines, &*arc_mu, b,
                );
                if !tbl.is_empty() {
                    dp[b] = tbl;
                }
            }
        }

        
        
        if hp.use_sdp {
            for b in 0..num_b {
                dp[b] = build_dp_stochastic_for_battery(
                    challenge, hp, batt_nodes, expected_premiums, b_to_lines, mu, b,
                );
            }
        }

        dp
    }

    
    
    fn build_dp_with_mu_for_battery(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
        b: usize,
    ) -> Vec<Vec<f64>> {
        let num_t = challenge.num_steps;
        let soc_levels = hp.soc_levels;
        let action_grid = hp.action_grid;
        let dt = 0.25_f64;

        let mut dp = vec![vec![0.0_f64; soc_levels]; num_t + 1];

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
                    let v_next = dp[t + 1][idx0c] * (1.0 - frac)
                        + dp[t + 1][idx1c] * frac;

                    let val = profit + v_next;
                    if val > max_val { max_val = val; }
                }
                dp[t][i] = max_val;
            }
        }
        dp
    }

    
    
    
    
    
    
    
    
    

    fn gh_quadrature(k: usize) -> (Vec<f64>, Vec<f64>) {
        match k {
            3 => (
                vec![-1.732_050_807_568_877_2, 0.0, 1.732_050_807_568_877_2],
                vec![0.166_666_666_666_666_7, 0.666_666_666_666_666_6, 0.166_666_666_666_666_7],
            ),
            _ => (
                vec![-1.0, 1.0],
                vec![0.5, 0.5],
            ),
        }
    }

    fn build_dp_stochastic_for_battery(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
        b: usize,
    ) -> Vec<Vec<f64>> {
        let num_t = challenge.num_steps;
        let soc_levels = hp.soc_levels;
        let action_grid = hp.action_grid;
        let dt = 0.25_f64;

        let bat = &challenge.batteries[b];
        let node = batt_nodes[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);

        let sigma: f64 = 0.15;
        let rho_jump: f64 = 0.02;
        let alpha: f64 = 3.5;

        let jump_prem_factor = if rho_jump > 0.0 && alpha > 1.0 {
            rho_jump * alpha / (alpha - 1.0)
        } else {
            0.0
        };

        let mut dp = vec![vec![0.0_f64; soc_levels]; num_t + 1];

        
        let std_gh = gh_quadrature(hp.sdp_k);

        for t in (0..num_t).rev() {
            let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                challenge.market.day_ahead_prices[t][node]
            } else {
                challenge.market.day_ahead_prices[t][0]
            };

            
            let (nodes_q, weights_q) = if hp.use_tail_quadrature {
                build_tail_mixture_quadrature(p_da, sigma, rho_jump, alpha)
            } else {
                std_gh.clone()
            };
            let K = nodes_q.len();

            let mu_adjust: f64 = b_to_lines[b].iter()
                .map(|&(l, impact)| {
                    if l < mu[t].len() { mu[t][l] * impact } else { 0.0 }
                })
                .sum();

            let extra = expected_premiums[t][b] - mu_adjust;
            let p_sell_base = p_da * (1.0 + hp.jump_premium) + extra;
            let p_buy_base = p_da + extra;

            let max_pwr_c = bat.power_charge_mw * hp.network_derating;
            let max_pwr_d = bat.power_discharge_mw * hp.network_derating;

            let base_abs = p_da.abs().max(1e-6);
            let scene_perturb: Vec<f64> = nodes_q.iter().map(|&z| {
                sigma * z * base_abs + base_abs * jump_prem_factor
            }).collect();

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
                    let v_next = dp[t + 1][idx0c] * (1.0 - frac) + dp[t + 1][idx1c] * frac;
                    if !v_next.is_finite() {
                        continue;
                    }

                    let mut exp_val = 0.0_f64;
                    for k in 0..K {
                        let perturbation = scene_perturb[k];
                        let price = if u > 0.0 {
                            p_sell_base + perturbation
                        } else {
                            p_buy_base + perturbation
                        };
                        let abs_u = u.abs();
                        let revenue = u * price * dt;
                        let tx = 0.25 * abs_u * dt;
                        let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                        let deg = deg_base * deg_base;
                        let profit_scenario = revenue - tx - deg;
                        exp_val += weights_q[k] * (profit_scenario + v_next);
                    }

                    if exp_val.is_finite() && exp_val > max_val {
                        max_val = exp_val;
                    }
                }

                if max_val == f64::NEG_INFINITY {
                    let single_price = if p_sell_base.is_sign_positive() {
                        p_sell_base
                    } else {
                        p_buy_base
                    };
                    let mid_u = (u_max + u_min) * 0.5;
                    let abs_mid = mid_u.abs();
                    let rev = mid_u * single_price * dt;
                    let tx = 0.25 * abs_mid * dt;
                    let deg_base = (abs_mid * dt) / bat.capacity_mwh.max(1e-9);
                    max_val = (rev - tx - deg_base * deg_base).max(0.0);
                }

                dp[t][i] = max_val;
            }
        }

        dp
    }

    
    
    
    
    
    fn build_tail_mixture_quadrature(
        _da_price: f64,
        sigma: f64,
        rho_jump: f64,
        alpha: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        
        if alpha <= 1.0 || sigma <= 0.0 || rho_jump <= 0.0 || rho_jump >= 1.0 {
            return gh_quadrature(3);
        }

        
        let mean_pareto = alpha / (alpha - 1.0);
        if !mean_pareto.is_finite() {
            return gh_quadrature(3);
        }

        
        let w_norm = 1.0 - rho_jump;
        let w_jump = rho_jump;

        
        let (gh2_nodes, gh2_weights) = gh_quadrature(2);

        let mut nodes = Vec::with_capacity(3);
        let mut weights = Vec::with_capacity(3);

        
        for (n, w) in gh2_nodes.iter().zip(gh2_weights.iter()) {
            nodes.push(*n);
            weights.push(w_norm * w);
        }

        
        
        
        let jump_node = mean_pareto / sigma;
        let jump_node = if jump_node.is_finite() && jump_node > 0.0 { jump_node } else { 2.0 };
        nodes.push(jump_node);
        weights.push(w_jump);

        
        let wsum: f64 = weights.iter().sum();
        if !wsum.is_finite() || wsum.abs() < 1e-12 {
            return gh_quadrature(3);
        }

        (nodes, weights)
    }

    
    
    
    
    
    
    
    
    fn refine_solution(
        challenge: &Challenge,
        state: &State,
        actions: Vec<f64>,
        fuel_budget: u64,
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let ptdf_b: Vec<Vec<(usize, f64)>> = (0..num_b).map(|b| {
            let node = challenge.batteries[b].node;
            (0..num_l).filter_map(|l| {
                let imp = challenge.network.ptdf[l][node];
                if imp.abs() > 1e-12 { Some((l, imp)) } else { None }
            }).collect()
        }).collect();

        
        let fuel_budget_p1 = fuel_budget * 2 / 3;
        let (limits, ptdf) = (
            &challenge.network.flow_limits,
            &challenge.network.ptdf,
        );

        let feasible = |act: &[f64]| -> bool {
            let inj = challenge.compute_total_injections(state, act);
            let flows: Vec<f64> = (0..num_l)
                .map(|l| (0..challenge.network.num_nodes).map(|k| ptdf[l][k] * inj[k]).sum())
                .collect();
            flows.iter().enumerate().all(|(l, &f)| f.abs() <= limits[l] + 1e-6 * limits[l])
        };

        let mut actions = actions;
        let mut best_profit_p1 = challenge.compute_profit(state, &actions);

        let cost_per_eval = 10;
        let max_steps_p1 = (fuel_budget_p1 as usize).saturating_div(cost_per_eval).min(200);
        let mut steps_used = 0;

        for _pass in 0..3 {
            if steps_used >= max_steps_p1 { break; }
            for b in 0..num_b {
                if steps_used >= max_steps_p1 { break; }
                let (lo, hi) = state.action_bounds[b];
                let range = (hi - lo).max(1e-6);
                let delta = (range * 0.02).min(0.5);
                for &dir in &[-1.0_f64, 1.0] {
                    steps_used += 1;
                    let new_val = (actions[b] + delta * dir).clamp(lo, hi);
                    if (new_val - actions[b]).abs() < 1e-9 { continue; }
                    let mut candidate = actions.clone();
                    candidate[b] = new_val;
                    if !feasible(&candidate) { continue; }
                    let profit = challenge.compute_profit(state, &candidate);
                    if profit > best_profit_p1 {
                        best_profit_p1 = profit;
                        actions = candidate;
                    }
                }
            }
        }

        
        let fuel_budget_p2 = fuel_budget / 3;
        let mut flows = challenge.network.compute_flows(&challenge.compute_total_injections(state, &actions));
        let mut best_actions = actions.clone();
        let mut best_profit = challenge.compute_profit(state, &best_actions);

        let mut mu_pos = vec![0.0_f64; num_l];
        let mut mu_neg = vec![0.0_f64; num_l];
        let mut step_size = 0.1;
        let max_it1 = ((fuel_budget_p2 as usize) / (num_b.max(1) * 10)).min(20).max(5);
        for _ in 0..max_it1 {
            for b in 0..num_b {
                let (lo, hi) = state.action_bounds[b];
                let node = challenge.batteries[b].node;
                let rt = *state.rt_prices.get(node).unwrap_or(&0.0);
                let pen: f64 = ptdf_b[b].iter().map(|&(l, i)| i * (mu_pos[l] - mu_neg[l])).sum();
                let dk = (dt / challenge.batteries[b].capacity_mwh.max(1e-9)).powi(2);
                let mut bu = actions[b];
                let eval_fn = |u: f64| -> f64 {
                    u.abs() * (if u > 0.0 { rt } else { -rt }) * dt - 0.25_f64 * u.abs() * dt - dk * u * u - pen * u
                };
                let mut bv = eval_fn(bu);
                if lo < 0.0 && lo < hi.min(0.0) {
                    let b_c = dt * (-rt - 0.25) - pen;
                    let c = if dk > 1e-12 { (-b_c / (2.0 * dk)).clamp(lo, hi.min(0.0)) } else { lo };
                    let v = eval_fn(c); if v > bv { bv = v; bu = c; }
                    let v = eval_fn(lo); if v > bv { bv = v; bu = lo; }
                }
                if hi > 0.0 && hi > lo.max(0.0) {
                    let b_d = dt * (rt - 0.25) - pen;
                    let c = if dk > 1e-12 { (b_d / (2.0 * dk)).clamp(lo.max(0.0), hi) } else { hi };
                    let v = eval_fn(c); if v > bv { bv = v; bu = c; }
                    let v = eval_fn(hi); if v > bv { bv = v; bu = hi; }
                }
                let d = bu - actions[b];
                if d.abs() > 1e-8 { for &(l, i) in &ptdf_b[b] { flows[l] += i * d; } actions[b] = bu; }
            }
            let mut mv = 0.0_f64;
            for l in 0..num_l {
                let lim = challenge.network.flow_limits[l]; if lim <= 1e-6 { continue; }
                let vp = (flows[l] - lim).max(0.0); let vn = (-flows[l] - lim).max(0.0);
                mu_pos[l] = (mu_pos[l] + step_size * vp).min(500.0); mu_neg[l] = (mu_neg[l] + step_size * vn).min(500.0);
                mv = mv.max(vp).max(vn);
            }
            step_size *= 0.85;
            let p = challenge.compute_profit(state, &actions);
            if p > best_profit { best_profit = p; best_actions = actions.clone(); }
            if mv < 1e-4 { break; }
        }
        actions = best_actions.clone();

        let max_sw = ((fuel_budget_p2 / 2) as usize / num_b.max(1)).min(2);
        let step_frac = 0.005;
        let mut flows = challenge.network.compute_flows(&challenge.compute_total_injections(state, &actions));
        for _ in 0..max_sw {
            let mut improved = false;
            for b in 0..num_b {
                let (lo, hi) = state.action_bounds[b];
                let base = actions[b];
                let mut best_u = base;
                let mut best_prof = challenge.compute_profit(state, &actions);
                let range = (hi - lo).max(1e-6);
                for delta in [step_frac * range, -step_frac * range] {
                    let u = (base + delta).clamp(lo, hi);
                    if (u - base).abs() < 1e-8 { continue; }
                    let mut feas = true;
                    for &(l, imp) in &ptdf_b[b] {
                        if (flows[l] + imp * (u - base)).abs() > challenge.network.flow_limits[l] + 1e-6 { feas = false; break; }
                    }
                    if !feas { continue; }
                    actions[b] = u;
                    let p = challenge.compute_profit(state, &actions);
                    actions[b] = base;
                    if p > best_prof { best_prof = p; best_u = u; }
                }
                if (best_u - base).abs() > 1e-8 {
                    let d = best_u - base;
                    for &(l, imp) in &ptdf_b[b] { flows[l] += imp * d; }
                    actions[b] = best_u;
                    improved = true;
                }
            }
            if !improved { break; }
        }

        let inj = challenge.compute_total_injections(state, &actions);
        let final_flows = challenge.network.compute_flows(&inj);
        let mut beta = 1.0_f64;
        for l in 0..num_l {
            let lim = challenge.network.flow_limits[l];
            if final_flows[l].abs() > lim + 1e-6 && final_flows[l].abs() > 1e-12 {
                beta = beta.min(lim / final_flows[l].abs());
            }
        }
        if beta < 1.0 { for b in 0..num_b { actions[b] *= beta; } }
        let p = challenge.compute_profit(state, &actions);
        if p > best_profit { best_actions = actions.clone(); }
        best_actions
    }
    
}
mod lp {
    
    
    
    

    const LP_MAX_PIVOTS: usize = 3000;
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
        let mut pivots_done: usize = 0;

        for pivot in 0..max_pivots {
            pivots_done = pivot + 1;

            
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
        (Some(x), pivots_done)
    }

    
    pub fn lp_solve(n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let (sol, _) = lp_solve_with_budget(n, m, c, a, b, LP_MAX_PIVOTS);
        sol
    }

    
    
    
    
    pub fn lp_solve_with_duals(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, Option<Vec<f64>>, usize) {
        if b.iter().any(|&x| x < -1e-6) {
            return (None, None, 0);
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

        for pivot_count in 0..max_pivots {
            let entering = match (0..n_vars).find(|&j| tab[m][j] < -LP_EPS) {
                Some(j) => j,
                None => {
                    
                    let duals = (0..m).map(|i| {
                        let slack_col = n + i;
                        let val = -tab[m][slack_col];
                        if val.is_finite() { val } else { 0.0 }
                    }).collect();
                    let mut x = vec![0.0_f64; n];
                    for (i, &bv) in basis.iter().enumerate() {
                        if bv < n {
                            x[bv] = tab[i][rhs_col].max(0.0);
                        }
                    }
                    return (Some(x), Some(duals), pivot_count);
                }
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
                None => return (None, None, 0),
            };

            let pivot_val = tab[leaving_row][entering];
            if pivot_val.abs() < LP_EPS {
                return (None, None, 0);
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

        
        let duals = (0..m).map(|i| {
            let slack_col = n + i;
            let val = -tab[m][slack_col];
            if val.is_finite() { val } else { 0.0 }
        }).collect();
        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i][rhs_col].max(0.0);
            }
        }
        (Some(x), Some(duals), max_pivots)
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
            use_policy: false,
            use_warmstart: false,
            use_mpc: false,
            mpc_horizon: 2,
            mpc_pivot_budget: 800,
            use_dw: false,
            dw_iters: 3,
            dw_max_lines: 10,
            dw_max_cols_per_batt: 5,
            dw_pivot_budget_per_solve: 2000,
            dw_total_pivot_budget: 8000,
            use_dw_prescreen: false,
            use_lns: false,
            lns_cg_iters: 2,
            lns_cg_column_limit: 6,
            lns_max_lines: 8,
            lns_lp_pivots_total: 4000,
            use_pivot_reserve: false,
            lp_max_lines: 9999,
            use_parallel_dp: false,
            use_sdp: false,
            sdp_k: 3,
            use_ldd_proximal: false,
            ldd_momentum: 0.5,
            ldd_clip_fraction: 0.2,
            use_tail_quadrature: false,
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
            asca_iters: 60,
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
            use_policy: false,
            use_warmstart: false,
            use_mpc: false,
            mpc_horizon: 2,
            mpc_pivot_budget: 800,
            use_dw: false,
            dw_iters: 3,
            dw_max_lines: 10,
            dw_max_cols_per_batt: 5,
            dw_pivot_budget_per_solve: 2000,
            dw_total_pivot_budget: 8000,
            use_dw_prescreen: false,
            use_lns: false,
            lns_cg_iters: 2,
            lns_cg_column_limit: 6,
            lns_max_lines: 8,
            lns_lp_pivots_total: 4000,
            use_pivot_reserve: false,
            lp_max_lines: 9999,
            use_parallel_dp: false,
            use_sdp: false,
            sdp_k: 3,
            use_ldd_proximal: false,
            ldd_momentum: 0.5,
            ldd_clip_fraction: 0.2,
            use_tail_quadrature: false,
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
            action_grid: 40,
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
            ldd_iters: 2,
            ldd_step_size: 0.1,
            use_kkt: true,
            kkt_cong_threshold: 0.70,
            kkt_price_scale: 0.8,
            max_admm_iters: 10,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: false,
            lp_soft_lambda: 1e5,
            lp_per_call_pivots: 500,
            lp_total_pivots: 8000,
            use_policy: false,
            use_warmstart: true,
            use_mpc: false,
            mpc_horizon: 2,
            mpc_pivot_budget: 800,
            use_dw: false,
            dw_iters: 3,
            dw_max_lines: 10,
            dw_max_cols_per_batt: 5,
            dw_pivot_budget_per_solve: 2000,
            dw_total_pivot_budget: 8000,
            use_dw_prescreen: true,
            use_lns: false,
            lns_cg_iters: 2,
            lns_cg_column_limit: 6,
            lns_max_lines: 8,
            lns_lp_pivots_total: 4000,
            use_pivot_reserve: true,
            lp_max_lines: 8,
            use_parallel_dp: true,
            use_sdp: true,
            sdp_k: 3,
            use_ldd_proximal: true,
            ldd_momentum: 0.5,
            ldd_clip_fraction: 0.2,
            use_tail_quadrature: true,
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
            use_policy: false,
            use_warmstart: false,
            use_mpc: false,
            mpc_horizon: 2,
            mpc_pivot_budget: 800,
            use_dw: false,
            dw_iters: 3,
            dw_max_lines: 10,
            dw_max_cols_per_batt: 5,
            dw_pivot_budget_per_solve: 2000,
            dw_total_pivot_budget: 8000,
            use_dw_prescreen: false,
            use_lns: false,
            lns_cg_iters: 2,
            lns_cg_column_limit: 6,
            lns_max_lines: 8,
            lns_lp_pivots_total: 4000,
            use_pivot_reserve: false,
            lp_max_lines: 9999,
            use_parallel_dp: false,
            use_sdp: false,
            sdp_k: 3,
            use_ldd_proximal: false,
            ldd_momentum: 0.5,
            ldd_clip_fraction: 0.2,
            use_tail_quadrature: false,
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
            use_policy: false,
            use_warmstart: false,
            use_mpc: false,
            mpc_horizon: 2,
            mpc_pivot_budget: 800,
            use_dw: false,
            dw_iters: 3,
            dw_max_lines: 10,
            dw_max_cols_per_batt: 5,
            dw_pivot_budget_per_solve: 2000,
            dw_total_pivot_budget: 8000,
            use_dw_prescreen: false,
            use_lns: false,
            lns_cg_iters: 2,
            lns_cg_column_limit: 6,
            lns_max_lines: 8,
            lns_lp_pivots_total: 4000,
            use_pivot_reserve: false,
            lp_max_lines: 9999,
            use_parallel_dp: false,
            use_sdp: false,
            sdp_k: 3,
            use_ldd_proximal: false,
            ldd_momentum: 0.5,
            ldd_clip_fraction: 0.2,
            use_tail_quadrature: false,
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
            "titan: unsupported num_batteries={} (expected one of 10/20/40/60/100)", n
        )),
    }
}

pub fn help() {
    println!("titan");
}

