use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod helpers {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use std::cell::RefCell;
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
        pub use_sdp: bool,
        pub use_lp: bool,
        pub lp_refine_sweeps: usize,
        pub use_cg: bool,
        pub cg_iters: usize,
    }

    impl TrackHp {
        pub fn override_from_map(&mut self, h: &Option<Map<String, Value>>) {
            let Some(m) = h else { return };
            if let Some(v) = m.get("soc_levels").and_then(|v| v.as_u64()) { self.soc_levels = (v as usize).max(3usize); }
            if let Some(v) = m.get("action_grid").and_then(|v| v.as_u64()) { self.action_grid = (v as usize).max(4usize); }
            if let Some(v) = m.get("asca_iters").and_then(|v| v.as_u64()) { self.asca_iters = v as usize; }
            if let Some(v) = m.get("ternary_iters").and_then(|v| v.as_u64()) { self.ternary_iters = v as usize; }
            if let Some(v) = m.get("convergence_tol").and_then(|v| v.as_f64()) { self.convergence_tol = v as f64; }
            if let Some(v) = m.get("anticipate_lmp").and_then(|v| v.as_bool()) { self.anticipate_lmp = v; }
            if let Some(v) = m.get("lmp_threshold").and_then(|v| v.as_f64()) { self.lmp_threshold = v as f64; }
            if let Some(v) = m.get("lmp_premium_scale").and_then(|v| v.as_f64()) { self.lmp_premium_scale = v as f64; }
            if let Some(v) = m.get("jump_premium").and_then(|v| v.as_f64()) { self.jump_premium = v as f64; }
            if let Some(v) = m.get("prune_ratio").and_then(|v| v.as_f64()) { self.prune_ratio = (v as f64).clamp(0.0_f64, 0.9_f64); }
            if let Some(v) = m.get("deflator_iters").and_then(|v| v.as_u64()) { self.deflator_iters = v as usize; }
            if let Some(v) = m.get("flow_margin").and_then(|v| v.as_f64()) { self.flow_margin = (v as f64).max(0.0_f64); }
            if let Some(v) = m.get("network_derating").and_then(|v| v.as_f64()) { self.network_derating = (v as f64).clamp(0.01_f64, 1.0_f64); }
            if let Some(v) = m.get("use_sdp").and_then(|v| v.as_bool()) { self.use_sdp = v; }
            if let Some(v) = m.get("use_lp").and_then(|v| v.as_bool()) { self.use_lp = v; }
            if let Some(v) = m.get("lp_refine_sweeps").and_then(|v| v.as_u64()) { self.lp_refine_sweeps = v as usize; }
            if let Some(v) = m.get("use_cg").and_then(|v| v.as_bool()) { self.use_cg = v; }
            if let Some(v) = m.get("cg_iters").and_then(|v| v.as_u64()) { self.cg_iters = v as usize; }
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
            let inner = guard.as_mut().expect("aycdicdb: STATE not initialised");
            if inner.cache.is_none() {
                inner.cache = Some(build_cache(challenge, state, &inner.hp));
            }
            let cache = inner.cache.as_ref().unwrap();
            let hp = &inner.hp;

            let zero_action = vec![0.0_f64; challenge.num_batteries];
            let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base_cur);

            let mut actions = if hp.use_cg {
                run_column_generation(challenge, state, cache, hp, &flows_base)
            } else {
                let mut actions_asca = vec![0.0; challenge.num_batteries];
                run_asca(challenge, state, cache, hp, &flows_base, &mut actions_asca);
                run_deflator(challenge, state, cache, hp, &flows_base, &mut actions_asca);

                let mut actions = actions_asca.clone();

                if hp.use_lp {
                    if let Some(lp_act) = joint_lp_dispatch(challenge, state, cache, &flows_base) {
                        let mut actions_lp_warm = lp_act.clone();
                        if hp.lp_refine_sweeps > 0usize {
                            let mut hp_warm = hp.clone();
                            hp_warm.asca_iters = hp.lp_refine_sweeps;
                            run_asca(challenge, state, cache, &hp_warm, &flows_base, &mut actions_lp_warm);
                        }

                        let profit_asca: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, actions_asca[b]))
                            .sum();
                        let profit_lp: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, lp_act[b]))
                            .sum();
                        let profit_lp_warm: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, actions_lp_warm[b]))
                            .sum();

                        if profit_lp_warm >= profit_asca && profit_lp_warm >= profit_lp {
                            actions = actions_lp_warm;
                        } else if profit_lp >= profit_asca {
                            actions = lp_act;
                        }
                    }
                }
                actions
            };
          
            run_deflator(challenge, state, cache, hp, &flows_base, &mut actions);
            Ok(actions)
        })
    }

    fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> AycdicdbCache {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let num_t = challenge.num_steps;

        let zero_action = vec![0.0_f64; num_b];
        let inj_base = challenge.compute_total_injections(state, &zero_action);
        let flows0 = challenge.network.compute_flows(&inj_base);
        
        let batt_nodes: Vec<usize> = challenge.batteries.iter().map(|b| b.node).collect();
        let mut ptdf_sparse: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_l];
        let mut b_to_lines: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_b];
        let mut dummy = zero_action.clone();
        for b in 0..num_b {
            dummy[b] = 1.0_f64;
            let inj1 = challenge.compute_total_injections(state, &dummy);
            let flows1 = challenge.network.compute_flows(&inj1);
            for l in 0..num_l {
                let impact = flows1[l] - flows0[l];
                if impact.abs() > 1e-8_f64 {
                    ptdf_sparse[l].push((b, impact));
                    b_to_lines[b].push((l, impact));
                }
            }
            dummy[b] = 0.0_f64;
        }

        let mut expected_premiums = vec![vec![0.0_f64; num_b]; num_t];
        if hp.anticipate_lmp && num_l > 0usize {
            let base_premium = 20.0_f64 * hp.lmp_premium_scale;
            for t in 0..num_t {
                let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                for l in 0..num_l {
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6_f64 { continue; }
                    let ratio = f_exo[l].abs() / limit;
                    if ratio > hp.lmp_threshold {
                        let proba = ((ratio - hp.lmp_threshold) / (1.0_f64 - hp.lmp_threshold).max(1e-6_f64))
                            .clamp(0.0_f64, 1.0_f64);
                        let premium = base_premium * proba;
                        let sign_f = f_exo[l].signum();
                        for &(b, impact) in &ptdf_sparse[l] {
                            if impact.abs() > 1e-6_f64 {
                                let nodal_shift = -impact * sign_f * premium;
                                expected_premiums[t][b] += nodal_shift;
                            }
                        }
                    }
                }
            }
        }

        let soc_levels = hp.soc_levels;
        let dt = 0.25_f64;
        let mut dp = vec![vec![vec![0.0_f64; soc_levels]; num_t + 1usize]; num_b];
        
        const GH5_Z: [f64; 5] = [0.0_f64, 0.9586_f64, -0.9586_f64, 2.0202_f64, -2.0202_f64];
        const GH5_W: [f64; 5] = [0.5333_f64, 0.2221_f64, 0.2221_f64, 0.0113_f64, 0.0113_f64];
        let sdp_sigma_eff = if hp.use_sdp {
            let sigma = challenge.market.params.volatility;
            let rho_j = challenge.market.params.jump_probability;
            let alpha_j = challenge.market.params.tail_index;
            let jump_var = if alpha_j > 2.0_f64 { rho_j * alpha_j / (alpha_j - 2.0_f64) } else { rho_j * 4.0_f64 };
            (sigma * sigma + jump_var).sqrt()
        } else { 0.0_f64 };

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = batt_nodes[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9_f64);
            let deg_coeff = (dt / bat.capacity_mwh.max(1e-9_f64)).powi(2);

            for t in (0..num_t).rev() {
                let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                    challenge.market.day_ahead_prices[t][node]
                } else {
                    challenge.market.day_ahead_prices[t][0]
                };
                let extra = expected_premiums[t][b];

                for i in 0..soc_levels {
                    let soc = bat.soc_min_mwh + soc_span * (i as f64) / ((soc_levels - 1usize) as f64);

                    let charge_soc_limit = if bat.efficiency_charge > 0.0_f64 {
                        (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                    } else { 0.0_f64 };
                    let discharge_soc_limit = if bat.efficiency_discharge > 0.0_f64 {
                        (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                    } else { 0.0_f64 };

                    let max_pwr_c = bat.power_charge_mw * hp.network_derating;
                    let max_pwr_d = bat.power_discharge_mw * hp.network_derating;

                    let u_min = -(max_pwr_c.min(charge_soc_limit.max(0.0_f64)));
                    let u_max = (max_pwr_d.min(discharge_soc_limit.max(0.0_f64))).max(u_min);

                    let v_next_slice = &dp[b][t + 1usize];

                    let max_val = if hp.use_sdp {
                        let mut val_sum = 0.0_f64;
                        for k in 0..5usize {
                            let p = (p_da * (1.0_f64 + sdp_sigma_eff * GH5_Z[k]) + extra).max(1e-6_f64);
                            val_sum += GH5_W[k] * dp_analytic_max(bat, p, p, soc, u_min, u_max, v_next_slice, soc_levels, soc_span, deg_coeff);
                        }
                        val_sum
                    } else {
                        let p_sell = p_da * (1.0_f64 + hp.jump_premium) + extra;
                        let p_buy = p_da + extra;
                        dp_analytic_max(bat, p_buy, p_sell, soc, u_min, u_max, v_next_slice, soc_levels, soc_span, deg_coeff)
                    };
                    dp[b][t][i] = max_val;
                }
            }
        }

        AycdicdbCache { dp, ptdf_sparse, b_to_lines, batt_nodes }
    }
    
    fn dp_analytic_max(
        bat: &Battery,
        p_buy: f64, p_sell: f64,
        soc: f64, u_min: f64, u_max: f64,
        v_next: &[f64],
        soc_levels: usize, soc_span: f64,
        deg_coeff: f64,
    ) -> f64 {
        let dt = 0.25_f64;
        let _cap = bat.capacity_mwh.max(1e-9_f64);

        let lambda = if soc_levels > 1usize {
            let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1usize) as f64);
            let idx0 = (idx_f.floor() as usize).min(soc_levels - 2usize);
            let delta_soc = soc_span / ((soc_levels - 1usize) as f64);
            (v_next[idx0 + 1usize] - v_next[idx0]) / delta_soc
        } else { 0.0_f64 };

        let eval = |u: f64| -> f64 {
            let price = if u > 0.0_f64 { p_sell } else { p_buy };
            let abs_u = u.abs();
            let profit = u * price * dt - 0.25_f64 * abs_u * dt - deg_coeff * u * u;
            let next_soc = if u < 0.0_f64 {
                soc + bat.efficiency_charge * (-u) * dt
            } else {
                soc - u / bat.efficiency_discharge.max(1e-9_f64) * dt
            };
            let next_soc = next_soc.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1usize) as f64);
            let idx0 = (idx_f.floor() as isize).max(0isize) as usize;
            let i0 = idx0.min(soc_levels - 1usize);
            let i1 = (idx0 + 1usize).min(soc_levels - 1usize);
            let frac = (idx_f - idx0 as f64).clamp(0.0_f64, 1.0_f64);
            profit + v_next[i0] * (1.0_f64 - frac) + v_next[i1] * frac
        };

        let mut best = eval(0.0_f64);

        if u_min < 0.0_f64 {
            let u_hi = 0.0_f64.min(u_max);
            if u_min < u_hi {
                let b_c = dt * (lambda * bat.efficiency_charge - p_buy - 0.25_f64);
                let x_star = if deg_coeff > 1e-30_f64 { b_c / (2.0_f64 * deg_coeff) } else { -u_min };
                let cand = (-x_star.clamp(0.0_f64, -u_min)).clamp(u_min, u_hi);
                let v = eval(cand); if v > best { best = v; }
                let v = eval(u_min); if v > best { best = v; }
            }
        }

        if u_max > 0.0_f64 {
            let u_lo = 0.0_f64.max(u_min);
            if u_lo < u_max {
                let eff_d = bat.efficiency_discharge.max(1e-9_f64);
                let b_d = dt * (p_sell - 0.25_f64 - lambda / eff_d);
                let x_star = if deg_coeff > 1e-30_f64 { b_d / (2.0_f64 * deg_coeff) } else { u_max };
                let cand = x_star.clamp(u_lo, u_max);
                let v = eval(cand); if v > best { best = v; }
                let v = eval(u_max); if v > best { best = v; }
            }
        }

        if best == f64::NEG_INFINITY { 0.0_f64 } else { best }
    }

    #[inline]
    fn asca_lambda(v_table: &[f64], soc: f64, soc_min: f64, soc_span: f64, soc_levels: usize) -> f64 {
        if soc_levels < 2usize { return 0.0_f64; }
        let idx_f = (soc - soc_min) / soc_span * ((soc_levels - 1usize) as f64);
        let idx0 = (idx_f.floor() as usize).min(soc_levels - 2usize);
        let delta_soc = soc_span / ((soc_levels - 1usize) as f64);
        (v_table[idx0 + 1usize] - v_table[idx0]) / delta_soc.max(1e-12_f64)
    }

    #[inline]
    fn eval_profit(challenge: &Challenge, state: &State, ca: &AycdicdbCache, b: usize, u: f64) -> f64 {
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0_f64 };
        let dt = 0.25_f64;
        let abs_u = u.abs();
        let revenue = u * rt_price * dt;
        let tx = 0.25_f64 * abs_u * dt;
        let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9_f64);
        let deg = deg_base * deg_base;
        let profit = revenue - tx - deg;

        let soc = state.socs[b];
        let next_soc_raw = if u < 0.0_f64 {
            soc + bat.efficiency_charge * (-u) * dt
        } else {
            soc - u / bat.efficiency_discharge.max(1e-9_f64) * dt
        };
        let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9_f64);
        let soc_levels = ca.dp[b][0].len();
        let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1usize) as f64);
        let idx0 = (idx_f.floor() as isize).max(0isize) as usize;
        let idx0c = idx0.min(soc_levels - 1usize);
        let idx1c = (idx0 + 1usize).min(soc_levels - 1usize);
        let frac = (idx_f - idx0 as f64).clamp(0.0_f64, 1.0_f64);
        let t_next = (state.time_step + 1usize).min(ca.dp[b].len() - 1usize);
        profit + ca.dp[b][t_next][idx0c] * (1.0_f64 - frac) + ca.dp[b][t_next][idx1c] * frac
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
        let mut flows: Vec<f64> = flows_base.to_vec();
        for b in 0..num_b {
            if actions[b].abs() > 1e-12_f64 {
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows[l] += p * actions[b]; }
                }
            }
        }

        let mut active = vec![true; num_b];
        if hp.prune_ratio > 0.0_f64 && num_b >= 2usize {
            let cutoff = ((num_b as f64) * hp.prune_ratio) as usize;
            if cutoff > 0usize {
                let mut caps: Vec<(usize, f64)> = challenge.batteries.iter().enumerate().map(|(i, b)| (i, b.capacity_mwh)).collect();
                caps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..cutoff.min(num_b) { active[caps[i].0] = false; }
            }
        }

        let footprint = |batt: usize| -> f64 {
            let mut fp = 1e-4_f64;
            for &(l, p) in &ca.b_to_lines[batt] {
                let limit = challenge.network.flow_limits[l];
                if limit > 1e-6_f64 {
                    let utilization = flows_base[l].abs() / limit;
                    fp += p.abs() * utilization.powi(2) * 10.0_f64;
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
                    if p.abs() < 1e-9_f64 { continue; }
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0_f64);
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

                let v0 = eval_profit(challenge, state, ca, b, 0.0_f64);
                if u_min <= 0.0_f64 && 0.0_f64 <= u_max && v0 > best_v { best_v = v0; best_u = 0.0_f64; }

                let bat = &challenge.batteries[b];
                let node = ca.batt_nodes[b];
                let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0_f64 };
                let deg_coeff = (0.25_f64 / bat.capacity_mwh.max(1e-9_f64)).powi(2);
                let soc_levels = ca.dp[b][0].len();
                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9_f64);
                let t_next = (state.time_step + 1usize).min(ca.dp[b].len() - 1usize);
                let lambda = asca_lambda(&ca.dp[b][t_next], state.socs[b], bat.soc_min_mwh, soc_span, soc_levels);
                let augment_kinks = hp.use_cg && soc_levels >= 64usize && (u_max - u_min) > 0.25_f64;

                if u_min < 0.0_f64 {
                    let lo = u_min; let hi = 0.0_f64.min(u_max);
                    if lo < hi {
                        let b_coeff = 0.25_f64 * (-rt_price - 0.25_f64 + lambda * bat.efficiency_charge);
                        let cand = if deg_coeff > 1e-30_f64 {
                            let x = b_coeff / (2.0_f64 * deg_coeff);
                            (-x.clamp(0.0_f64, -lo)).clamp(lo, hi)
                        } else { lo };
                        let mut local_cands = Vec::with_capacity(4usize);
                        push_unique_candidate(&mut local_cands, cand, lo, hi);
                        push_unique_candidate(&mut local_cands, lo, lo, hi);
                        if augment_kinks {
                            cg_add_nearby_kinks(
                                &mut local_cands, bat, state.socs[b], soc_span, soc_levels, cand, true, lo, hi,
                            );
                        }
                        for u in local_cands {
                            let v = eval_profit(challenge, state, ca, b, u);
                            if v > best_v { best_v = v; best_u = u; }
                        }
                    }
                }

                if u_max > 0.0_f64 {
                    let lo = 0.0_f64.max(u_min); let hi = u_max;
                    if lo < hi {
                        let eff_d = bat.efficiency_discharge.max(1e-9_f64);
                        let b_coeff = 0.25_f64 * (rt_price - 0.25_f64 - lambda / eff_d);
                        let cand = if deg_coeff > 1e-30_f64 {
                            let x = b_coeff / (2.0_f64 * deg_coeff);
                            x.clamp(lo, hi)
                        } else { hi };
                        let mut local_cands = Vec::with_capacity(4usize);
                        push_unique_candidate(&mut local_cands, cand, lo, hi);
                        push_unique_candidate(&mut local_cands, hi, lo, hi);
                        if augment_kinks {
                            cg_add_nearby_kinks(
                                &mut local_cands, bat, state.socs[b], soc_span, soc_levels, cand, false, lo, hi,
                            );
                        }
                        for u in local_cands {
                            let v = eval_profit(challenge, state, ca, b, u);
                            if v > best_v { best_v = v; best_u = u; }
                        }
                    }
                }

                let delta = best_u - actions[b];
                if delta.abs() > 1e-6_f64 {
                    actions[b] = best_u;
                    for &(l, p) in &ca.b_to_lines[b] { if l < num_l { flows[l] += p * delta; } }
                    if delta.abs() > max_change { max_change = delta.abs(); }
                }
            }
            if max_change < hp.convergence_tol { break; }
        }
    }

    #[inline]
    fn potential(challenge: &Challenge, state: &State, ca: &AycdicdbCache, b: usize) -> f64 {
        let (u_lo, u_hi) = state.action_bounds[b];
        let v_lo = eval_profit(challenge, state, ca, b, u_lo);
        let v_hi = eval_profit(challenge, state, ca, b, u_hi);
        let v0 = eval_profit(challenge, state, ca, b, 0.0_f64);
        (v_lo.max(v_hi) - v0).max(0.0_f64)
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

        let mut flows = vec![0.0_f64; num_l];
        for l in 0..num_l {
            let mut f = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { f += imp * actions[b]; }
            flows[l] = flows_base[l] + f;
        }

        let mut is_safe = true;
        for _ in 0..hp.deflator_iters {
            let mut violated_lines = Vec::new();
            is_safe = true;
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0_f64);
                if flows[l].abs() <= limit { continue; }
                is_safe = false;
                violated_lines.push(l);
            }
            if is_safe { break; }

            let mut weighted_relief = vec![0.0_f64; num_b];
            let mut is_culprit = vec![false; num_b];
            for &l in &violated_lines {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0_f64);
                let overflow = (flows[l].abs() - limit).max(0.0_f64);
                if overflow <= 1e-12_f64 { continue; }
                let sign = flows[l].signum();
                let weight = 1.0_f64 + (overflow / limit.max(1e-6_f64)).min(1.0_f64);

                for &(b, impact) in &ca.ptdf_sparse[l] {
                    let action = actions[b];
                    if action.abs() <= 1e-12_f64 { continue; }
                    let relief_rate = sign * impact * action.signum();
                    if relief_rate > 1e-9_f64 {
                        weighted_relief[b] += weight * relief_rate;
                        is_culprit[b] = true;
                    }
                }
            }

            let mut loss_density = vec![f64::INFINITY; num_b];
            for b in 0..num_b {
                if !is_culprit[b] { continue; }
                let val_curr = eval_profit(challenge, state, ca, b, actions[b]);
                let val_zero = eval_profit(challenge, state, ca, b, 0.0_f64);
                let loss = (val_curr - val_zero).max(0.0_f64);
                loss_density[b] = loss / weighted_relief[b].max(1e-9_f64);
            }

            let mut progressed = false;
            for &l in &violated_lines {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0_f64);
                let mut remaining = (flows[l].abs() - limit).max(0.0_f64);
                if remaining <= 1e-9_f64 { continue; }
                let sign = flows[l].signum();

                let mut culprits: Vec<(usize, f64, f64, f64)> = Vec::new();
                for &(b, impact) in &ca.ptdf_sparse[l] {
                    let action = actions[b];
                    if action.abs() <= 1e-12_f64 { continue; }
                    let relief_rate = sign * impact * action.signum();
                    if relief_rate > 1e-9_f64 {
                        let full_relief = relief_rate * action.abs();
                        culprits.push((b, relief_rate, full_relief, loss_density[b]));
                    }
                }
                culprits.sort_by(|a, b| {
                    let ord = a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal);
                    if ord == std::cmp::Ordering::Equal {
                        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        ord
                    }
                });

                for (b, relief_rate, _, _) in culprits {
                    if remaining <= 1e-9_f64 { break; }
                    let max_du = actions[b].abs();
                    if max_du <= 1e-12_f64 { continue; }
                    let du = (remaining / relief_rate).min(max_du);
                    if du <= 1e-12_f64 { continue; }
                    let mut new_action = actions[b] - actions[b].signum() * du;
                    if new_action.abs() < 1e-12_f64 { new_action = 0.0_f64; }
                    let delta = new_action - actions[b];
                    actions[b] = new_action;
                    for &(ll, pp) in &ca.b_to_lines[b] {
                        if ll < num_l { flows[ll] += pp * delta; }
                    }
                    remaining -= relief_rate * du;
                    progressed = true;
                }
            }

            if !progressed { break; }
        }

        if is_safe { return; }

        let f_act: Vec<f64> = (0..num_l).map(|l| {
            let mut s = 0.0_f64;
            for &(b, imp) in &ca.ptdf_sparse[l] { s += imp * actions[b]; }
            s
        }).collect();

        let mut beta = 1.0_f64;
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0_f64);
            let total = flows_base[l] + f_act[l];
            if total.abs() <= limit { continue; }
            if f_act[l].abs() < 1e-9_f64 { continue; }
            let target = if total > 0.0_f64 { limit } else { -limit };
            let candidate = (target - flows_base[l]) / f_act[l];
            if candidate < beta { beta = candidate; }
        }
        let beta = beta.clamp(0.0_f64, 1.0_f64);
        for b in 0..num_b { actions[b] *= beta; }

        for b in 0..num_b {
            let (lo, hi) = state.action_bounds[b];
            if actions[b] < lo { actions[b] = lo; }
            if actions[b] > hi { actions[b] = hi; }
        }
    }

    fn screen_active_lines_from_ranges(
        challenge: &Challenge,
        ca: &AycdicdbCache,
        flows_base: &[f64],
        ranges: &[(f64, f64)],
        flow_margin: f64,
    ) -> Vec<usize> {
        let num_l = challenge.network.flow_limits.len();
        if num_l <= 24usize {
            return (0..num_l)
                .filter(|&l| challenge.network.flow_limits[l] > flow_margin + 1e-6_f64)
                .collect();
        }

        let mut active_lines = Vec::with_capacity(num_l.min(64usize));
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - flow_margin).max(0.0_f64);
            if limit <= 1e-6_f64 { continue; }

            let base = if l < flows_base.len() { flows_base[l] } else { 0.0_f64 };
            if base.abs() >= 0.70_f64 * limit {
                active_lines.push(l);
                continue;
            }

            let mut max_pos = base;
            let mut max_neg = base;
            for &(b, impact) in &ca.ptdf_sparse[l] {
                let (u_lo, u_hi) = if b < ranges.len() { ranges[b] } else { (0.0_f64, 0.0_f64) };
                if impact >= 0.0_f64 {
                    max_pos += impact * u_hi;
                    max_neg += impact * u_lo;
                } else {
                    max_pos += impact * u_lo;
                    max_neg += impact * u_hi;
                }
            }

            if max_pos >= 0.95_f64 * limit || max_neg <= -0.95_f64 * limit {
                active_lines.push(l);
            }
        }

        active_lines
    }

    fn joint_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        flows_base: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let dt = 0.25_f64;
        let tx_cost = 0.25_f64;
        let n = 2usize * num_b;
        let ranges: Vec<(f64, f64)> = state.action_bounds.iter().copied().collect();
        let active_lines = screen_active_lines_from_ranges(challenge, ca, flows_base, &ranges, 0.0_f64);
        let m = 4usize * num_b + 2usize * active_lines.len();

        let mut c_obj = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        let t_next = (state.time_step + 1usize).min(ca.dp[0].len() - 1usize);

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0_f64 };
            let soc = state.socs[b];
            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9_f64);
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9_f64);
            let soc_levels = ca.dp[b][0].len();
            let dv = asca_lambda(&ca.dp[b][t_next], soc, bat.soc_min_mwh, soc_span, soc_levels);

            c_obj[b]         = (rt - tx_cost) * dt - dv / eta_d * dt;
            c_obj[num_b + b] = -(rt + tx_cost) * dt + dv * eta_c * dt;

            let (u_min, u_max) = state.action_bounds[b];
            let r = 4usize * b;
            a_mat[r][b] = 1.0_f64;
            b_vec[r] = u_max.max(0.0_f64);
            a_mat[r + 1usize][num_b + b] = 1.0_f64;
            b_vec[r + 1usize] = (-u_min).max(0.0_f64);
            a_mat[r + 2usize][b]         =  dt / eta_d;
            a_mat[r + 2usize][num_b + b] = -eta_c * dt;
            b_vec[r + 2usize] = (soc - bat.soc_min_mwh).max(0.0_f64);
            a_mat[r + 3usize][b]         = -dt / eta_d;
            a_mat[r + 3usize][num_b + b] =  eta_c * dt;
            b_vec[r + 3usize] = (bat.soc_max_mwh - soc).max(0.0_f64);
        }

        let row_f = 4usize * num_b;
        for (k, &l) in active_lines.iter().enumerate() {
            let limit = challenge.network.flow_limits[l];
            let exo = flows_base[l];
            let rp = row_f + 2usize * k;
            let rn = rp + 1usize;
            for &(b, impact) in &ca.ptdf_sparse[l] {
                a_mat[rp][b]         += impact;
                a_mat[rp][num_b + b] -= impact;
                a_mat[rn][b]         -= impact;
                a_mat[rn][num_b + b] += impact;
            }
            b_vec[rp] = (limit - exo).max(0.0_f64);
            b_vec[rn] = (limit + exo).max(0.0_f64);
        }

        let (opt_x, _) = super::lp::lp_solve_with_budget(n, m, &c_obj, &a_mat, &b_vec, 3000usize);
        let opt_x = opt_x?;

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let u = opt_x[b] - opt_x[num_b + b];
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }
        Some(actions)
    }    

    fn run_column_generation(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;

        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); num_b];
        let mut column_profits: Vec<Vec<f64>> = vec![Vec::new(); num_b];
        for b in 0..num_b {
            let (u_min, u_max) = state.action_bounds[b];
            columns[b].push(0.0_f64);
            column_profits[b].push(eval_profit(challenge, state, ca, b, 0.0_f64));
            if u_min < 0.0_f64 {
                columns[b].push(u_min);
                column_profits[b].push(eval_profit(challenge, state, ca, b, u_min));
            }
            if u_max > 0.0_f64 {
                columns[b].push(u_max);
                column_profits[b].push(eval_profit(challenge, state, ca, b, u_max));
            }

            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0_f64 };
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9_f64);
            let soc_levels = ca.dp[b][0].len();
            let t_next = (state.time_step + 1usize).min(ca.dp[b].len() - 1usize);
            let lambda = asca_lambda(&ca.dp[b][t_next], state.socs[b], bat.soc_min_mwh, soc_span, soc_levels);
            let deg_coeff = (0.25_f64 / bat.capacity_mwh.max(1e-9_f64)).powi(2);
            let u_opt = cg_analytic_seed(bat, rt_price, lambda, u_min, u_max, deg_coeff);
            if !columns[b].iter().any(|&u| (u - u_opt).abs() < 1e-8_f64) {
                columns[b].push(u_opt);
                column_profits[b].push(eval_profit(challenge, state, ca, b, u_opt));
            }
        }

        let mut lp_obj_prev = f64::NEG_INFINITY;
        let mut no_improve = 0u32;

        for _iter in 0..hp.cg_iters {
            let master = build_master_lp(challenge, ca, &columns, &column_profits, flows_base, hp.flow_margin);
            if master.nvars == 0usize || master.ncons == 0usize {
                break;
            }

            let (sol, duals, _pivots) = super::lp::lp_solve_with_duals(
                master.nvars, master.ncons, &master.c, &master.a, &master.b, 3000usize,
            );

            let (Some(x), Some(y)) = (sol, duals) else { break; };

            let mut sigma = vec![0.0f64; num_b];
            for b in 0..num_b {
                sigma[b] = if b < y.len() { y[b] } else { 0.0_f64 };
            }

            let mut penalty = vec![0.0f64; num_b];
            let dual_start = num_b;
            for (k, &l) in master.active_lines.iter().enumerate() {
                let y_pos = if dual_start + 2usize * k < y.len() { y[dual_start + 2usize * k] } else { 0.0_f64 };
                let y_neg = if dual_start + 2usize * k + 1usize < y.len() { y[dual_start + 2usize * k + 1usize] } else { 0.0_f64 };
                let net_dual = y_pos - y_neg;
                if net_dual.abs() <= 1e-12_f64 { continue; }
                for &(b_idx, impact) in &ca.ptdf_sparse[l] {
                    penalty[b_idx] += net_dual * impact;
                }
            }

            let mut added = false;
            for b in 0..num_b {
                let (u_min, u_max) = state.action_bounds[b];
                let (u_star, profit_star, reduced_star) = pricing_subproblem_analytic(
                    challenge, state, ca, b, penalty[b], u_min, u_max,
                );
                let reduced_cost = reduced_star - sigma[b];
                if reduced_cost > 1e-6_f64 {
                    if !columns[b].iter().any(|&u| (u - u_star).abs() < 1e-8_f64) {
                        columns[b].push(u_star);
                        column_profits[b].push(profit_star);
                        added = true;
                    }
                }
            }

            let mut lp_obj = 0.0_f64;
            for b in 0..num_b {
                let start = master.col_start[b];
                for (c_local, &idx) in master.col_map[b].iter().enumerate() {
                    let prof = if idx < column_profits[b].len() { column_profits[b][idx] } else { 0.0_f64 };
                    let var_idx = start + c_local;
                    if var_idx < x.len() {
                        lp_obj += prof * x[var_idx];
                    }
                }
            }
            if lp_obj.is_finite() && lp_obj_prev.is_finite() && lp_obj <= lp_obj_prev + 1e-6_f64 {
                no_improve += 1u32;
                if no_improve >= 3u32 { break; }
            } else {
                no_improve = 0u32;
                if lp_obj.is_finite() { lp_obj_prev = lp_obj; }
            }

            if !added { break; }
        }

        let master = build_master_lp(challenge, ca, &columns, &column_profits, flows_base, hp.flow_margin);
        if master.nvars == 0usize {
            return vec![0.0_f64; num_b];
        }
        let (sol, _, _) = super::lp::lp_solve_with_duals(
            master.nvars, master.ncons, &master.c, &master.a, &master.b, 3000usize,
        );
        let x = match sol {
            Some(x) => x,
            None => return vec![0.0_f64; num_b],
        };

        let mut actions = vec![0.0f64; num_b];
        let mut mixed_batteries = 0usize;
        for b in 0..num_b {
            let mut u_sum = 0.0_f64;
            let mut significant = 0usize;
            for (c_local, &idx) in master.col_map[b].iter().enumerate() {
                let var_idx = master.col_start[b] + c_local;
                if var_idx < x.len() && idx < columns[b].len() {
                    let w = x[var_idx].max(0.0_f64);
                    if w > 0.05_f64 { significant += 1usize; }
                    u_sum += w * columns[b][idx];
                }
            }
            if significant > 1usize { mixed_batteries += 1usize; }
            let (lo, hi) = state.action_bounds[b];
            actions[b] = if u_sum.is_finite() { u_sum.clamp(lo, hi) } else { 0.0_f64 };
        }

        let mut total_profit: f64 = (0..num_b)
            .map(|b| eval_profit(challenge, state, ca, b, actions[b]))
            .sum();

        if mixed_batteries > 0usize {
            let mut hp_warm = hp.clone();
            hp_warm.asca_iters = hp.asca_iters.min(4usize).max(1usize);
            let mut actions_refined = actions.clone();
            run_asca(challenge, state, ca, &hp_warm, flows_base, &mut actions_refined);
            let profit_refined: f64 = (0..num_b)
                .map(|b| eval_profit(challenge, state, ca, b, actions_refined[b]))
                .sum();
            if profit_refined.is_finite() && profit_refined > total_profit + 1e-9_f64 {
                actions = actions_refined;
                total_profit = profit_refined;
            }
        }

        if total_profit.is_finite() && total_profit < -1e-3_f64 {
            let mut hp_warm = hp.clone();
            hp_warm.asca_iters = hp.asca_iters.min(4usize).max(1usize);
            let mut actions_asca = actions.clone();
            run_asca(challenge, state, ca, &hp_warm, flows_base, &mut actions_asca);
            let profit_asca: f64 = (0..num_b)
                .map(|b| eval_profit(challenge, state, ca, b, actions_asca[b]))
                .sum();
            if profit_asca.is_finite() && profit_asca > total_profit {
                return actions_asca;
            }
        }

        actions
    }

    struct MasterLP {
        nvars: usize,
        ncons: usize,
        c: Vec<f64>,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        col_start: Vec<usize>,
        col_map: Vec<Vec<usize>>,
        active_lines: Vec<usize>,
    }

    fn build_master_lp(
        challenge: &Challenge,
        ca: &AycdicdbCache,
        columns: &[Vec<f64>],
        column_profits: &[Vec<f64>],
        flows_base: &[f64],
        flow_margin: f64,
    ) -> MasterLP {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let mut col_start = Vec::with_capacity(num_b);
        let mut col_map: Vec<Vec<usize>> = vec![Vec::new(); num_b];
        let mut nvars = 0usize;
        for b in 0..num_b {
            col_start.push(nvars);
            let nc = columns[b].len();
            col_map[b] = (0..nc).collect();
            nvars += nc;
        }

        let ranges: Vec<(f64, f64)> = columns.iter().map(|cols| {
            if cols.is_empty() {
                return (0.0_f64, 0.0_f64);
            }
            let mut lo = cols[0];
            let mut hi = cols[0];
            for &u in cols.iter().skip(1usize) {
                if u < lo { lo = u; }
                if u > hi { hi = u; }
            }
            (lo, hi)
        }).collect();
        let active_lines = screen_active_lines_from_ranges(challenge, ca, flows_base, &ranges, flow_margin);
        let mut line_pos = vec![usize::MAX; num_l];
        for (k, &l) in active_lines.iter().enumerate() {
            line_pos[l] = k;
        }

        let ncons = num_b + 2usize * active_lines.len();
        let mut c_obj = vec![0.0_f64; nvars];
        let mut a_mat = vec![vec![0.0_f64; nvars]; ncons];
        let mut b_vec = vec![0.0_f64; ncons];

        for b in 0..num_b {
            let start = col_start[b];
            let nc = columns[b].len();
            for j in 0..nc {
                a_mat[b][start + j] = 1.0_f64;
                if j < column_profits[b].len() {
                    c_obj[start + j] = column_profits[b][j];
                }
            }
            b_vec[b] = 1.0_f64;
        }

        let flow_start = num_b;
        for b in 0..num_b {
            let start = col_start[b];
            for j in 0..columns[b].len() {
                let u = columns[b][j];
                if u.abs() < 1e-12_f64 { continue; }
                let col = start + j;
                for &(l, impact) in &ca.b_to_lines[b] {
                    let k = line_pos[l];
                    if k == usize::MAX { continue; }
                    let rp = flow_start + 2usize * k;
                    let rn = rp + 1usize;
                    let coeff = impact * u;
                    a_mat[rp][col] += coeff;
                    a_mat[rn][col] -= coeff;
                }
            }
        }

        for (k, &l) in active_lines.iter().enumerate() {
            let limit = challenge.network.flow_limits[l] - flow_margin;
            let exo = if l < flows_base.len() { flows_base[l] } else { 0.0_f64 };
            let rp = flow_start + 2usize * k;
            let rn = rp + 1usize;
            b_vec[rp] = (limit - exo).max(0.0_f64);
            b_vec[rn] = (limit + exo).max(0.0_f64);
        }

        MasterLP { nvars, ncons, c: c_obj, a: a_mat, b: b_vec, col_start, col_map, active_lines }
    }

    #[inline]
    fn push_unique_candidate(cands: &mut Vec<f64>, u: f64, lo: f64, hi: f64) {
        if !u.is_finite() { return; }
        let uc = u.clamp(lo, hi);
        if cands.iter().all(|&v| (v - uc).abs() > 1e-8_f64) {
            cands.push(uc);
        }
    }

    #[inline]
    fn cg_add_nearby_kinks(
        cands: &mut Vec<f64>,
        bat: &Battery,
        soc: f64,
        soc_span: f64,
        soc_levels: usize,
        u: f64,
        charge_branch: bool,
        lo: f64,
        hi: f64,
    ) {
        if soc_levels < 2usize || soc_span <= 1e-12_f64 {
            return;
        }
        let dt = 0.25_f64;
        let eff_c = bat.efficiency_charge.max(1e-9_f64);
        let eff_d = bat.efficiency_discharge.max(1e-9_f64);
        let next_soc = if charge_branch {
            soc + eff_c * (-u).max(0.0_f64) * dt
        } else {
            soc - u.max(0.0_f64) / eff_d * dt
        }.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

        let levels_m1 = (soc_levels - 1usize) as f64;
        let idx_f = ((next_soc - bat.soc_min_mwh) / soc_span * levels_m1).clamp(0.0_f64, levels_m1);
        let idx0 = idx_f.floor() as isize;
        for idx in [idx0, idx0 + 1isize] {
            if idx < 0isize || idx >= soc_levels as isize {
                continue;
            }
            let grid_soc = bat.soc_min_mwh + soc_span * (idx as f64) / levels_m1;
            let kink_u = if charge_branch {
                -(grid_soc - soc) / (eff_c * dt)
            } else {
                (soc - grid_soc) * eff_d / dt
            };
            push_unique_candidate(cands, kink_u, lo, hi);
        }
    }

    fn pricing_subproblem_analytic(
        challenge: &Challenge,
        state: &State,
        ca: &AycdicdbCache,
        b: usize,
        penalty: f64,
        lo: f64,
        hi: f64,
    ) -> (f64, f64, f64) {
        if hi - lo < 1e-8_f64 {
            let profit = eval_profit(challenge, state, ca, b, lo);
            let profit = if profit.is_finite() { profit } else { 0.0_f64 };
            return (lo, profit, profit - penalty * lo);
        }

        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0_f64 };
        let soc = state.socs[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9_f64);
        let soc_levels = ca.dp[b][0].len();
        let t_next = (state.time_step + 1usize).min(ca.dp[b].len() - 1usize);
        let lambda = asca_lambda(&ca.dp[b][t_next], soc, bat.soc_min_mwh, soc_span, soc_levels);
        let deg_coeff = (0.25_f64 / bat.capacity_mwh.max(1e-9_f64)).powi(2);

        let mut cands = Vec::with_capacity(8usize);
        push_unique_candidate(&mut cands, lo, lo, hi);
        push_unique_candidate(&mut cands, hi, lo, hi);
        if lo <= 0.0_f64 && 0.0_f64 <= hi {
            push_unique_candidate(&mut cands, 0.0_f64, lo, hi);
        }

        if lo < 0.0_f64 {
            let hi_c = 0.0_f64.min(hi);
            if lo < hi_c {
                let b_coeff = 0.25_f64 * (lambda * bat.efficiency_charge - rt_price - 0.25_f64) + penalty;
                let cand = if deg_coeff > 1e-30_f64 {
                    let x = b_coeff / (2.0_f64 * deg_coeff);
                    (-x.clamp(0.0_f64, -lo)).clamp(lo, hi_c)
                } else { lo };
                push_unique_candidate(&mut cands, cand, lo, hi_c);
                cg_add_nearby_kinks(&mut cands, bat, soc, soc_span, soc_levels, cand, true, lo, hi_c);
            }
        }

        if hi > 0.0_f64 {
            let lo_d = 0.0_f64.max(lo);
            let eff_d = bat.efficiency_discharge.max(1e-9_f64);
            if lo_d < hi {
                let b_coeff = 0.25_f64 * (rt_price - 0.25_f64 - lambda / eff_d) - penalty;
                let cand = if deg_coeff > 1e-30_f64 {
                    let x = b_coeff / (2.0_f64 * deg_coeff);
                    x.clamp(lo_d, hi)
                } else { hi };
                push_unique_candidate(&mut cands, cand, lo_d, hi);
                cg_add_nearby_kinks(&mut cands, bat, soc, soc_span, soc_levels, cand, false, lo_d, hi);
            }
        }

        let mut best_u = if lo <= 0.0_f64 && 0.0_f64 <= hi { 0.0_f64 } else { lo };
        let mut best_profit = eval_profit(challenge, state, ca, b, best_u);
        if !best_profit.is_finite() { best_profit = 0.0_f64; }
        let mut best_reduced = best_profit - penalty * best_u;

        for &u in &cands {
            let profit = eval_profit(challenge, state, ca, b, u);
            if !profit.is_finite() { continue; }
            let reduced = profit - penalty * u;
            if reduced > best_reduced {
                best_u = u;
                best_profit = profit;
                best_reduced = reduced;
            }
        }

        (best_u, best_profit, best_reduced)
    }

    fn cg_analytic_seed(
        bat: &Battery,
        rt_price: f64,
        lambda: f64,
        u_min: f64,
        u_max: f64,
        deg_coeff: f64,
    ) -> f64 {
        let dt = 0.25_f64;

        let mut best_u = 0.0_f64;
        let two_deg = 2.0_f64 * deg_coeff;

        if u_min < 0.0_f64 {
            let hi = 0.0_f64.min(u_max);
            if u_min < hi && deg_coeff > 1e-30_f64 {
                let b_coeff = dt * (lambda * bat.efficiency_charge - rt_price - 0.25_f64);
                let raw_cand = b_coeff / two_deg;
                let cand = (-raw_cand.clamp(0.0_f64, -u_min)).clamp(u_min, hi);
                best_u = cand;
            }
        }

        if u_max > 0.0_f64 && deg_coeff > 1e-30_f64 {
            let eff_d = bat.efficiency_discharge.max(1e-9_f64);
            let lo = 0.0_f64.max(u_min);
            let b_coeff = dt * (rt_price - 0.25_f64 - lambda / eff_d);
            let raw_cand = b_coeff / two_deg;
            let cand = raw_cand.clamp(lo, u_max);
            let sell_net = rt_price - 0.25_f64 - lambda / eff_d;
            let buy_net = lambda * bat.efficiency_charge - rt_price - 0.25_f64;
            if sell_net > 0.0_f64 && sell_net.abs() >= buy_net.abs() {
                best_u = cand;
            }
        }

        best_u.clamp(u_min, u_max)
    }
}
mod lp {
    const LP_EPS: f64 = 1e-9_f64;

    pub fn lp_solve_with_budget(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, usize) {
        let (sol, _, piv) = lp_solve_with_duals(n, m, c, a, b, max_pivots);
        (sol, piv)
    }

    pub fn lp_solve_with_duals(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, Option<Vec<f64>>, usize) {
        if b.iter().any(|&x| x < -1e-6_f64) {
            return (None, None, 0usize);
        }

        let n_vars = n + m;
        let rhs_col = n_vars;
        let n_cols = n_vars + 1usize;

        let mut tab = vec![vec![0.0_f64; n_cols]; m + 1usize];

        for i in 0..m {
            for j in 0..n {
                tab[i][j] = a[i][j];
            }
            tab[i][n + i] = 1.0_f64;
            tab[i][rhs_col] = b[i].max(0.0_f64);
        }

        for j in 0..n {
            tab[m][j] = -c[j];
        }

        let mut basis: Vec<usize> = (n..n + m).collect();
        let mut pivots_used = 0usize;

        for pivot in 0..max_pivots {
            pivots_used = pivot + 1usize;
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
                None => return (None, None, 0usize),
            };

            let pivot_val = tab[leaving_row][entering];
            if pivot_val.abs() < LP_EPS {
                return (None, None, 0usize);
            }
            for j in 0..n_cols {
                tab[leaving_row][j] /= pivot_val;
            }

            for i in 0..=m {
                if i != leaving_row {
                    let factor = tab[i][entering];
                    if factor.abs() > 1e-15_f64 {
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
                x[bv] = tab[i][rhs_col].max(0.0_f64);
            }
        }

        let mut y = vec![0.0f64; m];
        for i in 0..m {
            let slack_col = n + i;
            if !basis.contains(&slack_col) {
                y[i] = -tab[m][slack_col];
                if y[i].abs() < LP_EPS { y[i] = 0.0_f64; }
            } else {
                y[i] = 0.0_f64;
            }
            if y[i] < 0.0_f64 { y[i] = 0.0_f64; }
        }

        (Some(x), Some(y), pivots_used)
    }

}
mod track_baseline {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 101usize,
            action_grid: 40usize,
            asca_iters: 25usize,
            ternary_iters: 25usize,
            convergence_tol: 1e-4_f64,
            anticipate_lmp: false,
            lmp_threshold: 0.85_f64,
            lmp_premium_scale: 0.45_f64,
            jump_premium: 0.02_f64,
            prune_ratio: 0.00_f64,
            deflator_iters: 15usize,
            flow_margin: 1e-4_f64,
            network_derating: 1.00_f64,
            use_sdp: true,
            use_lp: true,
            lp_refine_sweeps: 3usize,
            use_cg: true,
            cg_iters: 10usize,
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
            soc_levels: 201usize,
            action_grid: 30usize,
            asca_iters: 60usize,
            ternary_iters: 15usize,
            convergence_tol: 1e-3_f64,
            anticipate_lmp: true,
            lmp_threshold: 0.65_f64,
            lmp_premium_scale: 1.00_f64,
            jump_premium: 0.00_f64,
            prune_ratio: 0.00_f64,
            deflator_iters: 50usize,
            flow_margin: 1e-4_f64,
            network_derating: 1.00_f64,
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0usize,
            use_cg: false,
            cg_iters: 0usize,
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
            soc_levels: 201usize,
            action_grid: 30usize,
            asca_iters: 25usize,
            ternary_iters: 15usize,
            convergence_tol: 1e-3_f64,
            anticipate_lmp: true,
            lmp_threshold: 0.65_f64,
            lmp_premium_scale: 1.00_f64,
            jump_premium: 0.00_f64,
            prune_ratio: 0.00_f64,
            deflator_iters: 50usize,
            flow_margin: 1e-4_f64,
            network_derating: 0.22_f64,
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0usize,
            use_cg: false,
            cg_iters: 0usize,
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
            soc_levels: 201usize,
            action_grid: 30usize,
            asca_iters: 45usize,
            ternary_iters: 15usize,
            convergence_tol: 1e-3_f64,
            anticipate_lmp: true,
            lmp_threshold: 0.65_f64,
            lmp_premium_scale: 1.00_f64,
            jump_premium: 0.00_f64,
            prune_ratio: 0.00_f64,
            deflator_iters: 50usize,
            flow_margin: 1e-4_f64,
            network_derating: 0.10_f64,
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0usize,
            use_cg: false,
            cg_iters: 0usize,
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
            soc_levels: 201usize,
            action_grid: 30usize,
            asca_iters: 35usize,
            ternary_iters: 15usize,
            convergence_tol: 1e-3_f64,
            anticipate_lmp: true,
            lmp_threshold: 0.65_f64,
            lmp_premium_scale: 1.00_f64,
            jump_premium: 0.00_f64,
            prune_ratio: 0.00_f64,
            deflator_iters: 50usize,
            flow_margin: 1e-4_f64,
            network_derating: 0.10_f64,
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0usize,
            use_cg: false,
            cg_iters: 0usize,
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
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if n <= 15usize => track_baseline::solve(challenge, save_solution, hyperparameters),
        n if n <= 30usize => track_congested::solve(challenge, save_solution, hyperparameters),
        n if n <= 50usize => track_multiday::solve(challenge, save_solution, hyperparameters),
        n if n <= 80usize => track_dense::solve(challenge, save_solution, hyperparameters),
        n if n <= 150usize => track_capstone::solve(challenge, save_solution, hyperparameters),
        n => Err(anyhow!(
            "aycdicdb: unsupported num_batteries={} (expected one of 10/20/40/60/100)", n
        )),
    }
}