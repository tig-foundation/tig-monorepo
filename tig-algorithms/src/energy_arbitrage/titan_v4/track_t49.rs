// TIG's UI uses the pattern `tig_challenges::energy_arbitrage` to automatically detect your algorithm's challenge
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
        // per-timestep sigma ∝ (p_DA/mean_p)^alpha; 0.0 = constant sigma (backward-compat)
        pub sdp_sigma_het_alpha: f64,
        // global scale multiplier on sdp_sigma_eff (restored from i23 where scale=1.25 gave +0.55%)
        pub sdp_sigma_scale: f64,
        // OCO SOC tracking coeff; 0.0 = disabled (backward-compat)
        pub lambda_track: f64,
        // exact analytical pricing oracle (replace 30-iter golden section in CG)
        pub use_analytical_pricing: bool,
        // Morales VI energy bounds in master LP
        pub use_morales_vi: bool,
        // seed CG column pool with warm-start columns from previous timestep LP duals
        pub use_lp_dual_warmstart: bool,
        // per-battery LMP premium ∝ capacity; 0.0 = uniform (backward-compat)
        pub het_lmp_alpha: f64,
        // reuse final column pool from previous nonce (cross-nonce column warm-start)
        pub use_lp_basis_warmstart: bool,
        // per-battery sigma SDP scaling by c-rate (cap/power); 0.0 = uniform (backward-compat)
        pub het_crate_alpha: f64,
        // after CG dispatch, also run joint_lp_dispatch and return better-profit solution
        pub use_cg_lp_combine: bool,
        // i72: temporal aggregation — cluster T=96 timesteps into k representative steps for DP
        // 0 = disabled (use all T steps), else build DP on k_clusters+1 levels
        pub k_clusters: usize,
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
            if let Some(v) = m.get("use_sdp").and_then(|v| v.as_bool()) { self.use_sdp = v; }
            if let Some(v) = m.get("use_lp").and_then(|v| v.as_bool()) { self.use_lp = v; }
            if let Some(v) = m.get("lp_refine_sweeps").and_then(|v| v.as_u64()) { self.lp_refine_sweeps = v as usize; }
            if let Some(v) = m.get("use_cg").and_then(|v| v.as_bool()) { self.use_cg = v; }
            if let Some(v) = m.get("cg_iters").and_then(|v| v.as_u64()) { self.cg_iters = v as usize; }
            if let Some(v) = m.get("sdp_sigma_het_alpha").and_then(|v| v.as_f64()) { self.sdp_sigma_het_alpha = v.clamp(-2.0, 3.0); }
            if let Some(v) = m.get("sdp_sigma_scale").and_then(|v| v.as_f64()) { self.sdp_sigma_scale = v.clamp(0.01, 5.0); }
            if let Some(v) = m.get("lambda_track").and_then(|v| v.as_f64()) { self.lambda_track = v.clamp(0.0, 10.0); }
            if let Some(v) = m.get("use_analytical_pricing").and_then(|v| v.as_bool()) { self.use_analytical_pricing = v; }
            if let Some(v) = m.get("use_morales_vi").and_then(|v| v.as_bool()) { self.use_morales_vi = v; }
            if let Some(v) = m.get("use_lp_dual_warmstart").and_then(|v| v.as_bool()) { self.use_lp_dual_warmstart = v; }
            if let Some(v) = m.get("het_lmp_alpha").and_then(|v| v.as_f64()) { self.het_lmp_alpha = v.clamp(-2.0, 2.0); }
            if let Some(v) = m.get("use_lp_basis_warmstart").and_then(|v| v.as_bool()) { self.use_lp_basis_warmstart = v; }
            if let Some(v) = m.get("het_crate_alpha").and_then(|v| v.as_f64()) { self.het_crate_alpha = v.clamp(-2.0, 3.0); }
            if let Some(v) = m.get("use_cg_lp_combine").and_then(|v| v.as_bool()) { self.use_cg_lp_combine = v; }
            if let Some(v) = m.get("k_clusters").and_then(|v| v.as_u64()) { self.k_clusters = v as usize; }
        }
    }

    pub struct TitanCache {
        pub dp: Vec<Vec<Vec<f64>>>,
        pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
        pub b_to_lines: Vec<Vec<(usize, f64)>>,
        pub batt_nodes: Vec<usize>,
        // soc_ref[b][ck] = soc at argmax_i dp[b][ck][i] — DP mode trajectory (cluster-indexed)
        pub soc_ref: Vec<Vec<f64>>,
        // i72: cluster_map[t] = cluster index ck for actual timestep t (len = num_t+1)
        pub cluster_map: Vec<usize>,
    }

    struct Inner {
        hp: TrackHp,
        cache: Option<TitanCache>,
    }

    thread_local! {
        static STATE: RefCell<Option<Inner>> = RefCell::new(None);
        // penalty_prev[b] from last CG final LP — reset per nonce
        static CG_PREV_DUALS: RefCell<Option<Vec<f64>>> = RefCell::new(None);
        // columns[timestep][battery] from last nonce — NOT reset per nonce (cross-nonce)
        static CG_PREV_COLS: RefCell<Vec<Vec<Vec<f64>>>> = RefCell::new(Vec::new());
    }

    pub fn solve_with_hp(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hp: TrackHp,
    ) -> Result<()> {
        STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
        CG_PREV_DUALS.with(|pd| *pd.borrow_mut() = None);
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

            let mut actions = if hp.use_cg {
                let cg_actions = run_column_generation(challenge, state, cache, hp, &flows_base);
                if hp.use_cg_lp_combine {
                    if let Some(lp_act) = joint_lp_dispatch(challenge, state, cache, &flows_base) {
                        let profit_cg: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, cg_actions[b]))
                            .sum();
                        let profit_lp: f64 = (0..challenge.num_batteries)
                            .map(|b| eval_profit(challenge, state, cache, b, lp_act[b]))
                            .sum();
                        if profit_lp > profit_cg { lp_act } else { cg_actions }
                    } else {
                        cg_actions
                    }
                } else {
                    cg_actions
                }
            } else {
                let mut actions_asca = vec![0.0; challenge.num_batteries];
                run_asca(challenge, state, cache, hp, &flows_base, &mut actions_asca);
                run_deflator(challenge, state, cache, hp, &flows_base, &mut actions_asca);

                let mut actions = actions_asca.clone();

                if hp.use_lp {
                    if let Some(lp_act) = joint_lp_dispatch(challenge, state, cache, &flows_base) {
                        let mut actions_lp_warm = lp_act.clone();
                        if hp.lp_refine_sweeps > 0 {
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

            // Final deflator safety net: ensures ALL action paths (CG, ASCA, LP)
            // are within network margin before returning. Idempotent on already-feasible actions.
            run_deflator(challenge, state, cache, hp, &flows_base, &mut actions);
            Ok(actions)
        })
    }

    fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> TitanCache {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let num_t = challenge.num_steps;

        let zero_action = vec![0.0_f64; num_b];
        let inj_base = challenge.compute_total_injections(state, &zero_action);
        let flows0 = challenge.network.compute_flows(&inj_base);

        // Use battery.node directly — no injection simulation needed for node discovery
        let batt_nodes: Vec<usize> = challenge.batteries.iter().map(|b| b.node).collect();
        let mut ptdf_sparse: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_l];
        let mut b_to_lines: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_b];
        let mut dummy = zero_action.clone();
        for b in 0..num_b {
            dummy[b] = 1.0;
            let inj1 = challenge.compute_total_injections(state, &dummy);
            let flows1 = challenge.network.compute_flows(&inj1);
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
            // per-battery capacity scaling of LMP premium
            let cap_scale: Vec<f64> = if hp.het_lmp_alpha != 0.0 && num_b > 1 {
                let caps: Vec<f64> = (0..num_b).map(|b| challenge.batteries[b].capacity_mwh.max(1e-9)).collect();
                let mean_cap = caps.iter().sum::<f64>() / num_b as f64;
                caps.iter().map(|&c| 1.0 + hp.het_lmp_alpha * (c / mean_cap - 1.0)).collect()
            } else {
                vec![1.0_f64; num_b]
            };
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
                                let nodal_shift = -impact * sign_f * premium * cap_scale[b];
                                expected_premiums[t][b] += nodal_shift;
                            }
                        }
                    }
                }
            }
        }

        // i72: temporal aggregation — cluster T timesteps into k representative steps
        let k_eff = if hp.k_clusters == 0 || hp.k_clusters >= num_t { num_t } else { hp.k_clusters };
        let cluster_map: Vec<usize> = {
            let mut map = vec![0usize; num_t + 1];
            if k_eff == num_t {
                for t in 0..=num_t { map[t] = t; }
            } else {
                for t in 0..num_t { map[t] = (t * k_eff / num_t).min(k_eff - 1); }
                map[num_t] = k_eff;
            }
            map
        };
        let k_actual = if k_eff == num_t { num_t } else { k_eff };

        // Cluster representative DA prices [b][ck] and premiums [ck][b]
        let (cluster_da_prices, cluster_premiums): (Vec<Vec<f64>>, Vec<Vec<f64>>) = {
            let mut cda = vec![vec![0.0_f64; k_actual]; num_b];
            let mut cprem = vec![vec![0.0_f64; num_b]; k_actual];
            let mut counts = vec![0usize; k_actual];
            for t in 0..num_t {
                let ck = cluster_map[t].min(k_actual.saturating_sub(1));
                counts[ck] += 1;
                for b in 0..num_b {
                    let node = batt_nodes[b];
                    let p = if node < challenge.market.day_ahead_prices[t].len() {
                        challenge.market.day_ahead_prices[t][node]
                    } else {
                        challenge.market.day_ahead_prices[t][0]
                    };
                    cda[b][ck] += p;
                    cprem[ck][b] += expected_premiums[t][b];
                }
            }
            for ck in 0..k_actual {
                let n = counts[ck].max(1) as f64;
                for b in 0..num_b { cda[b][ck] /= n; cprem[ck][b] /= n; }
            }
            (cda, cprem)
        };

        let soc_levels = hp.soc_levels;
        let dt = 0.25_f64;
        let mut dp = vec![vec![vec![0.0_f64; soc_levels]; k_actual + 1]; num_b];

        // K=5 Gauss-Hermite SDP: σ_eff from market params, exact for degree-9 polynomials
        const GH5_Z: [f64; 5] = [0.0, 0.9586, -0.9586, 2.0202, -2.0202];
        const GH5_W: [f64; 5] = [0.5333, 0.2221, 0.2221, 0.0113, 0.0113];
        let sdp_sigma_eff = if hp.use_sdp {
            let sigma = challenge.market.params.volatility;
            let rho_j = challenge.market.params.jump_probability;
            let alpha_j = challenge.market.params.tail_index;
            let jump_var = if alpha_j > 2.0 { rho_j * alpha_j / (alpha_j - 2.0) } else { rho_j * 4.0 };
            (sigma * sigma + jump_var).sqrt() * hp.sdp_sigma_scale
        } else { 0.0 };

        // per-battery c-rate scale for sigma — slow batteries hedge more
        let c_rate_scale: Vec<f64> = if hp.use_sdp && hp.het_crate_alpha != 0.0 && num_b > 1 {
            let crates: Vec<f64> = (0..num_b).map(|b| {
                let bat = &challenge.batteries[b];
                let mean_pwr = ((bat.power_charge_mw + bat.power_discharge_mw) / 2.0).max(1e-9);
                bat.capacity_mwh.max(1e-9) / mean_pwr
            }).collect();
            let mean_cr = crates.iter().sum::<f64>() / num_b as f64;
            crates.iter().map(|&cr| (cr / mean_cr).powf(hp.het_crate_alpha)).collect()
        } else {
            vec![1.0_f64; num_b]
        };

        // per-battery mean DA price for heteroscedastic sigma scaling
        let mean_da_price: Vec<f64> = if hp.use_sdp && hp.sdp_sigma_het_alpha != 0.0 {
            (0..num_b).map(|b| {
                let node = batt_nodes[b];
                let sum: f64 = (0..num_t).map(|t| {
                    if node < challenge.market.day_ahead_prices[t].len() {
                        challenge.market.day_ahead_prices[t][node]
                    } else {
                        challenge.market.day_ahead_prices[t][0]
                    }
                }).sum();
                (sum / num_t as f64).max(1e-9)
            }).collect()
        } else {
            vec![1.0; num_b]
        };

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let deg_coeff = (dt / bat.capacity_mwh.max(1e-9)).powi(2);
            let mean_p = mean_da_price[b];

            // i72: backward over k_actual cluster steps (representative prices)
            for ck in (0..k_actual).rev() {
                let p_da = cluster_da_prices[b][ck];
                let extra = cluster_premiums[ck][b];

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
                    let u_max = (max_pwr_d.min(discharge_soc_limit.max(0.0))).max(u_min);

                    let v_next_slice = &dp[b][ck + 1];

                    let max_val = if hp.use_sdp {
                        let sigma_t = if hp.sdp_sigma_het_alpha != 0.0 {
                            sdp_sigma_eff * (p_da / mean_p.max(1e-9)).powf(hp.sdp_sigma_het_alpha) * c_rate_scale[b]
                        } else {
                            sdp_sigma_eff * c_rate_scale[b]
                        };
                        let mut val_sum = 0.0_f64;
                        for k in 0..5 {
                            let p = (p_da * (1.0 + sigma_t * GH5_Z[k]) + extra).max(1e-6);
                            val_sum += GH5_W[k] * dp_analytic_max(bat, p, p, soc, u_min, u_max, v_next_slice, soc_levels, soc_span, deg_coeff);
                        }
                        val_sum
                    } else {
                        let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
                        let p_buy = p_da + extra;
                        dp_analytic_max(bat, p_buy, p_sell, soc, u_min, u_max, v_next_slice, soc_levels, soc_span, deg_coeff)
                    };
                    dp[b][ck][i] = max_val;
                }
            }
        }

        // soc_ref[b][ck] = soc at argmax_i dp[b][ck][i] (cluster-indexed)
        let mut soc_ref = vec![vec![0.0_f64; k_actual + 1]; num_b];
        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            for ck in 0..=k_actual {
                let best_k = dp[b][ck].iter().enumerate()
                    .max_by(|(_, a), (_, bb)| a.partial_cmp(bb).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(soc_levels / 2);
                soc_ref[b][ck] = bat.soc_min_mwh + soc_span * (best_k as f64) / ((soc_levels - 1) as f64);
            }
        }

        TitanCache { dp, ptdf_sparse, b_to_lines, batt_nodes, soc_ref, cluster_map }
    }

    // Analytic DP step: analytic quadratic opt on each branch + exact eval at 3-5 candidates.
    // Replaces the action_grid enumeration — ~10× fewer evaluations.
    fn dp_analytic_max(
        bat: &Battery,
        p_buy: f64, p_sell: f64,
        soc: f64, u_min: f64, u_max: f64,
        v_next: &[f64],
        soc_levels: usize, soc_span: f64,
        deg_coeff: f64,
    ) -> f64 {
        let dt = 0.25_f64;
        let cap = bat.capacity_mwh.max(1e-9);

        // Compute lambda (dV/dSOC) at current SOC for future-value linearisation
        let lambda = if soc_levels > 1 {
            let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
            let idx0 = (idx_f.floor() as usize).min(soc_levels - 2);
            let delta_soc = soc_span / ((soc_levels - 1) as f64);
            (v_next[idx0 + 1] - v_next[idx0]) / delta_soc
        } else { 0.0 };

        let eval = |u: f64| -> f64 {
            let price = if u > 0.0 { p_sell } else { p_buy };
            let abs_u = u.abs();
            let profit = u * price * dt - 0.25 * abs_u * dt - deg_coeff * u * u;
            let next_soc = if u < 0.0 {
                soc + bat.efficiency_charge * (-u) * dt
            } else {
                soc - u / bat.efficiency_discharge.max(1e-9) * dt
            };
            let next_soc = next_soc.clamp(bat.soc_min_mwh, bat.soc_max_mwh);
            let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
            let idx0 = (idx_f.floor() as isize).max(0) as usize;
            let i0 = idx0.min(soc_levels - 1);
            let i1 = (idx0 + 1).min(soc_levels - 1);
            let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
            profit + v_next[i0] * (1.0 - frac) + v_next[i1] * frac
        };

        let mut best = eval(0.0);

        // Charge branch: u in [u_min, 0)
        if u_min < 0.0 {
            let u_hi = 0.0_f64.min(u_max);
            if u_min < u_hi {
                // B_charge = dt*(lambda*eff_c - p_buy - 0.25); argmax c = B/(2*deg_coeff)
                let b_c = dt * (lambda * bat.efficiency_charge - p_buy - 0.25);
                let x_star = if deg_coeff > 1e-30 { b_c / (2.0 * deg_coeff) } else { -u_min };
                let cand = (-x_star.clamp(0.0, -u_min)).clamp(u_min, u_hi);
                let v = eval(cand); if v > best { best = v; }
                let v = eval(u_min); if v > best { best = v; } // boundary
            }
        }

        // Discharge branch: u in (0, u_max]
        if u_max > 0.0 {
            let u_lo = 0.0_f64.max(u_min);
            if u_lo < u_max {
                let eff_d = bat.efficiency_discharge.max(1e-9);
                let b_d = dt * (p_sell - 0.25 - lambda / eff_d);
                let x_star = if deg_coeff > 1e-30 { b_d / (2.0 * deg_coeff) } else { u_max };
                let cand = x_star.clamp(u_lo, u_max);
                let v = eval(cand); if v > best { best = v; }
                let v = eval(u_max); if v > best { best = v; } // boundary
            }
        }

        if best == f64::NEG_INFINITY { 0.0 } else { best }
    }

    // Extract lambda (dV/dSOC) from DP table for ASCA analytic solve
    #[inline]
    fn asca_lambda(v_table: &[f64], soc: f64, soc_min: f64, soc_span: f64, soc_levels: usize) -> f64 {
        if soc_levels < 2 { return 0.0; }
        let idx_f = (soc - soc_min) / soc_span * ((soc_levels - 1) as f64);
        let idx0 = (idx_f.floor() as usize).min(soc_levels - 2);
        let delta_soc = soc_span / ((soc_levels - 1) as f64);
        (v_table[idx0 + 1] - v_table[idx0]) / delta_soc.max(1e-12)
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
        let t_next = ca.cluster_map.get(state.time_step + 1)
            .copied().unwrap_or(ca.dp[b].len() - 1).min(ca.dp[b].len() - 1);
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
        // Initialize flows to include contributions from current actions (supports warm-start from non-zero actions)
        let mut flows: Vec<f64> = flows_base.to_vec();
        for b in 0..num_b {
            if actions[b].abs() > 1e-12 {
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows[l] += p * actions[b]; }
                }
            }
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

                // Analytic quadratic optimum per branch + eval at candidate + boundary
                let bat = &challenge.batteries[b];
                let node = ca.batt_nodes[b];
                let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                let deg_coeff = (0.25_f64 / bat.capacity_mwh.max(1e-9)).powi(2);
                let soc_levels = ca.dp[b][0].len();
                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let t_next = ca.cluster_map.get(state.time_step + 1)
                    .copied().unwrap_or(ca.dp[b].len() - 1).min(ca.dp[b].len() - 1);
                let lambda = asca_lambda(&ca.dp[b][t_next], state.socs[b], bat.soc_min_mwh, soc_span, soc_levels);

                // OCO SOC tracking — soc deviation from DP mode trajectory
                let dt = 0.25_f64;
                let soc_dev = if hp.lambda_track > 0.0 {
                    let soc_ref_t = ca.soc_ref[b][t_next.min(ca.soc_ref[b].len() - 1)];
                    state.socs[b] - soc_ref_t
                } else { 0.0 };

                if u_min < 0.0 {
                    let lo = u_min; let hi = 0.0_f64.min(u_max);
                    if lo < hi {
                        let eta_c = bat.efficiency_charge;
                        // tracking adds -2*λ_track*soc_dev*eta_c*dt to b_coeff (via c=-u)
                        //      and λ_track*eta_c²*dt² to deg_coeff_eff
                        let b_coeff = dt * (-rt_price - 0.25 + lambda * eta_c)
                            - 2.0 * hp.lambda_track * soc_dev * eta_c * dt;
                        let deg_eff = deg_coeff + hp.lambda_track * eta_c * eta_c * dt * dt;
                        let cand = if deg_eff > 1e-30 {
                            let x = b_coeff / (2.0 * deg_eff);
                            (-x.clamp(0.0, -lo)).clamp(lo, hi)
                        } else { lo };
                        for &u in &[cand, lo] {
                            let v = eval_profit(challenge, state, ca, b, u);
                            if v > best_v { best_v = v; best_u = u; }
                        }
                    }
                }

                if u_max > 0.0 {
                    let lo = 0.0_f64.max(u_min); let hi = u_max;
                    if lo < hi {
                        let eff_d = bat.efficiency_discharge.max(1e-9);
                        // tracking adds +2*λ_track*soc_dev*dt/eta_d to b_coeff
                        //      and λ_track*dt²/eta_d² to deg_coeff_eff
                        let b_coeff = dt * (rt_price - 0.25 - lambda / eff_d)
                            + 2.0 * hp.lambda_track * soc_dev * dt / eff_d;
                        let deg_eff = deg_coeff + hp.lambda_track * dt * dt / (eff_d * eff_d);
                        let cand = if deg_eff > 1e-30 {
                            let x = b_coeff / (2.0 * deg_eff);
                            x.clamp(lo, hi)
                        } else { hi };
                        for &u in &[cand, hi] {
                            let v = eval_profit(challenge, state, ca, b, u);
                            if v > best_v { best_v = v; best_u = u; }
                        }
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

    fn joint_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        flows_base: &[f64],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let tx_cost = 0.25_f64;
        let n = 2 * num_b;
        let m = 4 * num_b + 2 * num_l;

        let mut c_obj = vec![0.0_f64; n];
        let mut a_mat = vec![vec![0.0_f64; n]; m];
        let mut b_vec = vec![0.0_f64; m];

        let t_next = ca.cluster_map.get(state.time_step + 1)
            .copied().unwrap_or(ca.dp[0].len() - 1).min(ca.dp[0].len() - 1);

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let soc = state.socs[b];
            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let dv = asca_lambda(&ca.dp[b][t_next], soc, bat.soc_min_mwh, soc_span, soc_levels);

            c_obj[b]         = (rt - tx_cost) * dt - dv / eta_d * dt;
            c_obj[num_b + b] = -(rt + tx_cost) * dt + dv * eta_c * dt;

            let (u_min, u_max) = state.action_bounds[b];
            let r = 4 * b;
            // d_b <= u_max (discharge power bound)
            a_mat[r][b] = 1.0;
            b_vec[r] = u_max.max(0.0);
            // c_b <= -u_min (charge power bound; u_min <= 0 so -u_min >= 0)
            a_mat[r + 1][num_b + b] = 1.0;
            b_vec[r + 1] = (-u_min).max(0.0);
            // d_b/eta_d - c_b*eta_c <= soc - soc_min (SOC lower bound prevents over-discharge)
            a_mat[r + 2][b]         =  dt / eta_d;
            a_mat[r + 2][num_b + b] = -eta_c * dt;
            b_vec[r + 2] = (soc - bat.soc_min_mwh).max(0.0);
            // -d_b/eta_d + c_b*eta_c <= soc_max - soc (SOC upper bound prevents over-charge)
            a_mat[r + 3][b]         = -dt / eta_d;
            a_mat[r + 3][num_b + b] =  eta_c * dt;
            b_vec[r + 3] = (bat.soc_max_mwh - soc).max(0.0);
        }

        let row_f = 4 * num_b;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            if limit <= 1e-6 { continue; }
            let exo = flows_base[l];
            let rp = row_f + 2 * l;
            let rn = rp + 1;
            for &(b, impact) in &ca.ptdf_sparse[l] {
                // Positive flow direction: sum PTDF*(d-c) <= limit - exo
                a_mat[rp][b]         += impact;
                a_mat[rp][num_b + b] -= impact;
                // Negative flow direction: sum PTDF*(c-d) <= limit + exo
                a_mat[rn][b]         -= impact;
                a_mat[rn][num_b + b] += impact;
            }
            b_vec[rp] = (limit - exo).max(0.0);
            b_vec[rn] = (limit + exo).max(0.0);
        }

        let (opt_x, _) = super::lp::lp_solve_with_budget(n, m, &c_obj, &a_mat, &b_vec, 3000);
        let opt_x = opt_x?;

        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let u = opt_x[b] - opt_x[num_b + b];
            let (lo, hi) = state.action_bounds[b];
            actions[b] = u.clamp(lo, hi);
        }
        Some(actions)
    }

    // ============================================================================
    // Dantzig-Wolfe Column Generation dispatch
    // ============================================================================

    fn run_column_generation(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        // 1. Initialise columns per battery
        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); num_b];
        for b in 0..num_b {
            let (u_min, u_max) = state.action_bounds[b];
            columns[b].push(0.0);
            if u_min < 0.0 { columns[b].push(u_min); }
            if u_max > 0.0 { columns[b].push(u_max); }
            // Add unconstrained-profit-optimal action as a seed column
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let t_next = ca.cluster_map.get(state.time_step + 1)
                .copied().unwrap_or(ca.dp[b].len() - 1).min(ca.dp[b].len() - 1);
            let lambda = asca_lambda(&ca.dp[b][t_next], state.socs[b], bat.soc_min_mwh, soc_span, soc_levels);
            let deg_coeff = (0.25_f64 / bat.capacity_mwh.max(1e-9)).powi(2);
            let u_opt = cg_analytic_seed(bat, rt_price, lambda, u_min, u_max, deg_coeff);
            if !columns[b].iter().any(|&u| (u - u_opt).abs() < 1e-8) {
                columns[b].push(u_opt);
            }
        }

        // warm-start seed — add columns derived from previous timestep's optimal duals
        if hp.use_lp_dual_warmstart {
            CG_PREV_DUALS.with(|pd| {
                if let Some(ref penalty_p) = *pd.borrow() {
                    for b in 0..num_b {
                        let (u_min, u_max) = state.action_bounds[b];
                        let pen = if b < penalty_p.len() { penalty_p[b] } else { 0.0 };
                        let (u_ws, _) = golden_section_subproblem(challenge, state, ca, b, pen, u_min, u_max);
                        if !columns[b].iter().any(|&u| (u - u_ws).abs() < 1e-8) {
                            columns[b].push(u_ws);
                        }
                    }
                }
            });
        }

        // seed column pool with columns from the same timestep of the previous nonce
        if hp.use_lp_basis_warmstart {
            let t = state.time_step;
            CG_PREV_COLS.with(|pc| {
                let guard = pc.borrow();
                if t < guard.len() && guard[t].len() == num_b {
                    for b in 0..num_b {
                        let (u_min, u_max) = state.action_bounds[b];
                        for &u in &guard[t][b] {
                            let u_c = u.clamp(u_min, u_max);
                            if !columns[b].iter().any(|&c| (c - u_c).abs() < 1e-8) {
                                columns[b].push(u_c);
                            }
                        }
                    }
                }
            });
        }

        let mut lp_obj_prev = f64::NEG_INFINITY;
        let mut no_improve = 0u32;

        for _iter in 0..hp.cg_iters {
            let master = build_master_lp(challenge, state, ca, &columns, flows_base, hp.flow_margin, hp);
            if master.nvars == 0 || master.ncons == 0 {
                break;
            }

            let (sol, duals, _pivots) = super::lp::lp_solve_with_duals(
                master.nvars, master.ncons, &master.c, &master.a, &master.b, 3000,
            );

            let (Some(x), Some(y)) = (sol, duals) else { break; };

            // Compute sigma (convexity duals) per battery — one constraint per battery
            let mut sigma = vec![0.0f64; num_b];
            for b in 0..num_b {
                sigma[b] = if b < y.len() { y[b] } else { 0.0 };
            }

            // Extract flow duals -> penalty per battery
            let mut penalty = vec![0.0f64; num_b];
            let dual_start = num_b;
            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let y_pos = if dual_start + 2 * l < y.len() { y[dual_start + 2 * l] } else { 0.0 };
                let y_neg = if dual_start + 2 * l + 1 < y.len() { y[dual_start + 2 * l + 1] } else { 0.0 };
                let net_dual = y_pos - y_neg;
                for &(b_idx, impact) in &ca.ptdf_sparse[l] {
                    penalty[b_idx] += net_dual * impact;
                }
            }

            // Pricing subproblems
            let mut added = false;
            for b in 0..num_b {
                let (u_min, u_max) = state.action_bounds[b];
                let (u_star, obj_star) = if hp.use_analytical_pricing {
                    pricing_analytical(challenge, state, ca, b, penalty[b], u_min, u_max)
                } else {
                    golden_section_subproblem(challenge, state, ca, b, penalty[b], u_min, u_max)
                };
                let reduced_cost = obj_star - sigma[b];
                if reduced_cost > 1e-6 {
                    if !columns[b].iter().any(|&u| (u - u_star).abs() < 1e-8) {
                        columns[b].push(u_star);
                        added = true;
                    }
                }
            }

            // Stagnation detection
            let mut lp_obj = 0.0;
            for b in 0..num_b {
                let start = master.col_start[b];
                for (c_local, &idx) in master.col_map[b].iter().enumerate() {
                    let u = columns[b][idx];
                    let prof = eval_profit(challenge, state, ca, b, u);
                    let var_idx = start + c_local;
                    if var_idx < x.len() {
                        lp_obj += prof * x[var_idx];
                    }
                }
            }
            if lp_obj.is_finite() && lp_obj_prev.is_finite() && lp_obj <= lp_obj_prev + 1e-6 {
                no_improve += 1;
                if no_improve >= 3 { break; }
            } else {
                no_improve = 0;
                if lp_obj.is_finite() { lp_obj_prev = lp_obj; }
            }

            if !added { break; }
        }

        // Final LP solve for convex combination
        let master = build_master_lp(challenge, state, ca, &columns, flows_base, hp.flow_margin, hp);
        if master.nvars == 0 {
            return vec![0.0; num_b];
        }
        let (sol, final_duals_opt, _) = super::lp::lp_solve_with_duals(
            master.nvars, master.ncons, &master.c, &master.a, &master.b, 3000,
        );

        // store final penalty duals for next timestep warm-start seed
        if hp.use_lp_dual_warmstart {
            if let Some(ref y_f) = final_duals_opt {
                let mut penalty_f = vec![0.0f64; num_b];
                let ds = num_b;
                for l in 0..num_l {
                    if challenge.network.flow_limits[l] <= 1e-6 { continue; }
                    let yp = if ds + 2 * l < y_f.len() { y_f[ds + 2 * l] } else { 0.0 };
                    let yn = if ds + 2 * l + 1 < y_f.len() { y_f[ds + 2 * l + 1] } else { 0.0 };
                    let nd = yp - yn;
                    for &(b_idx, impact) in &ca.ptdf_sparse[l] {
                        penalty_f[b_idx] += nd * impact;
                    }
                }
                CG_PREV_DUALS.with(|pd| *pd.borrow_mut() = Some(penalty_f));
            }
        }

        // store columns for cross-nonce warm-start at this timestep
        if hp.use_lp_basis_warmstart {
            let t = state.time_step;
            CG_PREV_COLS.with(|pc| {
                let mut guard = pc.borrow_mut();
                if guard.len() <= t { guard.resize(t + 1, Vec::new()); }
                guard[t] = columns.clone();
            });
        }

        let x = match sol {
            Some(x) => x,
            None => return vec![0.0; num_b],
        };

        // Construct action as convex combination of columns
        let mut actions = vec![0.0f64; num_b];
        for b in 0..num_b {
            let mut u_sum = 0.0;
            for (c_local, &idx) in master.col_map[b].iter().enumerate() {
                let var_idx = master.col_start[b] + c_local;
                if var_idx < x.len() && idx < columns[b].len() {
                    u_sum += x[var_idx] * columns[b][idx];
                }
            }
            let (lo, hi) = state.action_bounds[b];
            actions[b] = if u_sum.is_finite() { u_sum.clamp(lo, hi) } else { 0.0 };
        }

        // Risk check: if total profit is near-zero or negative, fall back to ASCA
        let total_profit: f64 = (0..num_b)
            .map(|b| eval_profit(challenge, state, ca, b, actions[b]))
            .sum();
        if total_profit.is_finite() && total_profit < -1e-3 {
            let mut actions_asca = vec![0.0; num_b];
            run_asca(challenge, state, ca, hp, flows_base, &mut actions_asca);
            return actions_asca;
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
    }

    fn build_master_lp(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        columns: &[Vec<f64>],
        flows_base: &[f64],
        flow_margin: f64,
        hp: &TrackHp,
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

        // Morales VI adds 2*num_b extra rows (discharge + charge energy bounds per battery)
        let vi_rows = if hp.use_morales_vi { 2 * num_b } else { 0 };
        let ncons = num_b + 2 * num_l + vi_rows;

        let mut c_obj = vec![0.0_f64; nvars];
        let mut a_mat = vec![vec![0.0_f64; nvars]; ncons];
        let mut b_vec = vec![0.0_f64; ncons];

        // Convexity upper-bound constraints (sum x <= 1 per battery)
        for b in 0..num_b {
            let start = col_start[b];
            let nc = columns[b].len();
            for j in 0..nc {
                a_mat[b][start + j] = 1.0;
            }
            b_vec[b] = 1.0;

            // Objective coefficients
            for j in 0..nc {
                let u = columns[b][j];
                c_obj[start + j] = eval_profit(challenge, state, ca, b, u);
            }
        }

        // Flow constraints
        let flow_start = num_b;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l] - flow_margin;
            let exo = if l < flows_base.len() { flows_base[l] } else { 0.0 };
            let rp = flow_start + 2 * l;
            let rn = flow_start + 2 * l + 1;

            if limit <= 1e-6 {
                b_vec[rp] = 0.0;
                b_vec[rn] = 0.0;
                continue;
            }

            for b in 0..num_b {
                let start = col_start[b];
                for j in 0..columns[b].len() {
                    let u = columns[b][j];
                    for &(line, impact) in &ca.b_to_lines[b] {
                        if line == l {
                            a_mat[rp][start + j] += impact * u;
                        }
                    }
                }
            }

            // RHS must be non-negative for simplex: ensure feasibility
            // Positive: sum(impact*u*x) <= limit - exo
            b_vec[rp] = (limit - exo).max(0.0);
            // Negative: -sum(impact*u*x) <= limit + exo
            for v in 0..nvars {
                a_mat[rn][v] = -a_mat[rp][v];
            }
            b_vec[rn] = (limit + exo).max(0.0);
        }

        // Morales VI — energy window constraints per battery (discharge + charge components)
        // For each battery b: sum_j lambda[b][j]*pos(u[b][j]) <= (soc-soc_min)*eff_d/dt
        //                     sum_j lambda[b][j]*neg(|u[b][j]|) <= (soc_max-soc)*eff_c/dt
        if hp.use_morales_vi {
            let dt = 0.25_f64;
            let vi_base = num_b + 2 * num_l;
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc = state.socs[b];
                let start = col_start[b];
                // Discharge energy bound row
                let r_d = vi_base + 2 * b;
                for j in 0..columns[b].len() {
                    let u = columns[b][j];
                    if u > 0.0 { a_mat[r_d][start + j] = u; }
                }
                b_vec[r_d] = ((soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt).max(0.0);
                // Charge energy bound row (negated for A x <= b form)
                let r_c = vi_base + 2 * b + 1;
                for j in 0..columns[b].len() {
                    let u = columns[b][j];
                    if u < 0.0 { a_mat[r_c][start + j] = -u; }
                }
                b_vec[r_c] = ((bat.soc_max_mwh - soc) * bat.efficiency_charge / dt).max(0.0);
            }
        }

        MasterLP { nvars, ncons, c: c_obj, a: a_mat, b: b_vec, col_start, col_map }
    }

    /// Golden-section search to maximise f(u) = eval_profit(b,u) - penalty * u.
    /// Returns (u_opt, f_opt).
    fn golden_section_subproblem(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        b: usize,
        penalty: f64,
        lo: f64,
        hi: f64,
    ) -> (f64, f64) {
        if hi - lo < 1e-8 {
            let f = eval_profit(challenge, state, ca, b, lo) - penalty * lo;
            return (lo, if f.is_finite() { f } else { 0.0 });
        }
        let invphi = 0.6180339887498948;
        let mut a = lo;
        let mut c = hi;
        let mut b1 = a + invphi * (c - a);
        let mut f1 = eval_profit(challenge, state, ca, b, b1) - penalty * b1;
        if !f1.is_finite() { f1 = 0.0; }
        let mut b2 = c - invphi * (c - a);
        let mut f2 = eval_profit(challenge, state, ca, b, b2) - penalty * b2;
        if !f2.is_finite() { f2 = 0.0; }

        for _ in 0..30 {
            if f1 > f2 {
                c = b2;
                b2 = b1;
                f2 = f1;
                b1 = a + invphi * (c - a);
                f1 = eval_profit(challenge, state, ca, b, b1) - penalty * b1;
                if !f1.is_finite() { f1 = 0.0; }
            } else {
                a = b1;
                b1 = b2;
                f1 = f2;
                b2 = c - invphi * (c - a);
                f2 = eval_profit(challenge, state, ca, b, b2) - penalty * b2;
                if !f2.is_finite() { f2 = 0.0; }
            }
            if (c - a).abs() < 1e-6 { break; }
        }
        let u_best = if f1 > f2 { b1 } else { b2 };
        let f_best = f1.max(f2);
        (u_best, f_best)
    }

    /// Analytic seed column: unconstrained profit-optimal action per battery.
    /// Uses the same derivative logic as dp_analytic_max / ASCA, ignoring flow constraints.
    fn cg_analytic_seed(
        bat: &Battery,
        rt_price: f64,
        lambda: f64,
        u_min: f64,
        u_max: f64,
        deg_coeff: f64,
    ) -> f64 {
        let dt = 0.25_f64;

        // Evaluate at 0, then charge and discharge branches
        let mut best_u = 0.0;
        let two_deg = 2.0 * deg_coeff;

        if u_min < 0.0 {
            let hi = 0.0f64.min(u_max);
            if u_min < hi && deg_coeff > 1e-30 {
                let b_coeff = dt * (lambda * bat.efficiency_charge - rt_price - 0.25);
                let raw_cand = b_coeff / two_deg;
                let cand = (-raw_cand.clamp(0.0, -u_min)).clamp(u_min, hi);
                best_u = cand;
            }
        }

        if u_max > 0.0 && deg_coeff > 1e-30 {
            let eff_d = bat.efficiency_discharge.max(1e-9);
            let lo = 0.0f64.max(u_min);
            let b_coeff = dt * (rt_price - 0.25 - lambda / eff_d);
            let raw_cand = b_coeff / two_deg;
            let cand = raw_cand.clamp(lo, u_max);
            // Pick whichever side has positive net gradient at 0
            let sell_net = rt_price - 0.25 - lambda / eff_d;
            let buy_net = lambda * bat.efficiency_charge - rt_price - 0.25;
            if sell_net > 0.0 && sell_net.abs() >= buy_net.abs() {
                best_u = cand;
            }
        }

        best_u.clamp(u_min, u_max)
    }

    /// Exact analytical pricing oracle — O(4 eval) vs 30-iter golden section.
    /// Solves h(u) = eval_profit(b,u) - penalty*u by linearising DP future-value via asca_lambda,
    /// then evaluating the exact quadratic optima on each branch + 3 boundary points.
    fn pricing_analytical(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        b: usize,
        penalty: f64,
        lo: f64,
        hi: f64,
    ) -> (f64, f64) {
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        let soc = state.socs[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let soc_levels = ca.dp[b][0].len();
        let t_next = ca.cluster_map.get(state.time_step + 1)
            .copied().unwrap_or(ca.dp[b].len() - 1).min(ca.dp[b].len() - 1);
        let lambda_dp = asca_lambda(&ca.dp[b][t_next], soc, bat.soc_min_mwh, soc_span, soc_levels);
        let deg_coeff = (0.25 / bat.capacity_mwh.max(1e-9)).powi(2);
        let dt = 0.25_f64;

        let eval_h = |u: f64| -> f64 {
            let f = eval_profit(challenge, state, ca, b, u) - penalty * u;
            if f.is_finite() { f } else { f64::NEG_INFINITY }
        };

        let mut best_u = 0.0_f64;
        let mut best_f = eval_h(0.0);
        macro_rules! try_u {
            ($u:expr) => { let f = eval_h($u); if f > best_f { best_f = f; best_u = $u; } };
        }
        try_u!(lo);
        try_u!(hi);

        // Charge branch (u <= 0): h ≈ -deg*u² + u*[(rt+0.25-λ*η_c)*dt - penalty]
        // u* = [(rt+0.25-λ*η_c)*dt - penalty] / (2*deg)
        if lo < 0.0 && deg_coeff > 1e-30 {
            let b_coeff = (rt_price + 0.25 - lambda_dp * bat.efficiency_charge) * dt - penalty;
            let u_star = (b_coeff / (2.0 * deg_coeff)).clamp(lo, 0.0_f64.min(hi));
            try_u!(u_star);
        }

        // Discharge branch (u >= 0): h ≈ -deg*u² + u*[(rt-0.25-λ/η_d)*dt - penalty]
        // u* = [(rt-0.25-λ/η_d)*dt - penalty] / (2*deg)
        if hi > 0.0 && deg_coeff > 1e-30 {
            let eff_d = bat.efficiency_discharge.max(1e-9);
            let b_coeff = (rt_price - 0.25 - lambda_dp / eff_d) * dt - penalty;
            let u_star = (b_coeff / (2.0 * deg_coeff)).clamp(0.0_f64.max(lo), hi);
            try_u!(u_star);
        }

        (best_u, best_f)
    }
}
mod lp {
    // Minimal dense simplex LP solver for joint battery dispatch.
    // Solves: maximize c^T x, s.t. A x <= b, x >= 0.
    // Assumes b_i >= 0 (holds for TIG typical instances).
    // Falls back to KKT/ASCA if infeasible or pivot limit reached.
    // Extended with dual variable extraction for column generation.

    const LP_MAX_PIVOTS: usize = 3000;
    const LP_EPS: f64 = 1e-9;

    /// Solve LP with a configurable pivot budget.
    /// Returns (solution, pivots_used) or (None, 0) on failure.
    pub fn lp_solve_with_budget(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, usize) {
        let (sol, _, piv) = lp_solve_with_duals(n, m, c, a, b, max_pivots);
        (sol, piv)
    }

    /// Solve LP with a configurable pivot budget, also returns dual variables.
    /// Returns (primal_solution, dual_solution, pivots_used).
    /// Dual variables for each constraint (row) are returned.
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
        let mut pivots_used = 0usize;

        for pivot in 0..max_pivots {
            pivots_used = pivot + 1;
            // Choose entering variable: Bland's rule = smallest index with negative reduced cost
            let entering = match (0..n_vars).find(|&j| tab[m][j] < -LP_EPS) {
                Some(j) => j,
                None => break, // Optimal
            };

            // Min-ratio test for leaving variable
            let leaving_row = (0..m)
                .filter(|&i| tab[i][entering] > LP_EPS)
                .min_by(|&i1, &i2| {
                    let r1 = tab[i1][rhs_col] / tab[i1][entering];
                    let r2 = tab[i2][rhs_col] / tab[i2][entering];
                    r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
                });

            let leaving_row = match leaving_row {
                Some(r) => r,
                None => return (None, None, 0), // Unbounded
            };

            // Pivot: normalize leaving row
            let pivot_val = tab[leaving_row][entering];
            if pivot_val.abs() < LP_EPS {
                return (None, None, 0); // Degenerate
            }
            for j in 0..n_cols {
                tab[leaving_row][j] /= pivot_val;
            }

            // Eliminate entering column from all other rows
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

        // Extract primal solution
        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i][rhs_col].max(0.0);
            }
        }

        // Extract dual variables: dual for constraint i is -tab[m][n+i] if slack is non-basic, else 0.
        // Corrected sign: reduced cost of slack = -y_i, so y_i = -tab[m][slack_col].
        let mut y = vec![0.0f64; m];
        for i in 0..m {
            let slack_col = n + i;
            if !basis.contains(&slack_col) {
                y[i] = -tab[m][slack_col];
                if y[i].abs() < LP_EPS { y[i] = 0.0; }
            } else {
                y[i] = 0.0;
            }
            // Ensure non-negativity
            if y[i] < 0.0 { y[i] = 0.0; }
        }

        (Some(x), Some(y), pivots_used)
    }

    /// Original lp_solve delegates to duals version with default cap.
    pub fn lp_solve(n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let (sol, _, _) = lp_solve_with_duals(n, m, c, a, b, LP_MAX_PIVOTS);
        sol
    }
}
mod track_baseline {
    use super::helpers::{solve_with_hp, TrackHp};
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{Challenge, Solution};

    fn defaults() -> TrackHp {
        TrackHp {
            soc_levels: 101,        // 101≈301 quality (track_notes); freed budget for K=5 SDP
            action_grid: 40,
            asca_iters: 25,         // max sweeps; early-exit when convergence_tol reached
            ternary_iters: 25,
            convergence_tol: 1e-4,  // tighter: ensure ASCA converges to local optimum
            anticipate_lmp: true,   // tuned default
            lmp_threshold: 0.65,    // tuned default
            lmp_premium_scale: 0.40, // tuned default
            jump_premium: 0.02,
            prune_ratio: 0.00,
            deflator_iters: 15,
            flow_margin: 1e-4,
            network_derating: 1.00,
            use_sdp: true,          // K=5 SDP: Gauss-Hermite 5-point, exact for degree-9 polynomials
            use_lp: true,           // joint LP dispatch: simultaneous cross-battery optimization
            lp_refine_sweeps: 3,    // ASCA sweeps warm-started from LP (reduced from 5 to cut cycle time)
            use_cg: true,           // Dantzig-Wolfe column generation per-step dispatch
            cg_iters: 20,           // cg_iters=20 re-test (was null at i03 pre-sigma-calib)
            sdp_sigma_het_alpha: 0.45, // tuned default
            sdp_sigma_scale: 1.1,   // tuned default
            lambda_track: 0.0,      // OCO SOC tracking; 0=disabled
            use_analytical_pricing: false, // ctrl=false, VarA=true
            use_morales_vi: false,         // ctrl=false, VarB=true
            use_lp_dual_warmstart: false,   // null — dead-list
            het_lmp_alpha: 0.0,             // dead-list
            use_lp_basis_warmstart: false,  // dead-list
            het_crate_alpha: 0.0,           // ctrl=0.0; variants: 0.25, 0.50, 0.75
            use_cg_lp_combine: false,       // ctrl=false, VarA=true
            k_clusters: 80
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
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0,
            use_cg: false,
            cg_iters: 0,
            sdp_sigma_het_alpha: 0.0,
            sdp_sigma_scale: 1.0,
            lambda_track: 0.0,
            use_analytical_pricing: false,
            use_morales_vi: false,
            use_lp_dual_warmstart: false,
            het_lmp_alpha: 0.0,
            use_lp_basis_warmstart: false,
            het_crate_alpha: 0.0,
            use_cg_lp_combine: false,
            k_clusters: 0,
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
            soc_levels: 201,
            action_grid: 30,
            asca_iters: 25,
            ternary_iters: 15,
            convergence_tol: 1e-3,
            anticipate_lmp: true,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.00,
            jump_premium: 0.00,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            network_derating: 0.22,
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0,
            use_cg: false,
            cg_iters: 0,
            sdp_sigma_het_alpha: 0.0,
            sdp_sigma_scale: 1.0,
            lambda_track: 0.0,
            use_analytical_pricing: false,
            use_morales_vi: false,
            use_lp_dual_warmstart: false,
            het_lmp_alpha: 0.0,
            use_lp_basis_warmstart: false,
            het_crate_alpha: 0.0,
            use_cg_lp_combine: false,
            k_clusters: 0,
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
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0,
            use_cg: false,
            cg_iters: 0,
            sdp_sigma_het_alpha: 0.0,
            sdp_sigma_scale: 1.0,
            lambda_track: 0.0,
            use_analytical_pricing: false,
            use_morales_vi: false,
            use_lp_dual_warmstart: false,
            het_lmp_alpha: 0.0,
            use_lp_basis_warmstart: false,
            het_crate_alpha: 0.0,
            use_cg_lp_combine: false,
            k_clusters: 0,
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
            use_sdp: false,
            use_lp: false,
            lp_refine_sweeps: 0,
            use_cg: false,
            cg_iters: 0,
            sdp_sigma_het_alpha: 0.0,
            sdp_sigma_scale: 1.0,
            lambda_track: 0.0,
            use_analytical_pricing: false,
            use_morales_vi: false,
            use_lp_dual_warmstart: false,
            het_lmp_alpha: 0.0,
            use_lp_basis_warmstart: false,
            het_crate_alpha: 0.0,
            use_cg_lp_combine: false,
            k_clusters: 0,
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
    pub het_lmp_alpha: Option<f64>,
    pub use_lp_basis_warmstart: Option<bool>,
    pub het_crate_alpha: Option<f64>,
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
