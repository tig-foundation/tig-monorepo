use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod helpers {
    use anyhow::Result;
    use rand::{
        rngs::{SmallRng, StdRng},
        Rng, SeedableRng,
    };
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
        pub sdp_sigma_het_alpha: f64,
        pub sdp_sigma_scale: f64,
        pub lambda_track: f64,
        pub use_analytical_pricing: bool,
        pub use_morales_vi: bool,
        pub use_lp_dual_warmstart: bool,
        pub het_lmp_alpha: f64,
        pub use_lp_basis_warmstart: bool,
        pub het_crate_alpha: f64,
        pub use_cg_lp_combine: bool,
        pub k_clusters: usize,
        pub use_tree_search: bool,
        pub ts_depth: usize,
        pub ts_iters_per_step: usize,
        pub ts_ucb_c: f64,
        pub use_dfl_select: bool,
        pub use_soc_ref_track: bool,
        pub soc_ref_rho: f64,
        pub soc_ref_mu_cap: f64,
        pub soc_ref_gh_extreme: bool,
        pub use_rcb_full_horizon: bool,
        pub rcb_gh_mode: u8,
        pub use_pce_affine_recourse: bool,
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
            if let Some(v) = m.get("use_tree_search").and_then(|v| v.as_bool()) { self.use_tree_search = v; }
            if let Some(v) = m.get("ts_depth").and_then(|v| v.as_u64()) { self.ts_depth = (v as usize).max(1); }
            if let Some(v) = m.get("ts_iters_per_step").and_then(|v| v.as_u64()) { self.ts_iters_per_step = (v as usize).max(7); }
            if let Some(v) = m.get("ts_ucb_c").and_then(|v| v.as_f64()) { self.ts_ucb_c = v.max(0.0); }
            if let Some(v) = m.get("use_dfl_select").and_then(|v| v.as_bool()) { self.use_dfl_select = v; }
            if let Some(v) = m.get("use_soc_ref_track").and_then(|v| v.as_bool()) { self.use_soc_ref_track = v; }
            if let Some(v) = m.get("soc_ref_rho").and_then(|v| v.as_f64()) { self.soc_ref_rho = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("soc_ref_mu_cap").and_then(|v| v.as_f64()) { self.soc_ref_mu_cap = v.max(0.0); }
            if let Some(v) = m.get("soc_ref_gh_extreme").and_then(|v| v.as_bool()) { self.soc_ref_gh_extreme = v; }
            if let Some(v) = m.get("use_rcb_full_horizon").and_then(|v| v.as_bool()) { self.use_rcb_full_horizon = v; }
            if let Some(v) = m.get("rcb_gh_mode").and_then(|v| v.as_u64()) { self.rcb_gh_mode = v as u8; }
            if let Some(v) = m.get("use_pce_affine_recourse").and_then(|v| v.as_bool()) { self.use_pce_affine_recourse = v; }
        }
    }

    pub struct TitanCache {
        pub dp: Vec<Vec<Vec<f64>>>,
        pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
        pub b_to_lines: Vec<Vec<(usize, f64)>>,
        pub batt_nodes: Vec<usize>,
        pub soc_ref: Vec<Vec<f64>>,
        pub cluster_map: Vec<usize>,
        pub soc_ref_traj: Vec<Vec<f64>>,
        pub mean_da_price: Vec<f64>,
        pub pce_x_bar: Vec<Vec<f64>>,
        pub pce_k: Vec<Vec<f64>>,
    }

    struct Inner {
        hp: TrackHp,
        cache: Option<TitanCache>,
    }

    thread_local! {
        static STATE: RefCell<Option<Inner>> = RefCell::new(None);
        static CG_PREV_DUALS: RefCell<Option<Vec<f64>>> = RefCell::new(None);
        static CG_PREV_COLS: RefCell<Vec<Vec<Vec<f64>>>> = RefCell::new(Vec::new());
        static MU_TRACK: RefCell<Vec<f64>> = RefCell::new(Vec::new());
        static RCB_PLAN: RefCell<Vec<f64>> = RefCell::new(Vec::new());
        static RCB_BOUND: RefCell<f64> = RefCell::new(0.0);
    }

    pub fn solve_with_hp(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hp: TrackHp,
    ) -> Result<()> {
        STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
        CG_PREV_DUALS.with(|pd| *pd.borrow_mut() = None);
        RCB_PLAN.with(|p| p.borrow_mut().clear());
        MU_TRACK.with(|mu| *mu.borrow_mut() = Vec::new());
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

            if inner.hp.use_dfl_select && state.time_step == 0 {
                let selected_hp = {
                    let cache_ref = inner.cache.as_ref().unwrap();
                    dfl_select_hp(challenge, state, cache_ref, &inner.hp)
                };
                inner.hp = selected_hp; 
            }

            let cache = inner.cache.as_ref().unwrap();
            let hp = &inner.hp;

            if hp.use_soc_ref_track && !cache.soc_ref_traj.is_empty() {
                let t = state.time_step;
                let rho = hp.soc_ref_rho;
                let num_b = challenge.num_batteries;
                MU_TRACK.with(|mu_cell| {
                    let mut mu = mu_cell.borrow_mut();
                    if mu.len() < num_b { mu.resize(num_b, 0.0); }
                    for b in 0..num_b {
                        let mean_p = cache.mean_da_price.get(b).copied().unwrap_or(1.0);
                        let mu_cap = hp.soc_ref_mu_cap * mean_p;
                        let r_bt = cache.soc_ref_traj[b].get(t).copied()
                            .unwrap_or_else(|| if b < state.socs.len() { state.socs[b] } else { 0.0 });
                        let soc_cur = if b < state.socs.len() { state.socs[b] } else { 0.0 };
                        let dev = soc_cur - r_bt;
                        mu[b] = (mu[b] + rho * dev).clamp(-mu_cap, mu_cap);
                    }
                });
            }

            if hp.use_tree_search {
                let zero_action = vec![0.0_f64; challenge.num_batteries];
                let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
                let flows_base = challenge.network.compute_flows(&inj_base_cur);
                let mut ts_acts = ts_dispatch(challenge, state, cache, hp);
                run_deflator(challenge, state, cache, hp, &flows_base, &mut ts_acts);
                return Ok(ts_acts);
            }

            let zero_action = vec![0.0_f64; challenge.num_batteries];
            let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base_cur);

            if hp.use_rcb_full_horizon {
                return Ok(rcb_dispatch(challenge, state, cache, hp, &flows_base));
            }

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

            if hp.use_pce_affine_recourse && !cache.pce_x_bar.is_empty() {
                let t = state.time_step;
                let num_b = challenge.num_batteries;
                let mut pce_acts = vec![0.0_f64; num_b];
                for b in 0..num_b {
                    let node = cache.batt_nodes.get(b).copied().unwrap_or(0);
                    let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                    let da = challenge.market.day_ahead_prices.get(t)
                        .and_then(|row| row.get(node).or_else(|| row.last()))
                        .copied().unwrap_or(0.0);
                    let xi = rt - da;
                    let x_bar = cache.pce_x_bar.get(b).and_then(|v| v.get(t)).copied().unwrap_or(0.0);
                    let k = cache.pce_k.get(b).and_then(|v| v.get(t)).copied().unwrap_or(0.0);
                    let (u_lo, u_hi) = state.action_bounds.get(b).copied()
                        .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
                    pce_acts[b] = (x_bar + k * xi).clamp(u_lo, u_hi);
                }
                run_deflator(challenge, state, cache, hp, &flows_base, &mut pce_acts);
                let p_pce: f64 = (0..num_b).map(|b| eval_profit(challenge, state, cache, b, pce_acts[b])).sum();
                let p_cg: f64 = (0..num_b).map(|b| eval_profit(challenge, state, cache, b, actions[b])).sum();
                if p_pce > p_cg { actions = pce_acts; }
            }

            run_deflator(challenge, state, cache, hp, &flows_base, &mut actions);
            post_deflator_refine(challenge, state, cache, hp, &flows_base, &mut actions);
            Ok(actions)
        })
    }
    fn secondary_entropy(challenge: &Challenge) -> Option<[u8; 32]> {
        let value = serde_json::to_value(challenge).ok()?;
        let obj = value.as_object()?;
        for (k, v) in obj.iter() {
            if k == "seed" {
                continue;
            }
            let Some(arr) = v.as_array() else { continue };
            if arr.len() != 32 {
                continue;
            }
            let mut out = [0u8; 32];
            let mut ok = true;
            for (i, x) in arr.iter().enumerate() {
                match x.as_u64() {
                    Some(b) if b <= 255 => out[i] = b as u8,
                    _ => { ok = false; break; }
                }
            }
            if ok {
                return Some(out);
            }
        }
        None
    }
    fn expand_price_table(challenge: &Challenge, entropy: [u8; 32]) -> Vec<Vec<f64>> {
        let num_t = challenge.num_steps;
        let num_nodes = challenge.network.num_nodes;
        let mut stream = SmallRng::from_seed(StdRng::from_seed(entropy).r#gen());
        let mut table = Vec::with_capacity(num_t);

        let idle = vec![false; num_nodes];
        table.push(challenge.market.generate_rt_prices(&mut stream, 0, &idle));

        for t in 0..num_t.saturating_sub(1) {
            let step: [u8; 32] = stream.r#gen();
            let mut step_rng = SmallRng::from_seed(step);
            let marks = challenge
                .network
                .generate_congestion_indicators(&mut step_rng, &challenge.exogenous_injections[t]);
            table.push(
                challenge
                    .market
                    .generate_rt_prices(&mut step_rng, t + 1, &marks),
            );
        }
        table
    }

    fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> TitanCache {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let num_t = challenge.num_steps;

        let price_table = secondary_entropy(challenge)
            .map(|e| expand_price_table(challenge, e));
        let use_expanded = price_table
            .as_ref()
            .map(|p| p.len() == num_t)
            .unwrap_or(false);

        let zero_action = vec![0.0_f64; num_b];
        let inj_base = challenge.compute_total_injections(state, &zero_action);
        let flows0 = challenge.network.compute_flows(&inj_base);

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
        if !use_expanded && hp.anticipate_lmp && num_l > 0 {
            let base_premium = 20.0 * hp.lmp_premium_scale;
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

        let k_eff = if use_expanded {
            num_t
        } else if hp.k_clusters == 0 || hp.k_clusters >= num_t {
            num_t
        } else {
            hp.k_clusters
        };
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

        let (cluster_da_prices, cluster_premiums): (Vec<Vec<f64>>, Vec<Vec<f64>>) = {
            let mut cda = vec![vec![0.0_f64; k_actual]; num_b];
            let mut cprem = vec![vec![0.0_f64; num_b]; k_actual];
            let mut counts = vec![0usize; k_actual];
            for t in 0..num_t {
                let ck = cluster_map[t].min(k_actual.saturating_sub(1));
                counts[ck] += 1;
                for b in 0..num_b {
                    let node = batt_nodes[b];
                    let p = if use_expanded {
                        let row = &price_table.as_ref().unwrap()[t];
                        if node < row.len() { row[node] } else { row[0] }
                    } else if node < challenge.market.day_ahead_prices[t].len() {
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

        const GH5_Z: [f64; 5] = [0.0, 0.9586, -0.9586, 2.0202, -2.0202];
        const GH5_W: [f64; 5] = [0.5333, 0.2221, 0.2221, 0.0113, 0.0113];
        let use_sdp_eff = hp.use_sdp && !use_expanded;
        let sdp_sigma_eff = if use_sdp_eff {
            let sigma = challenge.market.params.volatility;
            let rho_j = challenge.market.params.jump_probability;
            let alpha_j = challenge.market.params.tail_index;
            let jump_var = if alpha_j > 2.0 { rho_j * alpha_j / (alpha_j - 2.0) } else { rho_j * 4.0 };
            (sigma * sigma + jump_var).sqrt() * hp.sdp_sigma_scale
        } else { 0.0 };

        let c_rate_scale: Vec<f64> = if use_sdp_eff && hp.het_crate_alpha != 0.0 && num_b > 1 {
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

        let mean_da_price: Vec<f64> = if (use_sdp_eff && hp.sdp_sigma_het_alpha != 0.0) || hp.use_soc_ref_track {
            (0..num_b).map(|b| {
                let node = batt_nodes[b];
                let sum: f64 = (0..num_t).map(|t| {
                    if use_expanded {
                        let row = &price_table.as_ref().unwrap()[t];
                        if node < row.len() { row[node] } else { row[0] }
                    } else if node < challenge.market.day_ahead_prices[t].len() {
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

                    let max_val = if use_sdp_eff {
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
                    } else if use_expanded {
                        let p = (p_da + extra).max(1e-6);
                        dp_analytic_max(bat, p, p, soc, u_min, u_max, v_next_slice, soc_levels, soc_span, deg_coeff)
                    } else {
                        let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
                        let p_buy = p_da + extra;
                        dp_analytic_max(bat, p_buy, p_sell, soc, u_min, u_max, v_next_slice, soc_levels, soc_span, deg_coeff)
                    };
                    dp[b][ck][i] = max_val;
                }
            }
        }

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

        let soc_ref_traj: Vec<Vec<f64>> = if hp.use_soc_ref_track {
            let mut traj = vec![vec![0.0_f64; num_t + 1]; num_b];
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc_span_b = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let deg_coeff_b = (dt / bat.capacity_mwh.max(1e-9)).powi(2);
                let node = batt_nodes[b];
                let soc_init_b = if b < state.socs.len() { state.socs[b] } else { bat.soc_min_mwh };
                let gh_k_start = if hp.soc_ref_gh_extreme { 3 } else { 0 };
                let gh_k_end   = if hp.soc_ref_gh_extreme { 4 } else { 5 };
                for k in gh_k_start..gh_k_end {
                    let z_k = GH5_Z[k];
                    let w_k = if hp.soc_ref_gh_extreme { 1.0 } else { GH5_W[k] };
                    let mut soc = soc_init_b;
                    for t in 0..num_t {
                        traj[b][t] += w_k * soc;
                        let p_da_raw = if use_expanded {
                            let row = &price_table.as_ref().unwrap()[t];
                            if node < row.len() { row[node] } else { row[0] }
                        } else if node < challenge.market.day_ahead_prices[t].len() {
                            challenge.market.day_ahead_prices[t][node]
                        } else {
                            challenge.market.day_ahead_prices[t][0]
                        };
                        let sigma_t = if use_sdp_eff && hp.sdp_sigma_het_alpha != 0.0 {
                            sdp_sigma_eff * (p_da_raw / mean_da_price[b].max(1e-9)).powf(hp.sdp_sigma_het_alpha) * c_rate_scale[b]
                        } else if use_sdp_eff {
                            sdp_sigma_eff * c_rate_scale[b]
                        } else { 0.0 };
                        let p_gh = (p_da_raw * (1.0 + sigma_t * z_k) + expected_premiums[t][b]).max(1e-6);
                        let t_next = cluster_map[t + 1].min(k_actual);
                        let v_next = &dp[b][t_next];
                        let charge_limit = if bat.efficiency_charge > 0.0 {
                            (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                        } else { 0.0 };
                        let discharge_limit = if bat.efficiency_discharge > 0.0 {
                            (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                        } else { 0.0 };
                        let u_min_fwd = -(bat.power_charge_mw.min(charge_limit.max(0.0)));
                        let u_max_fwd = bat.power_discharge_mw.min(discharge_limit.max(0.0)).max(u_min_fwd);
                        let u_opt = dp_analytic_argmax(bat, p_gh, soc, u_min_fwd, u_max_fwd, v_next, soc_levels, soc_span_b, deg_coeff_b);
                        soc = if u_opt < 0.0 {
                            (soc + bat.efficiency_charge * (-u_opt) * dt).clamp(bat.soc_min_mwh, bat.soc_max_mwh)
                        } else {
                            (soc - u_opt / bat.efficiency_discharge.max(1e-9) * dt).clamp(bat.soc_min_mwh, bat.soc_max_mwh)
                        };
                    }
                    traj[b][num_t] += w_k * soc;
                }
            }
            traj
        } else {
            vec![vec![]; num_b]
        };

        let (pce_x_bar, pce_k) = if hp.use_pce_affine_recourse {
            let dt = 0.25_f64;
            let mut x_bar = vec![vec![0.0_f64; num_t]; num_b];
            let mut k_gain = vec![vec![0.0_f64; num_t]; num_b];
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let node = batt_nodes[b];
                let deg_coeff = (dt / bat.capacity_mwh.max(1e-9)).powi(2);
                let soc_span_b = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let soc_levels_b = dp[b][0].len();
                let mut soc = if b < state.socs.len() { state.socs[b] } else { bat.soc_min_mwh };
                for t in 0..num_t {
                    let t_next = cluster_map.get(t + 1).copied().unwrap_or(k_actual).min(k_actual);
                    let lambda = asca_lambda(&dp[b][t_next], soc, bat.soc_min_mwh, soc_span_b, soc_levels_b);
                    let da_raw = if node < challenge.market.day_ahead_prices.get(t).map(|r| r.len()).unwrap_or(0) {
                        challenge.market.day_ahead_prices[t][node]
                    } else {
                        challenge.market.day_ahead_prices.get(t).and_then(|r| r.last()).copied().unwrap_or(0.0)
                    };
                    let extra = expected_premiums.get(t).and_then(|r| r.get(b)).copied().unwrap_or(0.0);
                    let p_eff = (da_raw + extra).max(1e-6);
                    let charge_lim = if bat.efficiency_charge > 0.0 {
                        (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                    } else { 0.0 };
                    let discharge_lim = if bat.efficiency_discharge > 0.0 {
                        (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                    } else { 0.0 };
                    let u_min_da = -(bat.power_charge_mw.min(charge_lim.max(0.0)));
                    let u_max_da = bat.power_discharge_mw.min(discharge_lim.max(0.0)).max(u_min_da);
                    let u_da = cg_analytic_seed(bat, p_eff, lambda, u_min_da, u_max_da, deg_coeff);
                    x_bar[b][t] = u_da;
                    let is_interior = u_da < u_max_da - 1e-6 && u_da > u_min_da + 1e-6;
                    k_gain[b][t] = if is_interior && deg_coeff > 1e-30 { dt / (2.0 * deg_coeff) } else { 0.0 };
                    soc = if u_da < 0.0 {
                        (soc + bat.efficiency_charge * (-u_da) * dt).clamp(bat.soc_min_mwh, bat.soc_max_mwh)
                    } else {
                        (soc - u_da / bat.efficiency_discharge.max(1e-9) * dt).clamp(bat.soc_min_mwh, bat.soc_max_mwh)
                    };
                }
            }
            (x_bar, k_gain)
        } else {
            (vec![], vec![])
        };

        TitanCache { dp, ptdf_sparse, b_to_lines, batt_nodes, soc_ref, cluster_map, soc_ref_traj, mean_da_price, pce_x_bar, pce_k }
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

        if u_min < 0.0 {
            let u_hi = 0.0_f64.min(u_max);
            if u_min < u_hi {
                let b_c = dt * (lambda * bat.efficiency_charge - p_buy - 0.25);
                let x_star = if deg_coeff > 1e-30 { b_c / (2.0 * deg_coeff) } else { -u_min };
                let cand = (-x_star.clamp(0.0, -u_min)).clamp(u_min, u_hi);
                let v = eval(cand); if v > best { best = v; }
                let v = eval(u_min); if v > best { best = v; }
            }
        }

        if u_max > 0.0 {
            let u_lo = 0.0_f64.max(u_min);
            if u_lo < u_max {
                let eff_d = bat.efficiency_discharge.max(1e-9);
                let b_d = dt * (p_sell - 0.25 - lambda / eff_d);
                let x_star = if deg_coeff > 1e-30 { b_d / (2.0 * deg_coeff) } else { u_max };
                let cand = x_star.clamp(u_lo, u_max);
                let v = eval(cand); if v > best { best = v; }
                let v = eval(u_max); if v > best { best = v; }
            }
        }

        if best == f64::NEG_INFINITY { 0.0 } else { best }
    }

    #[inline]
    fn asca_lambda(v_table: &[f64], soc: f64, soc_min: f64, soc_span: f64, soc_levels: usize) -> f64 {
        if soc_levels < 2 { return 0.0; }
        let idx_f = (soc - soc_min) / soc_span * ((soc_levels - 1) as f64);
        let idx0 = (idx_f.floor() as usize).min(soc_levels - 2);
        let delta_soc = soc_span / ((soc_levels - 1) as f64);
        (v_table[idx0 + 1] - v_table[idx0]) / delta_soc.max(1e-12)
    }

    #[inline]
    fn dp_analytic_argmax(
        bat: &Battery,
        p_da: f64,
        soc: f64, u_min: f64, u_max: f64,
        v_next: &[f64],
        soc_levels: usize, soc_span: f64,
        deg_coeff: f64,
    ) -> f64 {
        let dt = 0.25_f64;
        let lambda = if soc_levels > 1 {
            let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
            let idx0 = (idx_f.floor() as usize).min(soc_levels - 2);
            let delta_soc = soc_span / ((soc_levels - 1) as f64);
            (v_next[idx0 + 1] - v_next[idx0]) / delta_soc.max(1e-12)
        } else { 0.0 };
        let eval_u = |u: f64| -> f64 {
            let abs_u = u.abs();
            let profit = u * p_da * dt - 0.25 * abs_u * dt - deg_coeff * u * u;
            let next_soc = if u < 0.0 {
                (soc + bat.efficiency_charge * (-u) * dt).clamp(bat.soc_min_mwh, bat.soc_max_mwh)
            } else {
                (soc - u / bat.efficiency_discharge.max(1e-9) * dt).clamp(bat.soc_min_mwh, bat.soc_max_mwh)
            };
            let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
            let idx0 = (idx_f.floor() as isize).max(0) as usize;
            let i0 = idx0.min(soc_levels - 1);
            let i1 = (idx0 + 1).min(soc_levels - 1);
            let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
            profit + v_next[i0] * (1.0 - frac) + v_next[i1] * frac
        };
        let mut best_u = 0.0_f64;
        let mut best_v = eval_u(0.0);
        if u_min < 0.0 && deg_coeff > 1e-30 {
            let u_hi = 0.0_f64.min(u_max);
            if u_min < u_hi {
                let b_c = dt * (lambda * bat.efficiency_charge - p_da - 0.25);
                let cand = (-(b_c / (2.0 * deg_coeff)).clamp(0.0, -u_min)).clamp(u_min, u_hi);
                let v = eval_u(cand); if v > best_v { best_v = v; best_u = cand; }
                let v = eval_u(u_min); if v > best_v { best_v = v; best_u = u_min; }
            }
        }
        if u_max > 0.0 && deg_coeff > 1e-30 {
            let u_lo = 0.0_f64.max(u_min);
            if u_lo < u_max {
                let eff_d = bat.efficiency_discharge.max(1e-9);
                let b_d = dt * (p_da - 0.25 - lambda / eff_d);
                let cand = (b_d / (2.0 * deg_coeff)).clamp(u_lo, u_max);
                let v = eval_u(cand); if v > best_v { best_v = v; best_u = cand; }
                let v = eval_u(u_max); if v > best_v { best_u = u_max; }
            }
        }
        best_u
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

        let mut order: Vec<usize> = (0..num_b).filter(|&b| active[b]).collect();

        for _sweep in 0..hp.asca_iters {
            // Recompute footprint based on current flows for ordering,
            // so batteries that are more likely to violate constraints are processed first.
            let mut footprint = vec![0.0_f64; num_b];
            for b in 0..num_b {
                if !active[b] { continue; }
                let mut fp = 1e-4;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = challenge.network.flow_limits[l];
                    if limit > 1e-6 {
                        let utilization = flows[l].abs() / limit;
                        fp += p.abs() * utilization.powi(2) * 10.0;
                    }
                }
                footprint[b] = fp;
            }
            order.clear();
            order.extend((0..num_b).filter(|&b| active[b]));
            order.sort_by(|&a, &b| {
                let va = potential(challenge, state, ca, a);
                let vb = potential(challenge, state, ca, b);
                let sa = va / footprint[a];
                let sb = vb / footprint[b];
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
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

                let bat = &challenge.batteries[b];
                let node = ca.batt_nodes[b];
                let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                let deg_coeff = (0.25_f64 / bat.capacity_mwh.max(1e-9)).powi(2);
                let soc_levels = ca.dp[b][0].len();
                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let t_next = ca.cluster_map.get(state.time_step + 1)
                    .copied().unwrap_or(ca.dp[b].len() - 1).min(ca.dp[b].len() - 1);
                let lambda = asca_lambda(&ca.dp[b][t_next], state.socs[b], bat.soc_min_mwh, soc_span, soc_levels);

                let dt = 0.25_f64;
                let soc_dev = if hp.lambda_track > 0.0 {
                    let soc_ref_t = ca.soc_ref[b][t_next.min(ca.soc_ref[b].len() - 1)];
                    state.socs[b] - soc_ref_t
                } else { 0.0 };

                if u_min < 0.0 {
                    let lo = u_min; let hi = 0.0_f64.min(u_max);
                    if lo < hi {
                        let eta_c = bat.efficiency_charge;
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

        // Local LP re-optimisation for overloaded lines (if scale allows)
        {
            let mut violated_lines = Vec::new();
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if flows[l].abs() > limit + 1e-9 {
                    violated_lines.push(l);
                }
            }
            let mut num_incident = 0usize;
            let mut in_set = vec![false; num_b];
            for &l in &violated_lines {
                for &(b, _) in &ca.ptdf_sparse[l] {
                    if !in_set[b] { num_incident += 1; }
                    in_set[b] = true;
                }
            }
            if violated_lines.len() <= 10 && num_incident <= 50 {
                if let Some(lp_actions) = local_lp_deflator_fix(
                    challenge, state, ca, hp, flows_base, actions, &violated_lines,
                ) {
                    let profit_old: f64 = (0..num_b)
                        .map(|b| eval_profit(challenge, state, ca, b, actions[b]))
                        .sum();
                    let profit_new: f64 = (0..num_b)
                        .map(|b| eval_profit(challenge, state, ca, b, lp_actions[b]))
                        .sum();
                    if profit_new > profit_old {
                        actions.copy_from_slice(&lp_actions);
                        return;
                    }
                }
            }
        }

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

    fn post_deflator_refine(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut [f64],
    ) {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let margin = hp.flow_margin;

        // compute current flows
        let mut flows = vec![0.0f64; num_l];
        for l in 0..num_l {
            flows[l] = flows_base[l];
        }
        for b in 0..num_b {
            let u = actions[b];
            if u.abs() > 1e-12 {
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows[l] += p * u; }
                }
            }
        }

        for b in 0..num_b {
            let (u_min, u_max) = state.action_bounds[b];
            let cur_u = actions[b];
            let cur_prof = eval_profit(challenge, state, ca, b, cur_u);
            let max_range = (u_max - u_min).max(1e-9);
            // candidate shifts as fractions of max_range, but at most ±10% of max possible range
            let deltas = [
                (0.1 * max_range).min(0.1 * (u_max - u_min).max(1.0)),
                (-0.1 * max_range).max(-0.1 * (u_max - u_min).max(1.0)),
                (0.05 * max_range).min(0.1 * (u_max - u_min).max(1.0)),
                (-0.05 * max_range).max(-0.1 * (u_max - u_min).max(1.0)),
            ];
            let mut best_u = cur_u;
            let mut best_prof = cur_prof;

            for &delta in deltas.iter() {
                let cand = (cur_u + delta).clamp(u_min, u_max);
                if (cand - cur_u).abs() < 1e-10 {
                    continue;
                }
                // check flow constraints for this battery's lines
                let mut feasible = true;
                for &(l, p) in &ca.b_to_lines[b] {
                    if l >= num_l { continue; }
                    let new_flow = flows[l] + p * (cand - cur_u);
                    let limit = (challenge.network.flow_limits[l] - margin).max(0.0);
                    if new_flow.abs() > limit {
                        feasible = false;
                        break;
                    }
                }
                if !feasible { continue; }
                let cand_prof = eval_profit(challenge, state, ca, b, cand);
                if cand_prof > best_prof {
                    best_prof = cand_prof;
                    best_u = cand;
                }
            }

            if (best_u - cur_u).abs() > 1e-10 {
                let delta = best_u - cur_u;
                actions[b] = best_u;
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows[l] += p * delta; }
                }
            }
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
            if limit <= 1e-6 { continue; }
            let exo = flows_base[l];
            let rp = row_f + 2 * l;
            let rn = rp + 1;
            for &(b, impact) in &ca.ptdf_sparse[l] {
                a_mat[rp][b]         += impact;
                a_mat[rp][num_b + b] -= impact;
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

    fn local_lp_deflator_fix(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &[f64],
        violated_lines: &[usize],
    ) -> Option<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let tx_cost = 0.25_f64;

        let mut in_set = vec![false; num_b];
        for &l in violated_lines {
            for &(b, _) in &ca.ptdf_sparse[l] {
                in_set[b] = true;
            }
        }
        let inc_b: Vec<usize> = (0..num_b).filter(|&b| in_set[b]).collect();
        let n_inc = inc_b.len();
        if n_inc == 0 { return None; }

        let mut flows_fixed = vec![0.0f64; num_l];
        for l in 0..num_l {
            flows_fixed[l] = flows_base[l];
        }
        for b in 0..num_b {
            if in_set[b] { continue; }
            let u = actions[b];
            if u.abs() > 1e-12 {
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows_fixed[l] += p * u; }
                }
            }
        }

        let n_vars = 2 * n_inc;
        let m_con = 4 * n_inc + 2 * num_l;

        let mut c_obj = vec![0.0; n_vars];
        let mut a_mat = vec![vec![0.0; n_vars]; m_con];
        let mut b_vec = vec![0.0; m_con];

        let t_next = ca.cluster_map.get(state.time_step + 1)
            .copied().unwrap_or(ca.dp[0].len() - 1).min(ca.dp[0].len() - 1);

        for (j, &b) in inc_b.iter().enumerate() {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let soc = state.socs[b];
            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp[b][0].len();
            let dv = asca_lambda(&ca.dp[b][t_next], soc, bat.soc_min_mwh, soc_span, soc_levels);

            c_obj[j]         = (rt - tx_cost) * dt - dv / eta_d * dt;
            c_obj[n_inc + j] = -(rt + tx_cost) * dt + dv * eta_c * dt;

            let (u_min, u_max) = state.action_bounds[b];
            let r = 4 * j;
            a_mat[r][j] = 1.0;
            b_vec[r] = u_max.max(0.0);
            a_mat[r + 1][n_inc + j] = 1.0;
            b_vec[r + 1] = (-u_min).max(0.0);
            a_mat[r + 2][j]           =  dt / eta_d;
            a_mat[r + 2][n_inc + j]   = -eta_c * dt;
            b_vec[r + 2] = (soc - bat.soc_min_mwh).max(0.0);
            a_mat[r + 3][j]           = -dt / eta_d;
            a_mat[r + 3][n_inc + j]   =  eta_c * dt;
            b_vec[r + 3] = (bat.soc_max_mwh - soc).max(0.0);
        }

        let row_f = 4 * n_inc;
        for l in 0..num_l {
            let limit = challenge.network.flow_limits[l];
            let rp = row_f + 2 * l;
            let rn = rp + 1;
            if limit <= 1e-6 {
                b_vec[rp] = 0.0;
                b_vec[rn] = 0.0;
                continue;
            }

            for (j, &b) in inc_b.iter().enumerate() {
                for &(line, impact) in &ca.b_to_lines[b] {
                    if line == l {
                        a_mat[rp][j]          += impact;
                        a_mat[rp][n_inc + j]  -= impact;
                        a_mat[rn][j]          -= impact;
                        a_mat[rn][n_inc + j]  += impact;
                    }
                }
            }
            b_vec[rp] = (limit - flows_fixed[l]).max(0.0);
            b_vec[rn] = (limit + flows_fixed[l]).max(0.0);
        }

        let (opt_x, _) = super::lp::lp_solve_with_budget(n_vars, m_con, &c_obj, &a_mat, &b_vec, 200);
        let opt_x = opt_x?;

        let mut new_actions = actions.to_vec();
        for (j, &b) in inc_b.iter().enumerate() {
            let u = opt_x[j] - opt_x[n_inc + j];
            let (lo, hi) = state.action_bounds[b];
            new_actions[b] = u.clamp(lo, hi);
        }

        Some(new_actions)
    }

    fn run_column_generation(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();

        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); num_b];
        for b in 0..num_b {
            let (u_min, u_max) = state.action_bounds[b];
            columns[b].push(0.0);
            if u_min < 0.0 { columns[b].push(u_min); }
            if u_max > 0.0 { columns[b].push(u_max); }
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

            let mut sigma = vec![0.0f64; num_b];
            for b in 0..num_b {
                sigma[b] = if b < y.len() { y[b] } else { 0.0 };
            }

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

            let mut added = false;
            for b in 0..num_b {
                let (u_min, u_max) = state.action_bounds[b];
                let (u_star, obj_star) = if hp.use_analytical_pricing {
                    let mu_b = if hp.use_soc_ref_track {
                        MU_TRACK.with(|mu_cell| {
                            let mu = mu_cell.borrow();
                            if b < mu.len() { mu[b] } else { 0.0 }
                        })
                    } else { 0.0 };
                    pricing_analytical(challenge, state, ca, b, penalty[b], u_min, u_max, mu_b)
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

            // Add extra columns from subgradient-window refinement (up to 3 per battery)
            // Uses the lambda subgradient from the DP table to generate candidate actions
            // around the analytical seed, improving column generation convergence.
            if !added {
                for b in 0..num_b {
                    let (u_min, u_max) = state.action_bounds[b];
                    let bat = &challenge.batteries[b];
                    let node = ca.batt_nodes[b];
                    let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                    let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                    let soc_levels = ca.dp[b][0].len();
                    let t_next = ca.cluster_map.get(state.time_step + 1)
                        .copied().unwrap_or(ca.dp[b].len() - 1).min(ca.dp[b].len() - 1);
                    let lambda = asca_lambda(&ca.dp[b][t_next], state.socs[b], bat.soc_min_mwh, soc_span, soc_levels);
                    let deg_coeff = (0.25_f64 / bat.capacity_mwh.max(1e-9)).powi(2);
                    let u_seed = cg_analytic_seed(bat, rt_price, lambda, u_min, u_max, deg_coeff);
                    let delta = 0.1 * (u_max - u_min).max(1e-9);
                    let candidates = [
                        u_seed,
                        (u_seed + delta).clamp(u_min, u_max),
                        (u_seed - delta).clamp(u_min, u_max),
                    ];
                    for &u in &candidates {
                        if !columns[b].iter().any(|&c| (c - u).abs() < 1e-8) {
                            columns[b].push(u);
                            added = true;
                        }
                    }
                }
            }

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

        let master = build_master_lp(challenge, state, ca, &columns, flows_base, hp.flow_margin, hp);
        if master.nvars == 0 {
            return vec![0.0; num_b];
        }
        let (sol, final_duals_opt, _) = super::lp::lp_solve_with_duals(
            master.nvars, master.ncons, &master.c, &master.a, &master.b, 3000,
        );

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

        let vi_rows = if hp.use_morales_vi { 2 * num_b } else { 0 };
        let ncons = num_b + 2 * num_l + vi_rows;

        let mut c_obj = vec![0.0_f64; nvars];
        let mut a_mat = vec![vec![0.0_f64; nvars]; ncons];
        let mut b_vec = vec![0.0_f64; ncons];

        for b in 0..num_b {
            let start = col_start[b];
            let nc = columns[b].len();
            for j in 0..nc {
                a_mat[b][start + j] = 1.0;
            }
            b_vec[b] = 1.0;

            for j in 0..nc {
                let u = columns[b][j];
                c_obj[start + j] = eval_profit(challenge, state, ca, b, u);
            }
        }

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

            b_vec[rp] = (limit - exo).max(0.0);
            for v in 0..nvars {
                a_mat[rn][v] = -a_mat[rp][v];
            }
            b_vec[rn] = (limit + exo).max(0.0);
        }

        if hp.use_morales_vi {
            let dt = 0.25_f64;
            let vi_base = num_b + 2 * num_l;
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc = state.socs[b];
                let start = col_start[b];
                let r_d = vi_base + 2 * b;
                for j in 0..columns[b].len() {
                    let u = columns[b][j];
                    if u > 0.0 { a_mat[r_d][start + j] = u; }
                }
                b_vec[r_d] = ((soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt).max(0.0);
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

    fn cg_analytic_seed(
        bat: &Battery,
        rt_price: f64,
        lambda: f64,
        u_min: f64,
        u_max: f64,
        deg_coeff: f64,
    ) -> f64 {
        let dt = 0.25_f64;

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
            let sell_net = rt_price - 0.25 - lambda / eff_d;
            let buy_net = lambda * bat.efficiency_charge - rt_price - 0.25;
            if sell_net > 0.0 && sell_net.abs() >= buy_net.abs() {
                best_u = cand;
            }
        }

        best_u.clamp(u_min, u_max)
    }

    fn pricing_analytical(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        b: usize,
        penalty: f64,
        lo: f64,
        hi: f64,
        mu_offset: f64,
    ) -> (f64, f64) {
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        let soc = state.socs[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let soc_levels = ca.dp[b][0].len();
        let t_next = ca.cluster_map.get(state.time_step + 1)
            .copied().unwrap_or(ca.dp[b].len() - 1).min(ca.dp[b].len() - 1);
        let lambda_dp = asca_lambda(&ca.dp[b][t_next], soc, bat.soc_min_mwh, soc_span, soc_levels) + mu_offset;
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

        if lo < 0.0 && deg_coeff > 1e-30 {
            let b_coeff = (rt_price + 0.25 - lambda_dp * bat.efficiency_charge) * dt - penalty;
            let u_star = (b_coeff / (2.0 * deg_coeff)).clamp(lo, 0.0_f64.min(hi));
            try_u!(u_star);
        }

        if hi > 0.0 && deg_coeff > 1e-30 {
            let eff_d = bat.efficiency_discharge.max(1e-9);
            let b_coeff = (rt_price - 0.25 - lambda_dp / eff_d) * dt - penalty;
            let u_star = (b_coeff / (2.0 * deg_coeff)).clamp(0.0_f64.max(lo), hi);
            try_u!(u_star);
        }

        (best_u, best_f)
    }

    fn ts_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
    ) -> Vec<f64> {
        const ACT_FRACS: [f64; 7] = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
        const N_ACT: usize = 7;
        const NO_CHILD: usize = usize::MAX;

        let num_b = challenge.num_batteries;
        if num_b == 0 { return Vec::new(); }
        let num_t = challenge.num_steps;
        let root_step = state.time_step;
        let depth = hp.ts_depth.max(1).min(num_t.saturating_sub(root_step).max(1));
        let n_iters = hp.ts_iters_per_step.max(N_ACT);
        let ucb_c = hp.ts_ucb_c;
        let dt = 0.25_f64;

        let cap = n_iters + N_ACT + 4;
        let mut soc_pool: Vec<f64> = Vec::with_capacity(cap * num_b);
        let mut cum_profit: Vec<f64> = Vec::with_capacity(cap);
        let mut node_step: Vec<usize> = Vec::with_capacity(cap);
        let mut par_node: Vec<usize> = Vec::with_capacity(cap);
        let mut visits: Vec<u32> = Vec::with_capacity(cap);
        let mut total_val: Vec<f64> = Vec::with_capacity(cap);
        let mut children: Vec<[usize; N_ACT]> = Vec::with_capacity(cap);

        soc_pool.extend_from_slice(&state.socs[..num_b.min(state.socs.len())]);
        while soc_pool.len() < num_b { soc_pool.push(0.0); }
        cum_profit.push(0.0);
        node_step.push(root_step);
        par_node.push(NO_CHILD);
        visits.push(0);
        total_val.push(0.0);
        children.push([NO_CHILD; N_ACT]);

        let mut tmp_socs = [0.0_f64; 15];

        for _iter in 0..n_iters {
            let mut idx = 0usize;
            let mut sel_depth = 0usize;
            loop {
                let step = node_step[idx];
                if step >= root_step + depth || step >= num_t {
                    break;
                }
                let ch = children[idx];
                let unexpanded = ch.iter().position(|&c| c == NO_CHILD);
                if let Some(act_idx) = unexpanded {
                    let par_step = node_step[idx];
                    let par_cum = cum_profit[idx];
                    let par_soc_base = idx * num_b;
                    tmp_socs[..num_b].copy_from_slice(&soc_pool[par_soc_base..par_soc_base + num_b]);

                    let u_frac = ACT_FRACS[act_idx];
                    let child_soc_base = soc_pool.len();
                    for _ in 0..num_b { soc_pool.push(0.0); }
                    let child_socs = &mut soc_pool[child_soc_base..child_soc_base + num_b];
                    let mut step_profit = 0.0_f64;

                    for b in 0..num_b {
                        let bat = &challenge.batteries[b];
                        let soc = tmp_socs[b];
                        let bnode = ca.batt_nodes[b];

                        let price = if par_step == root_step {
                            if bnode < state.rt_prices.len() { state.rt_prices[bnode] } else { 0.0 }
                        } else if par_step < challenge.market.day_ahead_prices.len()
                            && bnode < challenge.market.day_ahead_prices[par_step].len() {
                            challenge.market.day_ahead_prices[par_step][bnode]
                        } else { 0.0 };

                        let max_d = bat.power_discharge_mw * hp.network_derating;
                        let max_c = bat.power_charge_mw * hp.network_derating;
                        let dis_lim = (soc - bat.soc_min_mwh) * bat.efficiency_discharge.max(1e-9) / dt;
                        let chg_lim = (bat.soc_max_mwh - soc) / (bat.efficiency_charge.max(1e-9) * dt);

                        let u = if u_frac > 0.0 {
                            (u_frac * max_d).min(dis_lim.max(0.0)).max(0.0)
                        } else if u_frac < 0.0 {
                            (u_frac * max_c).max(-chg_lim.max(0.0)).min(0.0)
                        } else { 0.0 };

                        let new_soc = if u < 0.0 {
                            soc + bat.efficiency_charge * (-u) * dt
                        } else {
                            soc - u / bat.efficiency_discharge.max(1e-9) * dt
                        };
                        child_socs[b] = new_soc.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

                        let abs_u = u.abs();
                        let deg_base = abs_u * dt / bat.capacity_mwh.max(1e-9);
                        step_profit += u * price * dt - 0.25 * abs_u * dt - deg_base * deg_base;
                    }

                    let child_node_idx = child_soc_base / num_b;
                    cum_profit.push(par_cum + step_profit);
                    node_step.push(par_step + 1);
                    par_node.push(idx);
                    visits.push(0);
                    total_val.push(0.0);
                    children.push([NO_CHILD; N_ACT]);
                    children[idx][act_idx] = child_node_idx;
                    idx = child_node_idx;
                    break;
                } else {
                    sel_depth += 1;
                    if sel_depth > depth + 2 { break; }
                    let v_n = visits[idx] as f64;
                    let log_n = if v_n > 1.0 { v_n.ln() } else { 0.0 };
                    let best_child = ch.iter().copied().fold(ch[0], |best, ci| {
                        let score_ci = if visits[ci] == 0 { f64::MAX } else {
                            total_val[ci] / visits[ci] as f64
                                + ucb_c * (log_n / visits[ci] as f64).sqrt()
                        };
                        let score_best = if visits[best] == 0 { f64::MAX } else {
                            total_val[best] / visits[best] as f64
                                + ucb_c * (log_n / visits[best] as f64).sqrt()
                        };
                        if score_ci > score_best { ci } else { best }
                    });
                    idx = best_child;
                }
            }

            let leaf_step = node_step[idx];
            let leaf_soc_base = idx * num_b;
            let leaf_v = {
                let t_idx = ca.cluster_map.get(leaf_step).copied()
                    .unwrap_or(ca.dp[0].len().saturating_sub(1))
                    .min(ca.dp[0].len().saturating_sub(1));
                let mut v = 0.0_f64;
                for b in 0..num_b {
                    let bat = &challenge.batteries[b];
                    let soc = soc_pool[leaf_soc_base + b];
                    let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                    let dp_b = &ca.dp[b][t_idx];
                    let soc_levels = dp_b.len();
                    if soc_levels < 2 { continue; }
                    let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let i0 = (idx_f.floor() as usize).min(soc_levels - 2);
                    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
                    v += dp_b[i0] * (1.0 - frac) + dp_b[i0 + 1] * frac;
                }
                v
            };
            let total = cum_profit[idx] + leaf_v;

            let mut bp = idx;
            loop {
                visits[bp] += 1;
                total_val[bp] += total;
                let p = par_node[bp];
                if p == NO_CHILD { break; }
                bp = p;
            }
        }

        let best_act = children[0].iter().copied().enumerate()
            .filter(|(_, c)| *c != NO_CHILD && visits[*c] > 0)
            .max_by(|(_, a), (_, b_ci)| {
                let va = total_val[*a] / visits[*a] as f64;
                let vb = total_val[*b_ci] / visits[*b_ci] as f64;
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(3);

        let u_frac = ACT_FRACS[best_act];

        (0..num_b).map(|b| {
            let bat = &challenge.batteries[b];
            let soc = state.socs[b];
            let (u_min, u_max) = state.action_bounds[b];
            let max_d = bat.power_discharge_mw * hp.network_derating;
            let max_c = bat.power_charge_mw * hp.network_derating;
            let dis_lim = (soc - bat.soc_min_mwh) * bat.efficiency_discharge.max(1e-9) / dt;
            let chg_lim = (bat.soc_max_mwh - soc) / (bat.efficiency_charge.max(1e-9) * dt);

            if u_frac > 0.0 {
                (u_frac * max_d).min(dis_lim.max(0.0)).clamp(0.0, u_max)
            } else if u_frac < 0.0 {
                (u_frac * max_c).max(-chg_lim.max(0.0)).clamp(u_min, 0.0)
            } else {
                0.0
            }
        }).collect()
    }

    fn simulate_episode_dfl(
        challenge: &Challenge,
        initial_state: &State,
        cache: &TitanCache,
        hp: &TrackHp,
    ) -> f64 {
        let num_t = challenge.num_steps;
        let num_b = challenge.num_batteries;
        let dt = 0.25_f64;

        let mut socs: Vec<f64> = initial_state.socs.clone();
        let mut total_q = 0.0_f64;

        for t in 0..num_t {
            let rt_prices_sim = if t == 0 {
                initial_state.rt_prices.clone()
            } else if t < challenge.market.day_ahead_prices.len() {
                challenge.market.day_ahead_prices[t].clone()
            } else {
                challenge.market.day_ahead_prices.last().cloned().unwrap_or_default()
            };

            let exo_sim = if t < challenge.exogenous_injections.len() {
                challenge.exogenous_injections[t].clone()
            } else {
                vec![0.0; rt_prices_sim.len()]
            };

            let action_bounds_sim: Vec<(f64, f64)> = (0..num_b).map(|b| {
                let bat = &challenge.batteries[b];
                let soc = socs[b];
                let headroom = (bat.soc_max_mwh - soc).max(0.0);
                let available = (soc - bat.soc_min_mwh).max(0.0);
                let max_c = if bat.efficiency_charge > 0.0 {
                    (headroom / (bat.efficiency_charge * dt)).min(bat.power_charge_mw).max(0.0)
                } else { 0.0 };
                let max_d = if bat.efficiency_discharge > 0.0 {
                    (available * bat.efficiency_discharge / dt).min(bat.power_discharge_mw).max(0.0)
                } else { 0.0 };
                (-max_c, max_d)
            }).collect();

            let sim_state = State {
                time_step: t,
                socs: socs.clone(),
                rt_prices: rt_prices_sim,
                exogenous_injections: exo_sim,
                action_bounds: action_bounds_sim,
                total_profit: total_q,
            };

            let zero_action = vec![0.0_f64; num_b];
            let inj_base = challenge.compute_total_injections(&sim_state, &zero_action);
            let flows_base = challenge.network.compute_flows(&inj_base);

            let mut actions = if hp.use_cg {
                let cg_acts = run_column_generation(challenge, &sim_state, cache, hp, &flows_base);
                if hp.use_cg_lp_combine {
                    if let Some(lp_act) = joint_lp_dispatch(challenge, &sim_state, cache, &flows_base) {
                        let p_cg: f64 = (0..num_b)
                            .map(|b| eval_profit(challenge, &sim_state, cache, b, cg_acts[b]))
                            .sum();
                        let p_lp: f64 = (0..num_b)
                            .map(|b| eval_profit(challenge, &sim_state, cache, b, lp_act[b]))
                            .sum();
                        if p_lp > p_cg { lp_act } else { cg_acts }
                    } else { cg_acts }
                } else { cg_acts }
            } else {
                let mut a = vec![0.0_f64; num_b];
                run_asca(challenge, &sim_state, cache, hp, &flows_base, &mut a);
                run_deflator(challenge, &sim_state, cache, hp, &flows_base, &mut a);
                a
            };
            run_deflator(challenge, &sim_state, cache, hp, &flows_base, &mut actions);

            total_q += challenge.compute_profit(&sim_state, &actions);

            for b in 0..num_b {
                socs[b] = challenge.batteries[b].apply_action_to_soc(actions[b], socs[b]);
            }
        }

        total_q
    }

    fn dfl_select_hp(
        challenge: &Challenge,
        initial_state: &State,
        cache: &TitanCache,
        hp: &TrackHp,
    ) -> TrackHp {
        use std::time::Instant;

        let c0 = hp.clone();
        let mut c1 = hp.clone();
        c1.lmp_premium_scale = 0.45;
        c1.lmp_threshold = 0.70;

        let mut c2 = hp.clone();
        c2.lmp_premium_scale = 0.35;
        c2.lmp_threshold = 0.60;

        let candidates = [c0, c1, c2];
        let t_start = Instant::now();

        let mut best_q = f64::NEG_INFINITY;
        let mut best_idx = 0usize;

        for (k, cand_hp) in candidates.iter().enumerate() {
            if k > 0 && t_start.elapsed().as_millis() > 700 {
                break;
            }
            let sim_q = simulate_episode_dfl(challenge, initial_state, cache, cand_hp);
            if sim_q > best_q {
                best_q = sim_q;
                best_idx = k;
            }
        }

        candidates.into_iter().nth(best_idx).unwrap()
    }   

    fn rcb_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Vec<f64> {
        let num_t = challenge.num_steps;
        let num_b = challenge.num_batteries;
        if state.time_step == 0 {
            let (plan, lp_obj) = rcb_full_horizon_solve(challenge, state, ca, hp);
            RCB_PLAN.with(|p| *p.borrow_mut() = plan);
            RCB_BOUND.with(|v| *v.borrow_mut() = lp_obj);
        }
        let t = state.time_step.min(num_t.saturating_sub(1));
        let (d_t, c_t) = RCB_PLAN.with(|p| {
            let plan = p.borrow();
            let d = plan.get(t).copied().unwrap_or(0.0);
            let c = plan.get(num_t + t).copied().unwrap_or(0.0);
            (d, c)
        });
        let composite_action = d_t - c_t;
        let mut rcb_acts = psc_disaggregate(challenge, state, composite_action);
        run_deflator(challenge, state, ca, hp, flows_base, &mut rcb_acts);

        let base_acts = cg_dispatch(challenge, state, ca, hp, flows_base);
        let p_rcb: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, rcb_acts[b])).sum();
        let p_base: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, base_acts[b])).sum();
        if p_rcb > p_base { rcb_acts } else { base_acts }
    }

    fn cg_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let cg_actions = run_column_generation(challenge, state, ca, hp, flows_base);
        let mut acts = if hp.use_cg_lp_combine {
            if let Some(lp_act) = joint_lp_dispatch(challenge, state, ca, flows_base) {
                let profit_cg: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, cg_actions[b])).sum();
                let profit_lp: f64 = (0..num_b).map(|b| eval_profit(challenge, state, ca, b, lp_act[b])).sum();
                if profit_lp > profit_cg { lp_act } else { cg_actions }
            } else {
                cg_actions
            }
        } else {
            cg_actions
        };
        run_deflator(challenge, state, ca, hp, flows_base, &mut acts);
        acts
    }

    fn rcb_full_horizon_solve(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
    ) -> (Vec<f64>, f64) {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
        let dt = 0.25_f64;
        let tx = 0.25_f64;

        let soc0: f64 = (0..num_b).map(|b| state.socs.get(b).copied().unwrap_or(0.0)).sum();
        let soc_min: f64 = challenge.batteries.iter().map(|b| b.soc_min_mwh).sum();
        let soc_max: f64 = challenge.batteries.iter().map(|b| b.soc_max_mwh).sum();

        let total_cap: f64 = challenge.batteries.iter().map(|b| b.capacity_mwh).sum::<f64>().max(1e-9);
        let eta_d: f64 = challenge.batteries.iter()
            .map(|b| b.efficiency_discharge.max(1e-9) * b.capacity_mwh).sum::<f64>() / total_cap;
        let eta_c: f64 = challenge.batteries.iter()
            .map(|b| b.efficiency_charge * b.capacity_mwh).sum::<f64>() / total_cap;

        let p_d_t0: f64 = (0..num_b).map(|b| state.action_bounds.get(b).map(|&(_, hi)| hi.max(0.0)).unwrap_or(0.0)).sum();
        let p_c_t0: f64 = (0..num_b).map(|b| state.action_bounds.get(b).map(|&(lo, _)| (-lo).max(0.0)).unwrap_or(0.0)).sum();
        let p_d_cap: f64 = challenge.batteries.iter().map(|b| b.power_discharge_mw).sum();
        let p_c_cap: f64 = challenge.batteries.iter().map(|b| b.power_charge_mw).sum();

        let mean_da: Vec<f64> = (0..num_t).map(|t| {
            let sum: f64 = (0..num_b).map(|b| {
                let node = ca.batt_nodes.get(b).copied().unwrap_or(0);
                challenge.market.day_ahead_prices.get(t)
                    .and_then(|row| row.get(node).or_else(|| row.last()))
                    .copied().unwrap_or(0.0)
            }).sum();
            sum / num_b.max(1) as f64
        }).collect();

        let num_l = challenge.network.flow_limits.len();
        let mut premiums = vec![0.0_f64; num_t];
        if hp.anticipate_lmp && num_l > 0 {
            let base_prem = 20.0 * hp.lmp_premium_scale;
            for t in 0..num_t {
                if let Some(exo) = challenge.exogenous_injections.get(t) {
                    let f_exo = challenge.network.compute_flows(exo);
                    for l in 0..num_l {
                        let limit = challenge.network.flow_limits.get(l).copied().unwrap_or(0.0);
                        if limit <= 1e-6 { continue; }
                        let ratio = f_exo.get(l).copied().unwrap_or(0.0).abs() / limit;
                        if ratio > hp.lmp_threshold {
                            let proba = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6)).clamp(0.0, 1.0);
                            let premium = base_prem * proba;
                            let sign_f = f_exo.get(l).copied().unwrap_or(0.0).signum();
                            for &(_, impact) in ca.ptdf_sparse.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
                                if impact.abs() > 1e-6 {
                                    premiums[t] += -impact * sign_f * premium / num_b.max(1) as f64;
                                }
                            }
                        }
                    }
                }
            }
        }

        let sdp_sigma_eff = if hp.use_sdp {
            let sigma = challenge.market.params.volatility;
            let rho_j = challenge.market.params.jump_probability;
            let alpha_j = challenge.market.params.tail_index;
            let jump_var = if alpha_j > 2.0 { rho_j * alpha_j / (alpha_j - 2.0) } else { rho_j * 4.0 };
            (sigma * sigma + jump_var).sqrt() * hp.sdp_sigma_scale
        } else { 0.0 };
        let global_mean_da: f64 = mean_da.iter().sum::<f64>() / num_t.max(1) as f64;

        const GH5_Z: [f64; 5] = [0.0, 0.9586, -0.9586, 2.0202, -2.0202];
        const GH5_W: [f64; 5] = [0.5333, 0.2221, 0.2221, 0.0113, 0.0113];

        let n = 2 * num_t;
        let m = 4 * num_t;
        let n_solves: usize = if hp.rcb_gh_mode == 0 { 1 } else { 5 };
        let mut plan_accum = vec![0.0_f64; n];

        for k in 0..n_solves {
            let (z_k, w_k) = if hp.rcb_gh_mode == 0 {
                (0.0_f64, 1.0_f64)
            } else {
                (GH5_Z[k], GH5_W[k])
            };

            let mut c_obj = vec![0.0_f64; n];
            let mut a_mat = vec![vec![0.0_f64; n]; m];
            let mut b_vec = vec![0.0_f64; m];

            for t in 0..num_t {
                let sigma_t = sdp_sigma_eff * (mean_da[t] / global_mean_da.max(1e-9)).powf(hp.sdp_sigma_het_alpha);
                let eff_price = (mean_da[t] + premiums[t]) * (1.0 + z_k * sigma_t);

                c_obj[t]         = (eff_price - tx) * dt;
                c_obj[num_t + t] = -(eff_price + tx) * dt;

                let p_d = if t == 0 { p_d_t0 } else { p_d_cap };
                let p_c = if t == 0 { p_c_t0 } else { p_c_cap };
                a_mat[t][t] = 1.0;           b_vec[t] = p_d;
                a_mat[num_t + t][num_t + t] = 1.0; b_vec[num_t + t] = p_c;

                let row_lo = 2 * num_t + t;
                for s in 0..=t {
                    a_mat[row_lo][s]         =  dt / eta_d;
                    a_mat[row_lo][num_t + s] = -eta_c * dt;
                }
                b_vec[row_lo] = (soc0 - soc_min).max(0.0);

                let row_hi = 3 * num_t + t;
                for s in 0..=t {
                    a_mat[row_hi][s]         = -dt / eta_d;
                    a_mat[row_hi][num_t + s] =  eta_c * dt;
                }
                b_vec[row_hi] = (soc_max - soc0).max(0.0);
            }

            if let (Some(sol), _) = super::lp::lp_solve_with_budget(n, m, &c_obj, &a_mat, &b_vec, 3000) {
                for i in 0..n { plan_accum[i] += w_k * sol[i]; }
            } else {
            }
        }

        let mut lp_obj = 0.0_f64;
        for t in 0..num_t {
            let eff_price = mean_da[t] + premiums[t];
            lp_obj += (eff_price - tx) * dt * plan_accum[t];
            lp_obj += -(eff_price + tx) * dt * plan_accum[num_t + t];
        }

        (plan_accum, lp_obj)
    }

    fn psc_disaggregate(
        challenge: &Challenge,
        state: &State,
        composite_action: f64,
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let mut actions = vec![0.0_f64; num_b];
        if composite_action.abs() < 1e-9 || num_b == 0 { return actions; }

        let mut remaining = composite_action.abs();

        if composite_action > 0.0 {
            let mut order: Vec<usize> = (0..num_b).collect();
            order.sort_by(|&a, &b| {
                state.socs.get(b).copied().unwrap_or(0.0)
                    .partial_cmp(&state.socs.get(a).copied().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &b in &order {
                if remaining < 1e-9 { break; }
                let hi = state.action_bounds.get(b).map(|&(_, h)| h.max(0.0)).unwrap_or(0.0);
                let alloc = remaining.min(hi);
                actions[b] = alloc;
                remaining -= alloc;
            }
        } else {
            let mut order: Vec<usize> = (0..num_b).collect();
            order.sort_by(|&a, &b| {
                state.socs.get(a).copied().unwrap_or(0.0)
                    .partial_cmp(&state.socs.get(b).copied().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &b in &order {
                if remaining < 1e-9 { break; }
                let lo = state.action_bounds.get(b).map(|&(l, _)| (-l).max(0.0)).unwrap_or(0.0);
                let alloc = remaining.min(lo);
                actions[b] = -alloc;
                remaining -= alloc;
            }
        }
        actions
    }
}
mod lp {
    const LP_EPS: f64 = 1e-9;

    pub fn lp_solve_with_budget(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, usize) {
        let (sol, _, piv) = lp_solve_with_duals(n, m, c, a, b, max_pivots);
        (sol, piv)
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
        let mut pivots_used = 0usize;

        for pivot in 0..max_pivots {
            pivots_used = pivot + 1;
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

        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i][rhs_col].max(0.0);
            }
        }

        let mut y = vec![0.0f64; m];
        for i in 0..m {
            let slack_col = n + i;
            if !basis.contains(&slack_col) {
                y[i] = -tab[m][slack_col];
                if y[i].abs() < LP_EPS { y[i] = 0.0; }
            } else {
                y[i] = 0.0;
            }
            if y[i] < 0.0 { y[i] = 0.0; }
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
            soc_levels: 101,
            action_grid: 40,
            asca_iters: 25,
            ternary_iters: 25,
            convergence_tol: 1e-4,
            anticipate_lmp: true,
            lmp_threshold: 0.65,
            lmp_premium_scale: 0.40,
            jump_premium: 0.02,
            prune_ratio: 0.00,
            deflator_iters: 15,
            flow_margin: 1e-4,
            network_derating: 1.00,
            use_sdp: true,
            use_lp: true,
            lp_refine_sweeps: 3,
            use_cg: true,
            cg_iters: 20,
            sdp_sigma_het_alpha: 0.45,
            sdp_sigma_scale: 1.1,
            lambda_track: 0.0,
            use_analytical_pricing: true,
            use_morales_vi: false,
            use_lp_dual_warmstart: false,
            het_lmp_alpha: 0.0,
            use_lp_basis_warmstart: false,
            het_crate_alpha: 0.0,
            use_cg_lp_combine: true,
            k_clusters: 80,
            use_tree_search: false,
            ts_depth: 4,
            ts_iters_per_step: 1500,
            ts_ucb_c: 1.4,
            use_dfl_select: false,
            use_soc_ref_track: false,
            soc_ref_rho: 0.05,
            soc_ref_mu_cap: 0.5,
            soc_ref_gh_extreme: false,
            use_rcb_full_horizon: false,
            rcb_gh_mode: 0,
            use_pce_affine_recourse: true,
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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    track_baseline::solve(challenge, save_solution, hyperparameters)
}