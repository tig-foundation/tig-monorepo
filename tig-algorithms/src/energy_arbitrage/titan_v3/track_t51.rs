

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod helpers {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use std::cell::RefCell;
    use tig_challenges::energy_arbitrage::*;

    use std::arch::x86_64::*;

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
        pub ldd_bundle_radius: f64,
        
        pub ldd_temporal_eps: f64,
        
        pub ldd_use_polyak: bool,
        pub ldd_polyak_safety: f64,
        
        pub ldd_use_dual_avg: bool,
        pub ldd_da_decay: f64,  
        pub ldd_use_polyak_dp: bool,  
        
        
        
        pub ldd_use_bplm: bool,
        pub ldd_bplm_alpha: f64,
        
        
        pub ldd_use_lp_warmstart: bool,
        pub ldd_warmstart_scale: f64,
        
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
        
        
        pub use_threshold_decode: bool,
        
        
        
        pub use_analytic_dp_cell: bool,
        
        
        pub use_simd: bool,
        
        
        
        
        
        pub use_analytic_argmax: bool,
        
        
        
        
        
        pub use_slope_cache: bool,
        
        
        
        
        
        
        pub use_bat_hoist: bool,
        
        
        
        
        pub use_cf_dispatch: bool,
        
        pub cf_asca_iters: usize,
        
        
        
        
        
        pub use_ri_dp: bool,
        pub ri_cong_threshold: f64,
        
        
        
        
        
        pub phi_int_alpha: f64,
        
        
        
        pub use_soc_pruning: bool,
        
        
        
        
        
        
        pub use_mfg: bool,
        pub mfg_iters: usize,
        
        pub use_hybrid_asca: bool,
        pub hybrid_asca_iters: usize,
        
        
        
        
        pub use_per_bat_lmp_scale: bool,
        pub per_bat_lmp_alpha: f64,
        
        
        
        
        
        pub use_dual_cf: bool,
        pub dual_cf_scale: f64,
        
        
        
        pub use_morales_soc_pruning: bool,
        
        
        
        
        
        pub use_best_iter_dp: bool,
        
        
        
        
        
        
        pub use_cf_mu_warmstart: bool,
        pub cf_mu_warmstart_scale: f64,
        
        
        
        
        pub use_mu_teff: bool,
        pub mu_teff_eps: f64,
        
        
        
        pub use_rcb_fleet: bool,
        pub rcb_price_mode: u8,
        
        
        
        pub use_rcb_dp_fleet: bool,
        pub rcb_dp_post_iters: u8,
        
        
        
        pub ldd_use_dowg: bool,
        
        
        
        pub post_asca_admm_iters: u8,
        pub post_asca_admm_rho: f64,
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
            if let Some(v) = m.get("ldd_bundle_radius").and_then(|v| v.as_f64()) { self.ldd_bundle_radius = v.max(0.0); }
            if let Some(v) = m.get("ldd_temporal_eps").and_then(|v| v.as_f64()) { self.ldd_temporal_eps = v.max(0.0); }
            if let Some(v) = m.get("ldd_use_polyak").and_then(|v| v.as_bool()) { self.ldd_use_polyak = v; }
            if let Some(v) = m.get("ldd_polyak_safety").and_then(|v| v.as_f64()) { self.ldd_polyak_safety = v.max(0.0); }
            if let Some(v) = m.get("ldd_use_dual_avg").and_then(|v| v.as_bool()) { self.ldd_use_dual_avg = v; }
            if let Some(v) = m.get("ldd_da_decay").and_then(|v| v.as_f64()) { self.ldd_da_decay = v.max(0.0); }
            if let Some(v) = m.get("ldd_use_polyak_dp").and_then(|v| v.as_bool()) { self.ldd_use_polyak_dp = v; }
            if let Some(v) = m.get("ldd_use_bplm").and_then(|v| v.as_bool()) { self.ldd_use_bplm = v; }
            if let Some(v) = m.get("ldd_bplm_alpha").and_then(|v| v.as_f64()) { self.ldd_bplm_alpha = v.clamp(0.5, 0.999); }
            if let Some(v) = m.get("ldd_use_lp_warmstart").and_then(|v| v.as_bool()) { self.ldd_use_lp_warmstart = v; }
            if let Some(v) = m.get("ldd_warmstart_scale").and_then(|v| v.as_f64()) { self.ldd_warmstart_scale = v.max(0.0); }
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
            if let Some(v) = m.get("use_threshold_decode").and_then(|v| v.as_bool()) { self.use_threshold_decode = v; }
            if let Some(v) = m.get("use_analytic_dp_cell").and_then(|v| v.as_bool()) { self.use_analytic_dp_cell = v; }
            if let Some(v) = m.get("use_simd").and_then(|v| v.as_bool()) { self.use_simd = v; }
            if let Some(v) = m.get("use_analytic_argmax").and_then(|v| v.as_bool()) { self.use_analytic_argmax = v; }
            if let Some(v) = m.get("use_slope_cache").and_then(|v| v.as_bool()) { self.use_slope_cache = v; }
            if let Some(v) = m.get("use_bat_hoist").and_then(|v| v.as_bool()) { self.use_bat_hoist = v; }
            if let Some(v) = m.get("use_cf_dispatch").and_then(|v| v.as_bool()) { self.use_cf_dispatch = v; }
            if let Some(v) = m.get("cf_asca_iters").and_then(|v| v.as_u64()) { self.cf_asca_iters = v as usize; }
            if let Some(v) = m.get("use_ri_dp").and_then(|v| v.as_bool()) { self.use_ri_dp = v; }
            if let Some(v) = m.get("ri_cong_threshold").and_then(|v| v.as_f64()) { self.ri_cong_threshold = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("phi_int_alpha").and_then(|v| v.as_f64()) { self.phi_int_alpha = v.clamp(0.0, 1.0); }
            if let Some(v) = m.get("use_soc_pruning").and_then(|v| v.as_bool()) { self.use_soc_pruning = v; }
            if let Some(v) = m.get("use_mfg").and_then(|v| v.as_bool()) { self.use_mfg = v; }
            if let Some(v) = m.get("mfg_iters").and_then(|v| v.as_u64()) { self.mfg_iters = v as usize; }
            if let Some(v) = m.get("use_hybrid_asca").and_then(|v| v.as_bool()) { self.use_hybrid_asca = v; }
            if let Some(v) = m.get("hybrid_asca_iters").and_then(|v| v.as_u64()) { self.hybrid_asca_iters = v as usize; }
            if let Some(v) = m.get("use_per_bat_lmp_scale").and_then(|v| v.as_bool()) { self.use_per_bat_lmp_scale = v; }
            if let Some(v) = m.get("per_bat_lmp_alpha").and_then(|v| v.as_f64()) { self.per_bat_lmp_alpha = v; }
            if let Some(v) = m.get("use_dual_cf").and_then(|v| v.as_bool()) { self.use_dual_cf = v; }
            if let Some(v) = m.get("dual_cf_scale").and_then(|v| v.as_f64()) { self.dual_cf_scale = v.max(0.0); }
            if let Some(v) = m.get("use_morales_soc_pruning").and_then(|v| v.as_bool()) { self.use_morales_soc_pruning = v; }
            if let Some(v) = m.get("use_best_iter_dp").and_then(|v| v.as_bool()) { self.use_best_iter_dp = v; }
            if let Some(v) = m.get("use_cf_mu_warmstart").and_then(|v| v.as_bool()) { self.use_cf_mu_warmstart = v; }
            if let Some(v) = m.get("cf_mu_warmstart_scale").and_then(|v| v.as_f64()) { self.cf_mu_warmstart_scale = v.max(0.0); }
            if let Some(v) = m.get("use_mu_teff").and_then(|v| v.as_bool()) { self.use_mu_teff = v; }
            if let Some(v) = m.get("mu_teff_eps").and_then(|v| v.as_f64()) { self.mu_teff_eps = v.max(0.0); }
            if let Some(v) = m.get("use_rcb_fleet").and_then(|v| v.as_bool()) { self.use_rcb_fleet = v; }
            if let Some(v) = m.get("rcb_price_mode").and_then(|v| v.as_u64()) { self.rcb_price_mode = v as u8; }
            if let Some(v) = m.get("use_rcb_dp_fleet").and_then(|v| v.as_bool()) { self.use_rcb_dp_fleet = v; }
            if let Some(v) = m.get("rcb_dp_post_iters").and_then(|v| v.as_u64()) { self.rcb_dp_post_iters = v as u8; }
            if let Some(v) = m.get("ldd_use_dowg").and_then(|v| v.as_bool()) { self.ldd_use_dowg = v; }
            if let Some(v) = m.get("post_asca_admm_iters").and_then(|v| v.as_u64()) { self.post_asca_admm_iters = v as u8; }
            if let Some(v) = m.get("post_asca_admm_rho").and_then(|v| v.as_f64()) { self.post_asca_admm_rho = v.max(1e-9); }
        }
    }

    
    
    
    
    pub struct DpCube {
        pub data: Vec<f64>,
        pub stride_b: usize,
        pub stride_t: usize,
        pub num_b: usize,
        pub num_t_plus_1: usize,
        pub soc_levels: usize,
    }

    impl DpCube {
        #[inline(always)]
        pub fn new(num_b: usize, num_t_plus_1: usize, soc_levels: usize) -> Self {
            let stride_t = soc_levels;
            let stride_b = num_t_plus_1 * soc_levels;
            Self {
                data: vec![0.0_f64; num_b * stride_b],
                stride_b,
                stride_t,
                num_b,
                num_t_plus_1,
                soc_levels,
            }
        }

        #[inline(always)]
        pub fn cell(&self, b: usize, t: usize, i: usize) -> f64 {
            self.data[b * self.stride_b + t * self.stride_t + i]
        }

        #[inline(always)]
        pub fn set(&mut self, b: usize, t: usize, i: usize, v: f64) {
            self.data[b * self.stride_b + t * self.stride_t + i] = v;
        }

        #[inline(always)]
        pub fn slice_bt(&self, b: usize, t: usize) -> &[f64] {
            let off = b * self.stride_b + t * self.stride_t;
            &self.data[off..off + self.soc_levels]
        }
    }

    
    
    
    
    #[derive(Clone)]
    pub struct BatHoist {
        pub deg_coeff: f64,
        pub inv_2deg: f64,
        pub soc_min: f64,
        pub soc_max: f64,
        pub soc_span: f64,
        pub eff_c: f64,
        pub eff_d: f64,
        pub beta_charge: f64,
        pub beta_discharge: f64,
    }

    
    
    
    pub struct SocEnvelope {
        pub fwd_max: Vec<Vec<f64>>,
        pub bwd_min: Vec<Vec<f64>>,
    }

    pub struct TitanCache {
        pub dp: DpCube,
        
        
        
        
        
        pub dp_slope: Option<DpCube>,
        
        
        
        
        
        pub bat_hoist: Option<Vec<BatHoist>>,
        
        pub soc_envelope: Option<SocEnvelope>,
        pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
        pub b_to_lines: Vec<Vec<(usize, f64)>>,
        pub batt_nodes: Vec<usize>,
        
        pub rcb_p_agg: Option<Vec<f64>>,
    }

    struct Inner {
        hp: TrackHp,
        cache: Option<TitanCache>,
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
            let inner = guard.as_mut().expect("titan: STATE not initialised");
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

            
            
            let actions = if hp.use_cf_dispatch {
                
                
                
                let mut cf_act = cf_lp_dispatch(challenge, state, cache, hp, &flows_base);
                run_deflator(challenge, state, cache, hp, &flows_base, &mut cf_act);
                
                
                if hp.use_rcb_dp_fleet {
                    
                    
                    run_dsfd(challenge, state, cache, &mut cf_act);
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut cf_act);
                    if hp.rcb_dp_post_iters > 0 {
                        let mut hp_post = hp.clone();
                        hp_post.asca_iters = hp.rcb_dp_post_iters as usize;
                        run_asca(challenge, state, cache, &hp_post, &flows_base, &mut cf_act);
                        run_deflator(challenge, state, cache, hp, &flows_base, &mut cf_act);
                    }
                } else if hp.use_rcb_fleet {
                    
                    
                    run_rcb_fleet_dispatch(challenge, state, cache, &mut cf_act);
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut cf_act);
                } else if hp.use_mfg && hp.mfg_iters > 0 {
                    
                    
                    run_mfg_warm(challenge, state, cache, hp, &flows_base, &mut cf_act);
                    
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut cf_act);
                } else if hp.cf_asca_iters > 0 {
                    let mut hp_warm = hp.clone();
                    hp_warm.asca_iters = hp.cf_asca_iters;
                    run_asca(challenge, state, cache, &hp_warm, &flows_base, &mut cf_act);
                    
                    
                    
                    if hp.use_ri_dp {
                        run_ri_dp(challenge, state, cache, hp, &flows_base, &mut cf_act);
                    }
                    
                    
                    if hp.post_asca_admm_iters > 0 {
                        let mut hp_a = hp.clone();
                        hp_a.max_admm_iters = hp.post_asca_admm_iters as usize;
                        hp_a.admm_rho = hp.post_asca_admm_rho;
                        run_admm_dispatch(challenge, state, cache, &hp_a, &flows_base, &mut cf_act);
                    }
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut cf_act);
                }
                let cf_p: f64 = (0..challenge.num_batteries)
                    .map(|b| eval_profit(challenge, state, cache, b, cf_act[b])).sum();
                let base_p: f64 = (0..challenge.num_batteries)
                    .map(|b| eval_profit(challenge, state, cache, b, base_actions[b])).sum();
                let mut selected = if cf_p >= base_p { cf_act } else { base_actions };
                if hp.use_hybrid_asca && hp.hybrid_asca_iters > 0 {
                    let mut hp_hybrid = hp.clone();
                    hp_hybrid.asca_iters = hp.hybrid_asca_iters;
                    run_asca(challenge, state, cache, &hp_hybrid, &flows_base, &mut selected);
                    run_deflator(challenge, state, cache, hp, &flows_base, &mut selected);
                }
                selected
            } else if hp.use_lp {
                
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

    
    
    
    
    
    
    
    
    #[inline]
    fn analytic_dp_cell_max(
        bat: &Battery,
        p_buy: f64,
        p_sell: f64,
        soc: f64,
        u_min: f64,
        u_max: f64,
        v_next: &[f64],
        soc_levels: usize,
        soc_span: f64,
        deg_coeff: f64,
    ) -> f64 {
        let dt = 0.25_f64;

        
        let lambda = if soc_levels > 1 {
            let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
            let idx0 = (idx_f.floor() as usize).min(soc_levels - 2);
            let delta_soc = soc_span / ((soc_levels - 1) as f64);
            (v_next[idx0 + 1] - v_next[idx0]) / delta_soc.max(1e-12)
        } else {
            0.0
        };

        
        
        let eval = |u: f64| -> f64 {
            let price = if u > 0.0 { p_sell } else { p_buy };
            let abs_u = u.abs();
            let revenue = u * price * dt;
            let tx = 0.25 * abs_u * dt;
            let deg = deg_coeff * u * u;
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
            let v_n = v_next[idx0c] * (1.0 - frac) + v_next[idx1c] * frac;
            profit + v_n
        };

        
        
        let mut best = eval(0.0_f64.clamp(u_min, u_max));

        
        
        
        
        
        
        
        if u_min < 0.0 {
            let u_hi = 0.0_f64.min(u_max);
            if u_min < u_hi {
                let b_c = dt * (lambda * bat.efficiency_charge - p_buy - 0.25);
                let x_star = if deg_coeff > 1e-30 { b_c / (2.0 * deg_coeff) } else { -u_min };
                
                let cand = (-x_star).clamp(u_min, u_hi);
                let v = eval(cand);
                if v > best { best = v; }
                
                let v = eval(u_min);
                if v > best { best = v; }
            }
        }

        
        
        
        
        
        if u_max > 0.0 {
            let u_lo = 0.0_f64.max(u_min);
            if u_lo < u_max {
                let eff_d = bat.efficiency_discharge.max(1e-9);
                let b_d = dt * (p_sell - 0.25 - lambda / eff_d);
                let x_star = if deg_coeff > 1e-30 { b_d / (2.0 * deg_coeff) } else { u_max };
                let cand = x_star.clamp(u_lo, u_max);
                let v = eval(cand);
                if v > best { best = v; }
                
                let v = eval(u_max);
                if v > best { best = v; }
            }
        }

        if best == f64::NEG_INFINITY { 0.0 } else { best }
    }

    
    
    
    
    
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn simd_action_max_avx2(
        soc: f64,
        p_sell: f64,
        p_buy: f64,
        u_min: f64,
        span: f64,
        action_grid: usize,
        dt: f64,
        next_slice: &[f64],
        soc_levels: usize,
        soc_span: f64,
        soc_min_dp: f64,
        soc_max_dp: f64,
        eff_charge: f64,
        eff_disch_clamped: f64,
        capacity_clamped: f64,
    ) -> f64 {
        let n_vec = (action_grid + 1) / 4;

        let dt_v = _mm256_set1_pd(dt);
        let quarter_v = _mm256_set1_pd(0.25);
        let zero_v = _mm256_set1_pd(0.0);
        let one_v = _mm256_set1_pd(1.0);
        let u_min_v = _mm256_set1_pd(u_min);
        let span_v = _mm256_set1_pd(span);
        let action_grid_f_v = _mm256_set1_pd(action_grid as f64);
        let p_sell_v = _mm256_set1_pd(p_sell);
        let p_buy_v = _mm256_set1_pd(p_buy);
        let soc_v = _mm256_set1_pd(soc);
        let eff_charge_v = _mm256_set1_pd(eff_charge);
        let eff_disch_v = _mm256_set1_pd(eff_disch_clamped);
        let cap_v = _mm256_set1_pd(capacity_clamped);
        let soc_min_dp_v = _mm256_set1_pd(soc_min_dp);
        let soc_max_dp_v = _mm256_set1_pd(soc_max_dp);
        let soc_span_v = _mm256_set1_pd(soc_span);
        let soc_lev_m1_v = _mm256_set1_pd((soc_levels - 1) as f64);
        let sign_mask = _mm256_set1_pd(-0.0_f64);

        let mut max_v = _mm256_set1_pd(f64::NEG_INFINITY);
        let mut idxf_buf = [0.0_f64; 4];
        let mut frac_buf = [0.0_f64; 4];
        let mut v0_buf = [0.0_f64; 4];
        let mut v1_buf = [0.0_f64; 4];

        for chunk in 0..n_vec {
            let j0 = (chunk * 4) as f64;
            
            let j_v = _mm256_set_pd(j0 + 3.0, j0 + 2.0, j0 + 1.0, j0);

            
            let u_tmp = _mm256_mul_pd(span_v, j_v);
            let u_div = _mm256_div_pd(u_tmp, action_grid_f_v);
            let u_v = _mm256_add_pd(u_min_v, u_div);

            
            let pos_mask = _mm256_cmp_pd(u_v, zero_v, _CMP_GT_OQ);
            let price_v = _mm256_blendv_pd(p_buy_v, p_sell_v, pos_mask);

            
            let abs_u_v = _mm256_andnot_pd(sign_mask, u_v);

            
            let r_t = _mm256_mul_pd(u_v, price_v);
            let revenue_v = _mm256_mul_pd(r_t, dt_v);

            
            let tx_t = _mm256_mul_pd(quarter_v, abs_u_v);
            let tx_v = _mm256_mul_pd(tx_t, dt_v);

            
            let db_t = _mm256_mul_pd(abs_u_v, dt_v);
            let db_v = _mm256_div_pd(db_t, cap_v);
            let deg_v = _mm256_mul_pd(db_v, db_v);

            
            let p_t = _mm256_sub_pd(revenue_v, tx_v);
            let profit_v = _mm256_sub_pd(p_t, deg_v);

            
            
            
            let neg_mask = _mm256_cmp_pd(u_v, zero_v, _CMP_LT_OQ);
            let neg_u_v = _mm256_sub_pd(zero_v, u_v);
            let a_t1 = _mm256_mul_pd(eff_charge_v, neg_u_v);
            let a_t2 = _mm256_mul_pd(a_t1, dt_v);
            let res_a = _mm256_add_pd(soc_v, a_t2);
            let b_t1 = _mm256_div_pd(u_v, eff_disch_v);
            let b_t2 = _mm256_mul_pd(b_t1, dt_v);
            let res_b = _mm256_sub_pd(soc_v, b_t2);
            let next_soc_raw_v = _mm256_blendv_pd(res_b, res_a, neg_mask);
            
            let nsr_lo = _mm256_max_pd(next_soc_raw_v, soc_min_dp_v);
            let next_soc_v = _mm256_min_pd(nsr_lo, soc_max_dp_v);

            
            
            let i_t1 = _mm256_sub_pd(next_soc_v, soc_min_dp_v);
            let i_t2 = _mm256_div_pd(i_t1, soc_span_v);
            let idx_f_v = _mm256_mul_pd(i_t2, soc_lev_m1_v);

            
            _mm256_storeu_pd(idxf_buf.as_mut_ptr(), idx_f_v);
            for k in 0..4 {
                let idx_f = idxf_buf[k];
                let idx0 = (idx_f.floor() as isize).max(0) as usize;
                let idx0c = idx0.min(soc_levels - 1);
                let idx1c = (idx0 + 1).min(soc_levels - 1);
                let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                frac_buf[k] = frac;
                v0_buf[k] = next_slice[idx0c];
                v1_buf[k] = next_slice[idx1c];
            }
            let frac_v = _mm256_loadu_pd(frac_buf.as_ptr());
            let v0_v = _mm256_loadu_pd(v0_buf.as_ptr());
            let v1_v = _mm256_loadu_pd(v1_buf.as_ptr());

            
            let one_minus_f = _mm256_sub_pd(one_v, frac_v);
            let vn_t2 = _mm256_mul_pd(v0_v, one_minus_f);
            let vn_t3 = _mm256_mul_pd(v1_v, frac_v);
            let v_next_v = _mm256_add_pd(vn_t2, vn_t3);

            let val_v = _mm256_add_pd(profit_v, v_next_v);
            max_v = _mm256_max_pd(max_v, val_v);
        }

        
        let mut max_arr = [0.0_f64; 4];
        _mm256_storeu_pd(max_arr.as_mut_ptr(), max_v);
        let mut max_val = max_arr[0]
            .max(max_arr[1])
            .max(max_arr[2])
            .max(max_arr[3]);

        
        let tail_start = n_vec * 4;
        for j in tail_start..=action_grid {
            let u = if span > 0.0 {
                u_min + span * (j as f64) / (action_grid as f64)
            } else {
                u_min
            };
            let price = if u > 0.0 { p_sell } else { p_buy };
            let abs_u = u.abs();
            let revenue = u * price * dt;
            let tx = 0.25 * abs_u * dt;
            let deg_base = (abs_u * dt) / capacity_clamped;
            let deg = deg_base * deg_base;
            let profit = revenue - tx - deg;

            let next_soc_raw = if u < 0.0 {
                soc + eff_charge * (-u) * dt
            } else {
                soc - u / eff_disch_clamped * dt
            };
            let next_soc = next_soc_raw.clamp(soc_min_dp, soc_max_dp);

            let idx_f = (next_soc - soc_min_dp) / soc_span * ((soc_levels - 1) as f64);
            let idx0 = (idx_f.floor() as isize).max(0) as usize;
            let idx0c = idx0.min(soc_levels - 1);
            let idx1c = (idx0 + 1).min(soc_levels - 1);
            let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
            let v_next = next_slice[idx0c] * (1.0 - frac) + next_slice[idx1c] * frac;

            let val = profit + v_next;
            if val > max_val {
                max_val = val;
            }
        }
        max_val
    }

    fn build_dp_with_mu(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu: &[Vec<f64>],
        initial_socs: &[f64],
    ) -> DpCube {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
        let soc_levels = hp.soc_levels;
        let action_grid = hp.action_grid;
        let dt = 0.25_f64;

        
        let mut dp = DpCube::new(num_b, num_t + 1, soc_levels);
        let stride_b = dp.stride_b;
        let stride_t = dp.stride_t;

        
        let simd_enabled = hp.use_simd && is_x86_feature_detected!("avx2");

        
        
        
        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = batt_nodes[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let deg_coeff = (dt / bat.capacity_mwh.max(1e-9)).powi(2);

            
            
            let (reach_lo, reach_hi): (Vec<usize>, Vec<usize>) =
                if hp.use_morales_soc_pruning && b < initial_socs.len() {
                    let k0 = ((initial_socs[b] - bat.soc_min_mwh) / soc_span
                        * (soc_levels - 1) as f64)
                        .clamp(0.0, (soc_levels - 1) as f64)
                        .round() as usize;
                    let dc = (bat.power_charge_mw * bat.efficiency_charge * dt
                        / soc_span * (soc_levels - 1) as f64)
                        .ceil() as usize;
                    let dd = (bat.power_discharge_mw
                        / bat.efficiency_discharge.max(1e-9) * dt
                        / soc_span * (soc_levels - 1) as f64)
                        .ceil() as usize;
                    let mut lo = vec![k0; num_t + 1];
                    let mut hi = vec![k0; num_t + 1];
                    for step in 0..num_t {
                        hi[step + 1] = (hi[step] + dc).min(soc_levels - 1);
                        lo[step + 1] = lo[step].saturating_sub(dd);
                    }
                    (lo, hi)
                } else {
                    (vec![0usize; num_t + 1], vec![soc_levels - 1; num_t + 1])
                };

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

                let i_lo = reach_lo[t];
                let i_hi = reach_hi[t];
                for i in i_lo..=i_hi {
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

                    
                    let next_off = b * stride_b + (t + 1) * stride_t;
                    let max_val = if hp.use_analytic_dp_cell {
                        analytic_dp_cell_max(
                            bat,
                            p_buy,
                            p_sell,
                            soc,
                            u_min,
                            u_max,
                            &dp.data[next_off..next_off + soc_levels],
                            soc_levels,
                            soc_span,
                            deg_coeff,
                        )
                    } else {
                        let span = u_max - u_min;
                        
                        
                        let simd_ok = simd_enabled && action_grid >= 3;
                        if simd_ok {
                            
                            unsafe {
                                simd_action_max_avx2(
                                    soc,
                                    p_sell,
                                    p_buy,
                                    u_min,
                                    span,
                                    action_grid,
                                    dt,
                                    &dp.data[next_off..next_off + soc_levels],
                                    soc_levels,
                                    soc_span,
                                    bat.soc_min_mwh,
                                    bat.soc_max_mwh,
                                    bat.efficiency_charge,
                                    bat.efficiency_discharge.max(1e-9),
                                    bat.capacity_mwh.max(1e-9),
                                )
                            }
                        } else {
                            let mut max_val = f64::NEG_INFINITY;
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
                                let v_next = dp.data[next_off + idx0c] * (1.0 - frac)
                                    + dp.data[next_off + idx1c] * frac;

                                let val = profit + v_next;
                                if val > max_val { max_val = val; }
                            }
                            max_val
                        }
                    };
                    dp.data[b * stride_b + t * stride_t + i] = max_val;
                }
            }
        }
        dp
    }

    
    
    
    
    fn build_dp_with_mu_teff(
        challenge: &Challenge,
        hp: &TrackHp,
        batt_nodes: &[usize],
        expected_premiums_teff: &[Vec<f64>],
        b_to_lines: &[Vec<(usize, f64)>],
        mu_teff: &[Vec<f64>],
        initial_socs: &[f64],
        rep_times: &[usize],
    ) -> DpCube {
        let num_b = challenge.num_batteries;
        let t_eff = rep_times.len();
        let soc_levels = hp.soc_levels;
        let action_grid = hp.action_grid;
        let dt = 0.25_f64;

        let mut dp = DpCube::new(num_b, t_eff + 1, soc_levels);
        let stride_b = dp.stride_b;
        let stride_t = dp.stride_t;
        let simd_enabled = hp.use_simd && is_x86_feature_detected!("avx2");

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = batt_nodes[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let deg_coeff = (dt / bat.capacity_mwh.max(1e-9)).powi(2);

            let (reach_lo, reach_hi): (Vec<usize>, Vec<usize>) =
                if hp.use_morales_soc_pruning && b < initial_socs.len() {
                    let k0 = ((initial_socs[b] - bat.soc_min_mwh) / soc_span
                        * (soc_levels - 1) as f64)
                        .clamp(0.0, (soc_levels - 1) as f64)
                        .round() as usize;
                    let dc = (bat.power_charge_mw * bat.efficiency_charge * dt
                        / soc_span * (soc_levels - 1) as f64)
                        .ceil() as usize;
                    let dd = (bat.power_discharge_mw
                        / bat.efficiency_discharge.max(1e-9) * dt
                        / soc_span * (soc_levels - 1) as f64)
                        .ceil() as usize;
                    let mut lo = vec![k0; t_eff + 1];
                    let mut hi = vec![k0; t_eff + 1];
                    for step in 0..t_eff {
                        hi[step + 1] = (hi[step] + dc).min(soc_levels - 1);
                        lo[step + 1] = lo[step].saturating_sub(dd);
                    }
                    (lo, hi)
                } else {
                    (vec![0usize; t_eff + 1], vec![soc_levels - 1; t_eff + 1])
                };

            for c in (0..t_eff).rev() {
                let rt = rep_times[c];
                let p_da = if node < challenge.market.day_ahead_prices[rt].len() {
                    challenge.market.day_ahead_prices[rt][node]
                } else {
                    challenge.market.day_ahead_prices[rt][0]
                };

                let mu_adjust: f64 = b_to_lines[b].iter()
                    .map(|&(l, impact)| mu_teff[c][l] * impact)
                    .sum();

                let extra = expected_premiums_teff[c][b] - mu_adjust;
                let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
                let p_buy = p_da + extra;

                let max_pwr_c = bat.power_charge_mw * hp.network_derating;
                let max_pwr_d = bat.power_discharge_mw * hp.network_derating;

                let i_lo = reach_lo[c];
                let i_hi = reach_hi[c];
                for i in i_lo..=i_hi {
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

                    let next_off = b * stride_b + (c + 1) * stride_t;
                    let max_val = if hp.use_analytic_dp_cell {
                        analytic_dp_cell_max(
                            bat, p_buy, p_sell, soc, u_min, u_max,
                            &dp.data[next_off..next_off + soc_levels],
                            soc_levels, soc_span, deg_coeff,
                        )
                    } else {
                        let span = u_max - u_min;
                        let simd_ok = simd_enabled && action_grid >= 3;
                        if simd_ok {
                            unsafe {
                                simd_action_max_avx2(
                                    soc, p_sell, p_buy, u_min, span, action_grid, dt,
                                    &dp.data[next_off..next_off + soc_levels],
                                    soc_levels, soc_span, bat.soc_min_mwh, bat.soc_max_mwh,
                                    bat.efficiency_charge,
                                    bat.efficiency_discharge.max(1e-9),
                                    bat.capacity_mwh.max(1e-9),
                                )
                            }
                        } else {
                            let mut max_val = f64::NEG_INFINITY;
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
                                let v_next = dp.data[next_off + idx0c] * (1.0 - frac)
                                    + dp.data[next_off + idx1c] * frac;
                                let val = profit + v_next;
                                if val > max_val { max_val = val; }
                            }
                            max_val
                        }
                    };
                    dp.data[b * stride_b + c * stride_t + i] = max_val;
                }
            }
        }
        dp
    }

    
    
    
    fn sliding_window_cluster(mu: &[Vec<f64>], eps: f64) -> Vec<usize> {
        let n = mu.len();
        let mut map = vec![0usize; n];
        let mut c = 0usize;
        for t in 1..n {
            let diff = mu[t].iter().zip(&mu[t - 1]).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            if diff >= eps { c += 1; }
            map[t] = c;
        }
        map
    }

    
    
    fn ldd_simulate_flows(
        challenge: &Challenge,
        state: &State,
        dp: &DpCube,
        batt_nodes: &[usize],
        ptdf_sparse: &[Vec<(usize, f64)>],
        teff_map: Option<&[usize]>,
    ) -> Vec<Vec<f64>> {
        let num_b = challenge.num_batteries;
        let num_t = challenge.num_steps;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let sim_pts = 20usize;

        let mut socs: Vec<f64> = state.socs.clone();
        let mut flows_all = vec![vec![0.0_f64; num_l]; num_t];
        let stride_b = dp.stride_b;
        let stride_t = dp.stride_t;
        let soc_levels_cube = dp.soc_levels;

        for t in 0..num_t {
            let exo_flows = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let mut bat_actions = vec![0.0_f64; num_b];

            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let soc = socs[b];
                let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                let soc_levels = soc_levels_cube;
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
                    let t_next = if let Some(cm) = teff_map {
                        (cm[t] + 1).min(dp.num_t_plus_1 - 1)
                    } else {
                        (t + 1).min(num_t)
                    };
                    let off = b * stride_b + t_next * stride_t;
                    let v_next = dp.data[off + idx0c] * (1.0 - frac) + dp.data[off + idx1c] * frac;

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

        
        
        
        let per_bat_lmp_scale: Vec<f64> = if hp.use_per_bat_lmp_scale && num_b > 0 {
            let ptdf_mean_b: Vec<f64> = (0..num_b).map(|b| {
                if b_to_lines[b].is_empty() { 0.0 }
                else { b_to_lines[b].iter().map(|&(_, i)| i.abs()).sum::<f64>() / b_to_lines[b].len() as f64 }
            }).collect();
            let global_mean = ptdf_mean_b.iter().sum::<f64>() / num_b as f64;
            ptdf_mean_b.iter().map(|&p| {
                let raw = hp.lmp_premium_scale * (1.0 + hp.per_bat_lmp_alpha * (global_mean - p) / global_mean.max(1e-9));
                raw.clamp(0.40, 1.20)
            }).collect()
        } else {
            vec![hp.lmp_premium_scale; num_b]
        };

        let mut expected_premiums = vec![vec![0.0_f64; num_b]; num_t];
        if hp.anticipate_lmp && num_l > 0 {
            let base_premium = 20.0;
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
                                let nodal_shift = -impact * sign_f * premium * per_bat_lmp_scale[b];
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

        
        
        let (dp, final_mu) = if hp.ldd_iters > 0 && num_l > 0 {
            let mut mu = vec![vec![0.0_f64; num_l]; num_t];

            
            
            
            if hp.use_cf_mu_warmstart {
                let scale = hp.cf_mu_warmstart_scale;
                for t in 0..num_t {
                    let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                    for l in 0..num_l {
                        let limit = challenge.network.flow_limits[l];
                        if limit <= 1e-6 { continue; }
                        let viol = f_exo[l].abs() - limit;
                        if viol > 1e-6 {
                            mu[t][l] = f_exo[l].signum() * viol * scale;
                        }
                    }
                }
            }

            
            
            if hp.ldd_use_lp_warmstart {
                let mut hp_presim = hp.clone();
                hp_presim.soc_levels = 31;
                hp_presim.action_grid = 10;
                let mu_zero = vec![vec![0.0_f64; num_l]; num_t];
                let dp_presim = build_dp_with_mu(challenge, &hp_presim, &batt_nodes, &expected_premiums, &b_to_lines, &mu_zero, &state.socs);
                let flows_presim = ldd_simulate_flows(challenge, state, &dp_presim, &batt_nodes, &ptdf_sparse, None);
                for t in 0..num_t {
                    for l in 0..num_l {
                        let limit = challenge.network.flow_limits[l];
                        if limit <= 1e-6 { continue; }
                        let f = flows_presim[t][l];
                        if f > limit {
                            mu[t][l] = hp.ldd_warmstart_scale * hp.ldd_step_size * (f - limit);
                        } else if f < -limit {
                            mu[t][l] = -hp.ldd_warmstart_scale * hp.ldd_step_size * (-f - limit);
                        }
                    }
                }
            }

            let mut dp = build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu, &state.socs);

            
            let d_star: f64 = if hp.ldd_use_polyak_dp {
                (0..challenge.num_batteries).map(|b| {
                    let bat = &challenge.batteries[b];
                    let soc = state.socs[b];
                    let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                    let soc_levels = hp.soc_levels;
                    let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let idx0 = (idx_f.floor() as isize).max(0) as usize;
                    let idx0c = idx0.min(soc_levels - 1);
                    let idx1c = (idx0 + 1).min(soc_levels - 1);
                    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                    dp.cell(b, 0, idx0c) * (1.0 - frac) + dp.cell(b, 0, idx1c) * frac
                }).sum()
            } else { 0.0 };

            
            
            let mut active_lines: Vec<bool> = vec![true; num_l];
            let mut b_to_lines_filtered: Option<Vec<Vec<(usize, f64)>>> = None;

            
            
            
            let mut mu_best = mu.clone();
            let mut best_max_viol = f64::MAX;

            
            
            let mut dp_best_data: Option<Vec<f64>> = None;

            
            let mut cluster_map: Vec<usize> = (0..num_t).collect();

            
            let mut teff_cluster_map: Option<Vec<usize>> = None;
            let mut teff_rep_times: Vec<usize> = Vec::new();

            
            let mut mu_sum: Vec<Vec<f64>> = if hp.ldd_use_dual_avg {
                vec![vec![0.0_f64; num_l]; num_t]
            } else { Vec::new() };
            let mut da_count: usize = 0;

            
            let mut bplm_f_ub = f64::NEG_INFINITY;

            
            let mu_ref: Vec<Vec<f64>> = if hp.ldd_use_dowg { mu.clone() } else { Vec::new() };
            let mut dowg_e_acc = 1e-12_f64;

            for ldd_iter in 0..hp.ldd_iters {
                let flows_all = ldd_simulate_flows(challenge, state, &dp, &batt_nodes, &ptdf_sparse,
                    if ldd_iter > 0 { teff_cluster_map.as_deref() } else { None });

                
                let alpha = if hp.ldd_use_bplm {
                    
                    
                    let mut viol_sq = 0.0_f64;
                    for t in 0..num_t {
                        for l in 0..num_l {
                            if !active_lines[l] { continue; }
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let v = if f > limit { f - limit } else if f < -limit { -f - limit } else { 0.0 };
                            viol_sq += v * v;
                        }
                    }
                    let f_k: f64 = (0..challenge.num_batteries).map(|b| {
                        let bat = &challenge.batteries[b];
                        let soc = state.socs[b];
                        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                        let soc_levels = hp.soc_levels;
                        let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                        let idx0 = (idx_f.floor() as isize).max(0) as usize;
                        let idx0c = idx0.min(soc_levels - 1);
                        let idx1c = (idx0 + 1).min(soc_levels - 1);
                        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                        dp.cell(b, 0, idx0c) * (1.0 - frac) + dp.cell(b, 0, idx1c) * frac
                    }).sum();
                    bplm_f_ub = bplm_f_ub.max(f_k);
                    let step = hp.ldd_step_size;
                    let f_lb = f_k - step * viol_sq;
                    let f_lev = hp.ldd_bplm_alpha * bplm_f_ub + (1.0 - hp.ldd_bplm_alpha) * f_lb;
                    let f_pred = f_k + step * viol_sq;
                    if f_pred >= f_lev {
                        step
                    } else {
                        ((f_lev - f_k) / viol_sq.max(1e-12)).clamp(0.01, step)
                    }
                } else if hp.ldd_use_polyak_dp {
                    
                    
                    let mut viol_sq = 0.0_f64;
                    for t in 0..num_t {
                        for l in 0..num_l {
                            if !active_lines[l] { continue; }
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let v = if f > limit { f - limit } else if f < -limit { -f - limit } else { 0.0 };
                            viol_sq += v * v;
                        }
                    }
                    let d_k: f64 = (0..challenge.num_batteries).map(|b| {
                        let bat = &challenge.batteries[b];
                        let soc = state.socs[b];
                        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                        let soc_levels = hp.soc_levels;
                        let idx_f = (soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                        let idx0 = (idx_f.floor() as isize).max(0) as usize;
                        let idx0c = idx0.min(soc_levels - 1);
                        let idx1c = (idx0 + 1).min(soc_levels - 1);
                        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                        dp.cell(b, 0, idx0c) * (1.0 - frac) + dp.cell(b, 0, idx1c) * frac
                    }).sum();
                    let gap = (d_star - d_k).max(0.0);
                    (hp.ldd_polyak_safety * gap / viol_sq.max(1e-12)).clamp(0.01, 2.0)
                } else if hp.ldd_use_polyak {
                    
                    let mut viol_sq = 0.0_f64;
                    for t in 0..num_t {
                        for l in 0..num_l {
                            if !active_lines[l] { continue; }
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let v = if f > limit { f - limit } else if f < -limit { -f - limit } else { 0.0 };
                            viol_sq += v * v;
                        }
                    }
                    (hp.ldd_polyak_safety / viol_sq.max(1e-12)).clamp(0.01, 2.0)
                } else if hp.ldd_use_dowg {
                    
                    
                    let dist_sq: f64 = mu.iter().zip(mu_ref.iter())
                        .map(|(row, ref_row)| {
                            row.iter().zip(ref_row.iter())
                               .map(|(a, b)| (a - b) * (a - b))
                               .sum::<f64>()
                        }).sum();
                    let dist_k = dist_sq.sqrt();
                    let mut viol_sq = 0.0_f64;
                    for t in 0..num_t {
                        for l in 0..num_l {
                            if !active_lines[l] { continue; }
                            let limit = challenge.network.flow_limits[l];
                            if limit <= 1e-6 { continue; }
                            let f = flows_all[t][l];
                            let v = if f > limit { f - limit } else if f < -limit { -f - limit } else { 0.0 };
                            viol_sq += v * v;
                        }
                    }
                    let a = (dist_k / dowg_e_acc.sqrt()).clamp(0.01, 2.0);
                    dowg_e_acc += a * a * viol_sq.max(1e-12);
                    a
                } else {
                    hp.ldd_step_size / ((ldd_iter + 1) as f64)
                };
                let mut max_viol = 0.0_f64;
                for t in 0..num_t {
                    for l in 0..num_l {
                        if !active_lines[l] { continue; }
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

                
                
                if hp.ldd_bundle_radius > 0.0 && ldd_iter > 0 {
                    for t in 0..num_t {
                        for l in 0..num_l {
                            if !active_lines[l] { continue; }
                            let delta = mu[t][l] - mu_best[t][l];
                            mu[t][l] = mu_best[t][l] + delta.clamp(-hp.ldd_bundle_radius, hp.ldd_bundle_radius);
                        }
                    }
                }

                
                let is_new_best_viol = max_viol < best_max_viol;
                if is_new_best_viol {
                    best_max_viol = max_viol;
                    mu_best = mu.clone();
                }

                
                if hp.ldd_use_dual_avg {
                    let d = hp.ldd_da_decay;
                    if d > 0.0 {
                        for t in 0..num_t {
                            for l in 0..num_l {
                                if !active_lines[l] && ldd_iter > 0 { continue; }
                                mu_sum[t][l] = (1.0 - d) * mu_sum[t][l] + d * mu[t][l];
                            }
                        }
                    } else {
                        for t in 0..num_t {
                            for l in 0..num_l {
                                mu_sum[t][l] += mu[t][l];
                            }
                        }
                    }
                    da_count += 1;
                }

                if ldd_iter == 0 {
                    for l in 0..num_l {
                        active_lines[l] = (0..num_t).any(|t| mu[t][l].abs() > 1e-12);
                    }
                    b_to_lines_filtered = Some(b_to_lines.iter()
                        .map(|entries| entries.iter().filter(|&&(l, _)| active_lines[l]).cloned().collect())
                        .collect());
                    
                    if hp.ldd_temporal_eps > 0.0 {
                        cluster_map = sliding_window_cluster(&mu, hp.ldd_temporal_eps);
                    }
                    
                    if hp.use_mu_teff && hp.mu_teff_eps > 0.0 {
                        let cm = sliding_window_cluster(&mu, hp.mu_teff_eps);
                        let t_eff = *cm.iter().max().unwrap_or(&0) + 1;
                        
                        if t_eff < num_t * 95 / 100 {
                            let mut reps = vec![num_t; t_eff];
                            for t in 0..num_t {
                                let c = cm[t];
                                if reps[c] == num_t { reps[c] = t; }
                            }
                            teff_rep_times = reps;
                            teff_cluster_map = Some(cm);
                        }
                    }
                }

                let btl_eff = b_to_lines_filtered.as_deref().unwrap_or(&b_to_lines);
                
                
                dp = if hp.use_mu_teff && teff_cluster_map.is_some() && ldd_iter > 0 {
                    let cm = teff_cluster_map.as_ref().unwrap();
                    let t_eff = teff_rep_times.len();
                    let mut mu_teff = vec![vec![0.0_f64; num_l]; t_eff];
                    let mut cnts = vec![0usize; t_eff];
                    for t in 0..num_t { let c = cm[t]; for l in 0..num_l { mu_teff[c][l] += mu[t][l]; } cnts[c] += 1; }
                    for c in 0..t_eff { let n = cnts[c] as f64; for l in 0..num_l { mu_teff[c][l] /= n; } }
                    let exp_prem_teff: Vec<Vec<f64>> = teff_rep_times.iter()
                        .map(|&rt| expected_premiums[rt].clone()).collect();
                    build_dp_with_mu_teff(challenge, hp, &batt_nodes, &exp_prem_teff, btl_eff, &mu_teff, &state.socs, &teff_rep_times)
                } else if hp.ldd_temporal_eps > 0.0 && ldd_iter > 0 {
                    let nc = *cluster_map.iter().max().unwrap_or(&0) + 1;
                    let mut sums = vec![vec![0.0_f64; num_l]; nc];
                    let mut cnts = vec![0usize; nc];
                    for t in 0..num_t { let c = cluster_map[t]; for l in 0..num_l { sums[c][l] += mu[t][l]; } cnts[c] += 1; }
                    let mu_s: Vec<Vec<f64>> = (0..num_t).map(|t| { let c = cluster_map[t]; let n = cnts[c] as f64; (0..num_l).map(|l| sums[c][l] / n).collect() }).collect();
                    build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, btl_eff, &mu_s, &state.socs)
                } else {
                    build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, btl_eff, &mu, &state.socs)
                };

                
                
                if hp.use_best_iter_dp && is_new_best_viol {
                    dp_best_data = Some(dp.data.clone());
                }

                if max_viol < 1e-4 { break; }
            }
            
            if hp.use_best_iter_dp {
                if let Some(best_data) = dp_best_data {
                    dp.data = best_data;
                }
            }
            
            
            
            
            if teff_cluster_map.is_some() {
                let btl_final = b_to_lines_filtered.as_deref().unwrap_or(&b_to_lines);
                dp = if hp.ldd_temporal_eps > 0.0 {
                    let nc = *cluster_map.iter().max().unwrap_or(&0) + 1;
                    let mut sums = vec![vec![0.0_f64; num_l]; nc];
                    let mut cnts = vec![0usize; nc];
                    for t in 0..num_t { let c = cluster_map[t]; for l in 0..num_l { sums[c][l] += mu[t][l]; } cnts[c] += 1; }
                    let mu_s: Vec<Vec<f64>> = (0..num_t).map(|t| { let c = cluster_map[t]; let n = cnts[c] as f64; (0..num_l).map(|l| sums[c][l] / n).collect() }).collect();
                    build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, btl_final, &mu_s, &state.socs)
                } else {
                    build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, btl_final, &mu, &state.socs)
                };
            }
            
            
            
            if hp.ldd_use_dual_avg && da_count > 0 {
                let d = hp.ldd_da_decay;
                if d > 0.0 {
                    
                    for t in 0..num_t {
                        for l in 0..num_l { mu[t][l] = mu_sum[t][l]; }
                    }
                } else {
                    let inv = 1.0 / da_count as f64;
                    for t in 0..num_t {
                        for l in 0..num_l { mu[t][l] = mu_sum[t][l] * inv; }
                    }
                }
                let btl_da = b_to_lines_filtered.as_deref().unwrap_or(&b_to_lines);
                dp = if hp.ldd_temporal_eps > 0.0 {
                    let nc = *cluster_map.iter().max().unwrap_or(&0) + 1;
                    let mut sums2 = vec![vec![0.0_f64; num_l]; nc];
                    let mut cnts2 = vec![0usize; nc];
                    for t in 0..num_t { let c = cluster_map[t]; for l in 0..num_l { sums2[c][l] += mu[t][l]; } cnts2[c] += 1; }
                    let mu_s2: Vec<Vec<f64>> = (0..num_t).map(|t| { let c = cluster_map[t]; let n = cnts2[c] as f64; (0..num_l).map(|l| sums2[c][l] / n).collect() }).collect();
                    build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, btl_da, &mu_s2, &state.socs)
                } else {
                    build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, btl_da, &mu, &state.socs)
                };
            }
            (dp, mu)
        } else if hp.max_admm_iters > 0 && hp.anticipate_lmp && num_l > 0 {
            
            
            
            let mut hp_cheap = hp.clone();
            hp_cheap.soc_levels = 31;
            hp_cheap.action_grid = 15;
            let mu_zero = vec![vec![0.0_f64; num_l]; num_t];
            let dp_low = build_dp_with_mu(challenge, &hp_cheap, &batt_nodes, &expected_premiums, &b_to_lines, &mu_zero, &state.socs);
            let flows_all = ldd_simulate_flows(challenge, state, &dp_low, &batt_nodes, &ptdf_sparse, None);
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
            let dp = build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu, &state.socs);
            (dp, mu)
        } else {
            let mu = vec![vec![0.0_f64; num_l]; num_t];
            let dp = build_dp_with_mu(challenge, hp, &batt_nodes, &expected_premiums, &b_to_lines, &mu, &state.socs);
            (dp, mu)
        };

        
        
        
        
        let dp_slope = if hp.use_slope_cache { Some(build_dp_slope(&dp)) } else { None };
        
        
        
        
        
        
        
        
        
        
        
        
        let bat_hoist = if hp.use_bat_hoist {
            let dt = 0.25_f64;
            let levels_m1 = (dp.soc_levels - 1) as f64;
            let mut v: Vec<BatHoist> = Vec::with_capacity(num_b);
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let cap = bat.capacity_mwh.max(1e-9);
                let dt_over_cap = dt / cap;
                let deg_coeff = dt_over_cap * dt_over_cap;
                let inv_2deg = 1.0 / (2.0 * deg_coeff.max(1e-30));
                let soc_min = bat.soc_min_mwh;
                let soc_max = bat.soc_max_mwh;
                let soc_span = (soc_max - soc_min).max(1e-9);
                let eff_c = bat.efficiency_charge;
                let eff_d = bat.efficiency_discharge.max(1e-9);
                let beta_charge = eff_c * dt * levels_m1 / soc_span;
                let beta_discharge = dt * levels_m1 / (eff_d * soc_span);
                v.push(BatHoist {
                    deg_coeff,
                    inv_2deg,
                    soc_min,
                    soc_max,
                    soc_span,
                    eff_c,
                    eff_d,
                    beta_charge,
                    beta_discharge,
                });
            }
            Some(v)
        } else {
            None
        };
        
        let soc_envelope = if hp.use_soc_pruning {
            let dt = 0.25_f64;
            let mut fwd_max = vec![vec![0.0_f64; num_t + 1]; num_b];
            let mut bwd_min = vec![vec![0.0_f64; num_t + 1]; num_b];
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                
                fwd_max[b][0] = state.socs[b];
                for t in 1..=num_t {
                    fwd_max[b][t] = (fwd_max[b][t - 1] + bat.power_charge_mw * dt * bat.efficiency_charge)
                        .min(bat.soc_max_mwh);
                }
                
                bwd_min[b][num_t] = bat.soc_min_mwh;
                for t in (0..num_t).rev() {
                    bwd_min[b][t] = (bwd_min[b][t + 1] - bat.power_discharge_mw * dt / bat.efficiency_discharge.max(1e-9))
                        .max(bat.soc_min_mwh);
                }
            }
            Some(SocEnvelope { fwd_max, bwd_min })
        } else {
            None
        };
        
        
        let rcb_p_agg: Option<Vec<f64>> = if (hp.use_rcb_fleet || hp.use_rcb_dp_fleet) && num_b > 0 {
            let bat0 = &challenge.batteries[0];
            let eta_c = bat0.efficiency_charge;
            let eta_d = bat0.efficiency_discharge.max(1e-9);
            let p_c_max = challenge.batteries.iter().map(|b| b.power_charge_mw).sum::<f64>();
            let p_d_max = challenge.batteries.iter().map(|b| b.power_discharge_mw).sum::<f64>();
            let e_min = challenge.batteries.iter().map(|b| b.soc_min_mwh).sum::<f64>();
            let e_max = challenge.batteries.iter().map(|b| b.soc_max_mwh).sum::<f64>();
            let e0: f64 = state.socs.iter().sum();

            let price_fleet: Vec<f64> = (0..num_t).map(|t| {
                let sum: f64 = (0..num_b).map(|b| {
                    let node = batt_nodes[b];
                    let da = if node < challenge.market.day_ahead_prices[t].len() {
                        challenge.market.day_ahead_prices[t][node]
                    } else { challenge.market.day_ahead_prices[t].get(0).copied().unwrap_or(0.0) };
                    let ep = expected_premiums[t][b];
                    let mu_adj: f64 = b_to_lines[b].iter().map(|&(l, imp)| final_mu[t][l] * imp).sum();
                    da + ep - mu_adj
                }).sum::<f64>();
                sum / num_b as f64
            }).collect();

            Some(lp_solve_aggregate_fleet(
                num_t, e0, e_min, e_max, p_c_max, p_d_max, eta_c, eta_d, &price_fleet,
            ))
        } else {
            None
        };

        TitanCache { dp, dp_slope, bat_hoist, soc_envelope, ptdf_sparse, b_to_lines, batt_nodes, rcb_p_agg }
    }

    
    
    
    
    
    fn build_dp_slope(dp: &DpCube) -> DpCube {
        let num_b = dp.num_b;
        let num_t_plus_1 = dp.num_t_plus_1;
        let soc_levels = dp.soc_levels;
        let n_seg = soc_levels.saturating_sub(1).max(1);
        let mut slope = DpCube::new(num_b, num_t_plus_1, n_seg);
        if soc_levels < 2 { return slope; }
        for b in 0..num_b {
            for t in 0..num_t_plus_1 {
                for i in 0..(soc_levels - 1) {
                    let v = dp.cell(b, t, i + 1) - dp.cell(b, t, i);
                    slope.set(b, t, i, v);
                }
            }
        }
        slope
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
        let soc_levels = ca.dp.soc_levels;
        let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
        let idx0 = (idx_f.floor() as isize).max(0) as usize;
        let idx0c = idx0.min(soc_levels - 1);
        let idx1c = (idx0 + 1).min(soc_levels - 1);
        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
        let t_next = (state.time_step + 1).min(ca.dp.num_t_plus_1 - 1);
        profit + ca.dp.cell(b, t_next, idx0c) * (1.0 - frac) + ca.dp.cell(b, t_next, idx1c) * frac
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #[inline]
    fn analytic_argmax_piecewise_quadratic(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        b: usize,
        lo: f64,
        hi: f64,
    ) -> (f64, f64) {
        if !(lo < hi) {
            return (lo, eval_profit(challenge, state, ca, b, lo));
        }
        let bat = &challenge.batteries[b];
        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        let dt = 0.25_f64;
        
        
        
        let hoist = ca.bat_hoist.as_ref().map(|v| &v[b]);
        let (deg_coeff, inv_2deg, soc_min, soc_max, soc_span, eff_c, eff_d) = if let Some(h) = hoist {
            (h.deg_coeff, h.inv_2deg, h.soc_min, h.soc_max, h.soc_span, h.eff_c, h.eff_d)
        } else {
            let cap = bat.capacity_mwh.max(1e-9);
            let dt_over_cap = dt / cap;
            let deg_coeff = dt_over_cap * dt_over_cap; 
            let inv_2deg = 1.0 / (2.0 * deg_coeff.max(1e-30));
            let soc_min = bat.soc_min_mwh;
            let soc_max = bat.soc_max_mwh;
            let soc_span = (soc_max - soc_min).max(1e-9);
            let eff_c = bat.efficiency_charge;
            let eff_d = bat.efficiency_discharge.max(1e-9);
            (deg_coeff, inv_2deg, soc_min, soc_max, soc_span, eff_c, eff_d)
        };
        let soc = state.socs[b];
        let soc_levels = ca.dp.soc_levels;
        let max_idx = soc_levels - 1;
        let levels_m1 = max_idx as f64;
        let t_next = (state.time_step + 1).min(ca.dp.num_t_plus_1 - 1);
        let alpha = (soc - soc_min) / soc_span * levels_m1;

        let charge = hi <= 0.0;
        let lin_coef = if charge {
            (rt_price + 0.25) * dt
        } else {
            (rt_price - 0.25) * dt
        };

        let mut best_u_an = lo;
        let mut best_val_an = f64::NEG_INFINITY;

        if charge {
            
            let beta = if let Some(h) = hoist {
                h.beta_charge
            } else {
                eff_c * dt * levels_m1 / soc_span
            };
            let u_sat = if eff_c * dt > 1e-30 {
                -(soc_max - soc) / (eff_c * dt)
            } else {
                f64::NEG_INFINITY
            };
            
            let sat_hi = hi.min(u_sat);
            if lo < sat_hi {
                let b_sat = lin_coef;
                let c_sat = ca.dp.cell(b, t_next, max_idx);
                let u_arg = (b_sat * inv_2deg).clamp(lo, sat_hi);
                let val = -deg_coeff * u_arg * u_arg + b_sat * u_arg + c_sat;
                if val > best_val_an { best_val_an = val; best_u_an = u_arg; }
            }
            
            let free_lo = lo.max(u_sat);
            let free_hi = hi;
            if free_lo < free_hi && beta > 1e-30 {
                let idx_at_lo = (alpha - beta * free_lo).clamp(0.0, levels_m1);
                let idx_at_hi = (alpha - beta * free_hi).clamp(0.0, levels_m1);
                let k_hi_walk = (idx_at_lo.floor() as usize).min(max_idx);
                let k_lo_walk = (idx_at_hi.floor() as usize).min(max_idx);
                let inv_beta = 1.0 / beta;
                for k0 in k_lo_walk..=k_hi_walk {
                    let k0c = k0.min(max_idx);
                    let kp1c = (k0 + 1).min(max_idx);
                    let dp_k0 = ca.dp.cell(b, t_next, k0c);
                    
                    
                    
                    
                    let delta_dp = if let Some(slope) = ca.dp_slope.as_ref() {
                        if k0c < max_idx { slope.cell(b, t_next, k0c) } else { 0.0 }
                    } else {
                        let dp_kp1 = ca.dp.cell(b, t_next, kp1c);
                        dp_kp1 - dp_k0
                    };
                    let b_seg = lin_coef - delta_dp * beta;
                    let c_seg = dp_k0 + delta_dp * (alpha - k0 as f64);
                    let u_for_k0 = (alpha - k0 as f64) * inv_beta;
                    let u_for_kp1 = (alpha - (k0 + 1) as f64) * inv_beta;
                    let seg_lo = u_for_kp1.max(free_lo);
                    let seg_hi = u_for_k0.min(free_hi);
                    if !(seg_lo < seg_hi) { continue; }
                    let u_arg = (b_seg * inv_2deg).clamp(seg_lo, seg_hi);
                    let val = -deg_coeff * u_arg * u_arg + b_seg * u_arg + c_seg;
                    if val > best_val_an { best_val_an = val; best_u_an = u_arg; }
                }
            }
        } else {
            
            let beta = if let Some(h) = hoist {
                h.beta_discharge
            } else {
                dt * levels_m1 / (eff_d * soc_span)
            };
            let u_sat = (soc - soc_min) * eff_d / dt;
            
            let sat_lo = lo.max(u_sat);
            if sat_lo < hi {
                let b_sat = lin_coef;
                let c_sat = ca.dp.cell(b, t_next, 0);
                let u_arg = (b_sat * inv_2deg).clamp(sat_lo, hi);
                let val = -deg_coeff * u_arg * u_arg + b_sat * u_arg + c_sat;
                if val > best_val_an { best_val_an = val; best_u_an = u_arg; }
            }
            
            let free_lo = lo;
            let free_hi = hi.min(u_sat);
            if free_lo < free_hi && beta > 1e-30 {
                let idx_at_lo = (alpha - beta * free_lo).clamp(0.0, levels_m1);
                let idx_at_hi = (alpha - beta * free_hi).clamp(0.0, levels_m1);
                let k_hi_walk = (idx_at_lo.floor() as usize).min(max_idx);
                let k_lo_walk = (idx_at_hi.floor() as usize).min(max_idx);
                let inv_beta = 1.0 / beta;
                for k0 in k_lo_walk..=k_hi_walk {
                    let k0c = k0.min(max_idx);
                    let kp1c = (k0 + 1).min(max_idx);
                    let dp_k0 = ca.dp.cell(b, t_next, k0c);
                    
                    
                    
                    
                    let delta_dp = if let Some(slope) = ca.dp_slope.as_ref() {
                        if k0c < max_idx { slope.cell(b, t_next, k0c) } else { 0.0 }
                    } else {
                        let dp_kp1 = ca.dp.cell(b, t_next, kp1c);
                        dp_kp1 - dp_k0
                    };
                    let b_seg = lin_coef - delta_dp * beta;
                    let c_seg = dp_k0 + delta_dp * (alpha - k0 as f64);
                    let u_for_k0 = (alpha - k0 as f64) * inv_beta;
                    let u_for_kp1 = (alpha - (k0 + 1) as f64) * inv_beta;
                    let seg_lo = u_for_kp1.max(free_lo);
                    let seg_hi = u_for_k0.min(free_hi);
                    if !(seg_lo < seg_hi) { continue; }
                    let u_arg = (b_seg * inv_2deg).clamp(seg_lo, seg_hi);
                    let val = -deg_coeff * u_arg * u_arg + b_seg * u_arg + c_seg;
                    if val > best_val_an { best_val_an = val; best_u_an = u_arg; }
                }
            }
        }

        
        
        let v_best = eval_profit(challenge, state, ca, b, best_u_an);
        let v_lo = eval_profit(challenge, state, ca, b, lo);
        let v_hi = eval_profit(challenge, state, ca, b, hi);
        let mut best_u = best_u_an;
        let mut best_v = v_best;
        if v_lo > best_v { best_v = v_lo; best_u = lo; }
        if v_hi > best_v { best_v = v_hi; best_u = hi; }
        (best_u, best_v)
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
                        let (u, v) = if hp.use_threshold_decode {
                            threshold_decode_action(challenge, state, ca, b, lo, hi, hp, 0.0)
                        } else if hp.use_analytic_argmax {
                            analytic_argmax_piecewise_quadratic(challenge, state, ca, b, lo, hi)
                        } else {
                            ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters)
                        };
                        if v > best_v { best_v = v; best_u = u; }
                    }
                }

                if u_max > 0.0 {
                    let lo = 0.0_f64.max(u_min); let hi = u_max;
                    if lo < hi {
                        let (u, v) = if hp.use_threshold_decode {
                            threshold_decode_action(challenge, state, ca, b, lo, hi, hp, 0.0)
                        } else if hp.use_analytic_argmax {
                            analytic_argmax_piecewise_quadratic(challenge, state, ca, b, lo, hi)
                        } else {
                            ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters)
                        };
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

    
    
    
    
    fn threshold_decode_action(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        b: usize,
        lo: f64,
        hi: f64,
        _hp: &TrackHp,
        extra_lin_cost: f64,
    ) -> (f64, f64) {
        let bat = &challenge.batteries[b];
        let dt = 0.25_f64;
        let soc = state.socs[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
        let soc_levels = ca.dp.soc_levels;
        let t_next = (state.time_step + 1).min(ca.dp.num_t_plus_1 - 1);
        let denom = ((soc_levels as f64) - 1.0).max(1.0);
        let e_step = soc_span / denom;
        let idx_f = (soc - bat.soc_min_mwh) / e_step;
        let idx = (idx_f.round() as isize).clamp(0, (soc_levels as isize) - 1) as usize;
        let dq_de = if soc_levels >= 2 {
            let l_i = idx.saturating_sub(1);
            let h_i = (idx + 1).min(soc_levels - 1);
            if h_i > l_i {
                (ca.dp.cell(b, t_next, h_i) - ca.dp.cell(b, t_next, l_i)) / (((h_i - l_i) as f64) * e_step)
            } else { 0.0 }
        } else { 0.0 };

        let node = ca.batt_nodes[b];
        let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
        let eta_dis = bat.efficiency_discharge.max(1e-9);
        let eta_chg = bat.efficiency_charge;
        
        let p_eff = rt_price - extra_lin_cost / dt;

        
        
        let mut u = 0.0_f64.clamp(lo, hi);
        if hi > 1e-12 && p_eff > dq_de / eta_dis + 0.25 {
            u = hi;
        } else if lo < -1e-12 && p_eff < dq_de * eta_chg - 0.25 {
            u = lo;
        }
        let v = eval_profit(challenge, state, ca, b, u) - extra_lin_cost * u;
        (u, v)
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
                        let (u, v) = if hp.use_threshold_decode {
                            threshold_decode_action(challenge, state, ca, b, lo, hi, hp, nu_dot_ptdf)
                        } else {
                            ternary_search(
                                |u| eval_profit(challenge, state, ca, b, u) - nu_dot_ptdf * u,
                                lo, hi, hp.ternary_iters,
                            )
                        };
                        if v > best_v { best_v = v; best_u = u; }
                    }
                }
                if u_max > 0.0 {
                    let lo = 0.0_f64.max(u_min); let hi = u_max;
                    if lo < hi {
                        let (u, v) = if hp.use_threshold_decode {
                            threshold_decode_action(challenge, state, ca, b, lo, hi, hp, nu_dot_ptdf)
                        } else {
                            ternary_search(
                                |u| eval_profit(challenge, state, ca, b, u) - nu_dot_ptdf * u,
                                lo, hi, hp.ternary_iters,
                            )
                        };
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

        
        
        
        
        let phi_a = hp.phi_int_alpha;
        if phi_a > 0.0 {
            let mut beta_b = vec![1.0_f64; num_b];
            for l in 0..num_l {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                let total = flows_base[l] + f_act[l];
                if total.abs() <= limit { continue; }
                let sign = total.signum();
                let target = sign * limit;
                for &(b, impact) in &ca.ptdf_sparse[l] {
                    let contrib = impact * actions[b];
                    if contrib * sign <= 1e-9 { continue; }
                    let f_other = total - contrib;
                    let local_beta = if contrib.abs() > 1e-9 {
                        ((target - f_other) / contrib).clamp(0.0, 1.0)
                    } else { 1.0 };
                    if local_beta < beta_b[b] { beta_b[b] = local_beta; }
                }
            }
            for b in 0..num_b { actions[b] *= beta_b[b]; }
        } else {
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
        }

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
        let soc_levels = ca.dp.soc_levels;
        let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
        let idx0 = (idx_f.floor() as isize).max(0) as usize;
        let idx0c = idx0.min(soc_levels - 1);
        let idx1c = (idx0 + 1).min(soc_levels - 1);
        let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
        let t_next = (state.time_step + 1).min(ca.dp.num_t_plus_1 - 1);
        profit + ca.dp.cell(b, t_next, idx0c) * (1.0 - frac) + ca.dp.cell(b, t_next, idx1c) * frac
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

    
    
    
    
    
    
    fn run_ri_dp(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut Vec<f64>,
    ) {
        let num_l = challenge.network.flow_limits.len();
        let num_b = challenge.num_batteries;

        
        let max_util = (0..num_l).map(|l| {
            let limit = challenge.network.flow_limits[l];
            if limit <= 1e-6 { return 0.0; }
            let bat_f: f64 = ca.ptdf_sparse[l].iter().map(|&(b, imp)| imp * actions[b]).sum();
            (flows_base[l] + bat_f).abs() / limit
        }).fold(0.0_f64, f64::max);

        if max_util < hp.ri_cong_threshold { return; }

        
        for b in 0..num_b {
            let (u_lo, u_hi) = state.action_bounds[b];
            let mut best_u = 0.0_f64;
            let mut best_v = eval_profit(challenge, state, ca, b, 0.0);

            if u_lo < 0.0 {
                let hi = 0.0_f64.min(u_hi);
                if u_lo < hi {
                    let (u, v) = analytic_argmax_piecewise_quadratic(challenge, state, ca, b, u_lo, hi);
                    if v > best_v { best_v = v; best_u = u; }
                }
            }
            if u_hi > 0.0 {
                let lo = 0.0_f64.max(u_lo);
                if lo < u_hi {
                    let (u, v) = analytic_argmax_piecewise_quadratic(challenge, state, ca, b, lo, u_hi);
                    if v > best_v { best_u = u; }
                }
            }

            actions[b] = best_u;
        }
    }

    
    
    
    
    
    fn run_mfg_warm(
        challenge: &Challenge,
        state: &State,
        cache: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
        actions: &mut Vec<f64>,
    ) {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        if num_l == 0 || num_b == 0 { return; }

        let avg_price: f64 = {
            let s: f64 = (0..num_b).map(|b| {
                let n = cache.batt_nodes[b];
                if n < state.rt_prices.len() { state.rt_prices[n].abs() } else { 0.0 }
            }).sum();
            (s / num_b as f64).max(10.0)
        };
        let price_scale = avg_price * hp.kkt_price_scale.max(0.5);

        for _outer in 0..hp.mfg_iters {
            
            let mut total_f = flows_base.to_vec();
            for l in 0..num_l {
                for &(b, ptdf_val) in &cache.ptdf_sparse[l] {
                    total_f[l] += ptdf_val * actions[b];
                }
            }

            
            
            const MFG_THRESH: f64 = 0.85;
            let mut mu = vec![0.0_f64; num_l];
            for l in 0..num_l {
                let lim = challenge.network.flow_limits[l];
                if lim < 1e-6 { continue; }
                let util = total_f[l].abs() / lim;
                if util > MFG_THRESH {
                    let excess = ((util - MFG_THRESH) / (1.0 - MFG_THRESH).max(1e-6)).min(1.0);
                    mu[l] = excess * price_scale * total_f[l].signum();
                }
            }

            
            for b in 0..num_b {
                let node = cache.batt_nodes[b];
                let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
                let cong_adj: f64 = cache.b_to_lines[b].iter().map(|&(l, imp)| mu[l] * imp).sum();
                let p_aug = rt - cong_adj;
                actions[b] = kkt_best_action(challenge, state, cache, b, Some(p_aug), hp.ternary_iters);
            }

            
            run_deflator(challenge, state, cache, hp, flows_base, actions);
        }
    }

    
    
    
    
    
    
    
    
    
    fn lp_solve_aggregate_fleet(
        num_t: usize,
        e0: f64,
        e_min: f64,
        e_max: f64,
        p_c_max: f64,
        p_d_max: f64,
        eta_c: f64,
        eta_d: f64,
        price: &[f64],
    ) -> Vec<f64> {
        let dt = 0.25_f64;
        let n = 2 * num_t;  
        
        let m = 2 * num_t + 2 * num_t;

        let mut c = vec![0.0_f64; n];
        let mut a: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; m];
        let mut b_rhs = vec![0.0_f64; m];

        for t in 0..num_t {
            c[t] = price[t] * dt;            
            c[num_t + t] = -price[t] * dt;   
        }

        
        let mut row = 0usize;
        for t in 0..num_t { a[row][t] = 1.0; b_rhs[row] = p_d_max; row += 1; }
        for t in 0..num_t { a[row][num_t + t] = 1.0; b_rhs[row] = p_c_max; row += 1; }

        
        
        
        
        for tau in 0..num_t {
            for t in 0..=tau {
                a[row][t] = -dt / eta_d;           
                a[row][num_t + t] = eta_c * dt;    
            }
            b_rhs[row] = (e_max - e0).max(0.0);
            row += 1;
            for t in 0..=tau {
                a[row][t] = dt / eta_d;            
                a[row][num_t + t] = -eta_c * dt;   
            }
            b_rhs[row] = (e0 - e_min).max(0.0);
            row += 1;
        }

        match super::lp::lp_solve(n, m, &c, &a, &b_rhs) {
            Some(x) => (0..num_t).map(|t| x[t] - x[num_t + t]).collect(),
            None => vec![0.0_f64; num_t],
        }
    }

    
    
    fn run_rcb_fleet_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        actions: &mut [f64],
    ) {
        let Some(ref rcb_p_agg) = ca.rcb_p_agg else { return; };
        let t = state.time_step;
        if t >= rcb_p_agg.len() { return; }

        let p_t = rcb_p_agg[t];
        let num_b = challenge.num_batteries;
        let socs = &state.socs;

        let mut order: Vec<usize> = (0..num_b).collect();
        if p_t >= 0.0 {
            
            order.sort_by(|&a, &b| socs[b].partial_cmp(&socs[a]).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            
            order.sort_by(|&a, &b| socs[a].partial_cmp(&socs[b]).unwrap_or(std::cmp::Ordering::Equal));
        }

        for a in actions.iter_mut() { *a = 0.0; }
        let mut remaining = p_t;
        for &b in &order {
            if remaining.abs() < 1e-9 { break; }
            let (u_min, u_max) = state.action_bounds[b];
            let p_b = if remaining > 0.0 {
                remaining.min(u_max).max(0.0)
            } else {
                remaining.max(u_min).min(0.0)
            };
            actions[b] = p_b;
            remaining -= p_b;
        }
    }

    
    
    
    
    
    fn run_dsfd(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        actions: &mut [f64],
    ) {
        let Some(ref rcb_p_agg) = ca.rcb_p_agg else { return; };
        let t = state.time_step;
        if t >= rcb_p_agg.len() { return; }

        let p_t = rcb_p_agg[t];
        let num_b = challenge.num_batteries;
        let t_next = (t + 1).min(ca.dp.num_t_plus_1 - 1);
        let soc_levels = ca.dp.soc_levels;

        let weights: Vec<f64> = (0..num_b).map(|b| {
            let bat = &challenge.batteries[b];
            let soc = state.socs[b];
            let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let idx_f = (soc - bat.soc_min_mwh) / soc_span * (soc_levels - 1) as f64;
            let lo = (idx_f.floor() as isize).clamp(0, soc_levels as isize - 2) as usize;
            let hi = lo + 1;
            let dv = (ca.dp.cell(b, t_next, hi) - ca.dp.cell(b, t_next, lo)).max(0.0);
            let (u_min, u_max) = state.action_bounds[b];
            if p_t >= 0.0 {
                dv * u_max.max(0.0)
            } else {
                dv * (-u_min).max(0.0)
            }
        }).collect();

        let w_sum: f64 = weights.iter().sum();

        for b in 0..num_b {
            let (u_min, u_max) = state.action_bounds[b];
            actions[b] = if w_sum > 1e-9 {
                (p_t * weights[b] / w_sum).clamp(u_min, u_max)
            } else {
                (p_t / num_b as f64).clamp(u_min, u_max)
            };
        }
    }

    
    
    
    
    fn cf_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        flows_base: &[f64],
    ) -> Vec<f64> {
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let t = state.time_step;
        let t_next = (t + 1).min(ca.dp.num_t_plus_1 - 1);

        
        let mut dv_arr = vec![0.0_f64; num_b];
        let mut bounds_arr = vec![(0.0_f64, 0.0_f64); num_b];
        let mut rt_arr = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            rt_arr[b] = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let (mut u_min, mut u_max) = state.action_bounds[b];
            let soc = state.socs[b];
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let eta_c = bat.efficiency_charge;

            
            if let Some(ref env) = ca.soc_envelope {
                let tn = t + 1;
                if tn < env.bwd_min[b].len() {
                    let max_d = ((soc - env.bwd_min[b][tn]) * eta_d / dt).max(0.0);
                    if max_d < u_max { u_max = max_d; }
                }
                if tn < env.fwd_max[b].len() {
                    let max_c = ((env.fwd_max[b][tn] - soc) / (dt * eta_c.max(1e-9))).max(0.0);
                    let u_min_new = -max_c;
                    if u_min_new > u_min { u_min = u_min_new; }
                }
                if u_max < u_min { u_max = u_min; }
            }
            bounds_arr[b] = (u_min, u_max);

            
            let soc_levels = ca.dp.soc_levels;
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1) as f64;
            let idx_f = (soc - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo_idx = (idx_f.floor() as isize).max(0) as usize;
            let lo_idx = lo_idx.min(soc_levels - 1);
            let hi_idx = (lo_idx + 1).min(soc_levels - 1);
            dv_arr[b] = (ca.dp.cell(b, t_next, hi_idx) - ca.dp.cell(b, t_next, lo_idx)) / delta_s;
        }

        
        let mut actions = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let (u_min, u_max) = bounds_arr[b];
            let rt = rt_arr[b];
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let eta_c = bat.efficiency_charge;
            let dv = dv_arr[b];
            let gain_d = (rt - 0.25) * dt - dv * dt / eta_d;
            let gain_c = -(rt + 0.25) * dt + dv * eta_c * dt;
            let d_b = if gain_d > 0.0 { u_max.max(0.0) } else { 0.0 };
            let c_b = if gain_c > 0.0 { (-u_min).max(0.0) } else { 0.0 };
            actions[b] = (d_b - c_b).clamp(u_min, u_max);
        }

        
        if hp.use_dual_cf && num_l > 0 {
            
            let mut bat_inj = vec![0.0_f64; challenge.network.num_nodes];
            for b in 0..num_b {
                bat_inj[ca.batt_nodes[b]] += actions[b];
            }
            let bat_flows = challenge.network.compute_flows(&bat_inj);

            
            let mut dual_adj = vec![0.0_f64; num_b];
            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                
                let total_f = flows_base[l] + bat_flows[l];
                let viol = (total_f.abs() - limit).max(0.0);
                if viol < 1e-6 { continue; }
                let sign = total_f.signum();
                for &(b, impact) in &ca.ptdf_sparse[l] {
                    
                    dual_adj[b] += hp.dual_cf_scale * sign * impact * viol;
                }
            }

            
            for b in 0..num_b {
                if dual_adj[b].abs() < 1e-12 { continue; }
                let bat = &challenge.batteries[b];
                let (u_min, u_max) = bounds_arr[b];
                let rt_adj = rt_arr[b] - dual_adj[b] / dt;
                let eta_d = bat.efficiency_discharge.max(1e-9);
                let eta_c = bat.efficiency_charge;
                let dv = dv_arr[b];
                let gain_d = (rt_adj - 0.25) * dt - dv * dt / eta_d;
                let gain_c = -(rt_adj + 0.25) * dt + dv * eta_c * dt;
                let d_b = if gain_d > 0.0 { u_max.max(0.0) } else { 0.0 };
                let c_b = if gain_c > 0.0 { (-u_min).max(0.0) } else { 0.0 };
                actions[b] = (d_b - c_b).clamp(u_min, u_max);
            }
        }

        actions
    }

    fn joint_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
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

        let t_next = (state.time_step + 1).min(ca.dp.num_t_plus_1 - 1);

        
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

            
            let soc_levels = ca.dp.soc_levels;
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1) as f64;
            let idx_f = (soc - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo_idx = (idx_f.floor() as isize).max(0) as usize;
            let lo_idx = lo_idx.min(soc_levels - 1);
            let hi_idx = (lo_idx + 1).min(soc_levels - 1);
            let dv = (ca.dp.cell(b, t_next, hi_idx) - ca.dp.cell(b, t_next, lo_idx)) / delta_s;

            
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

        let t_next = (state.time_step + 1).min(ca.dp.num_t_plus_1 - 1);

        for b in 0..num_b {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
            let (u_min, u_max) = state.action_bounds[b];
            let u_max_pos = u_max.max(0.0);
            let u_max_neg = (-u_min).max(0.0);

            let eta_c = bat.efficiency_charge;
            let eta_d = bat.efficiency_discharge.max(1e-9);
            let soc_levels = ca.dp.soc_levels;
            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let delta_s = span / (soc_levels - 1) as f64;
            let idx_f = (state.socs[b] - bat.soc_min_mwh) / span * (soc_levels - 1) as f64;
            let lo = (idx_f.floor() as isize).max(0) as usize;
            let hi = (lo + 1).min(soc_levels - 1);
            let v_lo = ca.dp.cell(b, t_next, lo);
            let v_hi = ca.dp.cell(b, t_next, hi);
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

    
    
    fn multi_lp_dispatch(
        challenge: &Challenge,
        state: &State,
        ca: &TitanCache,
        hp: &TrackHp,
        horizon: usize,
    ) -> Option<Vec<f64>> {
        let t0 = state.time_step;
        let nb = challenge.num_batteries;
        let nl = challenge.network.num_lines;
        let H = horizon.min(challenge.num_steps - t0);
        if H == 0 { return Some(vec![0.0; nb]); }

        let da = &challenge.market.day_ahead_prices;
        let mut p_sell = vec![vec![0.0f64; nb]; H];
        let mut p_buy = vec![vec![0.0f64; nb]; H];
        for b in 0..nb {
            let node = ca.batt_nodes[b];
            for k in 0..H {
                let da_price = if t0 + k < da.len() && node < da[t0 + k].len() {
                    da[t0 + k][node]
                } else {
                    da[t0].get(0).copied().unwrap_or(0.0)
                };
                p_sell[k][b] = da_price * (1.0 + hp.jump_premium);
                p_buy[k][b] = da_price;
            }
        }

        let n = 2 * nb * H + H * nl + nb;
        let K = 11;

        let mut active_lines = Vec::new();
        for l in 0..nl {
            if challenge.network.flow_limits[l] > 1e-6 { active_lines.push(l); }
        }

        let mut m_count = 0;
        m_count += 2 * nb * H; 
        m_count += 2 * nb * H; 
        m_count += 2 * active_lines.len() * H; 
        m_count += (K - 1) * nb; 
        let m = m_count;

        let mut c = vec![0.0f64; n];
        let mut a = vec![vec![0.0f64; n]; m];
        let mut b_rhs = vec![0.0f64; m];

        let dt = 0.25;
        let kappa = 0.25;

        for b in 0..nb {
            for k in 0..H {
                let idx_d = k * 2 * nb + b;
                let idx_c = k * 2 * nb + nb + b;
                c[idx_d] = (p_sell[k][b] - kappa) * dt;
                c[idx_c] = -(p_buy[k][b] + kappa) * dt;
            }
        }
        let slack_penalty = hp.lp_soft_lambda;
        for k in 0..H {
            for &l in &active_lines {
                c[2 * nb * H + k * nl + l] = -slack_penalty;
            }
        }
        for b in 0..nb {
            c[2 * nb * H + H * nl + b] = 1.0;
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
            for k in 0..H {
                a[row][k * 2 * nb + b] = 1.0;
                b_rhs[row] = max_d[b];
                row += 1;
            }
        }
        for b in 0..nb {
            for k in 0..H {
                a[row][k * 2 * nb + nb + b] = 1.0;
                b_rhs[row] = max_c[b];
                row += 1;
            }
        }

        
        for b in 0..nb {
            for k in 1..=H {
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

        
        
        let mut exo_flows: Vec<Vec<f64>> = Vec::with_capacity(H);
        for k in 0..H {
            exo_flows.push(challenge.network.compute_flows(&challenge.exogenous_injections[t0 + k]));
        }
        for k in 0..H {
            let ef = &exo_flows[k];
            for &l in &active_lines {
                let limit = challenge.network.flow_limits[l];
                let idx_s = 2 * nb * H + k * nl + l;
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

        
        let t_term = t0 + H;
        for b in 0..nb {
            let bat = &challenge.batteries[b];
            let soc_sp = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let soc_levels = ca.dp.soc_levels;
            let step_f = (soc_levels - 1) as f64 / (K - 1) as f64;
            let dp_term_idx = t_term.min(ca.dp.num_t_plus_1 - 1);
            let dp_term = ca.dp.slice_bt(b, dp_term_idx);
            for i in 0..(K - 1) {
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
                let idx_v = 2 * nb * H + H * nl + b;
                a[row][idx_v] = 1.0;
                for k in 0..H {
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
        mut actions: Vec<f64>,
        fuel_budget: u64,
    ) -> Vec<f64> {
        
        let num_b = challenge.num_batteries;
        let num_l = challenge.network.flow_limits.len();
        let dt = 0.25_f64;
        let ptdf_b: Vec<Vec<(usize, f64)>> = (0..num_b)
            .map(|b| {
                let node = challenge.batteries[b].node;
                (0..num_l)
                    .filter_map(|l| {
                        let imp = challenge.network.ptdf[l][node];
                        if imp.abs() > 1e-12 { Some((l, imp)) } else { None }
                    })
                    .collect()
            })
            .collect();
        let mut flows = challenge.network.compute_flows(
            &challenge.compute_total_injections(state, &actions),
        );
        let mut best_actions = actions.clone();
        let mut best_profit = challenge.compute_profit(state, &best_actions);
        let mut mu_pos = vec![0.0_f64; num_l];
        let mut mu_neg = vec![0.0_f64; num_l];
        let mut step = 0.1;
        let pass1_budget = fuel_budget * 2 / 3;
        let max_iter = ((pass1_budget as usize) / (num_b.max(1) * 10)).min(30).max(5);
        for _ in 0..max_iter {
            for b in 0..num_b {
                let (lo, hi) = state.action_bounds[b];
                let node = challenge.batteries[b].node;
                let rt = *state.rt_prices.get(node).unwrap_or(&0.0);
                let pen: f64 = ptdf_b[b].iter().map(|&(l, i)| i * (mu_pos[l] - mu_neg[l])).sum();
                let deg_k = (dt / challenge.batteries[b].capacity_mwh.max(1e-9)).powi(2);
                let mut best_u = actions[b];
                let eval = |u: f64| -> f64 {
                    let a_abs = u.abs();
                    u * rt * dt - 0.25_f64 * a_abs * dt - deg_k * u * u - pen * u
                };
                let mut best_v = eval(best_u);
                if lo < 0.0 {
                    let hi_c = 0.0_f64.min(hi);
                    if lo < hi_c {
                        let b_c = dt * (-rt - 0.25) - deg_k.mul_add(0.0, pen) * 1.0;
                        if deg_k > 1e-12 {
                            let cand = (-b_c / (2.0 * deg_k)).clamp(lo, hi_c);
                            let v = eval(cand);
                            if v > best_v { best_v = v; best_u = cand; }
                        }
                        let v_lo = eval(lo);
                        if v_lo > best_v { best_v = v_lo; best_u = lo; }
                    }
                }
                if hi > 0.0 {
                    let lo_d = 0.0_f64.max(lo);
                    if lo_d < hi {
                        let b_d = dt * (rt - 0.25) - pen;
                        if deg_k > 1e-12 {
                            let cand = (b_d / (2.0 * deg_k)).clamp(lo_d, hi);
                            let v = eval(cand);
                            if v > best_v { best_v = v; best_u = cand; }
                        }
                        let v_hi = eval(hi);
                        if v_hi > best_v { best_v = v_hi; best_u = hi; }
                    }
                }
                let delta = best_u - actions[b];
                if delta.abs() > 1e-8 {
                    for &(l, imp) in &ptdf_b[b] { flows[l] += imp * delta; }
                    actions[b] = best_u;
                }
            }
            let mut max_viol = 0.0_f64;
            for l in 0..num_l {
                let lim = challenge.network.flow_limits[l];
                if lim <= 1e-6 { continue; }
                let f = flows[l];
                let vp = (f - lim).max(0.0);
                let vn = (-f - lim).max(0.0);
                mu_pos[l] = (mu_pos[l] + step * vp).min(500.0);
                mu_neg[l] = (mu_neg[l] + step * vn).min(500.0);
                max_viol = max_viol.max(vp).max(vn);
            }
            step *= 0.85;
            let p = challenge.compute_profit(state, &actions);
            if p > best_profit {
                best_profit = p;
                best_actions = actions.clone();
            }
            if max_viol < 1e-4 { break; }
        }
        
        let ptdf_b2: Vec<Vec<(usize, f64)>> = (0..num_b)
            .map(|b| {
                let node = challenge.batteries[b].node;
                (0..num_l)
                    .filter_map(|l| {
                        let imp = challenge.network.ptdf[l][node];
                        if imp.abs() > 1e-12 { Some((l, imp)) } else { None }
                    })
                    .collect()
            })
            .collect();
        let mut flows2 = challenge.network.compute_flows(
            &challenge.compute_total_injections(state, &actions),
        );
        let pass2_budget = fuel_budget / 3;
        let max_steps = (pass2_budget / 10).min(500) as usize;
        for s in 0..max_steps {
            let b = s % num_b;
            let (lo, hi) = state.action_bounds[b];
            let du = (hi - lo) * 0.05;
            let u_old = actions[b];
            for &delta in [-du, du].iter() {
                let u_try = (u_old + delta).clamp(lo, hi);
                if (u_try - u_old).abs() < 1e-8 { continue; }
                let feasible = !ptdf_b2[b].iter().any(|&(l, imp)| {
                    (flows2[l] - imp * u_old + imp * u_try).abs()
                        > challenge.network.flow_limits[l] * 1.001
                });
                if !feasible { continue; }
                actions[b] = u_try;
                for &(l, imp) in &ptdf_b2[b] {
                    flows2[l] += imp * (u_try - u_old);
                }
                let p = challenge.compute_profit(state, &actions);
                if p > best_profit {
                    best_profit = p;
                    best_actions = actions.clone();
                }
                break;
            }
        }
        
        let ptdf_b5: Vec<Vec<(usize, f64)>> = (0..num_b)
            .map(|b| {
                let node = challenge.batteries[b].node;
                (0..num_l)
                    .filter_map(|l| {
                        let imp = challenge.network.ptdf[l][node];
                        if imp.abs() > 1e-12 { Some((l, imp)) } else { None }
                    })
                    .collect()
            })
            .collect();
        let mut flows5 = challenge.network.compute_flows(
            &challenge.compute_total_injections(state, &best_actions),
        );
        let mut actions5 = best_actions.clone();
        let mut best_profit5 = best_profit;
        let max_sweeps = 5u64.min(fuel_budget / 2000);
        for _ in 0..max_sweeps {
            let mut improved = false;
            for b in 0..num_b {
                let bat = &challenge.batteries[b];
                let node = bat.node;
                let rt = *state.rt_prices.get(node).unwrap_or(&0.0);
                let (lo, hi) = state.action_bounds[b];
                if (hi - lo).abs() < 1e-12 { continue; }

                let eval_immediate = |u: f64| -> f64 {
                    let a_abs = u.abs();
                    u * rt * dt - 0.25_f64 * a_abs * dt
                        - (a_abs * dt / bat.capacity_mwh.max(1e-9)).powi(2)
                };

                let u_old = actions5[b];
                let (u_opt, _) = ternary_search(eval_immediate, lo, hi, 8);

                if (u_opt - u_old).abs() < 1e-9 { continue; }

                let feasible = ptdf_b5[b].iter().all(|&(l, imp)| {
                    (flows5[l] + imp * (u_opt - u_old)).abs()
                        <= challenge.network.flow_limits[l] * 1.001
                });
                if !feasible { continue; }

                let old_prof = eval_immediate(u_old);
                let new_prof = eval_immediate(u_opt);
                if new_prof > old_prof + 1e-12 {
                    for &(l, imp) in &ptdf_b5[b] {
                        flows5[l] += imp * (u_opt - u_old);
                    }
                    actions5[b] = u_opt;
                    let total_profit = challenge.compute_profit(state, &actions5);
                    if total_profit > best_profit5 {
                        best_profit5 = total_profit;
                        best_actions = actions5.clone();
                    }
                    improved = true;
                }
            }
            if !improved { break; }
        }
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

        for pivot in 0..max_pivots {
            
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

    
    pub fn lp_solve(n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let (sol, _) = lp_solve_with_budget(n, m, c, a, b, LP_MAX_PIVOTS);
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
            ldd_bundle_radius: 0.0,
            ldd_temporal_eps: 0.0,
            ldd_use_polyak: false,
            ldd_polyak_safety: 0.0,
            ldd_use_dual_avg: false,
            ldd_da_decay: 0.0,
            ldd_use_polyak_dp: false,
            ldd_use_bplm: false,
            ldd_bplm_alpha: 0.95,
            ldd_use_lp_warmstart: false,
            ldd_warmstart_scale: 1.0,
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
            use_threshold_decode: false,
            use_analytic_dp_cell: false,
            use_simd: true,
            use_analytic_argmax: false,
            use_slope_cache: false,
            use_bat_hoist: false,
            use_cf_dispatch: false,
            cf_asca_iters: 0,
            use_ri_dp: false,
            ri_cong_threshold: 0.80,
            phi_int_alpha: 0.0,
            use_soc_pruning: false,
            use_mfg: false,
            mfg_iters: 0,
            use_hybrid_asca: false,
            hybrid_asca_iters: 5,
            use_per_bat_lmp_scale: false,
            per_bat_lmp_alpha: 0.4,
            use_dual_cf: false,
            dual_cf_scale: 0.3,
            use_morales_soc_pruning: false,
            use_best_iter_dp: false,
            use_cf_mu_warmstart: false,
            cf_mu_warmstart_scale: 1.0,
            use_mu_teff: false,
            mu_teff_eps: 0.0,
            use_rcb_fleet: false,
            rcb_price_mode: 0,
            use_rcb_dp_fleet: false,
            rcb_dp_post_iters: 0,
            ldd_use_dowg: false,
            post_asca_admm_iters: 0,
            post_asca_admm_rho: 0.001,
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
            ldd_bundle_radius: 0.0,
            ldd_temporal_eps: 0.0,
            ldd_use_polyak: false,
            ldd_polyak_safety: 0.0,
            ldd_use_dual_avg: false,
            ldd_da_decay: 0.0,
            ldd_use_polyak_dp: false,
            ldd_use_bplm: false,
            ldd_bplm_alpha: 0.95,
            ldd_use_lp_warmstart: false,
            ldd_warmstart_scale: 1.0,
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
            use_threshold_decode: false,
            use_analytic_dp_cell: false,
            use_simd: true,
            use_analytic_argmax: false,
            use_slope_cache: false,
            use_bat_hoist: false,
            use_cf_dispatch: false,
            cf_asca_iters: 0,
            use_ri_dp: false,
            ri_cong_threshold: 0.80,
            phi_int_alpha: 0.0,
            use_soc_pruning: false,
            use_mfg: false,
            mfg_iters: 0,
            use_hybrid_asca: false,
            hybrid_asca_iters: 5,
            use_per_bat_lmp_scale: false,
            per_bat_lmp_alpha: 0.4,
            use_dual_cf: false,
            dual_cf_scale: 0.3,
            use_morales_soc_pruning: false,
            use_best_iter_dp: false,
            use_cf_mu_warmstart: false,
            cf_mu_warmstart_scale: 1.0,
            use_mu_teff: false,
            mu_teff_eps: 0.0,
            use_rcb_fleet: false,
            rcb_price_mode: 0,
            use_rcb_dp_fleet: false,
            rcb_dp_post_iters: 0,
            ldd_use_dowg: false,
            post_asca_admm_iters: 0,
            post_asca_admm_rho: 0.001,
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
            ldd_bundle_radius: 0.0,
            ldd_temporal_eps: 0.0,
            ldd_use_polyak: false,
            ldd_polyak_safety: 0.0,
            ldd_use_dual_avg: false,
            ldd_da_decay: 0.0,
            ldd_use_polyak_dp: false,
            ldd_use_bplm: false,
            ldd_bplm_alpha: 0.95,
            ldd_use_lp_warmstart: false,
            ldd_warmstart_scale: 1.0,
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
            use_threshold_decode: false,
            use_analytic_dp_cell: false,
            use_simd: true,
            use_analytic_argmax: false,
            use_slope_cache: false,
            use_bat_hoist: false,
            use_cf_dispatch: false,
            cf_asca_iters: 0,
            use_ri_dp: false,
            ri_cong_threshold: 0.80,
            phi_int_alpha: 0.0,
            use_soc_pruning: false,
            use_mfg: false,
            mfg_iters: 0,
            use_hybrid_asca: false,
            hybrid_asca_iters: 5,
            use_per_bat_lmp_scale: false,
            per_bat_lmp_alpha: 0.4,
            use_dual_cf: false,
            dual_cf_scale: 0.3,
            use_morales_soc_pruning: false,
            use_best_iter_dp: false,
            use_cf_mu_warmstart: false,
            cf_mu_warmstart_scale: 1.0,
            use_mu_teff: false,
            mu_teff_eps: 0.0,
            use_rcb_fleet: false,
            rcb_price_mode: 0,
            use_rcb_dp_fleet: false,
            rcb_dp_post_iters: 0,
            ldd_use_dowg: false,
            post_asca_admm_iters: 0,
            post_asca_admm_rho: 0.001,
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
            asca_iters: 5,
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
            ldd_bundle_radius: 0.0,
            ldd_temporal_eps: 0.0,
            ldd_use_polyak: false,
            ldd_polyak_safety: 0.0,
            ldd_use_dual_avg: false,
            ldd_da_decay: 0.0,
            ldd_use_polyak_dp: false,
            ldd_use_bplm: false,
            ldd_bplm_alpha: 0.95,
            ldd_use_lp_warmstart: false,
            ldd_warmstart_scale: 1.0,
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
            use_threshold_decode: false,
            use_analytic_dp_cell: false,
            use_simd: true,
            use_analytic_argmax: false,
            use_slope_cache: false,
            use_bat_hoist: false,
            use_cf_dispatch: false,
            cf_asca_iters: 0,
            use_ri_dp: false,
            ri_cong_threshold: 0.80,
            phi_int_alpha: 0.0,
            use_soc_pruning: false,
            use_mfg: false,
            mfg_iters: 0,
            use_hybrid_asca: false,
            hybrid_asca_iters: 5,
            use_per_bat_lmp_scale: false,
            per_bat_lmp_alpha: 0.4,
            use_dual_cf: false,
            dual_cf_scale: 0.3,
            use_morales_soc_pruning: false,
            use_best_iter_dp: false,
            use_cf_mu_warmstart: false,
            cf_mu_warmstart_scale: 1.0,
            use_mu_teff: false,
            mu_teff_eps: 0.0,
            use_rcb_fleet: false,
            rcb_price_mode: 0,
            use_rcb_dp_fleet: false,
            rcb_dp_post_iters: 0,
            ldd_use_dowg: false,
            post_asca_admm_iters: 0,
            post_asca_admm_rho: 0.001,
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
            soc_levels: 81,
            action_grid: 10,
            asca_iters: 5,
            ternary_iters: 15,
            convergence_tol: 1e-3,
            anticipate_lmp: true,
            lmp_threshold: 0.52,
            lmp_premium_scale: 0.70,
            jump_premium: 0.06,
            prune_ratio: 0.00,
            deflator_iters: 50,
            flow_margin: 1e-4,
            network_derating: 0.10,
            dual_iters: 0,
            da_step_size: 0.0,
            ldd_iters: 3,
            ldd_step_size: 0.5,
            ldd_bundle_radius: 0.01,
            ldd_temporal_eps: 0.01,
            ldd_use_polyak: false,
            ldd_polyak_safety: 1.0,
            ldd_use_dual_avg: false,
            ldd_da_decay: 0.0,
            ldd_use_polyak_dp: false,
            ldd_use_bplm: false,
            ldd_bplm_alpha: 0.95,
            ldd_use_lp_warmstart: false,
            ldd_warmstart_scale: 1.0,
            use_kkt: false,
            kkt_cong_threshold: 0.0,
            kkt_price_scale: 0.8,
            max_admm_iters: 0,
            admm_rho: 0.2,
            admm_primal_tol: 0.05,
            use_lp: false,
            lp_soft_lambda: 1e5,
            lp_per_call_pivots: 300,
            lp_total_pivots: 6000,
            lp_horizon: 0,
            use_threshold_decode: false,
            use_analytic_dp_cell: false,
            use_simd: true,
            use_analytic_argmax: true,
            use_slope_cache: false,
            use_bat_hoist: false,
            use_cf_dispatch: true,
            cf_asca_iters: 20,
            use_ri_dp: false,
            ri_cong_threshold: 0.80,
            phi_int_alpha: 0.0,
            use_soc_pruning: false,
            use_mfg: false,
            mfg_iters: 0,
            use_hybrid_asca: false,
            hybrid_asca_iters: 5,
            use_per_bat_lmp_scale: false,
            per_bat_lmp_alpha: 0.4,
            use_dual_cf: false,
            dual_cf_scale: 0.3,
            use_morales_soc_pruning: false,
            use_best_iter_dp: false,
            use_cf_mu_warmstart: false,
            cf_mu_warmstart_scale: 1.0,
            use_mu_teff: false,
            mu_teff_eps: 0.0,
            use_rcb_fleet: false,
            rcb_price_mode: 0,
            use_rcb_dp_fleet: false,
            rcb_dp_post_iters: 0,
            ldd_use_dowg: false,
            post_asca_admm_iters: 8,
            post_asca_admm_rho: 0.40,
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
    pub ldd_use_polyak: Option<bool>,
    pub ldd_polyak_safety: Option<f64>,
    pub ldd_use_dual_avg: Option<bool>,
    pub ldd_da_decay: Option<f64>,
    pub ldd_use_polyak_dp: Option<bool>,
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

