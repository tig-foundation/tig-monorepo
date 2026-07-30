use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::energy_arbitrage::*;

pub mod track_t49;
pub mod track_t50;
pub mod t51_engine;
pub mod t52_engine;
pub mod t53_engine;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

fn n(v: u64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }
fn b(v: bool) -> Value { Value::Bool(v) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // titan_v8: per-track solvers assembled from the best valid iter of each track
    // (T49=t49/i48, T50=t50/i86, T51=t51/i90, T52=t52/i76, T53=t53/i87).
    // Baked HP = the iter's own baked defaults merged with the winning bench override,
    // so hp_json={} reproduces the winning per-track Q. User HP always win (merge_hp).
    match challenge.num_batteries {
        n if n <= 15 => {
            let hp = merge_hp(hyperparameters, vec![
                ("emp_tail_nodes", self::n(1)),
                ("soc_levels", self::n(101)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(25)),
                ("ternary_iters", self::n(25)),
                ("convergence_tol", f(1e-4)),
                ("k_clusters", self::n(80)),
                ("deflator_iters", self::n(15)),
                ("lp_refine_sweeps", self::n(3)),
                ("cg_iters", self::n(20)),
                ("use_lp", b(true)),
                ("use_sdp", b(true)),
                ("use_cg", b(true)),
                ("network_derating", f(1.00)),
                ("use_analytical_pricing", b(true)),
                ("use_pce_affine_recourse", b(true)),
                ("use_emp_scenarios", b(true)),
                ("emp_tail_scale", f(2.0)),
                ("use_regret_weights", b(true)),
                ("regret_weight_mode", self::n(2)),
                ("emp_tail_nodes", self::n(0)),
            ]);
            track_t49::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 30 => {
            let hp = merge_hp(hyperparameters, vec![
                ("ct_ref_kappa", f(0.2)),
                ("soc_levels", self::n(201)),
                ("action_grid", self::n(20)),
                ("asca_iters", self::n(30)),
                ("ternary_iters", self::n(25)),
                ("deflator_iters", self::n(50)),
                ("network_derating", f(0.35)),
                ("max_admm_iters", self::n(10)),
                ("lp_total_pivots", self::n(15000)),
                ("dw_total_pivot_budget", self::n(8000)),
                ("lns_lp_pivots_total", self::n(6000)),
                ("use_lp", b(true)),
                ("use_dw", b(true)),
                ("use_lns", b(true)),
                ("use_kkt", b(true)),
                ("ct_step_eta", f(1.0)),
                ("dp_rho_jump", f(0.015)),
                ("ct_gdd_alpha", f(1.0)),
                ("lmp_threshold", f(0.60)),
                ("anticipate_lmp", b(true)),
                ("lmp_premium_scale", f(1.2)),
                ("use_primal_refine", b(true)),
                ("dw_mu_damping_alpha", f(0.9)),
                ("dw_boxstep_delta", f(0.25)),
                ("premium_shape_gamma", f(4.5)),
                ("use_cos_weights", b(true)),
                ("cos_line_weight_scale", f(2.0)),
                ("line_weight_w_min", f(0.5)),
                ("line_weight_w_max", f(2.0)),
                ("use_cos_cs_weights", b(false)),
                ("cos_alpha_under", f(1.0)),
                ("use_lmp_premiums_kkt", b(true)),
                ("use_ct_adaptive_per_line", b(true)),
                ("use_ptdf_constraint_tracking", b(true)),
            ]);
            track_t50::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => {
            let hp = merge_hp(hyperparameters, vec![
                // exact per-step LP: +54,917 tuned / +50,200 held-out, 496W/0L, DETCHECK PASS
                ("use_lp_dispatch", b(true)),
                ("lp_max_lines", self::n(0)),
                // benchmarker-submitted tuning, measured +6,653 over shipped defaults (p=6e-19)
                ("grad_outer_iters", self::n(200)),
                ("dp_soc_levels", self::n(81)),
                ("joint_pair_budget", self::n(1200)),
                ("lahc_init_alpha_span", f(0.0)),
                ("dp_soc_levels", self::n(97)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(80)),
                ("grad_ls_iters", self::n(6)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(2)),
                ("lookahead_horizon", self::n(24)),
                ("rh_stride", self::n(3)),
                ("pga_beta_end", f(0.6)),
                ("use_momentum", b(true)),
                ("use_bb_clamps", b(true)),
                ("soc_ref_lambda", f(0.05)),
                ("use_admm_solver", b(false)),
                ("use_cosine_beta", b(true)),
                ("soc_ref_dyn_stride", self::n(6)),
                ("joint_triplet_top_k", self::n(15)),
                ("use_rolling_horizon", b(true)),
                ("joint_triplet_budget", self::n(300)),
                ("use_joint_pair_polish", b(true)),
                ("use_joint_triplet_polish", b(true)),
                ("use_arb_seed", b(true)),
                ("arb_pct", self::n(75)),
                ("arb_inverse", b(true)),
                ("use_std_arb_third", b(true)),
                ("arb_pct_third", self::n(0)),
                ("use_da_arb_third", b(true)),
                ("use_rollout_additive", b(true)),
                ("rollout_window", self::n(12)),
                ("use_rollout_additive_2", b(true)),
                ("rollout_window_2", self::n(4)),
                ("use_rollout_additive_3", b(true)),
                ("rollout_window_3", self::n(2)),
                ("use_rollout_additive_4", b(true)),
                ("rollout_window_4", self::n(1)),
                ("use_basin_hop", b(true)),
                ("basin_hop_scale", f(0.05)),
                ("basin_hop_k", self::n(4)),
                ("pair_lahc_lh", self::n(5)),
                ("lahc_init_alpha_span", f(0.0)),
            ]);
            t51_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => {
            let hp = merge_hp(hyperparameters, vec![
                // exact per-step LP: +48,315 tuned / +63,445 held-out, 194W/1L, DETCHECK PASS
                ("use_lp_dispatch", b(true)),
                ("lp_max_lines", self::n(0)),
                ("arb_diversity_inverse", b(true)),
                ("use_arb_diversity_pair", b(true)),
                ("use_arb_diversity_seed", b(true)),
                ("dp_soc_levels", self::n(65)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(75)),
                ("grad_ls_iters", self::n(12)),
                ("bisect_iters", self::n(30)),
                // i30 KEPT: coord_polish 0→1 = +547Q prouvé hp_json.
                ("coord_polish_passes", self::n(1)),
                ("lookahead_horizon", self::n(24)),
                
                // hp_json on ver3797): 0.10=2387727(+530) 0.15=2386612 0.20=2385558 0.25=2387197(CTRL)
                // 0.40=2386806 0.55=2386632 0.70=2385293. t53 optimum 0.70 (P41) does NOT transfer
                // to t52 (n<=80, clusters=4): axis flat/rugged, deterministic peak at 0.10.
                ("cwv_lambda", f(0.10)),
                
                
                // (+5575, +0.234%). 0.25=2365859 0.35=+310 0.40=+4666 [0.50=+5575] 0.65=+3801
                // 0.85=-2239 1.00=CTRL 1.25=-4462 1.50=-6629. i80's cwv=0.10 (weak coupling) slid
                // the optimum down from 1.0; t53's 1.5 (P43, cwv=0.70) does NOT transfer.
                ("ct_step_eta", f(0.50)),
                ("use_dp_seed", b(false)),
                ("use_ptdf_ct", b(true)),
                ("ct_ref_kappa", f(0.0)),
                ("cwv_clusters", self::n(4)),
                ("use_momentum", b(true)),
                ("lmp_threshold", f(0.5)),
                ("lr_growth_cap", f(1.025)),
                ("use_bb_clamps", b(true)),
                ("use_zero_seed", b(false)),
                ("anticipate_lmp", b(true)),
                ("cwv_agg_levels", self::n(65)),
                ("use_cosine_beta", b(true)),
                ("use_composite_wv", b(true)),
                ("use_pwl_value_dp", b(false)),
                ("joint_pair_budget", self::n(1024)),
                ("lmp_premium_scale", f(2.0)),
                ("pwl_max_breakpoints", self::n(64)),
                ("congestion_grid_alpha", f(0.0)),
                ("use_joint_pair_polish", b(true)),
                // i31 combine P23: gram-incremental projection on top of P27 coord_polish=1.
                // iso-geometry (Q bit-exact vs i30), cuts wall-time ~36% (P23 measured on i25).
                ("use_gram_incremental_proj", b(true)),
                ("resync_period", self::n(16)),
                // i34 BB adaptive step-size sentinel: false=iso-i31 (CTRL). Sweep via hp_json.
                ("use_bb_step", b(false)),
                // i35 L7 disagg-order sentinel: 0=natural (CTRL iso-i31). Sweep {1,2,3} via hp_json.
                ("disagg_order_mode", self::n(0)),
                // i36 KEPT γ=4.0 (+28,436Q DOMINANT 2 AXES). i37 bake: 1.0→4.0.
                
                
                // unimodal (2.5..5.0 + fine 3.25/3.75), time-neutral 11.0s. Precedent t50/t53 gamma-peak shifts with stack.
                ("premium_shape_gamma", f(3.5)),
                // i45 KEPT P31 virtual-queue ct_vq_v=100.0 (+18,046Q +0.71%). i46 bake.
                ("ct_vq_v", f(100.0)),
            ]);
            t52_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 150 => {
            let hp = merge_hp(hyperparameters, vec![
                // improvement004: +134,562 vs the benchmarker-HP baseline, 185W/14L, DETCHECK PASS.
                // FIRST WINS -- this entry, not the duplicate below, is the effective one.
                ("use_gram_incremental_proj", b(true)),
                ("use_lp_dispatch", b(true)),
                ("lp_max_lines", self::n(0)),
                ("num_seeds", self::n(3)),
                ("pair_alpha_max_passes", self::n(6)),
                ("pair_price_sorted", b(true)),
                ("dp_power_derate", f(0.55)),
                ("use_coupling_cut", b(true)),
                ("lmp_threshold_hp", f(0.30)),
                ("dp_soc_levels", self::n(65)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("premium_shape_gamma", f(5.0)),
                ("grad_outer_iters", self::n(150)),
                ("grad_ls_iters", self::n(12)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(1)),
                ("lookahead_horizon", self::n(24)),
                ("num_seeds", self::n(1)),
                ("cwv_lambda", f(0.7)),
                ("ct_step_eta", f(1.5)),
                ("use_ptdf_ct", b(true)),
                ("ct_gdd_alpha", f(0.0)),
                ("ct_ref_kappa", f(0.0)),
                ("use_momentum", b(true)),
                ("anticipate_lmp", b(true)),
                ("use_cosine_beta", b(false)),
                ("use_composite_wv", b(true)),
                ("joint_pair_budget", self::n(5500)),
                ("use_joint_pair_polish", b(true)),
                ("proj_relax", f(1.0)),
                ("use_gram_incremental_proj", b(true)),
                ("use_basin_hop", b(true)),
                ("basin_hop_scale", f(0.05)),
                ("basin_hop_k", self::n(4)),
                ("pair_alpha_interval", b(true)),
                ("pair_alpha_max_passes", self::n(2)),
                ("joint_pair_early_exit_k", self::n(2000)),
                ("ct_vq_v", f(100.0)),
                ("use_dp_value_shift", b(true)),
                ("dp_value_curv_coef", f(0.65)),
                ("oco_full_rebuild", b(true)),
            ]);
            t53_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("titan_v8: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("titan_v8");
}
