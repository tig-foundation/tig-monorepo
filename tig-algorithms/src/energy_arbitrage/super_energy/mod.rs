use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::energy_arbitrage::*;

pub mod track_baseline;
pub mod track_congested;
pub mod track_multiday;
pub mod track_dense;
pub mod track_capstone;

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

/// Instance-adaptive LMP premium curvature in the unimodal band [2.5, 5.0].
/// Tighter effective line limits (vs nominal) → higher gamma. Capstone γ_cong=0.40
/// lands near the proven 4.0 peak; dense γ_cong=0.50 lands near 3.75.
fn dense_adaptive_premium_gamma(challenge: &Challenge) -> f64 {
    let n_l = challenge.network.num_lines.max(1) as f64;
    let mean_eff = challenge.network.flow_limits.iter().copied().sum::<f64>() / n_l;
    let mean_nom = challenge.network.nominal_flow_limits.iter().copied().sum::<f64>() / n_l;
    let cong = (1.0 - mean_eff / mean_nom.max(1e-9)).clamp(0.0, 1.0);
    (2.5 + 2.5 * cong).clamp(2.5, 5.0)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Per-track HP maps mirror TrackHp / Hyperparameters in each track_*.rs file.
    // Values = track defaults merged with champion overrides (hp_json={} reproduces those).
    // User-supplied hyperparameters always win via merge_hp.
    match challenge.num_batteries {
        // T49 baseline — track_baseline::TrackHp
        n if n <= 15 => {
            let hp = merge_hp(hyperparameters, vec![
                ("soc_levels", self::n(101)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(25)),
                ("convergence_tol", f(1e-4)),
                ("anticipate_lmp", b(false)),
                ("lmp_threshold", f(0.85)),
                ("lmp_premium_scale", f(0.45)),
                ("jump_premium", f(0.02)),
                ("prune_ratio", f(0.0)),
                ("deflator_iters", self::n(15)),
                ("flow_margin", f(1e-4)),
                ("network_derating", f(1.00)),
                ("use_sdp", b(true)),
                ("use_lp", b(true)),
                ("lp_refine_sweeps", self::n(3)),
                ("use_cg", b(true)),
                ("cg_iters", self::n(20)),
            ]);
            track_baseline::solve_challenge(challenge, save_solution, &hp)
        }
        // T50 congested — track_congested::TrackHp
        n if n <= 30 => {
            let hp = merge_hp(hyperparameters, vec![
                ("soc_levels", self::n(201)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(4)),
                ("ternary_iters", self::n(20)),
                ("convergence_tol", f(1e-3)),
                ("anticipate_lmp", b(true)),
                ("lmp_threshold", f(0.65)),
                ("lmp_premium_scale", f(1.2)),
                ("jump_premium", f(0.0)),
                ("prune_ratio", f(0.0)),
                ("deflator_iters", self::n(50)),
                ("flow_margin", f(1e-4)),
                ("flow_feas_tol", f(1e-6)),
                ("network_derating", f(0.35)),
                ("dual_iters", self::n(0)),
                ("da_step_size", f(0.01)),
                ("ldd_iters", self::n(4)),
                ("ldd_step_size", f(0.25)),
                ("use_kkt", b(true)),
                ("kkt_cong_threshold", f(0.70)),
                ("kkt_price_scale", f(0.8)),
                ("max_admm_iters", self::n(10)),
                ("admm_rho", f(0.2)),
                ("admm_primal_tol", f(0.05)),
                ("use_lp", b(true)),
                ("dantzig_in_dw", b(false)),
                ("dantzig_in_lns", b(false)),
                ("dantzig_in_kkt", b(false)),
                ("dantzig_bland_degen", b(false)),
                ("lp_soft_lambda", f(1e5)),
                ("lp_per_call_pivots", self::n(800)),
                ("lp_total_pivots", self::n(15000)),
                ("use_policy", b(false)),
                ("use_warmstart", b(true)),
                ("use_mpc", b(false)),
                ("mpc_horizon", self::n(2)),
                ("mpc_pivot_budget", self::n(800)),
                ("use_dw", b(true)),
                ("dw_iters", self::n(3)),
                ("dw_max_lines", self::n(10)),
                ("dw_max_cols_per_batt", self::n(5)),
                ("dw_pivot_budget_per_solve", self::n(2000)),
                ("dw_total_pivot_budget", self::n(8000)),
                ("use_dw_prescreen", b(true)),
                ("use_lns", b(true)),
                ("lns_cg_iters", self::n(3)),
                ("lns_cg_column_limit", self::n(6)),
                ("lns_max_lines", self::n(12)),
                ("lns_lp_pivots_total", self::n(6000)),
                ("use_pivot_reserve", b(true)),
                ("lp_max_lines", self::n(12)),
                ("use_parallel_dp", b(true)),
                ("use_sdp", b(true)),
                ("sdp_k", self::n(3)),
                ("use_ldd_proximal", b(true)),
                ("ldd_momentum", f(0.5)),
                ("ldd_clip_fraction", f(0.2)),
                ("use_tail_quadrature", b(true)),
                ("dp_sigma", f(0.15)),
                ("dp_rho_jump", f(0.015)),
                ("dp_alpha", f(3.5)),
                ("lns_dual_smooth_alpha", f(0.0)),
                ("use_primal_refine", b(true)),
                ("use_lmp_premiums_kkt", b(true)),
                ("use_prime_admm", b(false)),
                ("dw_mu_damping_alpha", f(0.6)),
                ("use_adaptive_lines", b(false)),
                ("use_binary_congestion_premium", b(false)),
                ("congestion_quantize_levels", self::n(0)),
                ("use_action_aware_premium", b(false)),
                ("coord_premium_mode", self::n(0)),
                ("coord_premium_scale", f(0.0)),
                ("use_slp_degradation", b(false)),
                ("use_ptdf_constraint_tracking", b(true)),
                ("ct_step_eta", f(0.5)),
                ("ct_ref_kappa", f(0.0)),
                ("ct_oc_kappa", f(0.0)),
                ("use_ct_adaptive_per_line", b(true)),
                ("ct_gdd_rho", f(0.0)),
                ("ct_gdd_alpha", f(1.0)),
            ]);
            track_congested::solve_challenge(challenge, save_solution, &hp)
        }
        // T51 multiday — track_multiday::Hyperparameters
        n if n <= 50 => {
            let hp = merge_hp(hyperparameters, vec![
                ("dp_soc_levels", self::n(97)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(80)),
                ("grad_ls_iters", self::n(6)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(2)),
                ("lookahead_horizon", self::n(24)),
                ("fuel_budget", self::n(0)),
                ("use_bb_clamps", b(true)),
                ("use_momentum", b(true)),
                ("anticipate_lmp", b(false)),
                ("lmp_threshold", f(0.65)),
                ("lmp_premium_scale", f(1.0)),
                ("use_joint_pair_polish", b(true)),
                ("joint_pair_budget", self::n(780)),
                ("use_joint_triplet_polish", b(true)),
                ("joint_triplet_budget", self::n(300)),
                ("joint_triplet_top_k", self::n(15)),
                ("use_admm_polish", b(false)),
                ("use_ejection_chain", b(false)),
                ("use_scvc", b(false)),
                ("scvc_alpha", f(0.5)),
                ("use_rolling_horizon", b(true)),
                ("rh_stride", self::n(3)),
                ("soc_ref_lambda", f(0.05)),
                ("soc_ref_dyn_stride", self::n(6)),
                ("use_cosine_beta", b(true)),
                ("pga_beta_end", f(0.6)),
                ("use_admm_solver", b(false)),
                ("admm_rho", f(0.45)),
                ("admm_iters", self::n(9)),
            ]);
            track_multiday::solve_challenge(challenge, save_solution, &hp)
        }
        // T52 dense — track_dense::Hyperparameters
        n if n <= 80 => {
            let hp = merge_hp(hyperparameters, vec![
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
                ("premium_shape_gamma", f(dense_adaptive_premium_gamma(challenge))),
                // Paired-panel calibration: lower V reacts faster to accumulated line stress.
                ("ct_vq_v", f(75.0)),
            ]);
            track_dense::solve_challenge(challenge, save_solution, &hp)
        }
        // T53 capstone — track_capstone::Hyperparameters
        n if n <= 150 => {
            let hp = merge_hp(hyperparameters, vec![
                ("use_gram_incremental_proj", b(true)),
                ("dp_soc_levels", self::n(65)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("premium_shape_gamma", f(4.5)),
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
                ("use_basin_hop", b(true)),
                ("basin_hop_scale", f(0.05)),
                ("basin_hop_k", self::n(4)),
                ("pair_alpha_interval", b(true)),
                ("pair_alpha_max_passes", self::n(2)),
                ("joint_pair_early_exit_k", self::n(2000)),
                ("ct_vq_v", f(75.0)),
                ("use_dp_value_shift", b(true)),
                ("dp_value_curv_coef", f(0.55)),
                ("oco_full_rebuild", b(true)),
                ("ct_round2_eta_frac", f(0.10)),
            ]);
            track_capstone::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("super_energy: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("super_energy — per-track energy_arbitrage solver");
    println!("Tracks: baseline(<=15), congested(<=30), multiday(<=50), dense(<=80), capstone(<=150)");
}
