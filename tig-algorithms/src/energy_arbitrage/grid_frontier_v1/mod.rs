// TIG Energy Arbitrage: grid_frontier_v1
// Public-state adaptive controller with portfolio selection, network-aware repair, active-set repricing, and frontier optimization.
// Designed as a drop-in tig-algorithms/src/energy_arbitrage/<algo_name>/mod.rs module.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use tig_challenges::energy_arbitrage::{
    constants, Battery, Challenge, NextRTPrices, Solution, State,
};

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub levels: Option<usize>,
    pub action_levels: Option<usize>,
    pub cand_levels: Option<usize>,
    pub passes: Option<usize>,
    pub flow_margin: Option<f64>,
    pub min_value: Option<f64>,

    pub force_dp: Option<bool>,
    pub force_shadow: Option<bool>,
    pub force_hybrid: Option<bool>,
    pub congested_shadow: Option<bool>,
    pub multiday_shadow: Option<bool>,

    pub shadow_q: Option<f64>,
    pub q_baseline: Option<f64>,
    pub q_congested: Option<f64>,
    pub q_multiday: Option<f64>,
    pub q_dense: Option<f64>,
    pub q_capstone: Option<f64>,
    pub shadow_inventory_weight: Option<f64>,

    pub support_baseline: Option<bool>,
    pub support_jump_scale: Option<f64>,
    pub support_jump_tail: Option<f64>,
    pub support_normal_spread: Option<f64>,
    pub support_congestion_scale: Option<f64>,

    pub include_all_small: Option<bool>,
    pub include_baseline_candidates: Option<bool>,

    pub adaptive_enabled: Option<bool>,
    pub expected_shadow_enabled: Option<bool>,
    pub spike_discount: Option<f64>,
    pub dynamic_q_shift: Option<f64>,

    pub short_horizon_enabled: Option<bool>,
    pub residual_retry_enabled: Option<bool>,
    pub counterflow_enabled: Option<bool>,
    pub selector_shadow_enabled: Option<bool>,
    pub soc_skew: Option<f64>,
    pub endgame_window: Option<usize>,
    pub endgame_floor: Option<f64>,
    pub density_eps: Option<f64>,
    pub relief_bonus: Option<f64>,
    pub counter_passes: Option<usize>,

    pub dual_enabled: Option<bool>,
    pub dual_price_scale: Option<f64>,
    pub dual_stress_threshold: Option<f64>,
    pub network_reserve_enabled: Option<bool>,
    pub reserve_window: Option<usize>,
    pub reserve_price_scale: Option<f64>,
    pub node_adaptation_enabled: Option<bool>,
    pub alts_per_battery: Option<usize>,
    pub local_refine_enabled: Option<bool>,
    pub refine_passes: Option<usize>,
    pub rollout_enabled: Option<bool>,
    pub rollout_horizon: Option<usize>,
    pub rollout_candidate_cap: Option<usize>,

    pub beam_pack_enabled: Option<bool>,
    pub beam_width: Option<usize>,
    pub beam_battery_cap: Option<usize>,
    pub beam_partial_enabled: Option<bool>,
    pub regret_swap_enabled: Option<bool>,
    pub regret_swap_passes: Option<usize>,

    pub blend_candidates_enabled: Option<bool>,
    pub blend_top_k: Option<usize>,
    pub blend_alpha: Option<f64>,

    pub ensemble_rollout_enabled: Option<bool>,
    pub rollout_robust_weight: Option<f64>,
    pub rollout_upside_weight: Option<f64>,
    pub nodal_dual_price_enabled: Option<bool>,
    pub nodal_dual_scale: Option<f64>,
    pub terminal_pressure_enabled: Option<bool>,
    pub terminal_pressure_scale: Option<f64>,
    pub pairflow_enabled: Option<bool>,
    pub pairflow_passes: Option<usize>,
    pub pairflow_candidate_cap: Option<usize>,
    pub pairflow_counter_cap: Option<usize>,
    pub pairflow_min_value: Option<f64>,
    pub pairflow_main_fracs: Option<usize>,

    pub inventory_target_enabled: Option<bool>,
    pub target_policy_weight: Option<f64>,
    pub target_lookahead: Option<usize>,
    pub target_mid_band: Option<f64>,

    pub selector_refine_enabled: Option<bool>,
    pub selector_refine_passes: Option<usize>,
    pub selector_refine_battery_cap: Option<usize>,

    pub portfolio_consensus_enabled: Option<bool>,
    pub consensus_top_k: Option<usize>,
    pub consensus_weight_decay: Option<f64>,

    pub active_set_enabled: Option<bool>,
    pub active_set_passes: Option<usize>,
    pub active_set_grid: Option<usize>,
    pub active_set_battery_cap: Option<usize>,
    pub active_set_dual_step: Option<f64>,
    pub active_set_dual_decay: Option<f64>,
    pub active_set_stress_threshold: Option<f64>,

    pub frontier_optimizer_enabled: Option<bool>,
    pub frontier_top_k: Option<usize>,
    pub frontier_passes: Option<usize>,
    pub frontier_step: Option<f64>,
}

#[derive(Clone, Copy)]
struct Config {
    levels: usize,
    action_levels: usize,
    cand_levels: usize,
    passes: usize,
    flow_margin: f64,
    min_value: f64,

    force_dp: bool,
    force_shadow: bool,
    force_hybrid: bool,
    congested_shadow: bool,
    multiday_shadow: bool,

    shadow_q: f64,
    q_baseline: f64,
    q_congested: f64,
    q_multiday: f64,
    q_dense: f64,
    q_capstone: f64,
    shadow_inventory_weight: f64,

    support_baseline: bool,
    support_jump_scale: f64,
    support_jump_tail: f64,
    support_normal_spread: f64,
    support_congestion_scale: f64,

    include_all_small: bool,
    include_baseline_candidates: bool,

    adaptive_enabled: bool,
    expected_shadow_enabled: bool,
    spike_discount: f64,
    dynamic_q_shift: f64,

    short_horizon_enabled: bool,
    residual_retry_enabled: bool,
    counterflow_enabled: bool,
    selector_shadow_enabled: bool,
    soc_skew: f64,
    endgame_window: usize,
    endgame_floor: f64,
    density_eps: f64,
    relief_bonus: f64,
    counter_passes: usize,

    dual_enabled: bool,
    dual_price_scale: f64,
    dual_stress_threshold: f64,
    network_reserve_enabled: bool,
    reserve_window: usize,
    reserve_price_scale: f64,
    node_adaptation_enabled: bool,
    alts_per_battery: usize,
    local_refine_enabled: bool,
    refine_passes: usize,
    rollout_enabled: bool,
    rollout_horizon: usize,
    rollout_candidate_cap: usize,

    beam_pack_enabled: bool,
    beam_width: usize,
    beam_battery_cap: usize,
    beam_partial_enabled: bool,
    regret_swap_enabled: bool,
    regret_swap_passes: usize,

    blend_candidates_enabled: bool,
    blend_top_k: usize,
    blend_alpha: f64,

    ensemble_rollout_enabled: bool,
    rollout_robust_weight: f64,
    rollout_upside_weight: f64,
    nodal_dual_price_enabled: bool,
    nodal_dual_scale: f64,
    terminal_pressure_enabled: bool,
    terminal_pressure_scale: f64,
    pairflow_enabled: bool,
    pairflow_passes: usize,
    pairflow_candidate_cap: usize,
    pairflow_counter_cap: usize,
    pairflow_min_value: f64,
    pairflow_main_fracs: usize,

    inventory_target_enabled: bool,
    target_policy_weight: f64,
    target_lookahead: usize,
    target_mid_band: f64,

    selector_refine_enabled: bool,
    selector_refine_passes: usize,
    selector_refine_battery_cap: usize,

    portfolio_consensus_enabled: bool,
    consensus_top_k: usize,
    consensus_weight_decay: f64,

    active_set_enabled: bool,
    active_set_passes: usize,
    active_set_grid: usize,
    active_set_battery_cap: usize,
    active_set_dual_step: f64,
    active_set_dual_decay: f64,
    active_set_stress_threshold: f64,

    frontier_optimizer_enabled: bool,
    frontier_top_k: usize,
    frontier_passes: usize,
    frontier_step: f64,
}

impl Config {
    fn from_hp(hp: &Option<Map<String, Value>>, challenge: &Challenge) -> Self {
        let get_f64 = |k: &str, d: f64| -> f64 {
            hp.as_ref()
                .and_then(|m| m.get(k))
                .and_then(|v| v.as_f64())
                .unwrap_or(d)
        };
        let get_usize = |k: &str, d: usize| -> usize {
            hp.as_ref()
                .and_then(|m| m.get(k))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let get_bool = |k: &str, d: bool| -> bool {
            hp.as_ref()
                .and_then(|m| m.get(k))
                .and_then(|v| v.as_bool())
                .unwrap_or(d)
        };

        let size = challenge.num_steps * challenge.num_batteries;
        let default_levels = 11;
        let default_action_levels = 5;
        let default_passes = 1;
        let track = detect_track(challenge);
        let default_flow_margin = match track {
            TrackKind::Baseline => 0.9980,
            TrackKind::Congested => 0.9975,
            TrackKind::Multiday => 0.9975,
            TrackKind::Dense => 0.9970,
            TrackKind::Capstone | TrackKind::Unknown => 0.9960,
        };
        let default_shadow_inventory_weight = match track {
            TrackKind::Baseline => 0.92,
            TrackKind::Congested => 0.94,
            TrackKind::Multiday => 0.96,
            TrackKind::Dense => 0.96,
            TrackKind::Capstone | TrackKind::Unknown => 0.98,
        };
        let default_soc_skew = match track {
            TrackKind::Baseline => 0.45,
            TrackKind::Congested => 0.38,
            TrackKind::Multiday => 0.32,
            TrackKind::Dense => 0.30,
            TrackKind::Capstone | TrackKind::Unknown => 0.25,
        };
        let default_endgame_window = match track {
            TrackKind::Baseline => 24,
            TrackKind::Congested => 28,
            TrackKind::Multiday => 32,
            TrackKind::Dense => 36,
            TrackKind::Capstone | TrackKind::Unknown => 40,
        };
        let default_density_eps = match track {
            TrackKind::Baseline => 0.015,
            TrackKind::Congested | TrackKind::Multiday => 0.020,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 0.025,
        };
        let default_relief_bonus = match track {
            TrackKind::Baseline => 2.5,
            TrackKind::Congested | TrackKind::Multiday => 3.0,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 4.0,
        };
        let default_counter_passes = match track {
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 2,
            _ => 1,
        };
        let default_dual_price_scale = match track {
            TrackKind::Baseline => 18.0,
            TrackKind::Congested => 28.0,
            TrackKind::Multiday => 34.0,
            TrackKind::Dense => 42.0,
            TrackKind::Capstone | TrackKind::Unknown => 48.0,
        };
        let default_reserve_window = match track {
            TrackKind::Baseline => 24,
            TrackKind::Congested => 32,
            TrackKind::Multiday => 48,
            TrackKind::Dense => 64,
            TrackKind::Capstone | TrackKind::Unknown => 72,
        };
        let default_reserve_price_scale = match track {
            TrackKind::Baseline => 5.0,
            TrackKind::Congested => 8.0,
            TrackKind::Multiday => 10.0,
            TrackKind::Dense => 13.0,
            TrackKind::Capstone | TrackKind::Unknown => 15.0,
        };
        let default_alts_per_battery = match track {
            TrackKind::Baseline => 3,
            TrackKind::Congested | TrackKind::Multiday => 3,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 2,
        };
        let default_refine_passes = match track {
            TrackKind::Baseline | TrackKind::Congested => 2,
            TrackKind::Multiday => 1,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 0,
        };
        let default_rollout_horizon = match track {
            TrackKind::Baseline => 3,
            TrackKind::Congested => 3,
            TrackKind::Multiday => 2,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 2,
        };
        let default_rollout_candidate_cap = match track {
            TrackKind::Baseline | TrackKind::Congested => 8,
            TrackKind::Multiday | TrackKind::Dense => 6,
            TrackKind::Capstone | TrackKind::Unknown => 5,
        };

        let default_beam_width = match track {
            TrackKind::Baseline => 12,
            TrackKind::Congested => 10,
            TrackKind::Multiday => 7,
            TrackKind::Dense => 5,
            TrackKind::Capstone | TrackKind::Unknown => 4,
        };
        let default_beam_battery_cap = match track {
            TrackKind::Baseline => 16,
            TrackKind::Congested => 26,
            TrackKind::Multiday => 42,
            TrackKind::Dense => 46,
            TrackKind::Capstone | TrackKind::Unknown => 42,
        };
        let default_regret_swap_passes = match track {
            TrackKind::Baseline | TrackKind::Congested | TrackKind::Multiday => 2,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 1,
        };
        let default_blend_top_k = match track {
            TrackKind::Baseline | TrackKind::Congested => 6,
            TrackKind::Multiday | TrackKind::Dense => 5,
            TrackKind::Capstone | TrackKind::Unknown => 4,
        };
        let default_pairflow_passes = match track {
            TrackKind::Baseline => 1,
            TrackKind::Congested | TrackKind::Multiday => 2,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 2,
        };
        let default_pairflow_candidate_cap = match track {
            TrackKind::Baseline => 10,
            TrackKind::Congested => 18,
            TrackKind::Multiday => 24,
            TrackKind::Dense => 28,
            TrackKind::Capstone | TrackKind::Unknown => 30,
        };
        let default_pairflow_counter_cap = match track {
            TrackKind::Baseline => 8,
            TrackKind::Congested => 12,
            TrackKind::Multiday => 16,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 18,
        };
        let default_terminal_pressure_scale = match track {
            TrackKind::Baseline => 0.35,
            TrackKind::Congested => 0.45,
            TrackKind::Multiday => 0.55,
            TrackKind::Dense => 0.60,
            TrackKind::Capstone | TrackKind::Unknown => 0.65,
        };
        let default_target_lookahead = match track {
            TrackKind::Baseline => 48,
            TrackKind::Congested => 64,
            TrackKind::Multiday => 96,
            TrackKind::Dense => 96,
            TrackKind::Capstone | TrackKind::Unknown => 96,
        };
        let default_target_policy_weight = match track {
            TrackKind::Baseline => 0.80,
            TrackKind::Congested => 0.95,
            TrackKind::Multiday => 1.05,
            TrackKind::Dense => 1.10,
            TrackKind::Capstone | TrackKind::Unknown => 1.15,
        };
        let default_selector_refine_enabled = match track {
            TrackKind::Baseline | TrackKind::Congested | TrackKind::Multiday => true,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => false,
        };
        let default_selector_refine_passes = match track {
            TrackKind::Baseline => 2,
            TrackKind::Congested | TrackKind::Multiday => 1,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 0,
        };
        let default_selector_refine_battery_cap = match track {
            TrackKind::Baseline => 10,
            TrackKind::Congested => 14,
            TrackKind::Multiday => 16,
            TrackKind::Dense => 0,
            TrackKind::Capstone | TrackKind::Unknown => 0,
        };
        let default_consensus_top_k = match track {
            TrackKind::Baseline | TrackKind::Congested => 6,
            TrackKind::Multiday | TrackKind::Dense => 5,
            TrackKind::Capstone | TrackKind::Unknown => 4,
        };
        let default_active_set_passes = match track {
            TrackKind::Baseline => 1,
            TrackKind::Congested | TrackKind::Multiday => 2,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 2,
        };
        let default_active_set_grid = match track {
            TrackKind::Baseline => 7,
            TrackKind::Congested | TrackKind::Multiday => 7,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 5,
        };
        let default_active_set_battery_cap = match track {
            TrackKind::Baseline => 10,
            TrackKind::Congested => 18,
            TrackKind::Multiday => 28,
            TrackKind::Dense => 34,
            TrackKind::Capstone | TrackKind::Unknown => 36,
        };
        let default_frontier_top_k = match track {
            TrackKind::Baseline => 5,
            TrackKind::Congested => 6,
            TrackKind::Multiday => 6,
            TrackKind::Dense => 6,
            TrackKind::Capstone | TrackKind::Unknown => 5,
        };
        let default_frontier_passes = match track {
            TrackKind::Baseline => 2,
            TrackKind::Congested | TrackKind::Multiday => 3,
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => 2,
        };

        Self {
            levels: get_usize("levels", default_levels).clamp(11, 61),
            action_levels: get_usize("action_levels", default_action_levels).clamp(5, 25),
            cand_levels: get_usize("cand_levels", default_action_levels).clamp(5, 25),
            passes: get_usize("passes", default_passes).clamp(1, 4),
            flow_margin: get_f64("flow_margin", default_flow_margin).clamp(0.95, 1.0),
            min_value: get_f64("min_value", 0.0).max(0.0),

            force_dp: get_bool("force_dp", false),
            force_shadow: get_bool("force_shadow", false),
            force_hybrid: get_bool("force_hybrid", false),
            congested_shadow: get_bool("congested_shadow", true),
            multiday_shadow: get_bool("multiday_shadow", true),

            shadow_q: get_f64("shadow_q", 0.70).clamp(0.30, 0.98),
            q_baseline: get_f64("q_baseline", 0.90).clamp(0.30, 0.98),
            q_congested: get_f64("q_congested", 0.65).clamp(0.30, 0.98),
            q_multiday: get_f64("q_multiday", 0.45).clamp(0.30, 0.98),
            q_dense: get_f64("q_dense", 0.45).clamp(0.30, 0.98),
            q_capstone: get_f64("q_capstone", 0.35).clamp(0.30, 0.98),
            shadow_inventory_weight: get_f64(
                "shadow_inventory_weight",
                default_shadow_inventory_weight,
            )
            .clamp(0.50, 1.25),

            support_baseline: get_bool("support_baseline", true),
            support_jump_scale: get_f64("support_jump_scale", 1.00).clamp(0.25, 1.75),
            support_jump_tail: get_f64("support_jump_tail", 1.00).clamp(0.25, 1.75),
            support_normal_spread: get_f64("support_normal_spread", 1.00).clamp(0.25, 1.50),
            support_congestion_scale: get_f64("support_congestion_scale", 1.00).clamp(0.25, 1.75),

            include_all_small: get_bool("include_all_small", true),
            include_baseline_candidates: get_bool("include_baseline_candidates", false),

            adaptive_enabled: get_bool("adaptive_enabled", false),
            expected_shadow_enabled: get_bool("expected_shadow_enabled", true),
            spike_discount: get_f64("spike_discount", 0.55).clamp(0.20, 1.00),
            dynamic_q_shift: get_f64("dynamic_q_shift", 0.10).clamp(0.0, 0.25),

            short_horizon_enabled: get_bool("short_horizon_enabled", true),
            residual_retry_enabled: get_bool("residual_retry_enabled", false),
            counterflow_enabled: get_bool("counterflow_enabled", false),
            selector_shadow_enabled: get_bool("selector_shadow_enabled", false),
            soc_skew: get_f64("soc_skew", default_soc_skew).clamp(0.0, 1.0),
            endgame_window: get_usize("endgame_window", default_endgame_window).clamp(0, 96),
            endgame_floor: get_f64("endgame_floor", 0.10).clamp(0.0, 1.0),
            density_eps: get_f64("density_eps", default_density_eps).clamp(0.001, 0.25),
            relief_bonus: get_f64("relief_bonus", default_relief_bonus).clamp(0.0, 20.0),
            counter_passes: get_usize("counter_passes", default_counter_passes).clamp(0, 4),

            dual_enabled: get_bool("dual_enabled", false),
            dual_price_scale: get_f64("dual_price_scale", default_dual_price_scale)
                .clamp(0.0, 120.0),
            dual_stress_threshold: get_f64("dual_stress_threshold", 0.72).clamp(0.20, 0.98),
            network_reserve_enabled: get_bool("network_reserve_enabled", false),
            reserve_window: get_usize("reserve_window", default_reserve_window).clamp(0, 128),
            reserve_price_scale: get_f64("reserve_price_scale", default_reserve_price_scale)
                .clamp(0.0, 80.0),
            node_adaptation_enabled: get_bool("node_adaptation_enabled", false),
            alts_per_battery: get_usize("alts_per_battery", default_alts_per_battery).clamp(1, 5),
            local_refine_enabled: get_bool("local_refine_enabled", false),
            refine_passes: get_usize("refine_passes", default_refine_passes).clamp(0, 4),
            rollout_enabled: get_bool("rollout_enabled", false),
            rollout_horizon: get_usize("rollout_horizon", default_rollout_horizon).clamp(1, 5),
            rollout_candidate_cap: get_usize(
                "rollout_candidate_cap",
                default_rollout_candidate_cap,
            )
            .clamp(1, 16),

            beam_pack_enabled: get_bool("beam_pack_enabled", false),
            beam_width: get_usize("beam_width", default_beam_width).clamp(1, 32),
            beam_battery_cap: get_usize("beam_battery_cap", default_beam_battery_cap).clamp(0, 128),
            beam_partial_enabled: get_bool("beam_partial_enabled", true),
            regret_swap_enabled: get_bool("regret_swap_enabled", false),
            regret_swap_passes: get_usize("regret_swap_passes", default_regret_swap_passes)
                .clamp(0, 4),

            blend_candidates_enabled: get_bool("blend_candidates_enabled", false),
            blend_top_k: get_usize("blend_top_k", default_blend_top_k).clamp(2, 10),
            blend_alpha: get_f64("blend_alpha", 0.35).clamp(0.10, 0.90),

            ensemble_rollout_enabled: get_bool("ensemble_rollout_enabled", false),
            rollout_robust_weight: get_f64("rollout_robust_weight", 0.35).clamp(0.0, 1.0),
            rollout_upside_weight: get_f64("rollout_upside_weight", 0.12).clamp(0.0, 0.75),
            nodal_dual_price_enabled: get_bool("nodal_dual_price_enabled", false),
            nodal_dual_scale: get_f64("nodal_dual_scale", 1.0).clamp(0.0, 3.0),
            terminal_pressure_enabled: get_bool("terminal_pressure_enabled", false),
            terminal_pressure_scale: get_f64(
                "terminal_pressure_scale",
                default_terminal_pressure_scale,
            )
            .clamp(0.0, 2.0),
            pairflow_enabled: get_bool("pairflow_enabled", false),
            pairflow_passes: get_usize("pairflow_passes", default_pairflow_passes).clamp(0, 4),
            pairflow_candidate_cap: get_usize(
                "pairflow_candidate_cap",
                default_pairflow_candidate_cap,
            )
            .clamp(0, 96),
            pairflow_counter_cap: get_usize("pairflow_counter_cap", default_pairflow_counter_cap)
                .clamp(0, 96),
            pairflow_min_value: get_f64("pairflow_min_value", 0.0).max(0.0),
            pairflow_main_fracs: get_usize("pairflow_main_fracs", 4).clamp(1, 6),

            inventory_target_enabled: get_bool("inventory_target_enabled", false),
            target_policy_weight: get_f64("target_policy_weight", default_target_policy_weight)
                .clamp(0.0, 3.0),
            target_lookahead: get_usize("target_lookahead", default_target_lookahead).clamp(8, 192),
            target_mid_band: get_f64("target_mid_band", 0.08).clamp(0.0, 0.35),

            selector_refine_enabled: get_bool("selector_refine_enabled", false),
            selector_refine_passes: get_usize(
                "selector_refine_passes",
                default_selector_refine_passes,
            )
            .clamp(0, 4),
            selector_refine_battery_cap: get_usize(
                "selector_refine_battery_cap",
                default_selector_refine_battery_cap,
            )
            .clamp(0, 128),

            portfolio_consensus_enabled: get_bool("portfolio_consensus_enabled", false),
            consensus_top_k: get_usize("consensus_top_k", default_consensus_top_k).clamp(2, 10),
            consensus_weight_decay: get_f64("consensus_weight_decay", 0.72).clamp(0.25, 1.0),

            active_set_enabled: get_bool("active_set_enabled", false),
            active_set_passes: get_usize("active_set_passes", default_active_set_passes)
                .clamp(0, 5),
            active_set_grid: get_usize("active_set_grid", default_active_set_grid).clamp(3, 15),
            active_set_battery_cap: get_usize(
                "active_set_battery_cap",
                default_active_set_battery_cap,
            )
            .clamp(0, 128),
            active_set_dual_step: get_f64("active_set_dual_step", 34.0).clamp(0.0, 140.0),
            active_set_dual_decay: get_f64("active_set_dual_decay", 0.55).clamp(0.0, 0.98),
            active_set_stress_threshold: get_f64("active_set_stress_threshold", 0.74)
                .clamp(0.35, 0.98),

            frontier_optimizer_enabled: get_bool("frontier_optimizer_enabled", false),
            frontier_top_k: get_usize("frontier_top_k", default_frontier_top_k).clamp(2, 10),
            frontier_passes: get_usize("frontier_passes", default_frontier_passes).clamp(0, 6),
            frontier_step: get_f64("frontier_step", 0.35).clamp(0.05, 0.90),
        }
    }
}

pub fn help() {
    println!(
        "{}",
        r#"grid_frontier_v1: public-state Energy Arbitrage controller using shadow-price dispatch, compact policy selection, and network-feasible schedule repair. Default hyperparameters are configured for benchmark validity and runtime safety."#
    );
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let zero_action = vec![0.0; challenge.num_batteries];
    let zero_solution = Solution {
        schedule: vec![zero_action; challenge.num_steps],
    };
    save_solution(&zero_solution)?;

    let cfg = Config::from_hp(hyperparameters, challenge);
    let planner = Planner::new(challenge, cfg);
    let online = RefCell::new(OnlineState::new());
    let checkpoint_schedule: RefCell<Vec<Vec<f64>>> =
        RefCell::new(Vec::with_capacity(challenge.num_steps));
    let solution = challenge.grid_optimize(&|c, s| {
        {
            let mut o = online.borrow_mut();
            o.observe(c, s);
        }
        let o = online.borrow();
        let action = if cfg.adaptive_enabled {
            planner.policy_dynamic(c, s, &o)
        } else {
            planner.policy(c, s)
        }?;
        if action.len() == s.action_bounds.len()
            && action
                .iter()
                .zip(s.action_bounds.iter())
                .all(|(&a, &(lo, hi))| a >= lo && a <= hi)
            && is_action_feasible(c, s, &action)
        {
            let mut prefix = checkpoint_schedule.borrow_mut();
            prefix.push(action.clone());
            if prefix.len() == c.num_steps || prefix.len() % 16 == 0 {
                let mut schedule = Vec::with_capacity(c.num_steps);
                schedule.extend(prefix.iter().cloned());
                schedule.extend((prefix.len()..c.num_steps).map(|_| vec![0.0; c.num_batteries]));
                save_solution(&Solution { schedule })?;
            }
        }
        Ok(action)
    })?;
    save_solution(&solution)?;
    Ok(())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let cfg = Config::from_hp(&None, challenge);
    let planner = Planner::new(challenge, cfg);
    planner.policy(challenge, state)
}

struct Planner {
    cfg: Config,
    value_tables: Vec<BatteryValue>,
    shadow_baseline_table: Vec<Vec<f64>>,
    shadow_congested_table: Vec<Vec<f64>>,
    shadow_multiday_table: Vec<Vec<f64>>,
    shadow_dense_table: Vec<Vec<f64>>,
    shadow_capstone_table: Vec<Vec<f64>>,
    exp_shadow_baseline_table: Vec<Vec<f64>>,
    exp_shadow_congested_table: Vec<Vec<f64>>,
    exp_shadow_multiday_table: Vec<Vec<f64>>,
    exp_shadow_dense_table: Vec<Vec<f64>>,
    exp_shadow_capstone_table: Vec<Vec<f64>>,
    expected_prices: Vec<Vec<f64>>,
    future_discharge_need_table: Vec<Vec<f64>>,
    future_charge_need_table: Vec<Vec<f64>>,
    use_support_dp: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrackKind {
    Baseline,
    Congested,
    Multiday,
    Dense,
    Capstone,
    Unknown,
}

#[derive(Clone, Debug)]
struct OnlineState {
    last_t: Option<usize>,
    observations: usize,
    ratio_sum: f64,
    spike_observations: usize,
    max_ratio: f64,
    node_ratio_sum: Vec<f64>,
    node_observations: Vec<usize>,
    node_spike_observations: Vec<usize>,
}

impl OnlineState {
    fn new() -> Self {
        Self {
            last_t: None,
            observations: 0,
            ratio_sum: 0.0,
            spike_observations: 0,
            max_ratio: 0.0,
            node_ratio_sum: Vec::new(),
            node_observations: Vec::new(),
            node_spike_observations: Vec::new(),
        }
    }

    fn observe(&mut self, challenge: &Challenge, state: &State) {
        if self.last_t == Some(state.time_step) {
            return;
        }
        self.last_t = Some(state.time_step);
        if state.time_step >= challenge.num_steps || state.rt_prices.is_empty() {
            return;
        }
        let n = challenge.network.num_nodes;
        if self.node_ratio_sum.len() < n {
            self.node_ratio_sum.resize(n, 0.0);
            self.node_observations.resize(n, 0);
            self.node_spike_observations.resize(n, 0);
        }
        let da = &challenge.market.day_ahead_prices[state.time_step];
        for i in 0..state.rt_prices.len().min(da.len()) {
            let denom = da[i].abs().max(1.0);
            let ratio = state.rt_prices[i] / denom;
            self.observations += 1;
            self.ratio_sum += ratio;
            self.node_ratio_sum[i] += ratio;
            self.node_observations[i] += 1;
            let is_spike = ratio > 1.45 || state.rt_prices[i] > da[i] + 35.0;
            if is_spike {
                self.spike_observations += 1;
                self.node_spike_observations[i] += 1;
            }
            if ratio > self.max_ratio {
                self.max_ratio = ratio;
            }
        }
    }

    fn spike_rate(&self) -> f64 {
        if self.observations == 0 {
            0.0
        } else {
            self.spike_observations as f64 / self.observations as f64
        }
    }

    fn avg_ratio(&self) -> f64 {
        if self.observations == 0 {
            1.0
        } else {
            self.ratio_sum / self.observations as f64
        }
    }

    fn node_scale(&self, node: usize) -> f64 {
        let local = if node < self.node_observations.len() && self.node_observations[node] >= 2 {
            self.node_ratio_sum[node] / self.node_observations[node] as f64
        } else {
            self.avg_ratio()
        };
        let spike_boost = if node < self.node_spike_observations.len()
            && node < self.node_observations.len()
            && self.node_observations[node] > 0
        {
            1.0 + 0.50
                * (self.node_spike_observations[node] as f64 / self.node_observations[node] as f64)
                    .clamp(0.0, 0.20)
        } else {
            1.0
        };
        (0.65 * local + 0.35 * self.avg_ratio()).clamp(0.75, 1.35) * spike_boost
    }
}

struct BatteryValue {
    min_soc: f64,
    step: f64,
    levels: usize,
    steps: usize,
    values: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Cand {
    density: f64,
    value: f64,
    battery: usize,
    action: f64,
}

#[derive(Clone)]
struct BeamState {
    actions: Vec<f64>,
    flows: Vec<f64>,
    score: f64,
}

fn push_ranked_candidate(v: &mut Vec<Cand>, cand: Cand, keep: usize) {
    v.push(cand);
    v.sort_by(|a, b| {
        b.density
            .partial_cmp(&a.density)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal))
    });
    let keep = keep.max(1);
    if v.len() > keep {
        v.truncate(keep);
    }
}

impl Planner {
    fn new(challenge: &Challenge, cfg: Config) -> Self {
        let use_support_dp =
            cfg.support_baseline && challenge.num_steps == 96 && challenge.num_batteries <= 12;

        let congestion_probs = if use_support_dp {
            build_congestion_probability_table(challenge)
        } else {
            Vec::new()
        };

        let mut value_tables = Vec::with_capacity(challenge.num_batteries);
        for b in 0..challenge.num_batteries {
            value_tables.push(BatteryValue::build(
                challenge,
                b,
                cfg,
                use_support_dp,
                &congestion_probs,
            ));
        }

        let look_cap = if cfg.short_horizon_enabled && challenge.num_steps > 96 {
            96
        } else {
            0
        };
        let shadow_baseline_table =
            build_future_quantile_table(challenge, cfg.q_baseline, look_cap);
        let shadow_congested_table =
            build_future_quantile_table(challenge, cfg.q_congested, look_cap);
        let shadow_multiday_table =
            build_future_quantile_table(challenge, cfg.q_multiday, look_cap);
        let shadow_dense_table = build_future_quantile_table(challenge, cfg.q_dense, look_cap);
        let shadow_capstone_table =
            build_future_quantile_table(challenge, cfg.q_capstone, look_cap);

        let expected_prices = build_expected_rt_price_table(challenge);
        let exp_shadow_baseline_table = build_future_quantile_table_from_values(
            challenge,
            &expected_prices,
            cfg.q_baseline,
            look_cap,
        );
        let exp_shadow_congested_table = build_future_quantile_table_from_values(
            challenge,
            &expected_prices,
            cfg.q_congested,
            look_cap,
        );
        let exp_shadow_multiday_table = build_future_quantile_table_from_values(
            challenge,
            &expected_prices,
            cfg.q_multiday,
            look_cap,
        );
        let exp_shadow_dense_table = build_future_quantile_table_from_values(
            challenge,
            &expected_prices,
            cfg.q_dense,
            look_cap,
        );
        let exp_shadow_capstone_table = build_future_quantile_table_from_values(
            challenge,
            &expected_prices,
            cfg.q_capstone,
            look_cap,
        );

        Self {
            cfg,
            value_tables,
            shadow_baseline_table,
            shadow_congested_table,
            shadow_multiday_table,
            shadow_dense_table,
            shadow_capstone_table,
            exp_shadow_baseline_table,
            exp_shadow_congested_table,
            exp_shadow_multiday_table,
            exp_shadow_dense_table,
            exp_shadow_capstone_table,
            expected_prices,
            future_discharge_need_table: build_future_network_reserve_table(challenge, cfg, true),
            future_charge_need_table: build_future_network_reserve_table(challenge, cfg, false),
            use_support_dp,
        }
    }

    fn policy(&self, challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        if self.cfg.force_shadow {
            return self.shadow_pack_policy(challenge, state);
        }
        if self.cfg.force_dp {
            return self.dp_pack_policy(challenge, state);
        }
        if self.cfg.force_hybrid {
            return self.hybrid_select_policy(challenge, state);
        }

        // Dimension-based routing maps to TIG tracks from observable instance dimensions.
        // BASELINE gets the stochastic support DP selector.
        if self.use_support_dp {
            return self.hybrid_select_policy(challenge, state);
        }

        // CONGESTED: H=96, m=20. The tuned shadow route is less brittle under line constraints.
        if self.cfg.congested_shadow && challenge.num_steps == 96 && challenge.num_batteries >= 15 {
            return self.shadow_pack_policy(challenge, state);
        }

        // MULTIDAY: H=192, m=40. Shadow routing won more surrogate seeds than DP selection.
        if self.cfg.multiday_shadow && challenge.num_steps > 96 && challenge.num_batteries <= 40 {
            return self.shadow_pack_policy(challenge, state);
        }

        // DENSE and CAPSTONE: compare candidate vectors with DP continuation value.
        self.hybrid_select_policy(challenge, state)
    }

    fn policy_dynamic(
        &self,
        challenge: &Challenge,
        state: &State,
        online: &OnlineState,
    ) -> Result<Vec<f64>> {
        if self.cfg.force_shadow || self.cfg.force_dp || self.cfg.force_hybrid {
            return self.policy(challenge, state);
        }

        let track = detect_track(challenge);
        let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(14);

        // Always include the routed DA-shadow vector. This is the robust backbone.
        candidates.push(self.shadow_pack_policy(challenge, state)?);

        // Expected-RT shadow is useful when jump/congestion parameters dominate DA seasonality.
        if self.cfg.expected_shadow_enabled {
            let table = self.routed_expected_shadow_table(challenge);
            candidates.push(self.shadow_pack_policy_with_table(challenge, state, table)?);
        }

        // RT-spike vector deliberately discounts inventory value on obvious current spikes.
        candidates.push(self.spike_capture_policy(challenge, state, online)?);

        // Explicit inventory target from current RT percentile versus future expected prices.
        // This adds a planning abstraction that the marginal shadow policies miss: move toward
        // high SOC when current RT is unusually cheap, low SOC when it is unusually dear, with
        // reserve and terminal adjustments.
        if self.cfg.inventory_target_enabled {
            candidates.push(self.inventory_target_policy(challenge, state, online)?);
        }

        // Active-set Lagrangian repricing estimates shadow prices for stressed PTDF cuts,
        // reprices nodal battery actions, packs the resulting auction, and updates cut prices
        // from repaired portfolio flows.
        if self.cfg.active_set_enabled && self.cfg.active_set_passes > 0 {
            candidates.push(self.active_set_reprice_policy(challenge, state, online)?);
        }

        match track {
            TrackKind::Baseline => {
                candidates.push(self.dp_pack_policy(challenge, state)?);
                candidates.push(self.rt_only_pack_policy(challenge, state)?);
                if self.cfg.include_baseline_candidates {
                    candidates.push(greedy_like_policy(challenge, state));
                    candidates.push(conservative_like_policy(challenge, state));
                }
            }
            TrackKind::Congested => {
                // Congested is sensitive to over-reserving; compare DA and expected tables, but avoid heavy DP.
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.shadow_baseline_table,
                )?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.exp_shadow_congested_table,
                )?);
                if online.max_ratio > 1.8 || online.spike_rate() > 0.025 {
                    candidates.push(self.rt_only_pack_policy(challenge, state)?);
                }
            }
            TrackKind::Multiday => {
                // Multi-day benefits from lower q cycling; include capstone-low and dense-mid alternatives.
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.shadow_capstone_table,
                )?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.shadow_dense_table,
                )?);
                if online.avg_ratio() > 1.05 || online.spike_rate() > 0.02 {
                    candidates.push(self.shadow_pack_policy_with_table(
                        challenge,
                        state,
                        &self.exp_shadow_multiday_table,
                    )?);
                }
            }
            TrackKind::Dense => {
                candidates.push(self.dp_pack_policy(challenge, state)?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.shadow_multiday_table,
                )?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.shadow_capstone_table,
                )?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.exp_shadow_dense_table,
                )?);
                if online.max_ratio > 1.7 || online.spike_rate() > 0.025 {
                    candidates.push(self.rt_only_pack_policy(challenge, state)?);
                }
            }
            TrackKind::Capstone | TrackKind::Unknown => {
                candidates.push(self.dp_pack_policy(challenge, state)?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.shadow_multiday_table,
                )?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.shadow_dense_table,
                )?);
                candidates.push(self.shadow_pack_policy_with_table(
                    challenge,
                    state,
                    &self.exp_shadow_capstone_table,
                )?);
                candidates.push(self.rt_only_pack_policy(challenge, state)?);
            }
        }

        candidates.push(vec![0.0f64; challenge.num_batteries]);

        Ok(self.select_best_action(challenge, state, candidates, Some(online)))
    }

    fn inventory_target_policy(
        &self,
        challenge: &Challenge,
        state: &State,
        online: &OnlineState,
    ) -> Result<Vec<f64>> {
        let n_bat = challenge.num_batteries;
        let mut actions = vec![0.0f64; n_bat];

        let zero = vec![0.0f64; n_bat];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let mut cur_flows = base_flows.clone();

        let mut cands: Vec<Cand> = Vec::with_capacity(n_bat * self.cfg.alts_per_battery.max(2));
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
        let look = self.cfg.target_lookahead.min(rem.max(1));
        let t = state.time_step.min(challenge.num_steps.saturating_sub(1));
        let table = self.routed_expected_shadow_table(challenge);

        for i in 0..n_bat {
            let bat = &challenge.batteries[i];
            let node = bat.node;
            let (lo, hi) = state.action_bounds[i];
            if hi <= 1e-9 && lo >= -1e-9 {
                continue;
            }

            let rt_eff =
                self.effective_rt_price(challenge, &base_flows, node, state.rt_prices[node]);
            let rank =
                future_price_rank(&self.expected_prices, state.time_step, node, rt_eff, look);
            let mut target_frac = 1.0 - rank;

            // Avoid churn around the median: when price is close to its future waterline,
            // target the current SOC fraction and let the other candidate families decide.
            if (rank - 0.50).abs() < self.cfg.target_mid_band {
                let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
                target_frac = ((state.socs[i] - bat.soc_min_mwh) / span).clamp(0.0, 1.0);
            }

            // Reserve bias: if future network states indicate scarce discharge capacity at
            // this node, protect more SOC; if future charge need dominates, preserve headroom.
            if self.cfg.network_reserve_enabled
                && t < self.future_discharge_need_table.len()
                && node < self.future_discharge_need_table[t].len()
            {
                let discharge_need = self.future_discharge_need_table[t][node].max(0.0);
                let charge_need = self.future_charge_need_table[t][node].max(0.0);
                let reserve_bias = ((discharge_need - charge_need)
                    / (self.cfg.reserve_price_scale + 1.0))
                    .clamp(-0.22, 0.22);
                target_frac = (target_frac + reserve_bias).clamp(0.0, 1.0);
            }

            // Terminal liquidation cap: as expiry approaches, force the waterline down so
            // energy does not strand after the last executable step.
            if self.cfg.endgame_window > 0 && rem < self.cfg.endgame_window {
                let rem_frac = rem as f64 / self.cfg.endgame_window.max(1) as f64;
                let cap = rem_frac.max(self.cfg.endgame_floor * 0.50).clamp(0.0, 1.0);
                target_frac = target_frac.min(cap);
            }

            let span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
            let target_soc = bat.soc_min_mwh + span * target_frac.clamp(0.0, 1.0);
            let gap = target_soc - state.socs[i];
            let desired = if gap > 0.0 {
                -(gap / (bat.efficiency_charge.max(1e-9) * constants::DELTA_T)).abs()
            } else {
                ((-gap) * bat.efficiency_discharge / constants::DELTA_T).abs()
            };
            let desired = desired.clamp(lo, hi);
            if desired.abs() < 1e-9 {
                continue;
            }

            let future_mid = table[t][node].max(0.0);
            let future_mean =
                future_price_mean(&self.expected_prices, state.time_step, node, look).max(0.0);
            let spread_value = (future_mid - rt_eff)
                .abs()
                .max((future_mean - rt_eff).abs())
                .max(3.0);
            let current_dist = (state.socs[i] - target_soc).abs();
            let mut local: Vec<Cand> = Vec::new();

            for frac in [0.35_f64, 0.65_f64, 1.0_f64, 1.25_f64].iter() {
                let u = (desired * *frac).clamp(lo, hi);
                if u.abs() < 1e-9 {
                    continue;
                }
                let new_soc = apply_action_to_soc_manual(bat, u, state.socs[i]);
                let dist_gain = (current_dist - (new_soc - target_soc).abs()).max(0.0);
                let mut value = profit_at_price(bat, u, rt_eff)
                    + self.cfg.target_policy_weight * dist_gain * spread_value;
                value += self.network_action_bonus(challenge, &base_flows, node, u);
                value += self.reserve_action_bonus(challenge, state, i, u);
                value +=
                    self.terminal_pressure_action_bonus(challenge, state, i, u, rt_eff.max(0.0));

                // When a node has repeatedly shown positive RT/DA bias, tolerate slightly
                // more sell-side urgency; when it is cheap-biased, tolerate more charge-side
                // inventory building.
                if self.cfg.node_adaptation_enabled {
                    let node_bias = online.node_scale(node) - 1.0;
                    if (node_bias > 0.05 && u > 0.0) || (node_bias < -0.05 && u < 0.0) {
                        value += node_bias.abs() * spread_value * dist_gain * 0.35;
                    }
                }

                if value <= self.cfg.min_value {
                    continue;
                }
                let usage = stress_usage(challenge, &base_flows, node, u);
                push_ranked_candidate(
                    &mut local,
                    Cand {
                        density: value / (self.cfg.density_eps + usage),
                        value,
                        battery: i,
                        action: u,
                    },
                    self.cfg.alts_per_battery.max(2),
                );
            }
            cands.extend(local.into_iter());
        }

        self.pack_candidates(challenge, state, &mut actions, &mut cur_flows, cands)
    }

    fn active_set_reprice_policy(
        &self,
        challenge: &Challenge,
        state: &State,
        online: &OnlineState,
    ) -> Result<Vec<f64>> {
        let n_bat = challenge.num_batteries;
        if n_bat == 0 || self.cfg.active_set_battery_cap == 0 || self.cfg.active_set_passes == 0 {
            return Ok(vec![0.0f64; n_bat]);
        }

        let zero = vec![0.0f64; n_bat];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let table = self.routed_expected_shadow_table(challenge);
        let t = state.time_step.min(challenge.num_steps.saturating_sub(1));
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);

        let mut duals = vec![0.0f64; challenge.network.num_lines];
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let stress = base_flows[l] / lim;
            let mag = stress.abs();
            if mag > self.cfg.active_set_stress_threshold {
                let excess = mag - self.cfg.active_set_stress_threshold;
                duals[l] = stress.signum() * self.cfg.active_set_dual_step * excess * excess;
            }
        }

        let mut best = zero.clone();
        let mut best_score = self.vector_objective_dynamic(challenge, state, &best, online)
            + 1e-9 * challenge.compute_profit(state, &best);

        // Prioritize batteries whose nodal price/spread and PTDF footprint make them
        // useful to the active cut auction.  Large tracks cap this list for fuel.
        let mut order: Vec<(f64, usize)> = Vec::with_capacity(n_bat);
        for i in 0..n_bat {
            let bat = &challenge.batteries[i];
            let node = bat.node;
            let (lo, hi) = state.action_bounds[i];
            let room = (hi - lo).abs();
            if room <= 1e-12 {
                continue;
            }
            let future = table[t][node];
            let rt = state.rt_prices[node];
            let spread = (rt - future).abs();
            let footprint = stress_usage(challenge, &base_flows, node, hi.abs().max(lo.abs()));
            let online_bias = (online.node_scale(node) - 1.0).abs();
            let priority =
                room * (1.0 + spread / 35.0) * (1.0 + 1.6 * footprint) * (1.0 + online_bias);
            order.push((priority, i));
        }
        order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        let cap = self.cfg.active_set_battery_cap.min(order.len());

        for pass in 0..self.cfg.active_set_passes {
            let mut actions = vec![0.0f64; n_bat];
            let mut cur_flows = base_flows.clone();
            let mut cands: Vec<Cand> = Vec::with_capacity(cap * self.cfg.active_set_grid.max(3));

            for &(_, i) in order.iter().take(cap) {
                let bat = &challenge.batteries[i];
                let node = bat.node;
                let (lo, hi) = state.action_bounds[i];
                if hi <= 1e-9 && lo >= -1e-9 {
                    continue;
                }

                let mut nodal_cut_price = 0.0f64;
                if node != challenge.network.slack_bus {
                    for l in 0..challenge.network.num_lines {
                        nodal_cut_price += duals[l] * challenge.network.ptdf[l][node];
                    }
                }
                let mut rt_eff =
                    self.effective_rt_price(challenge, &base_flows, node, state.rt_prices[node]);
                rt_eff -= nodal_cut_price;

                let future_price = table[t][node];
                let grid = action_candidates(lo, hi, self.cfg.active_set_grid);
                let mut local: Vec<Cand> = Vec::new();
                for u in grid.into_iter() {
                    if u.abs() < 1e-10 {
                        continue;
                    }
                    let mut value = shadow_action_value_ctx(
                        bat,
                        u,
                        rt_eff,
                        future_price,
                        self.cfg.shadow_inventory_weight,
                        state.socs[i],
                        self.cfg,
                        rem,
                    );
                    value += self.network_action_bonus(challenge, &base_flows, node, u);
                    value += self.reserve_action_bonus(challenge, state, i, u);
                    value += self.terminal_pressure_action_bonus(
                        challenge,
                        state,
                        i,
                        u,
                        rt_eff.max(0.0),
                    );

                    // Small exploration premium in later passes for actions that are explicitly
                    // counter-directional to the active cuts.  This gives the auction room to
                    // discover netted portfolios that greedy packing would reject.
                    if pass > 0 && node != challenge.network.slack_bus {
                        let mut relief = 0.0f64;
                        for l in 0..challenge.network.num_lines {
                            let lim = challenge.network.flow_limits[l].max(1e-9);
                            let before = (cur_flows[l] / lim).abs();
                            let after =
                                ((cur_flows[l] + challenge.network.ptdf[l][node] * u) / lim).abs();
                            relief += (before - after).max(0.0);
                        }
                        value += self.cfg.relief_bonus * 0.35 * relief;
                    }

                    if value <= self.cfg.min_value {
                        continue;
                    }
                    let usage = stress_usage(challenge, &base_flows, node, u);
                    push_ranked_candidate(
                        &mut local,
                        Cand {
                            density: value / (self.cfg.density_eps + usage),
                            value,
                            battery: i,
                            action: u,
                        },
                        self.cfg.alts_per_battery.max(2),
                    );
                }
                cands.extend(local.into_iter());
            }

            let candidate =
                self.pack_candidates(challenge, state, &mut actions, &mut cur_flows, cands)?;
            let candidate = repair_global_scale(challenge, state, candidate);
            let score = self.vector_objective_dynamic(challenge, state, &candidate, online)
                + 1e-9 * challenge.compute_profit(state, &candidate);
            if score > best_score {
                best_score = score;
                best = candidate.clone();
            }

            // Reprice the active cuts from the repaired candidate.  Unlike the static
            // nodal-dual adder, this loop updates the congestion market after seeing
            // the portfolio it just induced.
            let inj = challenge.compute_total_injections(state, &candidate);
            let flows = challenge.network.compute_flows(&inj);
            for l in 0..challenge.network.num_lines {
                let lim = challenge.network.flow_limits[l].max(1e-9);
                let stress = flows[l] / lim;
                let mag = stress.abs();
                if mag > self.cfg.active_set_stress_threshold {
                    let excess = mag - self.cfg.active_set_stress_threshold;
                    let update = stress.signum() * self.cfg.active_set_dual_step * excess * excess;
                    duals[l] = self.cfg.active_set_dual_decay * duals[l] + update;
                } else {
                    duals[l] *= self.cfg.active_set_dual_decay * 0.55;
                }
            }
        }

        Ok(repair_global_scale(challenge, state, best))
    }

    fn augment_candidate_blends(&self, mut candidates: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let k = self.cfg.blend_top_k.min(candidates.len());
        if k < 2 {
            return candidates;
        }
        let base = candidates[..k].to_vec();
        let alphas = [self.cfg.blend_alpha, 0.50, 1.0 - self.cfg.blend_alpha];
        for i in 0..k {
            for j in (i + 1)..k {
                if base[i].len() != base[j].len() {
                    continue;
                }
                for &a in alphas.iter() {
                    let b = 1.0 - a;
                    let mut x = Vec::with_capacity(base[i].len());
                    for idx in 0..base[i].len() {
                        x.push(a * base[i][idx] + b * base[j][idx]);
                    }
                    candidates.push(x);
                }
            }
        }
        candidates
    }

    fn select_best_action(
        &self,
        challenge: &Challenge,
        state: &State,
        candidates: Vec<Vec<f64>>,
        online: Option<&OnlineState>,
    ) -> Vec<f64> {
        if candidates.is_empty() {
            return vec![0.0f64; challenge.num_batteries];
        }

        // Score and repair policy vectors before building convex blends from the highest-scoring
        // repaired vectors. Blending by insertion order can waste blend slots on weak candidates
        // and miss useful RT-vs-inventory compromises.
        let score_repaired = |raw: Vec<f64>| -> (f64, Vec<f64>) {
            let repaired = repair_global_scale(challenge, state, raw);
            let one_step = match online {
                Some(o) => self.vector_objective_dynamic(challenge, state, &repaired, o),
                None => {
                    self.vector_objective(challenge, state, &repaired)
                        + 1e-9 * challenge.compute_profit(state, &repaired)
                }
            };
            (one_step, repaired)
        };

        let mut scored: Vec<(f64, Vec<f64>)> = candidates
            .into_iter()
            .map(|a| score_repaired(a))
            .filter(|(s, a)| s.is_finite() && a.len() == challenge.num_batteries)
            .collect();

        if scored.is_empty() {
            return vec![0.0f64; challenge.num_batteries];
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Add portfolio consensus actions before pairwise convex blends. The top repaired/scored
        // policies are treated as a small action distribution; the weighted barycenter and
        // coordinate median are robust middle points.
        if self.cfg.portfolio_consensus_enabled {
            let k = self.cfg.consensus_top_k.min(scored.len());
            if k >= 2 {
                let mut consensus = vec![0.0f64; challenge.num_batteries];
                let mut total_w = 0.0f64;
                let mut w = 1.0f64;
                for (_, a) in scored.iter().take(k) {
                    for i in 0..challenge.num_batteries {
                        consensus[i] += w * a[i];
                    }
                    total_w += w;
                    w *= self.cfg.consensus_weight_decay;
                }
                if total_w > 1e-12 {
                    for u in consensus.iter_mut() {
                        *u /= total_w;
                    }
                    let pair = score_repaired(consensus);
                    if pair.0.is_finite() {
                        scored.push(pair);
                    }
                }

                let mut median_vec = vec![0.0f64; challenge.num_batteries];
                for i in 0..challenge.num_batteries {
                    let mut vals: Vec<f64> = scored.iter().take(k).map(|(_, a)| a[i]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    median_vec[i] = vals[vals.len() / 2];
                }
                let pair = score_repaired(median_vec);
                if pair.0.is_finite() {
                    scored.push(pair);
                }
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            }
        }

        if self.cfg.blend_candidates_enabled {
            let k = self.cfg.blend_top_k.min(scored.len());
            if k >= 2 {
                let base: Vec<Vec<f64>> = scored.iter().take(k).map(|(_, a)| a.clone()).collect();
                let alphas = [
                    self.cfg.blend_alpha,
                    0.50,
                    1.0 - self.cfg.blend_alpha,
                    0.25,
                    0.75,
                ];
                let mut blends: Vec<(f64, Vec<f64>)> = Vec::new();
                for i in 0..k {
                    for j in (i + 1)..k {
                        if base[i].len() != base[j].len() {
                            continue;
                        }
                        for &a in alphas.iter() {
                            let b = 1.0 - a;
                            let mut x = Vec::with_capacity(base[i].len());
                            for idx in 0..base[i].len() {
                                x.push(a * base[i][idx] + b * base[j][idx]);
                            }
                            let pair = score_repaired(x);
                            if pair.0.is_finite() {
                                blends.push(pair);
                            }
                        }
                    }
                }
                scored.extend(blends.into_iter());
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            }
        }

        // Frontier optimizer: treat the top repaired/scored policy vectors as a small basis and
        // optimize inside their convex hull. Pairwise blends and consensus vectors are useful but
        // still discrete; this performs a simplex exchange search over the policy frontier.
        // Because the basis vectors are repaired first, convex combinations remain within battery
        // bounds and PTDF feasibility up to numerical tolerance.
        if self.cfg.frontier_optimizer_enabled && self.cfg.frontier_passes > 0 {
            let k = self.cfg.frontier_top_k.min(scored.len());
            if k >= 2 {
                let bases: Vec<Vec<f64>> = scored.iter().take(k).map(|(_, a)| a.clone()).collect();
                let x = self.frontier_simplex_optimize(challenge, state, &bases, online);
                let pair = score_repaired(x);
                if pair.0.is_finite() {
                    scored.push(pair);
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                }
            }
        }

        if !self.cfg.rollout_enabled || self.cfg.rollout_horizon <= 1 {
            return self.selector_refine_action(challenge, state, scored[0].1.clone(), online);
        }

        // Rollout only the top repaired/scored portfolio candidates.  This keeps
        // CAPSTONE fuel under control while still letting the final selector
        // compare blends and auction/pairflow vectors on expected future paths.
        let cap = self.cfg.rollout_candidate_cap.min(scored.len());
        let mut best_action = scored[0].1.clone();
        let mut best_score = f64::NEG_INFINITY;
        for (idx, (one_step, action)) in scored.iter().enumerate() {
            let score = if idx < cap {
                match online {
                    Some(o) => self.rollout_objective(challenge, state, action, o),
                    None => *one_step,
                }
            } else {
                *one_step
            };
            if score > best_score {
                best_score = score;
                best_action = action.clone();
            }
        }
        self.selector_refine_action(challenge, state, best_action, online)
    }

    fn frontier_simplex_optimize(
        &self,
        challenge: &Challenge,
        state: &State,
        bases: &[Vec<f64>],
        online: Option<&OnlineState>,
    ) -> Vec<f64> {
        let k = self.cfg.frontier_top_k.min(bases.len());
        if k == 0 || challenge.num_batteries == 0 {
            return vec![0.0f64; challenge.num_batteries];
        }
        if k == 1 {
            return bases[0].clone();
        }

        let n = challenge.num_batteries;
        let eval = |x: &[f64]| -> (f64, Vec<f64>) {
            let repaired = repair_global_scale(challenge, state, x.to_vec());
            let mut score = match online {
                Some(o) => self.vector_objective_dynamic(challenge, state, &repaired, o),
                None => {
                    self.vector_objective(challenge, state, &repaired)
                        + 1e-9 * challenge.compute_profit(state, &repaired)
                }
            };
            if !score.is_finite() {
                score = f64::NEG_INFINITY;
            }
            (score, repaired)
        };

        let mut weights = vec![0.0f64; k];
        weights[0] = 1.0;
        let (mut best_score, mut current) = eval(&bases[0]);
        let mut step = self.cfg.frontier_step.clamp(0.05, 0.90);

        for _ in 0..self.cfg.frontier_passes {
            let mut improved = false;

            for src in 0..k {
                if weights[src] <= 1e-9 {
                    continue;
                }
                for dst in 0..k {
                    if src == dst {
                        continue;
                    }
                    let deltas = [step, 0.5 * step, 0.25 * step];
                    for &d0 in deltas.iter() {
                        let delta = d0.min(weights[src]);
                        if delta <= 1e-7 {
                            continue;
                        }
                        let mut cand_weights = weights.clone();
                        cand_weights[src] -= delta;
                        cand_weights[dst] += delta;

                        let mut x = vec![0.0f64; n];
                        for b in 0..k {
                            let w = cand_weights[b];
                            if w.abs() < 1e-12 {
                                continue;
                            }
                            for i in 0..n {
                                x[i] += w * bases[b][i];
                            }
                        }

                        let (score, repaired) = eval(&x);
                        if score > best_score + 1e-7 {
                            best_score = score;
                            current = repaired;
                            weights = cand_weights;
                            improved = true;
                        }
                    }
                }
            }

            // Also test a smooth move toward the current rank-weighted barycenter.
            // This catches cases where several weaker-looking families together
            // dominate any one transfer direction.  Keep the simplex weights in
            // sync with the accepted barycentric move.
            let mut bary_weights = vec![0.0f64; k];
            let mut total_w = 0.0f64;
            let mut w = 1.0f64;
            for b in 0..k {
                bary_weights[b] = w;
                total_w += w;
                w *= self.cfg.consensus_weight_decay;
            }
            if total_w > 1e-12 {
                let mut cand_weights = weights.clone();
                for b in 0..k {
                    let target_w = bary_weights[b] / total_w;
                    cand_weights[b] = (1.0 - step) * cand_weights[b] + step * target_w;
                }
                let mut bary = vec![0.0f64; n];
                for b in 0..k {
                    let wb = cand_weights[b];
                    if wb.abs() < 1e-12 {
                        continue;
                    }
                    for i in 0..n {
                        bary[i] += wb * bases[b][i];
                    }
                }
                let (score, repaired) = eval(&bary);
                if score > best_score + 1e-7 {
                    best_score = score;
                    current = repaired;
                    weights = cand_weights;
                    improved = true;
                }
            }

            if improved {
                step *= 0.82;
            } else {
                step *= 0.50;
            }
            if step < 0.0125 {
                break;
            }
        }

        current
    }

    fn selector_refine_action(
        &self,
        challenge: &Challenge,
        state: &State,
        actions: Vec<f64>,
        online: Option<&OnlineState>,
    ) -> Vec<f64> {
        if !self.cfg.selector_refine_enabled
            || self.cfg.selector_refine_passes == 0
            || self.cfg.selector_refine_battery_cap == 0
            || challenge.num_batteries == 0
        {
            return actions;
        }

        let Some(o) = online else {
            return actions;
        };

        let mut best = repair_global_scale(challenge, state, actions);
        let mut best_score = self.vector_objective_dynamic(challenge, state, &best, o)
            + 1e-9 * challenge.compute_profit(state, &best);

        let zero = vec![0.0f64; challenge.num_batteries];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let mut order: Vec<(f64, usize)> = Vec::with_capacity(challenge.num_batteries);
        for i in 0..challenge.num_batteries {
            let bat = &challenge.batteries[i];
            let node = bat.node;
            let price = state.rt_prices.get(node).copied().unwrap_or(0.0).abs();
            let (lo, hi) = state.action_bounds[i];
            let room = (hi - lo).abs();
            let stress = stress_usage(challenge, &base_flows, node, best[i]);
            let priority =
                (best[i].abs() + 0.15 * room) * (1.0 + price / 100.0) * (1.0 + 2.0 * stress);
            order.push((priority, i));
        }
        order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        let cap = self.cfg.selector_refine_battery_cap.min(order.len());

        for _ in 0..self.cfg.selector_refine_passes {
            let mut changed = false;
            for &(_, i) in order.iter().take(cap) {
                let cur = best[i];
                let (lo, hi) = state.action_bounds[i];
                let mut trials = Vec::with_capacity(14);
                for x in [
                    0.0,
                    cur * 0.35,
                    cur * 0.65,
                    cur * 0.85,
                    cur * 1.10,
                    cur * 1.30,
                    lo * 0.25,
                    lo * 0.50,
                    lo * 0.85,
                    lo,
                    hi * 0.25,
                    hi * 0.50,
                    hi * 0.85,
                    hi,
                ] {
                    push_unique(&mut trials, x.clamp(lo, hi));
                }

                let mut local_best = cur;
                let mut local_best_score = best_score;
                for u in trials {
                    if (u - cur).abs() < 1e-9 {
                        continue;
                    }
                    let mut trial = best.clone();
                    trial[i] = u;
                    trial = repair_global_scale(challenge, state, trial);
                    if !is_action_feasible(challenge, state, &trial) {
                        continue;
                    }
                    let score = self.vector_objective_dynamic(challenge, state, &trial, o)
                        + 1e-9 * challenge.compute_profit(state, &trial);
                    if score > local_best_score + 1e-7 {
                        local_best_score = score;
                        local_best = trial[i];
                        best = trial;
                    }
                }
                if (local_best - cur).abs() > 1e-9 {
                    best_score = local_best_score;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        repair_global_scale(challenge, state, best)
    }

    fn rollout_objective(
        &self,
        challenge: &Challenge,
        state: &State,
        first_action: &[f64],
        online: &OnlineState,
    ) -> f64 {
        if self.cfg.ensemble_rollout_enabled {
            return self.rollout_objective_ensemble(challenge, state, first_action, online);
        }
        let mut total = challenge.compute_profit(state, first_action);
        if self.cfg.rollout_horizon <= 1 || state.time_step + 1 >= challenge.num_steps {
            return total + self.terminal_inventory_value(challenge, state, first_action, online);
        }

        let next_prices = self.forecast_prices_for_step(challenge, state.time_step + 1, online);
        let mut sim_state =
            match challenge.take_step(state, first_action, NextRTPrices::Override(next_prices)) {
                Ok(s) => s,
                Err(_) => return f64::NEG_INFINITY,
            };

        for _ in 1..self.cfg.rollout_horizon {
            if sim_state.time_step >= challenge.num_steps {
                break;
            }
            let table = self.routed_expected_shadow_table(challenge);
            let action = match self.shadow_pack_policy_with_table(challenge, &sim_state, table) {
                Ok(a) => a,
                Err(_) => return f64::NEG_INFINITY,
            };
            total += challenge.compute_profit(&sim_state, &action);
            if sim_state.time_step + 1 >= challenge.num_steps {
                break;
            }
            let prices = self.forecast_prices_for_step(challenge, sim_state.time_step + 1, online);
            sim_state =
                match challenge.take_step(&sim_state, &action, NextRTPrices::Override(prices)) {
                    Ok(s) => s,
                    Err(_) => return f64::NEG_INFINITY,
                };
        }

        total + self.terminal_state_value(challenge, &sim_state, online)
    }

    fn rollout_objective_ensemble(
        &self,
        challenge: &Challenge,
        state: &State,
        first_action: &[f64],
        online: &OnlineState,
    ) -> f64 {
        // Use a deterministic forecast ensemble and adapt the mean/worst/upside weights to the
        // current regime. High line stress and large-track regimes get more worst-case weight;
        // terminal windows and observed RT spikes get more upside/liquidation weight.
        let mut scores = Vec::with_capacity(4);
        for mode in 0..4 {
            let score = self.rollout_objective_path(challenge, state, first_action, online, mode);
            if score.is_finite() {
                scores.push(score);
            }
        }
        if scores.is_empty() {
            return f64::NEG_INFINITY;
        }
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
        let worst = scores[0];
        let best = *scores.last().unwrap_or(&mean);

        let track = detect_track(challenge);
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
        let zero = vec![0.0f64; challenge.num_batteries];
        let inj = challenge.compute_total_injections(state, &zero);
        let flows = challenge.network.compute_flows(&inj);
        let (_, stress) = most_stressed_line(challenge, &flows);

        let mut robust = self.cfg.rollout_robust_weight;
        let mut upside = self.cfg.rollout_upside_weight;

        match track {
            TrackKind::Baseline => {
                robust *= 0.80;
                upside *= 0.75;
            }
            TrackKind::Congested | TrackKind::Multiday => {
                robust += 0.04;
            }
            TrackKind::Dense | TrackKind::Capstone | TrackKind::Unknown => {
                robust += 0.08;
                upside += 0.02;
            }
        }

        if stress > 0.90 {
            robust += 0.12;
            upside *= 0.90;
        } else if stress < 0.55 {
            robust *= 0.75;
        }

        if online.spike_rate() > 0.025 || online.max_ratio > 1.75 {
            upside += 0.06;
            robust *= 0.92;
        }

        if self.cfg.endgame_window > 0 && rem < self.cfg.endgame_window {
            let end_frac = 1.0 - rem as f64 / self.cfg.endgame_window.max(1) as f64;
            robust *= 1.0 - 0.35 * end_frac;
            upside += 0.06 * end_frac;
        }

        robust = robust.clamp(0.0, 0.80);
        upside = upside.clamp(0.0, 0.75);
        let mean_weight = (1.0 - robust).max(0.0);

        mean_weight * mean + robust * worst + upside * (best - mean).max(0.0)
    }

    fn rollout_objective_path(
        &self,
        challenge: &Challenge,
        state: &State,
        first_action: &[f64],
        online: &OnlineState,
        mode: usize,
    ) -> f64 {
        let mut total = challenge.compute_profit(state, first_action);
        if self.cfg.rollout_horizon <= 1 || state.time_step + 1 >= challenge.num_steps {
            return total + self.terminal_inventory_value(challenge, state, first_action, online);
        }

        let next_prices =
            self.forecast_prices_for_step_mode(challenge, state.time_step + 1, online, mode);
        let mut sim_state =
            match challenge.take_step(state, first_action, NextRTPrices::Override(next_prices)) {
                Ok(s) => s,
                Err(_) => return f64::NEG_INFINITY,
            };

        for _ in 1..self.cfg.rollout_horizon {
            if sim_state.time_step >= challenge.num_steps {
                break;
            }
            let table = match mode {
                1 => self.routed_shadow_table(challenge),
                2 => self.routed_expected_shadow_table(challenge),
                3 => self.routed_shadow_table(challenge),
                _ => self.routed_expected_shadow_table(challenge),
            };
            let action = match self.shadow_pack_policy_with_table(challenge, &sim_state, table) {
                Ok(a) => a,
                Err(_) => return f64::NEG_INFINITY,
            };
            total += challenge.compute_profit(&sim_state, &action);
            if sim_state.time_step + 1 >= challenge.num_steps {
                break;
            }
            let prices = self.forecast_prices_for_step_mode(
                challenge,
                sim_state.time_step + 1,
                online,
                mode,
            );
            sim_state =
                match challenge.take_step(&sim_state, &action, NextRTPrices::Override(prices)) {
                    Ok(s) => s,
                    Err(_) => return f64::NEG_INFINITY,
                };
        }

        total + self.terminal_state_value(challenge, &sim_state, online)
    }

    fn forecast_prices_for_step_mode(
        &self,
        challenge: &Challenge,
        t: usize,
        online: &OnlineState,
        mode: usize,
    ) -> Vec<f64> {
        let n = challenge.network.num_nodes;
        if t >= self.expected_prices.len() || t >= challenge.market.day_ahead_prices.len() {
            return vec![0.0; n];
        }
        let mut out = Vec::with_capacity(n);
        let spike_rate = online.spike_rate().clamp(0.0, 0.15);
        for node in 0..n {
            let scale = if self.cfg.node_adaptation_enabled {
                online.node_scale(node)
            } else {
                online.avg_ratio().clamp(0.85, 1.25)
            };
            let da = challenge.market.day_ahead_prices[t][node];
            let exp = self.expected_prices[t][node] * scale;
            let price = match mode {
                // Downside: cheap RT, useful for testing whether charging actions still make sense.
                1 => (0.82 * exp + 0.18 * da) * (0.88 - 0.25 * spike_rate).clamp(0.72, 0.95),
                // Upside: deterministic spike surrogate, larger on historically spiky nodes.
                2 => {
                    let local_scale = if self.cfg.node_adaptation_enabled {
                        online.node_scale(node).clamp(0.90, 1.45)
                    } else {
                        scale.clamp(0.90, 1.35)
                    };
                    exp * (1.08 + 1.35 * spike_rate) + da.abs().max(1.0) * 0.10 * local_scale
                }
                // DA reserve path: less noisy and more inventory-preserving.
                3 => 0.55 * da + 0.45 * exp,
                _ => exp,
            };
            out.push(clamp_price(price));
        }
        out
    }

    fn forecast_prices_for_step(
        &self,
        challenge: &Challenge,
        t: usize,
        online: &OnlineState,
    ) -> Vec<f64> {
        let n = challenge.network.num_nodes;
        if t >= self.expected_prices.len() {
            return vec![0.0; n];
        }
        let mut out = Vec::with_capacity(n);
        for node in 0..n {
            let scale = if self.cfg.node_adaptation_enabled {
                online.node_scale(node)
            } else {
                online.avg_ratio().clamp(0.85, 1.25)
            };
            out.push(clamp_price(self.expected_prices[t][node] * scale));
        }
        out
    }

    fn terminal_inventory_value(
        &self,
        challenge: &Challenge,
        state: &State,
        action: &[f64],
        online: &OnlineState,
    ) -> f64 {
        if state.time_step >= challenge.num_steps {
            return 0.0;
        }
        let mut shadow_state = state.clone();
        for i in 0..challenge.num_batteries {
            shadow_state.socs[i] =
                apply_action_to_soc_manual(&challenge.batteries[i], action[i], state.socs[i]);
        }
        self.terminal_state_value(challenge, &shadow_state, online)
    }

    fn terminal_state_value(
        &self,
        challenge: &Challenge,
        state: &State,
        online: &OnlineState,
    ) -> f64 {
        if state.time_step >= challenge.num_steps || state.socs.is_empty() {
            return 0.0;
        }
        let table = self.routed_expected_shadow_table(challenge);
        let t = state.time_step.min(challenge.num_steps - 1);
        let mut value = 0.0;
        for i in 0..challenge.num_batteries {
            let bat = &challenge.batteries[i];
            let rel_soc = (state.socs[i] - bat.soc_min_mwh).max(0.0);
            let mut fp = table[t][bat.node];
            if self.cfg.node_adaptation_enabled {
                fp *= online.node_scale(bat.node).clamp(0.80, 1.30);
            }
            let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
            let fp_adj = marginal_future_price(bat, state.socs[i], fp, self.cfg, rem);
            value += self.cfg.shadow_inventory_weight * rel_soc * bat.efficiency_discharge * fp_adj;
            if self.cfg.terminal_pressure_enabled {
                value -= self.terminal_stranding_penalty(challenge, state, i, fp_adj);
            }
            if self.cfg.network_reserve_enabled {
                value += rel_soc * self.future_discharge_need_table[t][bat.node];
                let headroom = (bat.soc_max_mwh - state.socs[i]).max(0.0);
                value += headroom * self.future_charge_need_table[t][bat.node];
            }
        }
        value
    }

    fn vector_objective_dynamic(
        &self,
        challenge: &Challenge,
        state: &State,
        action: &[f64],
        online: &OnlineState,
    ) -> f64 {
        let immediate = challenge.compute_profit(state, action);
        let mut obj = if self.cfg.selector_shadow_enabled && !self.use_support_dp {
            let table = if self.cfg.expected_shadow_enabled {
                self.routed_expected_shadow_table(challenge)
            } else {
                self.routed_shadow_table(challenge)
            };
            let future_scale = online.avg_ratio().clamp(0.85, 1.25);
            self.vector_objective_shadow(challenge, state, action, table, future_scale)
        } else {
            self.vector_objective(challenge, state, action)
        };

        let spike_bias = (online.spike_rate() * 2.0).clamp(0.0, 0.15);
        obj += spike_bias * immediate.max(0.0);

        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
        if rem < 16 {
            obj += (16 - rem) as f64 * 0.002 * immediate;
        }
        obj
    }

    fn spike_capture_policy(
        &self,
        challenge: &Challenge,
        state: &State,
        online: &OnlineState,
    ) -> Result<Vec<f64>> {
        let n_bat = challenge.num_batteries;
        let mut actions = vec![0.0f64; n_bat];

        let zero = vec![0.0f64; n_bat];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let mut cur_flows = base_flows.clone();

        let table = if self.cfg.expected_shadow_enabled {
            self.routed_expected_shadow_table(challenge)
        } else {
            self.routed_shadow_table(challenge)
        };

        let mut cands: Vec<Cand> = Vec::with_capacity(n_bat * self.cfg.alts_per_battery);
        let adaptive_extra = if online.spike_rate() > 0.03 {
            self.cfg.dynamic_q_shift
        } else {
            0.0
        };

        for i in 0..n_bat {
            let bat = &challenge.batteries[i];
            let node = bat.node;
            let node_scale = if self.cfg.node_adaptation_enabled {
                online.node_scale(node).clamp(0.80, 1.30)
            } else {
                1.0
            };
            let future_price = table[state.time_step][node] * node_scale * (1.0 + adaptive_extra);
            let rt = self.effective_rt_price(challenge, &base_flows, node, state.rt_prices[node]);
            let (lo, hi) = state.action_bounds[i];
            let mut local: Vec<Cand> = Vec::new();

            for u0 in action_candidates(lo, hi, self.cfg.cand_levels) {
                let u = u0.clamp(lo, hi);
                if u.abs() < 1e-9 {
                    continue;
                }

                let ratio = rt / future_price.abs().max(1.0);
                let mut invw = self.cfg.shadow_inventory_weight;
                if u > 0.0 && (ratio > 1.10 || rt > future_price + 25.0) {
                    invw *= self.cfg.spike_discount;
                } else if u < 0.0 && (ratio < 0.70 || rt < 0.0) {
                    invw *= 1.05 + adaptive_extra;
                }

                let mut value = shadow_action_value_ctx(
                    bat,
                    u,
                    rt,
                    future_price,
                    invw,
                    state.socs[i],
                    self.cfg,
                    challenge.num_steps.saturating_sub(state.time_step + 1),
                );
                value += self.network_action_bonus(challenge, &base_flows, node, u);
                value += self.reserve_action_bonus(challenge, state, i, u);
                value += self.terminal_pressure_action_bonus(
                    challenge,
                    state,
                    i,
                    u,
                    state.rt_prices[node].max(0.0),
                );
                if value <= self.cfg.min_value {
                    continue;
                }

                let usage = stress_usage(challenge, &base_flows, node, u);
                let cand = Cand {
                    density: value / (self.cfg.density_eps + usage),
                    value,
                    battery: i,
                    action: u,
                };
                push_ranked_candidate(&mut local, cand, self.cfg.alts_per_battery);
            }
            cands.extend(local.into_iter());
        }

        self.pack_candidates(challenge, state, &mut actions, &mut cur_flows, cands)
    }

    fn hybrid_select_policy(&self, challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(7);

        candidates.push(self.dp_pack_policy(challenge, state)?);
        candidates.push(self.shadow_pack_policy(challenge, state)?);

        // On large tracks, use the DP value function to choose among several shadow-price aggressiveness levels.
        // This is cheap compared with building the value tables and protects against q misrouting.
        if challenge.num_steps > 96 && challenge.num_batteries > 40 {
            candidates.push(self.shadow_pack_policy_with_table(
                challenge,
                state,
                &self.shadow_multiday_table,
            )?);
            candidates.push(self.shadow_pack_policy_with_table(
                challenge,
                state,
                &self.shadow_dense_table,
            )?);
            candidates.push(self.shadow_pack_policy_with_table(
                challenge,
                state,
                &self.shadow_congested_table,
            )?);
        }

        candidates.push(self.rt_only_pack_policy(challenge, state)?);

        if self.cfg.include_baseline_candidates
            && challenge.num_batteries * challenge.num_steps <= 3000
        {
            candidates.push(greedy_like_policy(challenge, state));
            candidates.push(conservative_like_policy(challenge, state));
        }

        candidates.push(vec![0.0f64; challenge.num_batteries]);

        Ok(self.select_best_action(challenge, state, candidates, None))
    }

    fn vector_objective(&self, challenge: &Challenge, state: &State, action: &[f64]) -> f64 {
        if self.cfg.selector_shadow_enabled && !self.use_support_dp {
            let table = if self.cfg.expected_shadow_enabled {
                self.routed_expected_shadow_table(challenge)
            } else {
                self.routed_shadow_table(challenge)
            };
            return self.vector_objective_shadow(challenge, state, action, table, 1.0);
        }

        let mut obj = challenge.compute_profit(state, action);
        let t_next = state.time_step + 1;

        for i in 0..challenge.num_batteries {
            let bat = &challenge.batteries[i];
            let next_soc = apply_action_to_soc_manual(bat, action[i], state.socs[i]);
            obj += self.value_tables[i].value_at(t_next, next_soc)
                - self.value_tables[i].value_at(t_next, state.socs[i]);
            obj += self.reserve_action_bonus(challenge, state, i, action[i]);
            obj += self.terminal_pressure_action_bonus(
                challenge,
                state,
                i,
                action[i],
                state.rt_prices[bat.node].max(0.0),
            );
        }

        obj + self.network_vector_bonus(challenge, state, action)
    }

    fn vector_objective_shadow(
        &self,
        challenge: &Challenge,
        state: &State,
        action: &[f64],
        table: &[Vec<f64>],
        future_scale: f64,
    ) -> f64 {
        let mut obj = challenge.compute_profit(state, action);
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
        for i in 0..challenge.num_batteries {
            let bat = &challenge.batteries[i];
            let old_soc = state.socs[i];
            let next_soc = apply_action_to_soc_manual(bat, action[i], old_soc);
            let fp = table[state.time_step][bat.node] * future_scale;
            let mf = marginal_future_price(bat, old_soc, fp, self.cfg, rem);
            obj += self.cfg.shadow_inventory_weight
                * (next_soc - old_soc)
                * bat.efficiency_discharge
                * mf;
            obj += self.reserve_action_bonus(challenge, state, i, action[i]);
            obj += self.terminal_pressure_action_bonus(challenge, state, i, action[i], mf);
        }
        obj + self.network_vector_bonus(challenge, state, action)
    }

    fn network_vector_bonus(&self, challenge: &Challenge, state: &State, action: &[f64]) -> f64 {
        if !self.cfg.dual_enabled {
            return 0.0;
        }
        let zero = vec![0.0f64; challenge.num_batteries];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let inj = challenge.compute_total_injections(state, action);
        let flows = challenge.network.compute_flows(&inj);
        let mut bonus = 0.0f64;
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let stress = (base_flows[l].abs() / lim).clamp(0.0, 2.0);
            if stress <= self.cfg.dual_stress_threshold {
                continue;
            }
            let relief = (base_flows[l].abs() - flows[l].abs()) / lim;
            let intensity = ((stress - self.cfg.dual_stress_threshold)
                / (1.0 - self.cfg.dual_stress_threshold).max(1e-6))
            .clamp(0.0, 2.0);
            bonus += self.cfg.dual_price_scale * intensity * intensity * relief;
        }
        bonus
    }

    fn network_action_bonus(
        &self,
        challenge: &Challenge,
        base_flows: &[f64],
        node: usize,
        u: f64,
    ) -> f64 {
        if !self.cfg.dual_enabled || node == challenge.network.slack_bus || u.abs() < 1e-12 {
            return 0.0;
        }
        let mut bonus = 0.0f64;
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let f = base_flows[l];
            let stress = (f.abs() / lim).clamp(0.0, 2.0);
            if stress <= self.cfg.dual_stress_threshold {
                continue;
            }
            let delta = challenge.network.ptdf[l][node] * u;
            let before = f.abs();
            let after = (f + delta).abs();
            let relief = (before - after) / lim;
            let intensity = ((stress - self.cfg.dual_stress_threshold)
                / (1.0 - self.cfg.dual_stress_threshold).max(1e-6))
            .clamp(0.0, 2.0);
            bonus += self.cfg.dual_price_scale * intensity * intensity * relief;
        }
        bonus
    }

    fn effective_rt_price(
        &self,
        challenge: &Challenge,
        base_flows: &[f64],
        node: usize,
        rt_price: f64,
    ) -> f64 {
        if !self.cfg.nodal_dual_price_enabled || node == challenge.network.slack_bus {
            return rt_price;
        }
        let adder = self.nodal_dual_price_adder(challenge, base_flows, node);
        clamp_price(rt_price + self.cfg.nodal_dual_scale * adder)
    }

    fn nodal_dual_price_adder(
        &self,
        challenge: &Challenge,
        base_flows: &[f64],
        node: usize,
    ) -> f64 {
        if !self.cfg.dual_enabled || node == challenge.network.slack_bus {
            return 0.0;
        }
        let dt = constants::DELTA_T.max(1e-9);
        let mut adder = 0.0f64;
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let f = base_flows[l];
            let stress = (f.abs() / lim).clamp(0.0, 2.0);
            if stress <= self.cfg.dual_stress_threshold {
                continue;
            }
            let intensity = ((stress - self.cfg.dual_stress_threshold)
                / (1.0 - self.cfg.dual_stress_threshold).max(1e-6))
            .clamp(0.0, 2.0);
            // Positive injection that relieves a stressed line should see a positive price adder.
            let marginal_relief_per_mw = -f.signum() * challenge.network.ptdf[l][node] / lim;
            adder +=
                self.cfg.dual_price_scale * intensity * intensity * marginal_relief_per_mw / dt;
        }
        adder.clamp(-80.0, 80.0)
    }

    fn terminal_pressure_action_bonus(
        &self,
        challenge: &Challenge,
        state: &State,
        bat_idx: usize,
        u: f64,
        price_ref: f64,
    ) -> f64 {
        if !self.cfg.terminal_pressure_enabled
            || u.abs() < 1e-12
            || state.time_step >= challenge.num_steps
        {
            return 0.0;
        }
        let bat = &challenge.batteries[bat_idx];
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
        if self.cfg.endgame_window == 0 || rem >= self.cfg.endgame_window {
            return 0.0;
        }
        let dt = constants::DELTA_T;
        let urgency = 1.0 - rem as f64 / self.cfg.endgame_window.max(1) as f64;
        let available_soc = (state.socs[bat_idx] - bat.soc_min_mwh).max(0.0);
        let future_discharge_soc_capacity =
            rem as f64 * bat.power_discharge_mw * dt / bat.efficiency_discharge.max(1e-9);
        let stranded = (available_soc - future_discharge_soc_capacity).max(0.0);
        let soc_delta = if u > 0.0 {
            (u * dt / bat.efficiency_discharge.max(1e-9)).min(available_soc)
        } else {
            -((-u) * bat.efficiency_charge * dt)
        };
        if u > 0.0 {
            let frac = if available_soc > 1e-9 {
                (stranded / available_soc).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.cfg.terminal_pressure_scale
                * urgency
                * frac
                * soc_delta
                * price_ref.max(1.0)
                * bat.efficiency_discharge
        } else {
            // Late charging is penalized unless it is strongly justified elsewhere.
            -0.35
                * self.cfg.terminal_pressure_scale
                * urgency
                * (-soc_delta).max(0.0)
                * price_ref.max(1.0)
        }
    }

    fn terminal_stranding_penalty(
        &self,
        challenge: &Challenge,
        state: &State,
        bat_idx: usize,
        price_ref: f64,
    ) -> f64 {
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
        if self.cfg.endgame_window == 0 || rem >= self.cfg.endgame_window {
            return 0.0;
        }
        let bat = &challenge.batteries[bat_idx];
        let dt = constants::DELTA_T;
        let available_soc = (state.socs[bat_idx] - bat.soc_min_mwh).max(0.0);
        let future_discharge_soc_capacity =
            rem as f64 * bat.power_discharge_mw * dt / bat.efficiency_discharge.max(1e-9);
        let stranded = (available_soc - future_discharge_soc_capacity).max(0.0);
        let urgency = 1.0 - rem as f64 / self.cfg.endgame_window.max(1) as f64;
        self.cfg.terminal_pressure_scale
            * urgency
            * stranded
            * price_ref.max(1.0)
            * bat.efficiency_discharge
    }

    fn reserve_action_bonus(
        &self,
        challenge: &Challenge,
        state: &State,
        bat_idx: usize,
        u: f64,
    ) -> f64 {
        if !self.cfg.network_reserve_enabled || state.time_step >= challenge.num_steps {
            return 0.0;
        }
        let bat = &challenge.batteries[bat_idx];
        let node = bat.node;
        let dt = constants::DELTA_T;
        let soc_delta = if u < 0.0 {
            (-u) * bat.efficiency_charge * dt
        } else {
            -u * dt / bat.efficiency_discharge
        };
        let t = state.time_step.min(challenge.num_steps - 1);
        let high_soc_shadow = self.future_discharge_need_table[t][node];
        let headroom_shadow = self.future_charge_need_table[t][node];
        soc_delta * (high_soc_shadow - headroom_shadow)
    }

    fn local_refine_actions(
        &self,
        challenge: &Challenge,
        state: &State,
        mut actions: Vec<f64>,
    ) -> Vec<f64> {
        if !self.cfg.local_refine_enabled || self.cfg.refine_passes == 0 {
            return actions;
        }
        actions = repair_global_scale(challenge, state, actions);
        let mut best_obj = self.vector_objective(challenge, state, &actions)
            + 1e-9 * challenge.compute_profit(state, &actions);

        for _ in 0..self.cfg.refine_passes {
            let mut changed = false;
            for i in 0..challenge.num_batteries {
                let cur = actions[i];
                let (lo, hi) = state.action_bounds[i];
                let mut trials = Vec::with_capacity(10);
                for x in [
                    0.0,
                    cur * 0.50,
                    cur * 0.75,
                    cur * 1.15,
                    cur * 1.35,
                    lo * 0.50,
                    hi * 0.50,
                    lo,
                    hi,
                ] {
                    push_unique(&mut trials, x.clamp(lo, hi));
                }
                let mut best_u = cur;
                for u in trials {
                    if (u - cur).abs() < 1e-9 {
                        continue;
                    }
                    let mut trial = actions.clone();
                    trial[i] = u;
                    let inj = challenge.compute_total_injections(state, &trial);
                    let flows = challenge.network.compute_flows(&inj);
                    if challenge.network.verify_flows(&flows).is_err() {
                        continue;
                    }
                    let obj = self.vector_objective(challenge, state, &trial)
                        + 1e-9 * challenge.compute_profit(state, &trial);
                    if obj > best_obj + 1e-7 {
                        best_obj = obj;
                        best_u = u;
                    }
                }
                if (best_u - cur).abs() > 1e-9 {
                    actions[i] = best_u;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        actions
    }

    fn dp_pack_policy(&self, challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        let n_bat = challenge.num_batteries;
        let t = state.time_step;
        let mut actions = vec![0.0f64; n_bat];

        let zero = vec![0.0f64; n_bat];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let mut cur_flows = base_flows.clone();

        let include_all = self.use_support_dp && self.cfg.include_all_small;
        let mut cands: Vec<Cand> = Vec::with_capacity(
            n_bat
                * (if include_all {
                    self.cfg.cand_levels
                } else {
                    self.cfg.alts_per_battery
                }),
        );

        for i in 0..n_bat {
            let bat = &challenge.batteries[i];
            let (lo, hi) = state.action_bounds[i];
            if lo.abs() < 1e-12 && hi.abs() < 1e-12 {
                continue;
            }

            let base_future = self.value_tables[i].value_at(t + 1, state.socs[i]);
            let mut local: Vec<Cand> = Vec::new();

            for u0 in action_candidates(lo, hi, self.cfg.cand_levels) {
                let u = u0.clamp(lo, hi);
                if u.abs() < 1e-9 {
                    continue;
                }

                let next_soc = apply_action_to_soc_manual(bat, u, state.socs[i]);
                let rt_eff = self.effective_rt_price(
                    challenge,
                    &base_flows,
                    bat.node,
                    state.rt_prices[bat.node],
                );
                let immediate = profit_at_price(bat, u, rt_eff);
                let future = self.value_tables[i].value_at(t + 1, next_soc);
                let mut value = immediate + future - base_future;
                value += self.network_action_bonus(challenge, &base_flows, bat.node, u);
                value += self.reserve_action_bonus(challenge, state, i, u);
                value +=
                    self.terminal_pressure_action_bonus(challenge, state, i, u, rt_eff.max(0.0));

                if value <= self.cfg.min_value {
                    continue;
                }

                let usage = stress_usage(challenge, &base_flows, bat.node, u);
                let cand = Cand {
                    density: value / (self.cfg.density_eps + usage),
                    value,
                    battery: i,
                    action: u,
                };

                if include_all {
                    cands.push(cand);
                } else {
                    push_ranked_candidate(&mut local, cand, self.cfg.alts_per_battery);
                }
            }

            if !include_all {
                cands.extend(local.into_iter());
            }
        }

        self.pack_candidates(challenge, state, &mut actions, &mut cur_flows, cands)
    }

    fn routed_shadow_table(&self, challenge: &Challenge) -> &Vec<Vec<f64>> {
        if self.use_support_dp {
            &self.shadow_baseline_table
        } else if challenge.num_steps == 96 && challenge.num_batteries >= 15 {
            &self.shadow_congested_table
        } else if challenge.num_steps > 96 && challenge.num_batteries <= 40 {
            &self.shadow_multiday_table
        } else if challenge.num_steps > 96 && challenge.num_batteries <= 60 {
            &self.shadow_dense_table
        } else {
            &self.shadow_capstone_table
        }
    }

    fn routed_expected_shadow_table(&self, challenge: &Challenge) -> &Vec<Vec<f64>> {
        if self.use_support_dp {
            &self.exp_shadow_baseline_table
        } else if challenge.num_steps == 96 && challenge.num_batteries >= 15 {
            &self.exp_shadow_congested_table
        } else if challenge.num_steps > 96 && challenge.num_batteries <= 40 {
            &self.exp_shadow_multiday_table
        } else if challenge.num_steps > 96 && challenge.num_batteries <= 60 {
            &self.exp_shadow_dense_table
        } else {
            &self.exp_shadow_capstone_table
        }
    }

    fn shadow_pack_policy(&self, challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        let table = self.routed_shadow_table(challenge);
        self.shadow_pack_policy_with_table(challenge, state, table)
    }

    fn shadow_pack_policy_with_table(
        &self,
        challenge: &Challenge,
        state: &State,
        shadow_q_table: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        let n_bat = challenge.num_batteries;
        let mut actions = vec![0.0f64; n_bat];

        let zero = vec![0.0f64; n_bat];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let mut cur_flows = base_flows.clone();

        let mut cands: Vec<Cand> = Vec::with_capacity(n_bat * self.cfg.alts_per_battery);

        for i in 0..n_bat {
            let bat = &challenge.batteries[i];
            let node = bat.node;
            let future_price = shadow_q_table[state.time_step][node];
            let (lo, hi) = state.action_bounds[i];
            let mut local: Vec<Cand> = Vec::new();

            for u0 in action_candidates(lo, hi, self.cfg.cand_levels) {
                let u = u0.clamp(lo, hi);
                if u.abs() < 1e-9 {
                    continue;
                }
                let rt_eff =
                    self.effective_rt_price(challenge, &cur_flows, node, state.rt_prices[node]);
                let mut value = shadow_action_value_ctx(
                    bat,
                    u,
                    rt_eff,
                    future_price,
                    self.cfg.shadow_inventory_weight,
                    state.socs[i],
                    self.cfg,
                    challenge.num_steps.saturating_sub(state.time_step + 1),
                );
                value += self.network_action_bonus(challenge, &base_flows, node, u);
                value += self.reserve_action_bonus(challenge, state, i, u);
                value += self.terminal_pressure_action_bonus(
                    challenge,
                    state,
                    i,
                    u,
                    state.rt_prices[node].max(0.0),
                );
                if value <= self.cfg.min_value {
                    continue;
                }

                let usage = stress_usage(challenge, &base_flows, node, u);
                let cand = Cand {
                    density: value / (self.cfg.density_eps + usage),
                    value,
                    battery: i,
                    action: u,
                };
                push_ranked_candidate(&mut local, cand, self.cfg.alts_per_battery);
            }

            cands.extend(local.into_iter());
        }

        self.pack_candidates(challenge, state, &mut actions, &mut cur_flows, cands)
    }

    fn rt_only_pack_policy(&self, challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        let n_bat = challenge.num_batteries;
        let mut actions = vec![0.0f64; n_bat];

        let zero = vec![0.0f64; n_bat];
        let base_inj = challenge.compute_total_injections(state, &zero);
        let base_flows = challenge.network.compute_flows(&base_inj);
        let mut cur_flows = base_flows.clone();

        let mut cands: Vec<Cand> = Vec::with_capacity(n_bat * self.cfg.alts_per_battery);

        for i in 0..n_bat {
            let bat = &challenge.batteries[i];
            let (lo, hi) = state.action_bounds[i];
            let mut local: Vec<Cand> = Vec::new();

            for u0 in action_candidates(lo, hi, self.cfg.cand_levels) {
                let u = u0.clamp(lo, hi);
                if u.abs() < 1e-9 {
                    continue;
                }
                let rt_eff = self.effective_rt_price(
                    challenge,
                    &base_flows,
                    bat.node,
                    state.rt_prices[bat.node],
                );
                let mut value = profit_at_price(bat, u, rt_eff);
                value += self.network_action_bonus(challenge, &base_flows, bat.node, u);
                value += self.reserve_action_bonus(challenge, state, i, u);
                value +=
                    self.terminal_pressure_action_bonus(challenge, state, i, u, rt_eff.max(0.0));
                if value <= self.cfg.min_value {
                    continue;
                }
                let usage = stress_usage(challenge, &base_flows, bat.node, u);
                let cand = Cand {
                    density: value / (self.cfg.density_eps + usage),
                    value,
                    battery: i,
                    action: u,
                };
                push_ranked_candidate(&mut local, cand, self.cfg.alts_per_battery);
            }

            cands.extend(local.into_iter());
        }

        self.pack_candidates(challenge, state, &mut actions, &mut cur_flows, cands)
    }

    fn pack_candidates(
        &self,
        challenge: &Challenge,
        state: &State,
        actions: &mut Vec<f64>,
        cur_flows: &mut Vec<f64>,
        mut cands: Vec<Cand>,
    ) -> Result<Vec<f64>> {
        if self.cfg.beam_pack_enabled && !cands.is_empty() {
            return self.beam_pack_candidates(challenge, state, actions, cur_flows, cands);
        }

        cands.sort_by(|a, b| {
            b.density
                .partial_cmp(&a.density)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal))
        });

        let mut rejected: Vec<Cand> = Vec::new();
        let mut pending = cands;
        for _ in 0..self.cfg.passes {
            let mut next_pending: Vec<Cand> = Vec::new();
            let mut changed = false;

            for cand in pending.into_iter() {
                if actions[cand.battery].abs() > 1e-9 {
                    continue;
                }

                let node = challenge.batteries[cand.battery].node;
                let alpha = max_feasible_fraction(
                    challenge,
                    cur_flows,
                    node,
                    cand.action,
                    self.cfg.flow_margin,
                );

                if alpha > 1e-8 {
                    let du = cand.action * alpha;
                    actions[cand.battery] = du;
                    if node != challenge.network.slack_bus {
                        for l in 0..challenge.network.num_lines {
                            cur_flows[l] += challenge.network.ptdf[l][node] * du;
                        }
                    }
                    changed = true;
                    if alpha < 0.95 {
                        let mut r = cand;
                        r.action = cand.action * (1.0 - alpha);
                        r.value = cand.value * (1.0 - alpha).max(0.0);
                        rejected.push(r);
                    }
                } else {
                    next_pending.push(cand);
                }
            }

            if !changed || next_pending.is_empty() {
                rejected.extend(next_pending.into_iter());
                break;
            }
            pending = next_pending;
        }

        let rejected_pool = rejected.clone();

        if self.cfg.residual_retry_enabled {
            rejected.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));
            let retry_cap = challenge.num_batteries.min(rejected.len());
            for cand in rejected.into_iter().take(retry_cap) {
                if actions[cand.battery].abs() > 1e-9 {
                    continue;
                }
                let node = challenge.batteries[cand.battery].node;
                for frac in [0.35, 0.20, 0.10] {
                    let du = cand.action * frac;
                    if du.abs() < 1e-10 || cand.value * frac <= self.cfg.min_value {
                        continue;
                    }
                    let alpha =
                        max_feasible_fraction(challenge, cur_flows, node, du, self.cfg.flow_margin);
                    if alpha > 0.999 {
                        actions[cand.battery] = du;
                        if node != challenge.network.slack_bus {
                            for l in 0..challenge.network.num_lines {
                                cur_flows[l] += challenge.network.ptdf[l][node] * du;
                            }
                        }
                        break;
                    }
                }
            }
        }

        if self.cfg.counterflow_enabled {
            let table = if self.cfg.expected_shadow_enabled {
                self.routed_expected_shadow_table(challenge)
            } else {
                self.routed_shadow_table(challenge)
            };
            let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
            for _ in 0..self.cfg.counter_passes {
                let mut worst_line = 0usize;
                let mut worst_stress = 0.0f64;
                for l in 0..challenge.network.num_lines {
                    let stress = cur_flows[l].abs() / challenge.network.flow_limits[l].max(1e-9);
                    if stress > worst_stress {
                        worst_stress = stress;
                        worst_line = l;
                    }
                }
                if worst_stress < self.cfg.flow_margin * 0.96 {
                    break;
                }

                let signed = cur_flows[worst_line].signum();
                if signed.abs() < 1e-12 {
                    break;
                }

                let mut best: Option<(f64, usize, f64)> = None;
                for i in 0..challenge.num_batteries {
                    if actions[i].abs() > 1e-9 {
                        continue;
                    }
                    let bat = &challenge.batteries[i];
                    let node = bat.node;
                    if node == challenge.network.slack_bus {
                        continue;
                    }
                    let p = challenge.network.ptdf[worst_line][node];
                    if p.abs() < 1e-12 {
                        continue;
                    }
                    let (lo, hi) = state.action_bounds[i];
                    let mut opts: Vec<f64> = Vec::with_capacity(2);
                    if signed * p > 0.0 && lo < -1e-12 {
                        opts.push(lo * 0.25);
                        opts.push(lo * 0.10);
                    } else if signed * p < 0.0 && hi > 1e-12 {
                        opts.push(hi * 0.25);
                        opts.push(hi * 0.10);
                    }
                    for u in opts {
                        if u.abs() < 1e-10 {
                            continue;
                        }
                        let future_price = table[state.time_step][node];
                        let value = shadow_action_value_ctx(
                            bat,
                            u,
                            state.rt_prices[node],
                            future_price,
                            self.cfg.shadow_inventory_weight,
                            state.socs[i],
                            self.cfg,
                            rem,
                        );
                        let before = cur_flows[worst_line].abs();
                        let after = (cur_flows[worst_line] + p * u).abs();
                        let relief = ((before - after)
                            / challenge.network.flow_limits[worst_line].max(1e-9))
                        .max(0.0);
                        let score = value + self.cfg.relief_bonus * relief;
                        if score <= self.cfg.min_value {
                            continue;
                        }
                        let alpha = max_feasible_fraction(
                            challenge,
                            cur_flows,
                            node,
                            u,
                            self.cfg.flow_margin,
                        );
                        if alpha > 0.999 && best.map(|b| score > b.0).unwrap_or(true) {
                            best = Some((score, i, u));
                        }
                    }
                }

                let Some((_, i, u)) = best else {
                    break;
                };
                actions[i] = u;
                let node = challenge.batteries[i].node;
                if node != challenge.network.slack_bus {
                    for l in 0..challenge.network.num_lines {
                        cur_flows[l] += challenge.network.ptdf[l][node] * u;
                    }
                }
            }
        }

        if self.cfg.pairflow_enabled && self.cfg.pairflow_passes > 0 {
            let polished =
                self.pairflow_polish_actions(challenge, state, actions.clone(), &rejected_pool);
            *actions = polished;
        }
        Ok(repair_global_scale(
            challenge,
            state,
            self.local_refine_actions(challenge, state, actions.clone()),
        ))
    }

    fn beam_pack_candidates(
        &self,
        challenge: &Challenge,
        state: &State,
        actions: &mut Vec<f64>,
        cur_flows: &mut Vec<f64>,
        cands: Vec<Cand>,
    ) -> Result<Vec<f64>> {
        let n = challenge.num_batteries;
        if n == 0 || cands.is_empty() {
            return Ok(vec![0.0; n]);
        }

        let all_cands = cands.clone();
        let mut by_bat: Vec<Vec<Cand>> = vec![Vec::new(); n];
        for cand in cands.into_iter() {
            if cand.battery < n {
                push_ranked_candidate(
                    &mut by_bat[cand.battery],
                    cand,
                    self.cfg.alts_per_battery.max(1),
                );
            }
        }

        let mut order: Vec<usize> = (0..n).filter(|&i| !by_bat[i].is_empty()).collect();
        order.sort_by(|&a, &b| {
            let av = by_bat[a][0].value;
            let bv = by_bat[b][0].value;
            bv.partial_cmp(&av)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    by_bat[b][0]
                        .density
                        .partial_cmp(&by_bat[a][0].density)
                        .unwrap_or(Ordering::Equal)
                })
        });

        let cap = self.cfg.beam_battery_cap.min(order.len());
        let width = self.cfg.beam_width.max(1);
        let alpha_floor = if self.cfg.beam_partial_enabled {
            0.08
        } else {
            0.999
        };

        let mut beams = vec![BeamState {
            actions: actions.clone(),
            flows: cur_flows.clone(),
            score: 0.0,
        }];

        for &b in order.iter().take(cap) {
            let opts = &by_bat[b];
            let mut next: Vec<BeamState> = Vec::with_capacity(beams.len() * (opts.len() + 1));

            for st in beams.iter() {
                // Explicit zero option. This matters because some high standalone values
                // consume scarce flow capacity that is better reserved for later batteries.
                next.push(st.clone());

                if st.actions[b].abs() > 1e-9 {
                    continue;
                }

                for &cand in opts.iter() {
                    let node = challenge.batteries[b].node;
                    let alpha = max_feasible_fraction(
                        challenge,
                        &st.flows,
                        node,
                        cand.action,
                        self.cfg.flow_margin,
                    );
                    if alpha < alpha_floor {
                        continue;
                    }
                    let used_alpha = if alpha >= 0.999 {
                        1.0
                    } else {
                        alpha.clamp(0.0, 1.0)
                    };
                    let du = cand.action * used_alpha;
                    if du.abs() < 1e-10 {
                        continue;
                    }
                    let scaled_value = cand.value * used_alpha.abs();
                    if scaled_value <= self.cfg.min_value {
                        continue;
                    }

                    let mut ns = st.clone();
                    ns.actions[b] = du;
                    if node != challenge.network.slack_bus {
                        for l in 0..challenge.network.num_lines {
                            ns.flows[l] += challenge.network.ptdf[l][node] * du;
                        }
                    }
                    // Score combines candidate economics and a small density term. The density
                    // term breaks ties toward less line-intensive partial assignments.
                    ns.score += scaled_value + 0.01 * cand.density;
                    next.push(ns);
                }
            }

            next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            if next.len() > width {
                next.truncate(width);
            }
            beams = next;
        }

        if beams.is_empty() {
            return Ok(repair_global_scale(challenge, state, actions.clone()));
        }

        beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        let candidate_beams = width.min(beams.len());
        let mut best_action = repair_global_scale(challenge, state, beams[0].actions.clone());
        let mut best_obj = f64::NEG_INFINITY;

        // Safety candidate: the previous greedy-density packer is still evaluated.
        // This keeps the architecture upgrade from losing easy uncongested cases where
        // pure density packing is already near-optimal.
        let mut baseline_actions = actions.clone();
        let mut baseline_flows = cur_flows.clone();
        self.greedy_finish_candidates(
            challenge,
            &mut baseline_actions,
            &mut baseline_flows,
            all_cands.clone(),
        );
        baseline_actions = self.regret_swap_actions(challenge, state, baseline_actions, &all_cands);
        baseline_actions =
            self.pairflow_polish_actions(challenge, state, baseline_actions, &all_cands);
        baseline_actions = self.counterflow_polish_actions(challenge, state, baseline_actions);
        baseline_actions = repair_global_scale(challenge, state, baseline_actions);
        baseline_actions = self.local_refine_actions(challenge, state, baseline_actions);
        baseline_actions = repair_global_scale(challenge, state, baseline_actions);
        let baseline_obj = self.vector_objective(challenge, state, &baseline_actions)
            + 1e-9 * challenge.compute_profit(state, &baseline_actions);
        if baseline_obj > best_obj {
            best_obj = baseline_obj;
            best_action = baseline_actions;
        }

        for st in beams.iter().take(candidate_beams) {
            let mut trial_actions = st.actions.clone();
            let inj = challenge.compute_total_injections(state, &trial_actions);
            let mut trial_flows = challenge.network.compute_flows(&inj);

            // Greedily fill any batteries that the beam left empty. This gives the beam
            // the hard combinatorial choices while preserving the cheap high-throughput
            // behavior of the previous packer on uncontentious residual capacity.
            let mut tail: Vec<Cand> = Vec::new();
            for cand in all_cands.iter().copied() {
                if cand.battery < trial_actions.len() && trial_actions[cand.battery].abs() <= 1e-9 {
                    tail.push(cand);
                }
            }
            self.greedy_finish_candidates(challenge, &mut trial_actions, &mut trial_flows, tail);

            trial_actions = self.regret_swap_actions(challenge, state, trial_actions, &all_cands);
            trial_actions =
                self.pairflow_polish_actions(challenge, state, trial_actions, &all_cands);
            trial_actions = self.counterflow_polish_actions(challenge, state, trial_actions);
            trial_actions = repair_global_scale(challenge, state, trial_actions);
            trial_actions = self.local_refine_actions(challenge, state, trial_actions);
            trial_actions = repair_global_scale(challenge, state, trial_actions);

            let obj = self.vector_objective(challenge, state, &trial_actions)
                + 1e-9 * challenge.compute_profit(state, &trial_actions);
            if obj > best_obj {
                best_obj = obj;
                best_action = trial_actions;
            }
        }

        Ok(best_action)
    }

    fn pairflow_polish_actions(
        &self,
        challenge: &Challenge,
        state: &State,
        mut actions: Vec<f64>,
        cands: &[Cand],
    ) -> Vec<f64> {
        if !self.cfg.pairflow_enabled
            || self.cfg.pairflow_passes == 0
            || self.cfg.pairflow_candidate_cap == 0
            || self.cfg.pairflow_counter_cap == 0
            || cands.is_empty()
        {
            return actions;
        }
        actions = repair_global_scale(challenge, state, actions);
        let inj = challenge.compute_total_injections(state, &actions);
        let mut cur_flows = challenge.network.compute_flows(&inj);
        let mut pool = cands.to_vec();
        pool.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.density.partial_cmp(&a.density).unwrap_or(Ordering::Equal))
        });

        let main_fracs_all = [1.00, 0.75, 0.55, 0.35, 0.20, 0.10];
        let counter_fracs = [0.20, 0.35, 0.55, 0.80, 1.00];

        for _ in 0..self.cfg.pairflow_passes {
            let mut best_pair: Option<(f64, usize, f64, usize, f64)> = None;

            for cand in pool.iter().take(self.cfg.pairflow_candidate_cap).copied() {
                let i = cand.battery;
                if i >= challenge.num_batteries {
                    continue;
                }
                if actions[i].abs() > 1e-9 && actions[i] * cand.action <= 0.0 {
                    continue;
                }
                let node_i = challenge.batteries[i].node;
                let (lo_i, hi_i) = state.action_bounds[i];
                let delta_lo_i = lo_i - actions[i];
                let delta_hi_i = hi_i - actions[i];

                for &frac in main_fracs_all.iter().take(self.cfg.pairflow_main_fracs) {
                    let u1 = (cand.action * frac).clamp(delta_lo_i, delta_hi_i);
                    if u1.abs() < 1e-10 || (actions[i].abs() > 1e-9 && actions[i] * u1 < -1e-12) {
                        continue;
                    }
                    let val1 = self.estimate_action_value_live(challenge, state, &cur_flows, i, u1);
                    if val1 <= -self.cfg.pairflow_min_value {
                        continue;
                    }

                    let mut flows1 = cur_flows.clone();
                    add_node_action_to_flows(challenge, &mut flows1, node_i, u1);

                    if flows_feasible_margin(challenge, &flows1, self.cfg.flow_margin) {
                        if val1 > self.cfg.pairflow_min_value
                            && best_pair.map(|b| val1 > b.0).unwrap_or(true)
                        {
                            best_pair = Some((val1, i, u1, usize::MAX, 0.0));
                        }
                        continue;
                    }

                    let (worst_line, _) = most_stressed_line(challenge, &flows1);
                    let signed = flows1[worst_line].signum();
                    if signed.abs() < 1e-12 {
                        continue;
                    }

                    let mut counter_opts: Vec<(f64, usize, f64)> = Vec::new();
                    for j in 0..challenge.num_batteries {
                        if j == i {
                            continue;
                        }
                        let node_j = challenge.batteries[j].node;
                        if node_j == challenge.network.slack_bus {
                            continue;
                        }
                        let p = challenge.network.ptdf[worst_line][node_j];
                        if p.abs() < 1e-12 {
                            continue;
                        }
                        let (lo_j, hi_j) = state.action_bounds[j];
                        let delta_lo_j = lo_j - actions[j];
                        let delta_hi_j = hi_j - actions[j];
                        let mut base_dirs: Vec<f64> = Vec::with_capacity(2);
                        if signed * p > 0.0 && delta_lo_j < -1e-12 {
                            base_dirs.push(delta_lo_j);
                        }
                        if signed * p < 0.0 && delta_hi_j > 1e-12 {
                            base_dirs.push(delta_hi_j);
                        }
                        for base_u in base_dirs {
                            for &cf in counter_fracs.iter() {
                                let u2 = base_u * cf;
                                if u2.abs() < 1e-10
                                    || (actions[j].abs() > 1e-9 && actions[j] * u2 < -1e-12)
                                {
                                    continue;
                                }
                                let before = flows1[worst_line].abs();
                                let after = (flows1[worst_line] + p * u2).abs();
                                let relief = ((before - after)
                                    / challenge.network.flow_limits[worst_line].max(1e-9))
                                .max(0.0);
                                if relief <= 1e-10 {
                                    continue;
                                }
                                counter_opts.push((relief, j, u2));
                            }
                        }
                    }

                    counter_opts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                    if counter_opts.len() > self.cfg.pairflow_counter_cap {
                        counter_opts.truncate(self.cfg.pairflow_counter_cap);
                    }

                    for &(_, j, u2) in counter_opts.iter() {
                        let node_j = challenge.batteries[j].node;
                        let mut flows2 = flows1.clone();
                        add_node_action_to_flows(challenge, &mut flows2, node_j, u2);
                        if !flows_feasible_margin(challenge, &flows2, self.cfg.flow_margin) {
                            continue;
                        }

                        let val2 =
                            self.estimate_action_value_live(challenge, state, &flows1, j, u2);
                        let relief_bonus = self.pair_relief_bonus(challenge, &cur_flows, &flows2);
                        let score = val1 + val2 + relief_bonus;
                        if score > self.cfg.pairflow_min_value
                            && best_pair.map(|b| score > b.0).unwrap_or(true)
                        {
                            best_pair = Some((score, i, u1, j, u2));
                        }
                    }
                }
            }

            let Some((_, i, u1, j, u2)) = best_pair else {
                break;
            };
            actions[i] += u1;
            let node_i = challenge.batteries[i].node;
            add_node_action_to_flows(challenge, &mut cur_flows, node_i, u1);
            if j != usize::MAX {
                actions[j] += u2;
                let node_j = challenge.batteries[j].node;
                add_node_action_to_flows(challenge, &mut cur_flows, node_j, u2);
            }
        }

        repair_global_scale(challenge, state, actions)
    }

    fn estimate_action_value_live(
        &self,
        challenge: &Challenge,
        state: &State,
        flows: &[f64],
        bat_idx: usize,
        u: f64,
    ) -> f64 {
        if u.abs() < 1e-12 || state.time_step >= challenge.num_steps {
            return 0.0;
        }
        let bat = &challenge.batteries[bat_idx];
        let table = if self.cfg.expected_shadow_enabled {
            self.routed_expected_shadow_table(challenge)
        } else {
            self.routed_shadow_table(challenge)
        };
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);
        let future_price = table[state.time_step][bat.node];
        let rt_eff = self.effective_rt_price(challenge, flows, bat.node, state.rt_prices[bat.node]);
        let mut value = shadow_action_value_ctx(
            bat,
            u,
            rt_eff,
            future_price,
            self.cfg.shadow_inventory_weight,
            state.socs[bat_idx],
            self.cfg,
            rem,
        );
        value += self.reserve_action_bonus(challenge, state, bat_idx, u);
        value += self.terminal_pressure_action_bonus(challenge, state, bat_idx, u, rt_eff.max(0.0));
        value += self.network_action_bonus(challenge, flows, bat.node, u);
        value
    }

    fn pair_relief_bonus(&self, challenge: &Challenge, before: &[f64], after: &[f64]) -> f64 {
        if !self.cfg.dual_enabled {
            return 0.0;
        }
        let mut bonus = 0.0f64;
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let stress = (before[l].abs() / lim).clamp(0.0, 2.0);
            if stress <= self.cfg.dual_stress_threshold {
                continue;
            }
            let relief = (before[l].abs() - after[l].abs()) / lim;
            if relief <= 0.0 {
                continue;
            }
            let intensity = ((stress - self.cfg.dual_stress_threshold)
                / (1.0 - self.cfg.dual_stress_threshold).max(1e-6))
            .clamp(0.0, 2.0);
            bonus += 0.50 * self.cfg.dual_price_scale * intensity * intensity * relief;
        }
        bonus
    }

    fn greedy_finish_candidates(
        &self,
        challenge: &Challenge,
        actions: &mut Vec<f64>,
        cur_flows: &mut Vec<f64>,
        mut cands: Vec<Cand>,
    ) {
        cands.sort_by(|a, b| {
            b.density
                .partial_cmp(&a.density)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal))
        });

        let mut rejected: Vec<Cand> = Vec::new();
        let mut pending = cands;
        for _ in 0..self.cfg.passes {
            let mut next_pending: Vec<Cand> = Vec::new();
            let mut changed = false;

            for cand in pending.into_iter() {
                if cand.battery >= actions.len() || actions[cand.battery].abs() > 1e-9 {
                    continue;
                }
                let node = challenge.batteries[cand.battery].node;
                let alpha = max_feasible_fraction(
                    challenge,
                    cur_flows,
                    node,
                    cand.action,
                    self.cfg.flow_margin,
                );
                if alpha > 1e-8 {
                    let du = cand.action * alpha;
                    if du.abs() <= 1e-10 || cand.value * alpha <= self.cfg.min_value {
                        continue;
                    }
                    actions[cand.battery] = du;
                    if node != challenge.network.slack_bus {
                        for l in 0..challenge.network.num_lines {
                            cur_flows[l] += challenge.network.ptdf[l][node] * du;
                        }
                    }
                    changed = true;
                    if alpha < 0.95 {
                        let mut r = cand;
                        r.action = cand.action * (1.0 - alpha);
                        r.value = cand.value * (1.0 - alpha).max(0.0);
                        rejected.push(r);
                    }
                } else {
                    next_pending.push(cand);
                }
            }

            if !changed || next_pending.is_empty() {
                rejected.extend(next_pending.into_iter());
                break;
            }
            pending = next_pending;
        }

        if self.cfg.residual_retry_enabled {
            rejected.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));
            let retry_cap = challenge.num_batteries.min(rejected.len());
            for cand in rejected.into_iter().take(retry_cap) {
                if cand.battery >= actions.len() || actions[cand.battery].abs() > 1e-9 {
                    continue;
                }
                let node = challenge.batteries[cand.battery].node;
                for frac in [0.35, 0.20, 0.10] {
                    let du = cand.action * frac;
                    if du.abs() < 1e-10 || cand.value * frac <= self.cfg.min_value {
                        continue;
                    }
                    let alpha =
                        max_feasible_fraction(challenge, cur_flows, node, du, self.cfg.flow_margin);
                    if alpha > 0.999 {
                        actions[cand.battery] = du;
                        if node != challenge.network.slack_bus {
                            for l in 0..challenge.network.num_lines {
                                cur_flows[l] += challenge.network.ptdf[l][node] * du;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    fn counterflow_polish_actions(
        &self,
        challenge: &Challenge,
        state: &State,
        mut actions: Vec<f64>,
    ) -> Vec<f64> {
        if !self.cfg.counterflow_enabled || self.cfg.counter_passes == 0 {
            return actions;
        }

        let inj = challenge.compute_total_injections(state, &actions);
        let mut cur_flows = challenge.network.compute_flows(&inj);
        let table = if self.cfg.expected_shadow_enabled {
            self.routed_expected_shadow_table(challenge)
        } else {
            self.routed_shadow_table(challenge)
        };
        let rem = challenge.num_steps.saturating_sub(state.time_step + 1);

        for _ in 0..self.cfg.counter_passes {
            let mut worst_line = 0usize;
            let mut worst_stress = 0.0f64;
            for l in 0..challenge.network.num_lines {
                let stress = cur_flows[l].abs() / challenge.network.flow_limits[l].max(1e-9);
                if stress > worst_stress {
                    worst_stress = stress;
                    worst_line = l;
                }
            }
            if worst_stress < self.cfg.flow_margin * 0.96 {
                break;
            }

            let signed = cur_flows[worst_line].signum();
            if signed.abs() < 1e-12 {
                break;
            }

            let mut best: Option<(f64, usize, f64)> = None;
            for i in 0..challenge.num_batteries {
                if actions[i].abs() > 1e-9 {
                    continue;
                }
                let bat = &challenge.batteries[i];
                let node = bat.node;
                if node == challenge.network.slack_bus {
                    continue;
                }
                let p = challenge.network.ptdf[worst_line][node];
                if p.abs() < 1e-12 {
                    continue;
                }
                let (lo, hi) = state.action_bounds[i];
                let mut opts: Vec<f64> = Vec::with_capacity(2);
                if signed * p > 0.0 && lo < -1e-12 {
                    opts.push(lo * 0.25);
                    opts.push(lo * 0.10);
                } else if signed * p < 0.0 && hi > 1e-12 {
                    opts.push(hi * 0.25);
                    opts.push(hi * 0.10);
                }
                for u in opts {
                    if u.abs() < 1e-10 {
                        continue;
                    }
                    let future_price = table[state.time_step][node];
                    let rt_eff =
                        self.effective_rt_price(challenge, &cur_flows, node, state.rt_prices[node]);
                    let value = shadow_action_value_ctx(
                        bat,
                        u,
                        rt_eff,
                        future_price,
                        self.cfg.shadow_inventory_weight,
                        state.socs[i],
                        self.cfg,
                        rem,
                    );
                    let before = cur_flows[worst_line].abs();
                    let after = (cur_flows[worst_line] + p * u).abs();
                    let relief = ((before - after)
                        / challenge.network.flow_limits[worst_line].max(1e-9))
                    .max(0.0);
                    let score = value + self.cfg.relief_bonus * relief;
                    if score <= self.cfg.min_value {
                        continue;
                    }
                    let alpha =
                        max_feasible_fraction(challenge, &cur_flows, node, u, self.cfg.flow_margin);
                    if alpha > 0.999 && best.map(|b| score > b.0).unwrap_or(true) {
                        best = Some((score, i, u));
                    }
                }
            }

            let Some((_, i, u)) = best else {
                break;
            };
            actions[i] = u;
            let node = challenge.batteries[i].node;
            if node != challenge.network.slack_bus {
                for l in 0..challenge.network.num_lines {
                    cur_flows[l] += challenge.network.ptdf[l][node] * u;
                }
            }
        }

        actions
    }

    fn regret_swap_actions(
        &self,
        challenge: &Challenge,
        state: &State,
        mut actions: Vec<f64>,
        cands: &[Cand],
    ) -> Vec<f64> {
        if !self.cfg.regret_swap_enabled || self.cfg.regret_swap_passes == 0 || cands.is_empty() {
            return actions;
        }

        actions = repair_global_scale(challenge, state, actions);
        let mut best_obj = self.vector_objective(challenge, state, &actions)
            + 1e-9 * challenge.compute_profit(state, &actions);

        let mut ranked = cands.to_vec();
        ranked.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.density.partial_cmp(&a.density).unwrap_or(Ordering::Equal))
        });
        let cap = (challenge.num_batteries * 2).max(8).min(ranked.len());

        for _ in 0..self.cfg.regret_swap_passes {
            let mut changed = false;
            for cand in ranked.iter().take(cap) {
                if cand.battery >= actions.len() || cand.value <= self.cfg.min_value {
                    continue;
                }
                let current = actions[cand.battery];
                if (current - cand.action).abs() < 1e-8 {
                    continue;
                }

                let mut best_trial_obj = best_obj;
                let mut best_trial: Option<Vec<f64>> = None;

                let mut trial = actions.clone();
                trial[cand.battery] = cand.action;
                if is_action_feasible(challenge, state, &trial) {
                    let obj = self.vector_objective(challenge, state, &trial)
                        + 1e-9 * challenge.compute_profit(state, &trial);
                    if obj > best_trial_obj {
                        best_trial_obj = obj;
                        best_trial = Some(trial.clone());
                    }
                } else if let Some((line, sign)) =
                    most_violated_line_for_action(challenge, state, &trial)
                {
                    // Try displacing exactly one lower-value action that worsens the violated line.
                    for j in 0..challenge.num_batteries {
                        if j == cand.battery || actions[j].abs() < 1e-9 {
                            continue;
                        }
                        let node_j = challenge.batteries[j].node;
                        if node_j == challenge.network.slack_bus {
                            continue;
                        }
                        let signed_push = sign * challenge.network.ptdf[line][node_j] * actions[j];
                        if signed_push <= 1e-12 {
                            continue;
                        }
                        let mut trial2 = trial.clone();
                        trial2[j] = 0.0;
                        if !is_action_feasible(challenge, state, &trial2) {
                            continue;
                        }
                        let obj = self.vector_objective(challenge, state, &trial2)
                            + 1e-9 * challenge.compute_profit(state, &trial2);
                        if obj > best_trial_obj {
                            best_trial_obj = obj;
                            best_trial = Some(trial2);
                        }
                    }

                    // Last resort: apply line-wise repair. This can still uncover profitable
                    // replacements when the original greedy packing filled capacity poorly.
                    let repaired = repair_global_scale(challenge, state, trial);
                    let obj = self.vector_objective(challenge, state, &repaired)
                        + 1e-9 * challenge.compute_profit(state, &repaired);
                    if obj > best_trial_obj {
                        best_trial_obj = obj;
                        best_trial = Some(repaired);
                    }
                }

                if let Some(t) = best_trial {
                    if best_trial_obj > best_obj + 1e-7 {
                        actions = t;
                        best_obj = best_trial_obj;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        actions
    }
}

impl BatteryValue {
    fn build(
        challenge: &Challenge,
        bat_idx: usize,
        cfg: Config,
        use_support_dp: bool,
        congestion_probs: &[Vec<f64>],
    ) -> Self {
        let bat = &challenge.batteries[bat_idx];
        let levels = cfg.levels.max(2);
        let steps = challenge.num_steps;
        let min_soc = bat.soc_min_mwh;
        let max_soc = bat.soc_max_mwh;
        let step = if levels > 1 {
            (max_soc - min_soc) / (levels as f64 - 1.0)
        } else {
            1.0
        };

        let mut grid = vec![0.0f64; levels];
        for i in 0..levels {
            grid[i] = min_soc + step * i as f64;
        }

        let mut values = vec![0.0f64; (steps + 1) * levels];

        for t in (0..steps).rev() {
            for si in 0..levels {
                let soc = grid[si];
                let (lo, hi) = bounds_from_soc(bat, soc);

                if use_support_dp {
                    let scenarios = price_scenarios(challenge, t, bat.node, congestion_probs, cfg);
                    let mut expected = 0.0f64;

                    for &(price, weight) in scenarios.iter() {
                        if weight <= 0.0 {
                            continue;
                        }
                        let mut best = 0.0f64;
                        for u0 in action_candidates(lo, hi, cfg.action_levels) {
                            let u = u0.clamp(lo, hi);
                            let next_soc = apply_action_to_soc_manual(bat, u, soc);
                            let imm = profit_at_price(bat, u, price);
                            let fut = interp_flat(&values, levels, min_soc, step, t + 1, next_soc);
                            let val = imm + fut;
                            if val > best {
                                best = val;
                            }
                        }
                        expected += weight * best;
                    }
                    values[t * levels + si] = expected.max(0.0);
                } else {
                    let price = challenge.market.day_ahead_prices[t][bat.node];
                    let mut best = 0.0f64;
                    for u0 in action_candidates(lo, hi, cfg.action_levels) {
                        let u = u0.clamp(lo, hi);
                        let next_soc = apply_action_to_soc_manual(bat, u, soc);
                        let imm = profit_at_price(bat, u, price);
                        let fut = interp_flat(&values, levels, min_soc, step, t + 1, next_soc);
                        let val = imm + fut;
                        if val > best {
                            best = val;
                        }
                    }
                    values[t * levels + si] = best;
                }
            }
        }

        Self {
            min_soc,
            step,
            levels,
            steps,
            values,
        }
    }

    fn value_at(&self, t: usize, soc: f64) -> f64 {
        if t > self.steps {
            return 0.0;
        }
        interp_flat(&self.values, self.levels, self.min_soc, self.step, t, soc)
    }
}

fn interp_flat(values: &[f64], levels: usize, min_soc: f64, step: f64, t: usize, soc: f64) -> f64 {
    let base = t * levels;
    if levels == 0 || base + levels > values.len() {
        return 0.0;
    }
    if levels == 1 || step <= 0.0 {
        return values[base];
    }

    let x = (soc - min_soc) / step;
    if x <= 0.0 {
        return values[base];
    }
    let max_x = (levels - 1) as f64;
    if x >= max_x {
        return values[base + levels - 1];
    }

    let i = x.floor() as usize;
    let frac = x - i as f64;
    values[base + i] * (1.0 - frac) + values[base + i + 1] * frac
}

fn future_price_rank(values: &[Vec<f64>], t: usize, node: usize, price: f64, look: usize) -> f64 {
    if values.is_empty() || t + 1 >= values.len() || look == 0 {
        return 0.50;
    }
    let end = (t + 1 + look).min(values.len());
    let mut below = 0usize;
    let mut count = 0usize;
    for s in (t + 1)..end {
        if node < values[s].len() {
            count += 1;
            if values[s][node] <= price {
                below += 1;
            }
        }
    }
    if count == 0 {
        0.50
    } else {
        below as f64 / count as f64
    }
}

fn future_price_mean(values: &[Vec<f64>], t: usize, node: usize, look: usize) -> f64 {
    if values.is_empty() || t + 1 >= values.len() || look == 0 {
        return 0.0;
    }
    let end = (t + 1 + look).min(values.len());
    let mut sum = 0.0;
    let mut count = 0usize;
    for s in (t + 1)..end {
        if node < values[s].len() {
            sum += values[s][node];
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn price_scenarios(
    challenge: &Challenge,
    t: usize,
    node: usize,
    congestion_probs: &[Vec<f64>],
    cfg: Config,
) -> [(f64, f64); 4] {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    let da = challenge.market.day_ahead_prices[t][node];
    let sigma = challenge.market.params.volatility;
    let jump_p = challenge.market.params.jump_probability.clamp(0.0, 1.0);
    let alpha = challenge.market.params.tail_index.max(1.000_001);
    let pareto_mean = alpha / (alpha - 1.0);

    let cong_prob = congestion_probs
        .get(t)
        .and_then(|row| row.get(node))
        .copied()
        .unwrap_or(0.0);
    let premium = constants::GAMMA_PRICE * INV_SQRT_2PI * cong_prob * cfg.support_congestion_scale;

    let z = 3.0_f64.sqrt() * cfg.support_normal_spread;
    let p0 = clamp_price(da * (1.0 - sigma * z) + premium);
    let p1 = clamp_price(da + premium);
    let p2 = clamp_price(da * (1.0 + sigma * z) + premium);
    let pj = clamp_price(
        da + premium + da * pareto_mean * cfg.support_jump_scale * cfg.support_jump_tail,
    );

    [
        (p0, (1.0 - jump_p) / 6.0),
        (p1, (1.0 - jump_p) * 2.0 / 3.0),
        (p2, (1.0 - jump_p) / 6.0),
        (pj, jump_p),
    ]
}

fn clamp_price(x: f64) -> f64 {
    x.clamp(constants::LAMBDA_MIN, constants::LAMBDA_MAX)
}

fn build_congestion_probability_table(challenge: &Challenge) -> Vec<Vec<f64>> {
    let h = challenge.num_steps;
    let n = challenge.network.num_nodes;
    let mut out = vec![vec![0.0f64; n]; h];

    for t in 1..h {
        let flows = challenge
            .network
            .compute_flows(&challenge.exogenous_injections[t - 1]);
        let mut no_cong = vec![1.0f64; n];

        for l in 0..challenge.network.num_lines {
            let denom = (challenge.network.congestion_threshold * challenge.network.flow_limits[l])
                .max(1e-12);
            let p = (flows[l].abs() / denom).powf(10.0).clamp(0.0, 1.0);
            let (from, to) = challenge.network.lines[l];
            no_cong[from] *= 1.0 - p;
            no_cong[to] *= 1.0 - p;
        }

        for node in 0..n {
            out[t][node] = 1.0 - no_cong[node];
        }
    }

    out
}

fn action_candidates(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let n = n.max(2);
    let mut out = Vec::with_capacity(n + 9);

    for k in 0..n {
        let frac = if n <= 1 {
            0.0
        } else {
            k as f64 / (n - 1) as f64
        };
        push_unique(&mut out, lo + (hi - lo) * frac);
    }

    for u in [
        0.0,
        lo,
        hi,
        lo * 0.75,
        hi * 0.75,
        lo * 0.5,
        hi * 0.5,
        lo * 0.25,
        hi * 0.25,
        lo * 0.10,
        hi * 0.10,
    ] {
        push_unique(&mut out, u.clamp(lo, hi));
    }

    out
}

fn push_unique(v: &mut Vec<f64>, x: f64) {
    if !v.iter().any(|&y| (y - x).abs() < 1e-8) {
        v.push(x);
    }
}

fn bounds_from_soc(bat: &Battery, soc: f64) -> (f64, f64) {
    let dt = constants::DELTA_T;
    let headroom = (bat.soc_max_mwh - soc).max(0.0);
    let available = (soc - bat.soc_min_mwh).max(0.0);

    let max_charge_from_soc = if bat.efficiency_charge > 0.0 {
        headroom / (bat.efficiency_charge * dt)
    } else {
        0.0
    };
    let max_discharge_from_soc = if bat.efficiency_discharge > 0.0 {
        available * bat.efficiency_discharge / dt
    } else {
        0.0
    };

    let max_charge = max_charge_from_soc.min(bat.power_charge_mw).max(0.0);
    let max_discharge = max_discharge_from_soc.min(bat.power_discharge_mw).max(0.0);
    (-max_charge, max_discharge)
}

fn apply_action_to_soc_manual(bat: &Battery, action: f64, soc: f64) -> f64 {
    let c = (-action).max(0.0);
    let d = action.max(0.0);
    let dt = constants::DELTA_T;
    let next = soc + bat.efficiency_charge * c * dt - d * dt / bat.efficiency_discharge;
    next.clamp(bat.soc_min_mwh, bat.soc_max_mwh)
}

fn profit_at_price(bat: &Battery, u: f64, price: f64) -> f64 {
    if u.abs() < 1e-15 {
        return 0.0;
    }
    let dt = constants::DELTA_T;
    let revenue = u * price * dt;
    let abs_u = u.abs();
    let tx = constants::KAPPA_TX * abs_u * dt;
    let deg_base = (abs_u * dt) / bat.capacity_mwh;
    let deg = constants::KAPPA_DEG * deg_base.powf(constants::BETA_DEG);
    revenue - tx - deg
}

fn shadow_action_value(
    bat: &Battery,
    u: f64,
    rt_price: f64,
    future_price: f64,
    inventory_weight: f64,
) -> f64 {
    let immediate = profit_at_price(bat, u, rt_price);
    let dt = constants::DELTA_T;
    let soc_delta = if u < 0.0 {
        (-u) * bat.efficiency_charge * dt
    } else {
        -u * dt / bat.efficiency_discharge
    };
    immediate + inventory_weight * soc_delta * bat.efficiency_discharge * future_price
}

fn marginal_future_price(
    bat: &Battery,
    soc: f64,
    future_price: f64,
    cfg: Config,
    remaining_steps: usize,
) -> f64 {
    let denom = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
    let rel = ((soc - bat.soc_min_mwh) / denom).clamp(0.0, 1.0);
    let mut mult = 1.0 + cfg.soc_skew * (0.5 - rel);

    if cfg.endgame_window > 0 && remaining_steps < cfg.endgame_window {
        let decay = cfg.endgame_floor
            + (1.0 - cfg.endgame_floor) * remaining_steps as f64 / cfg.endgame_window.max(1) as f64;
        mult *= decay + (1.0 - decay) * (0.55 - rel).max(0.0);
    }

    (future_price * mult.clamp(0.05, 2.0)).max(0.0)
}

fn shadow_action_value_ctx(
    bat: &Battery,
    u: f64,
    rt_price: f64,
    future_price: f64,
    inventory_weight: f64,
    soc: f64,
    cfg: Config,
    remaining_steps: usize,
) -> f64 {
    let immediate = profit_at_price(bat, u, rt_price);
    let dt = constants::DELTA_T;
    let soc_delta = if u < 0.0 {
        (-u) * bat.efficiency_charge * dt
    } else {
        -u * dt / bat.efficiency_discharge
    };
    let mf = marginal_future_price(bat, soc, future_price, cfg, remaining_steps);
    immediate + inventory_weight * soc_delta * bat.efficiency_discharge * mf
}

fn detect_track(challenge: &Challenge) -> TrackKind {
    match (
        challenge.network.num_nodes,
        challenge.network.num_lines,
        challenge.num_batteries,
        challenge.num_steps,
    ) {
        (20, 30, 10, 96) => TrackKind::Baseline,
        (40, 60, 20, 96) => TrackKind::Congested,
        (80, 120, 40, 192) => TrackKind::Multiday,
        (100, 200, 60, 192) => TrackKind::Dense,
        (150, 300, 100, 192) => TrackKind::Capstone,
        _ => TrackKind::Unknown,
    }
}

fn build_expected_rt_price_table(challenge: &Challenge) -> Vec<Vec<f64>> {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    let h = challenge.num_steps;
    let n = challenge.network.num_nodes;
    let mut out = challenge.market.day_ahead_prices.clone();

    let alpha = challenge.market.params.tail_index.max(1.000_001);
    let pareto_mean = alpha / (alpha - 1.0);
    let jump_mult = challenge.market.params.jump_probability.clamp(0.0, 1.0) * pareto_mean;

    for t in 0..h {
        for node in 0..n {
            out[t][node] *= 1.0 + jump_mult;
        }
    }

    for t in 1..h {
        let flows = challenge
            .network
            .compute_flows(&challenge.exogenous_injections[t - 1]);
        let mut no_cong = vec![1.0f64; n];
        for l in 0..challenge.network.num_lines {
            let denom = (challenge.network.congestion_threshold * challenge.network.flow_limits[l])
                .max(1e-12);
            let p = (flows[l].abs() / denom).powf(10.0).clamp(0.0, 1.0);
            let (from, to) = challenge.network.lines[l];
            no_cong[from] *= 1.0 - p;
            no_cong[to] *= 1.0 - p;
        }
        for node in 0..n {
            let p_cong = 1.0 - no_cong[node];
            out[t][node] += constants::GAMMA_PRICE * INV_SQRT_2PI * p_cong;
            out[t][node] = out[t][node].clamp(constants::LAMBDA_MIN, constants::LAMBDA_MAX);
        }
    }
    out
}

fn build_future_network_reserve_table(
    challenge: &Challenge,
    cfg: Config,
    discharge_need: bool,
) -> Vec<Vec<f64>> {
    let h = challenge.num_steps;
    let n = challenge.network.num_nodes;
    let mut instant = vec![vec![0.0f64; n]; h];
    if !cfg.network_reserve_enabled || cfg.reserve_window == 0 || cfg.reserve_price_scale <= 0.0 {
        return instant;
    }

    for t in 0..h {
        let flows = challenge
            .network
            .compute_flows(&challenge.exogenous_injections[t]);
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let f = flows[l];
            let stress = (f.abs() / lim).clamp(0.0, 2.0);
            if stress <= cfg.dual_stress_threshold {
                continue;
            }
            let signed = if f >= 0.0 { 1.0 } else { -1.0 };
            let intensity = ((stress - cfg.dual_stress_threshold)
                / (1.0 - cfg.dual_stress_threshold).max(1e-6))
            .clamp(0.0, 2.0);
            for node in 0..n {
                if node == challenge.network.slack_bus {
                    continue;
                }
                let p = challenge.network.ptdf[l][node];
                if p.abs() < 1e-12 {
                    continue;
                }
                let relieves = if discharge_need {
                    signed * p < 0.0
                } else {
                    signed * p > 0.0
                };
                if relieves {
                    instant[t][node] += cfg.reserve_price_scale * intensity * intensity * p.abs();
                }
            }
        }
    }

    let mut out = vec![vec![0.0f64; n]; h];
    for t in 0..h {
        let end = h.min(t + 1 + cfg.reserve_window);
        for s in (t + 1)..end {
            let age = (s - t) as f64;
            let discount = 0.985_f64.powf(age);
            for node in 0..n {
                out[t][node] += discount * instant[s][node];
            }
        }
    }
    out
}

fn build_future_quantile_table_from_values(
    challenge: &Challenge,
    values: &[Vec<f64>],
    q: f64,
    look_cap: usize,
) -> Vec<Vec<f64>> {
    let h = challenge.num_steps;
    let n = challenge.network.num_nodes;
    let mut out = vec![vec![0.0f64; n]; h];

    for t in 0..h {
        for node in 0..n {
            let end = if look_cap == 0 {
                h
            } else {
                h.min(t + 1 + look_cap)
            };
            if t + 1 >= end {
                out[t][node] = 0.0;
                continue;
            }
            let mut vals: Vec<f64> = ((t + 1)..end).map(|s| values[s][node]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let idx = ((vals.len() - 1) as f64 * q).round() as usize;
            out[t][node] = vals[idx.min(vals.len() - 1)];
        }
    }
    out
}

fn build_future_quantile_table(challenge: &Challenge, q: f64, look_cap: usize) -> Vec<Vec<f64>> {
    let h = challenge.num_steps;
    let n = challenge.network.num_nodes;
    let da = &challenge.market.day_ahead_prices;
    let mut out = vec![vec![0.0f64; n]; h];

    for t in 0..h {
        for node in 0..n {
            let end = if look_cap == 0 {
                h
            } else {
                h.min(t + 1 + look_cap)
            };
            if t + 1 >= end {
                out[t][node] = 0.0;
                continue;
            }
            let mut vals: Vec<f64> = ((t + 1)..end).map(|s| da[s][node]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let idx = ((vals.len() - 1) as f64 * q).round() as usize;
            out[t][node] = vals[idx.min(vals.len() - 1)];
        }
    }
    out
}

fn stress_usage(challenge: &Challenge, base_flows: &[f64], node: usize, u: f64) -> f64 {
    if node == challenge.network.slack_bus || u.abs() < 1e-12 {
        return 0.0;
    }

    let mut worst = 0.0f64;
    for l in 0..challenge.network.num_lines {
        let limit = challenge.network.flow_limits[l].max(1e-9);
        let delta = challenge.network.ptdf[l][node] * u;
        let before = base_flows[l].abs();
        let after = (base_flows[l] + delta).abs();
        let inc = ((after - before) / limit).max(0.0);
        if inc > worst {
            worst = inc;
        }
    }
    worst
}

fn add_node_action_to_flows(challenge: &Challenge, flows: &mut [f64], node: usize, u: f64) {
    if node == challenge.network.slack_bus || u.abs() < 1e-12 {
        return;
    }
    for l in 0..challenge.network.num_lines {
        flows[l] += challenge.network.ptdf[l][node] * u;
    }
}

fn flows_feasible_margin(challenge: &Challenge, flows: &[f64], margin: f64) -> bool {
    for l in 0..challenge.network.num_lines {
        if flows[l].abs() > challenge.network.flow_limits[l] * margin + 1e-8 {
            return false;
        }
    }
    true
}

fn most_stressed_line(challenge: &Challenge, flows: &[f64]) -> (usize, f64) {
    let mut idx = 0usize;
    let mut best = 0.0f64;
    for l in 0..challenge.network.num_lines {
        let stress = flows[l].abs() / challenge.network.flow_limits[l].max(1e-9);
        if stress > best {
            best = stress;
            idx = l;
        }
    }
    (idx, best)
}

fn max_feasible_fraction(
    challenge: &Challenge,
    flows: &[f64],
    node: usize,
    u: f64,
    margin: f64,
) -> f64 {
    if node == challenge.network.slack_bus || u.abs() < 1e-12 {
        return 1.0;
    }

    let mut alpha = 1.0f64;
    for l in 0..challenge.network.num_lines {
        let delta = challenge.network.ptdf[l][node] * u;
        if delta.abs() < 1e-12 {
            continue;
        }

        let limit = challenge.network.flow_limits[l] * margin;
        let f = flows[l];
        let a = if delta > 0.0 {
            (limit - f) / delta
        } else {
            (-limit - f) / delta
        };
        if a < alpha {
            alpha = a;
        }
    }
    alpha.clamp(0.0, 1.0)
}

fn is_action_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let inj = challenge.compute_total_injections(state, action);
    let flows = challenge.network.compute_flows(&inj);
    challenge.network.verify_flows(&flows).is_ok()
}

fn most_violated_line_for_action(
    challenge: &Challenge,
    state: &State,
    action: &[f64],
) -> Option<(usize, f64)> {
    let inj = challenge.compute_total_injections(state, action);
    let flows = challenge.network.compute_flows(&inj);
    let mut best_line: Option<usize> = None;
    let mut best_over = 0.0f64;
    let mut best_sign = 1.0f64;
    for l in 0..challenge.network.num_lines {
        let lim = challenge.network.flow_limits[l].max(1e-9);
        let f = flows[l];
        let over = f.abs() - lim;
        if over > best_over {
            best_over = over;
            best_line = Some(l);
            best_sign = if f >= 0.0 { 1.0 } else { -1.0 };
        }
    }
    best_line.map(|l| (l, best_sign))
}

fn battery_action_profit(challenge: &Challenge, state: &State, bat_idx: usize, u: f64) -> f64 {
    if u.abs() < 1e-12 || bat_idx >= challenge.batteries.len() {
        return 0.0;
    }
    let bat = &challenge.batteries[bat_idx];
    let price = state.rt_prices.get(bat.node).copied().unwrap_or(0.0);
    let dt = constants::DELTA_T;
    let abs_u = u.abs();
    let revenue = u * price * dt;
    let tx_cost = constants::KAPPA_TX * abs_u * dt;
    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
    let deg_cost = constants::KAPPA_DEG * deg_base.powf(constants::BETA_DEG);
    revenue - tx_cost - deg_cost
}

fn repair_counterflow_projection(
    challenge: &Challenge,
    state: &State,
    actions: Vec<f64>,
) -> Vec<f64> {
    let mut repaired = actions.clone();
    let zero = vec![0.0f64; actions.len()];
    let zero_inj = challenge.compute_total_injections(state, &zero);
    let zero_flows = challenge.network.compute_flows(&zero_inj);
    if challenge.network.verify_flows(&zero_flows).is_err() {
        return zero;
    }

    let max_iter = (2usize * challenge.network.num_lines.max(1)).min(96);
    for _ in 0..max_iter {
        let inj = challenge.compute_total_injections(state, &repaired);
        let flows = challenge.network.compute_flows(&inj);
        if challenge.network.verify_flows(&flows).is_ok() {
            return repaired;
        }

        let mut worst_line: Option<usize> = None;
        let mut worst_over = 0.0f64;
        let mut worst_sign = 1.0f64;
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let f = flows[l];
            let over = f.abs() - lim * (1.0 - 4.0 * constants::EPS_FLOW);
            if over > worst_over {
                worst_over = over;
                worst_line = Some(l);
                worst_sign = if f >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        let Some(line) = worst_line else {
            break;
        };
        if worst_over <= 1e-9 {
            break;
        }

        // Choose counter-actions by least immediate-profit cost per MW of relief.
        // Positive signed_delta worsens the violated line; negative signed_delta relieves it.
        let mut options: Vec<(f64, f64, usize, f64, f64)> = Vec::new();
        for i in 0..challenge.num_batteries {
            let node = challenge.batteries[i].node;
            if node == challenge.network.slack_bus {
                continue;
            }
            let signed_ptdf = worst_sign * challenge.network.ptdf[line][node];
            if signed_ptdf.abs() < 1e-12 {
                continue;
            }
            let (lo, hi) = state.action_bounds[i];
            let cur = repaired[i];
            let target = if signed_ptdf > 0.0 { lo } else { hi };
            let max_delta = (target - cur).clamp(lo - cur, hi - cur);
            if max_delta.abs() < 1e-12 {
                continue;
            }
            let relief = -(signed_ptdf * max_delta);
            if relief <= 1e-12 {
                continue;
            }
            let before = battery_action_profit(challenge, state, i, cur);
            let after = battery_action_profit(challenge, state, i, cur + max_delta);
            let value_delta = after - before;
            let cost_density = (-value_delta).max(0.0) / relief.max(1e-9);
            options.push((cost_density, -value_delta, i, max_delta, relief));
        }

        if options.is_empty() {
            break;
        }
        options.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        });

        let mut need = worst_over * 1.010;
        let mut changed = false;
        for &(_, _, i, max_delta, relief) in options.iter().take(18) {
            if need <= 1e-9 {
                break;
            }
            let frac = (need / relief.max(1e-9)).clamp(0.0, 1.0);
            if frac <= 1e-9 {
                continue;
            }
            repaired[i] += max_delta * frac;
            need -= relief * frac;
            changed = true;
        }
        if !changed {
            break;
        }
    }

    repaired
}

fn repair_global_scale(challenge: &Challenge, state: &State, actions: Vec<f64>) -> Vec<f64> {
    let inj = challenge.compute_total_injections(state, &actions);
    let flows = challenge.network.compute_flows(&inj);
    if challenge.network.verify_flows(&flows).is_ok() {
        return actions;
    }

    let zero = vec![0.0f64; actions.len()];
    let zero_inj = challenge.compute_total_injections(state, &zero);
    let zero_flows = challenge.network.compute_flows(&zero_inj);
    if challenge.network.verify_flows(&zero_flows).is_err() {
        return zero;
    }

    // Try counterflow projection before curtailment. Many strong portfolios violate one binding
    // PTDF cut by a small amount; this step first looks for bounded counter-actions at other
    // batteries that relieve the violated line at low immediate-profit cost.
    let projected = repair_counterflow_projection(challenge, state, actions.clone());
    let inj_p = challenge.compute_total_injections(state, &projected);
    let flows_p = challenge.network.compute_flows(&inj_p);
    if challenge.network.verify_flows(&flows_p).is_ok() {
        return projected;
    }

    // Prefer a line-wise repair before global scaling.  Global scaling is safe, but it
    // throws away profitable actions that are unrelated to the violated constraint.
    // This loop only damps battery actions that push the currently most-violated line
    // further in the violating direction, leaving counterflow and non-binding actions intact.
    let mut repaired = projected;
    let max_iter = 4usize * challenge.network.num_lines.max(1);

    for _ in 0..max_iter {
        let inj_r = challenge.compute_total_injections(state, &repaired);
        let flows_r = challenge.network.compute_flows(&inj_r);
        if challenge.network.verify_flows(&flows_r).is_ok() {
            return repaired;
        }

        let mut worst_line: Option<usize> = None;
        let mut worst_over = 0.0f64;
        let mut worst_sign = 1.0f64;
        for l in 0..challenge.network.num_lines {
            let lim = challenge.network.flow_limits[l].max(1e-9);
            let f = flows_r[l];
            let over = f.abs() - lim * (1.0 - 4.0 * constants::EPS_FLOW);
            if over > worst_over {
                worst_over = over;
                worst_line = Some(l);
                worst_sign = if f >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        let Some(line) = worst_line else {
            break;
        };
        if worst_over <= 0.0 {
            break;
        }

        let mut push_sum = 0.0f64;
        for (i, &u) in repaired.iter().enumerate() {
            if u.abs() < 1e-12 {
                continue;
            }
            let node = challenge.batteries[i].node;
            if node == challenge.network.slack_bus {
                continue;
            }
            let signed_push = worst_sign * challenge.network.ptdf[line][node] * u;
            if signed_push > 0.0 {
                push_sum += signed_push;
            }
        }

        if push_sum <= 1e-12 {
            break;
        }

        let damp = (1.0 - (worst_over / push_sum) * 1.005).clamp(0.0, 0.995);
        let mut changed = false;
        for i in 0..repaired.len() {
            let u = repaired[i];
            if u.abs() < 1e-12 {
                continue;
            }
            let node = challenge.batteries[i].node;
            if node == challenge.network.slack_bus {
                continue;
            }
            let signed_push = worst_sign * challenge.network.ptdf[line][node] * u;
            if signed_push > 0.0 {
                repaired[i] *= damp;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let inj_r = challenge.compute_total_injections(state, &repaired);
    let flows_r = challenge.network.compute_flows(&inj_r);
    if challenge.network.verify_flows(&flows_r).is_ok() {
        return repaired;
    }

    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        let scaled: Vec<f64> = repaired.iter().map(|u| u * mid).collect();
        let inj2 = challenge.compute_total_injections(state, &scaled);
        let fl2 = challenge.network.compute_flows(&inj2);
        if challenge.network.verify_flows(&fl2).is_ok() {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    repaired.iter().map(|u| u * lo).collect()
}

fn greedy_like_policy(challenge: &Challenge, state: &State) -> Vec<f64> {
    let t = state.time_step;
    let horizon = 12usize;
    let da = &challenge.market.day_ahead_prices;
    let current_da = da[t][0];
    let end = (t + horizon).min(challenge.num_steps);

    let mut future_sum = 0.0;
    let mut future_count = 0.0;
    for s in (t + 1)..end {
        future_sum += da[s][0];
        future_count += 1.0;
    }
    let future_avg = if future_count > 0.0 {
        future_sum / future_count
    } else {
        current_da
    };

    let mut action = vec![0.0f64; challenge.num_batteries];
    for i in 0..challenge.num_batteries {
        let (lo, hi) = state.action_bounds[i];
        if current_da < future_avg - 5.0 {
            action[i] = lo;
        } else if current_da > future_avg + 5.0 {
            action[i] = hi;
        }
    }

    repair_global_scale(challenge, state, action)
}

fn conservative_like_policy(challenge: &Challenge, state: &State) -> Vec<f64> {
    let t = state.time_step;
    let da = &challenge.market.day_ahead_prices[t];
    let avg = da.iter().sum::<f64>() / da.len().max(1) as f64;
    let mut action = vec![0.0f64; challenge.num_batteries];

    for i in 0..challenge.num_batteries {
        let bat = &challenge.batteries[i];
        let (lo, hi) = state.action_bounds[i];
        let node_price = da[bat.node];
        let can_full_charge = lo <= -bat.power_charge_mw + constants::EPS_SOC;
        let can_full_discharge = hi >= bat.power_discharge_mw - constants::EPS_SOC;

        if node_price < 0.95 * avg && can_full_charge {
            action[i] = -bat.power_charge_mw;
        } else if node_price > 1.05 * avg && can_full_discharge {
            action[i] = bat.power_discharge_mw;
        }
        action[i] = action[i].clamp(lo, hi);
    }

    action = repair_global_scale(challenge, state, action);

    let mut profit = challenge.compute_profit(state, &action);
    let mut iters = 0;
    while state.total_profit + profit < 0.0 && iters < 200 {
        for u in action.iter_mut() {
            if u.abs() > 1e-12 {
                *u *= 0.95;
            } else {
                *u = 0.0;
            }
        }
        profit = challenge.compute_profit(state, &action);
        iters += 1;
    }

    action
}
