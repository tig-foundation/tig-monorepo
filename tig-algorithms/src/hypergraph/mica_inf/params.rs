// Per-track settings and the hyperparameters shared by both code paths.

use super::Hp;

// Stage budgets selected by the `effort` hyperparameter.
pub struct Effort {
    pub refine: usize,
    pub ils: usize,
    pub ils_quick: usize,
    pub polish: usize,
    pub post_balance: usize,
}

const fn effort(refine: usize, ils: usize, ils_quick: usize, polish: usize, post_balance: usize) -> Effort {
    Effort { refine, ils, ils_quick, polish, post_balance }
}

// Defaults of the shared hyperparameters, i.e. what `{}` selects on a track.
pub struct Defaults {
    pub effort: [Effort; 6],
    // Fuel ceiling of the refinement loop: fixed cost plus a per-round cost.
    pub fuel_fixed: u64,
    pub fuel_per_round: u64,
    pub swap_scale: u64,
    pub fm_rounds: usize,
    pub tabu_tenure: usize,
    pub move_limit: usize,
    pub passes: usize,
    pub neg_gain_thresh: i32,
}

// Shared hyperparameters, resolved once per solve.
pub struct Knobs {
    pub effort: &'static Effort,
    pub clusters: i32,
    pub refinement_rounds: usize,
    pub ils_iterations: usize,
    pub ils_quick_refine: usize,
    pub post_ils_polish: usize,
    pub swap_scale: u64,
    pub fm_rounds: usize,
    pub fm_steps: usize,
    pub fm_seeds: usize,
    pub fm_stop: f64,
    pub jet_rounds: usize,
    pub cyc_rounds: usize,
    pub tabu_tenure: usize,
    pub move_limit: usize,
    pub max_stagnant_rounds: usize,
    pub num_high_hedges: usize,
    pub passes: usize,
    pub neg_gain_thresh: i32,
}

impl Knobs {
    pub fn read(hp: &Hp, d: &'static Defaults, num_hyperedges: usize) -> Knobs {
        let mut clusters = hp.int("clusters", 4, 256, 64) as i32;
        if clusters % 4 != 0 {
            clusters += 4 - (clusters % 4);
        }
        let effort = &d.effort[hp.int("effort", 0, 5, 5) as usize];
        let refinement_rounds = hp.int("refinement", 50, 400_000, effort.refine as i64) as usize;
        // Caps the round count to what the granted fuel affords.
        let refinement_rounds = {
            let budget = super::fuel_remaining();
            if budget == 0 {
                refinement_rounds
            } else {
                let usable = (budget / 10) * 7;
                let affordable = usable.saturating_sub(d.fuel_fixed) / d.fuel_per_round;
                refinement_rounds.min(affordable.max(50) as usize)
            }
        };
        Knobs {
            effort,
            clusters,
            refinement_rounds,
            ils_iterations: hp.int("ils_iterations", 1, 500, effort.ils as i64) as usize,
            ils_quick_refine: hp.int("ils_quick_refine", 10, 500, effort.ils_quick as i64) as usize,
            post_ils_polish: hp.int("post_ils_polish", 20, 500, effort.polish as i64) as usize,
            swap_scale: hp.int("swap_scale", 0, 400, d.swap_scale as i64) as u64,
            fm_rounds: hp.int("fm_rounds", 1, 200, d.fm_rounds as i64) as usize,
            fm_steps: hp.int("fm_steps", 10, 4000, 400) as usize,
            fm_seeds: hp.int("fm_seeds", 1, 4000, 25) as usize,
            fm_stop: hp.int("fm_stop", 1, 400, 25) as f64 / 100.0,
            jet_rounds: hp.int("jet_rounds", 1, 64, 8) as usize,
            cyc_rounds: hp.int("cyc_rounds", 0, 64, 4) as usize,
            tabu_tenure: hp.int("tabu_tenure", 1, 30, d.tabu_tenure as i64) as usize,
            move_limit: hp.int("move_limit", 256, 1_000_000, d.move_limit as i64) as usize,
            max_stagnant_rounds: hp.int("max_stagnant_rounds", 2, 500, 30) as usize,
            num_high_hedges: (hp.int("num_high_hedges", 50, 50_000, 500) as usize).min(num_hyperedges),
            passes: hp.int("passes", 1, 16, d.passes as i64) as usize,
            neg_gain_thresh: hp.int("neg_gain_thresh", 0, 200, d.neg_gain_thresh as i64) as i32,
        }
    }
}

pub static D_10K: Defaults = Defaults {
    effort: [
        effort(3000, 3, 20, 30, 0),
        effort(4000, 3, 25, 40, 0),
        effort(5000, 5, 60, 100, 0),
        effort(8000, 5, 60, 150, 0),
        effort(10000, 5, 60, 200, 0),
        effort(18000, 15, 180, 220, 0),
    ],
    fuel_fixed: 25_000_000_000,
    fuel_per_round: 4_000_000,
    swap_scale: 100,
    fm_rounds: 40,
    tabu_tenure: 7,
    move_limit: 800_000,
    passes: 2,
    neg_gain_thresh: 3,
};

// Settings of the 20k / 50k / 100k / 200k code path.
pub struct Track {
    pub d: Defaults,
    pub slack_scale: usize,
    pub cool_pow: u32,
    // Target tie-break mode of the move kernel.
    pub zgm_mode: i32,
    pub tabu_fail_tenure: usize,
    pub tabu_fail_mark_len: usize,
    pub extra_window: usize,
    // Task-to-thread grouping by degree in the move kernel.
    pub group_by_degree: bool,
    pub moves_threads_per_node: u32,
    // Kernel names. One PTX module holds all tracks, so each name carries a track suffix.
    pub k_cluster: &'static str,
    pub k_preferences: &'static str,
    pub k_assignments: &'static str,
    pub k_edge_flags: &'static str,
    // Whether the edge-flag kernel takes the list of hyperedges with more than 32 pins.
    pub edge_flags_large_list: bool,
    pub k_edge_flags_incr: Option<&'static str>,
    pub k_moves: &'static str,
    pub k_balance: &'static str,
    pub k_connectivity: &'static str,
    pub k_reduce_conn: &'static str,
    pub k_swap_gains: &'static str,
    pub k_choose_elite: &'static str,
    pub k_assign_elite: &'static str,
}

pub static P_20K: Track = Track {
    d: Defaults {
        effort: [
            effort(500, 3, 20, 30, 32),
            effort(1000, 3, 25, 40, 32),
            effort(2000, 5, 50, 100, 64),
            effort(3000, 5, 50, 150, 64),
            effort(5000, 5, 60, 200, 0),
            effort(30000, 6, 70, 300, 0),
        ],
        fuel_fixed: 30_000_000_000,
        fuel_per_round: 4_500_000,
        swap_scale: 100,
        fm_rounds: 5,
        tabu_tenure: 12,
        move_limit: 200_000,
        passes: 3,
        neg_gain_thresh: 5,
    },
    slack_scale: 1,
    cool_pow: 2,
    zgm_mode: 0,
    tabu_fail_tenure: 3,
    tabu_fail_mark_len: 2048,
    extra_window: 49152,
    group_by_degree: false,
    moves_threads_per_node: 8,
    k_cluster: "hyperedge_clustering_20k",
    k_preferences: "compute_node_preferences_20k",
    k_assignments: "execute_node_assignments_20k",
    k_edge_flags: "precompute_edge_flags_wpe_20k",
    edge_flags_large_list: false,
    k_edge_flags_incr: None,
    k_moves: "compute_refinement_moves_warp8_20k",
    k_balance: "balance_final_20k",
    k_connectivity: "compute_connectivity_20k",
    k_reduce_conn: "reduce_connectivity_sum_20k",
    k_swap_gains: "compute_swap_gains_extended_20k",
    k_choose_elite: "choose_elite_per_hyperedge_20k",
    k_assign_elite: "assign_from_elite_votes_20k",
};

pub static P_50K: Track = Track {
    d: Defaults {
        effort: [
            effort(500, 3, 20, 30, 32),
            effort(1000, 3, 25, 40, 32),
            effort(2000, 5, 50, 100, 64),
            effort(4000, 5, 50, 150, 64),
            effort(6000, 5, 60, 200, 0),
            effort(24000, 10, 100, 200, 128),
        ],
        fuel_fixed: 30_000_000_000,
        fuel_per_round: 11_000_000,
        swap_scale: 100,
        fm_rounds: 10,
        tabu_tenure: 8,
        move_limit: 800_000,
        passes: 1,
        neg_gain_thresh: 5,
    },
    slack_scale: 8,
    cool_pow: 2,
    zgm_mode: 0,
    tabu_fail_tenure: 3,
    tabu_fail_mark_len: 4096,
    extra_window: 57344,
    group_by_degree: true,
    moves_threads_per_node: 8,
    k_cluster: "hyperedge_clustering_50k",
    k_preferences: "compute_node_preferences_50k",
    k_assignments: "execute_node_assignments_50k",
    k_edge_flags: "precompute_edge_flags_wpe_50k",
    edge_flags_large_list: true,
    k_edge_flags_incr: None,
    k_moves: "compute_refinement_moves_warp8_50k",
    k_balance: "balance_final_50k",
    k_connectivity: "compute_connectivity_50k",
    k_reduce_conn: "reduce_connectivity_sum_50k",
    k_swap_gains: "compute_swap_gains_extended_50k",
    k_choose_elite: "choose_elite_per_hyperedge_50k",
    k_assign_elite: "assign_from_elite_votes_50k",
};

pub static P_100K: Track = Track {
    d: Defaults {
        effort: [
            effort(500, 3, 20, 30, 32),
            effort(1000, 3, 25, 40, 32),
            effort(2000, 5, 50, 100, 64),
            effort(5000, 5, 50, 150, 64),
            effort(7000, 5, 60, 200, 0),
            effort(15000, 6, 70, 300, 0),
        ],
        fuel_fixed: 90_000_000_000,
        fuel_per_round: 20_000_000,
        swap_scale: 100,
        fm_rounds: 5,
        tabu_tenure: 12,
        move_limit: 200_000,
        passes: 1,
        neg_gain_thresh: 5,
    },
    slack_scale: 24,
    cool_pow: 3,
    zgm_mode: 1,
    tabu_fail_tenure: 4,
    tabu_fail_mark_len: 4096,
    extra_window: 61440,
    group_by_degree: false,
    moves_threads_per_node: 1,
    k_cluster: "hyperedge_clustering_100k",
    k_preferences: "compute_node_preferences_100k",
    k_assignments: "execute_node_assignments_100k",
    k_edge_flags: "precompute_edge_flags_100k",
    edge_flags_large_list: false,
    k_edge_flags_incr: Some("precompute_edge_flags_hlist_100k"),
    k_moves: "compute_refinement_moves_smem_100k",
    k_balance: "balance_final_100k",
    k_connectivity: "compute_connectivity_100k",
    k_reduce_conn: "reduce_connectivity_sum_100k",
    k_swap_gains: "compute_swap_gains_extended_100k",
    k_choose_elite: "choose_elite_per_hyperedge_100k",
    k_assign_elite: "assign_from_elite_votes_100k",
};

pub static P_200K: Track = Track {
    d: Defaults {
        effort: [
            effort(500, 5, 50, 25, 64),
            effort(1000, 5, 50, 50, 64),
            effort(2000, 5, 50, 100, 64),
            effort(5000, 5, 50, 150, 64),
            effort(7000, 5, 60, 200, 0),
            effort(15000, 6, 70, 300, 0),
        ],
        fuel_fixed: 200_000_000_000,
        fuel_per_round: 38_000_000,
        swap_scale: 0,
        fm_rounds: 5,
        tabu_tenure: 14,
        move_limit: 131_072,
        passes: 1,
        neg_gain_thresh: 5,
    },
    slack_scale: 20,
    cool_pow: 2,
    zgm_mode: 0,
    tabu_fail_tenure: 4,
    tabu_fail_mark_len: 4096,
    extra_window: 65536,
    group_by_degree: true,
    moves_threads_per_node: 8,
    k_cluster: "hyperedge_clustering_200k",
    k_preferences: "compute_node_preferences_200k",
    k_assignments: "execute_node_assignments_200k",
    k_edge_flags: "precompute_edge_flags_200k",
    edge_flags_large_list: false,
    k_edge_flags_incr: Some("precompute_edge_flags_hlist_200k"),
    k_moves: "compute_refinement_moves_warp8_200k",
    k_balance: "balance_final_200k",
    k_connectivity: "compute_connectivity_200k",
    k_reduce_conn: "reduce_connectivity_sum_200k",
    k_swap_gains: "compute_swap_gains_extended_200k",
    k_choose_elite: "choose_elite_per_hyperedge_200k",
    k_assign_elite: "assign_from_elite_votes_200k",
};
