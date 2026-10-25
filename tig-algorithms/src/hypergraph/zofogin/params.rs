// Per-track settings and the hyperparameters shared by both code paths.

use super::Hp;
use serde_json::{Map, Value};

pub const GROUP_SIZES: [i64; 6] = [0, 1, 2, 4, 8, 16];
pub const ILS_GROUP_SIZES: [i64; 5] = [0, 2, 4, 8, 16];
pub const GROUP_SIZE_10K: i64 = 8;

pub struct Effort {
    pub refine: usize,
    pub ils: usize,
    pub ils_quick: usize,
    pub polish: usize,
    pub post_balance: usize,
}

const fn effort(
    refine: usize,
    ils: usize,
    ils_quick: usize,
    polish: usize,
    post_balance: usize,
) -> Effort {
    Effort {
        refine,
        ils,
        ils_quick,
        polish,
        post_balance,
    }
}

pub struct Defaults {
    pub round_cap: Option<usize>,
    pub explicit_permille: Option<usize>,
    pub long_budget_trims: Option<LongBudgetTrims>,
    pub short_budget_trims: Option<ShortBudgetTrims>,
    pub grp_ils: Option<i64>,
    pub light_tail: Option<LightTail>,
    pub effort: [Effort; 6],
    pub fuel_fixed: u64,
    pub fuel_per_round: u64,
    pub swap_scale: u64,
    pub fm_rounds: usize,
    pub tabu_tenure: usize,
    pub move_limit: usize,
    pub passes: usize,
    pub neg_gain_thresh: i32,
}

pub struct LongBudgetTrims {
    pub min_rounds: usize,
    pub ils_iterations: Option<i64>,
    pub skip: Option<i64>,
    pub post_ils_polish: Option<i64>,
    pub gc: Option<i64>,
    pub ml_levels: Option<i64>,
    pub ml2_passes: Option<i64>,
    pub ml2_rounds: Option<i64>,
}

pub struct ShortBudgetTrims {
    pub max_rounds: usize,
    pub ils_iterations: Option<i64>,
    pub post_ils_polish: Option<i64>,
    pub swap_scale: Option<i64>,
    pub ml_levels: Option<i64>,
}

pub struct LightTail {
    pub fm_rounds: Option<i64>,
    pub ml_fine: Option<i64>,
    pub jet_rounds: Option<i64>,
    pub cyc_rounds: Option<i64>,
}

pub fn configured_hp(
    hp: &Hp<'_>,
    d: &Defaults,
    knobs: &mut Knobs,
) -> anyhow::Result<Option<Map<String, Value>>> {
    let trims = d
        .long_budget_trims
        .as_ref()
        .filter(|t| knobs.refinement_rounds >= t.min_rounds);
    let short = d
        .short_budget_trims
        .as_ref()
        .filter(|t| knobs.refinement_rounds < t.max_rounds);
    anyhow::ensure!(
        trims.is_none() || short.is_none(),
        "overlapping budget settings"
    );
    if trims.is_none() && short.is_none() && d.light_tail.is_none() && d.grp_ils.is_none() {
        return Ok(None);
    }
    let mut values = (*hp.0).clone().unwrap_or_default();
    let mut insert =
        |key: &str, value: Option<i64>, valid: &dyn Fn(i64) -> bool| -> anyhow::Result<()> {
            if let Some(v) = value {
                anyhow::ensure!(valid(v), "invalid default {}={}", key, v);
                if !values.get(key).and_then(Value::as_i64).map_or(false, valid) {
                    values.insert(key.to_owned(), Value::from(v));
                }
            }
            Ok(())
        };
    let range = |lo: i64, hi: i64| move |v: i64| (lo..=hi).contains(&v);
    if let Some(t) = trims {
        insert("ils_iterations", t.ils_iterations, &range(0, 500))?;
        insert("skip", t.skip, &range(0, 127))?;
        insert("post_ils_polish", t.post_ils_polish, &range(0, 500))?;
        insert("gc", t.gc, &range(0, 1))?;
        insert("ml_levels", t.ml_levels, &range(1, 4))?;
        insert("ml2_passes", t.ml2_passes, &range(1, 16))?;
        insert("ml2_rounds", t.ml2_rounds, &range(1, 20))?;
    }
    if let Some(t) = short {
        insert("ils_iterations", t.ils_iterations, &range(0, 500))?;
        insert("post_ils_polish", t.post_ils_polish, &range(0, 500))?;
        insert("swap_scale", t.swap_scale, &range(0, 400))?;
        insert("ml_levels", t.ml_levels, &range(1, 4))?;
    }
    insert("grp_ils", d.grp_ils, &|v| ILS_GROUP_SIZES.contains(&v))?;
    if let Some(t) = &d.light_tail {
        insert("fm_rounds", t.fm_rounds, &range(1, 200))?;
        insert("ml_fine", t.ml_fine, &range(0, 5))?;
        insert("jet_rounds", t.jet_rounds, &range(1, 64))?;
        insert("cyc_rounds", t.cyc_rounds, &range(0, 64))?;
    }
    let resolved = Some(values);
    let h = Hp(&resolved);
    knobs.ils_iterations = h.int("ils_iterations", 0, 500, knobs.ils_iterations as i64) as usize;
    knobs.post_ils_polish = h.int("post_ils_polish", 0, 500, knobs.post_ils_polish as i64) as usize;
    knobs.swap_scale = h.int("swap_scale", 0, 400, knobs.swap_scale as i64) as u64;
    knobs.fm_rounds = h.int("fm_rounds", 1, 200, knobs.fm_rounds as i64) as usize;
    knobs.jet_rounds = h.int("jet_rounds", 1, 64, knobs.jet_rounds as i64) as usize;
    knobs.cyc_rounds = h.int("cyc_rounds", 0, 64, knobs.cyc_rounds as i64) as usize;
    Ok(resolved)
}

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
        let refinement_rounds = {
            let budget = super::fuel_remaining();
            let explicit = hp.get("refinement", 50, 400_000).is_some();
            let refinement_rounds = match (explicit, d.explicit_permille, d.round_cap) {
                (true, Some(permille), _) => {
                    assert!(
                        (100..=1000).contains(&permille),
                        "explicit_permille outside 100..1000"
                    );
                    refinement_rounds.min((refinement_rounds * permille / 1000).max(50))
                }
                (_, _, Some(cap)) => {
                    assert!(
                        (50..=400_000).contains(&cap),
                        "round_cap outside 50..400000"
                    );
                    refinement_rounds.min(cap)
                }
                _ => refinement_rounds,
            };
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
            ils_iterations: hp.int("ils_iterations", 0, 500, effort.ils as i64) as usize,
            ils_quick_refine: hp.int("ils_quick_refine", 10, 500, effort.ils_quick as i64) as usize,
            post_ils_polish: hp.int("post_ils_polish", 0, 500, effort.polish as i64) as usize,
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
            num_high_hedges: (hp.int("num_high_hedges", 50, 50_000, 500) as usize)
                .min(num_hyperedges),
            passes: hp.int("passes", 1, 16, d.passes as i64) as usize,
            neg_gain_thresh: hp.int("neg_gain_thresh", 0, 200, d.neg_gain_thresh as i64) as i32,
        }
    }
}

pub static D_10K: Defaults = Defaults {
    short_budget_trims: None,
    grp_ils: Some(8),
    round_cap: None,
    explicit_permille: None,
    long_budget_trims: None,
    light_tail: None,
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

pub struct Ml2Defaults {
    pub passes: i64,
    pub levels: i64,
    pub cmax: i64,
    pub rounds: i64,
    pub seeds: i64,
    pub steps: i64,
    pub cycles: i64,
    pub cycles_last: i64,
}

pub struct Track {
    pub d: Defaults,
    pub ml2: Option<Ml2Defaults>,
    pub slack_scale: usize,
    pub cool_pow: u32,
    pub tabu_fail_tenure: usize,
    pub tabu_fail_mark_len: usize,
    pub extra_window: usize,
    pub picks_prefix: usize,
    pub filt_blocks: u32,
    pub hub_degree: i32,
    pub lanes4_degree: i32,
    pub lanes2_degree: i32,
    pub ml_tail: i64,
    pub grp_n: i64,
    pub k_cluster: &'static str,
    pub k_preferences: &'static str,
    pub k_assignments: &'static str,
    pub k_edge_flags: &'static str,
    pub edge_flags_large_list: bool,
    pub k_edge_flags_incr: Option<&'static str>,
    pub k_expand: Option<&'static str>,
    pub k_moves: &'static str,
    pub k_balance: &'static str,
    pub k_connectivity: &'static str,
    pub k_reduce_conn: &'static str,
    pub k_swap_gains: &'static str,
    pub k_choose_elite: &'static str,
    pub k_assign_elite: &'static str,
}

pub static P_20K: Track = Track {
    ml2: None,
    d: Defaults {
        short_budget_trims: None,
        grp_ils: None,
        round_cap: Some(9000),
        explicit_permille: Some(1000),
        long_budget_trims: Some(LongBudgetTrims {
            min_rounds: 5000,
            ils_iterations: Some(0),
            skip: Some(64),
            post_ils_polish: Some(0),
            gc: None,
            ml_levels: None,
            ml2_passes: None,
            ml2_rounds: None,
        }),
        light_tail: None,
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
    tabu_fail_tenure: 3,
    tabu_fail_mark_len: 2048,
    extra_window: 49152,
    picks_prefix: 2048,
    filt_blocks: 16,
    hub_degree: 64,
    lanes4_degree: 32,
    lanes2_degree: 16,
    ml_tail: 0,
    grp_n: 8,
    k_cluster: "hyperedge_clustering_20k",
    k_preferences: "compute_node_preferences_20k",
    k_assignments: "execute_node_assignments_20k",
    k_edge_flags: "precompute_edge_flags_wpe_20k",
    edge_flags_large_list: true,
    k_edge_flags_incr: None,
    k_expand: None,
    k_moves: "compute_refinement_moves_warp8_20k",
    k_balance: "balance_final_20k",
    k_connectivity: "compute_connectivity_20k",
    k_reduce_conn: "reduce_connectivity_sum_20k",
    k_swap_gains: "compute_swap_gains_extended_20k",
    k_choose_elite: "choose_elite_per_hyperedge_20k",
    k_assign_elite: "assign_from_elite_votes_20k",
};

pub static P_50K: Track = Track {
    ml2: None,
    d: Defaults {
        short_budget_trims: None,
        grp_ils: None,
        round_cap: None,
        explicit_permille: None,
        long_budget_trims: Some(LongBudgetTrims {
            min_rounds: 5000,
            ils_iterations: Some(0),
            skip: Some(64),
            post_ils_polish: Some(0),
            gc: None,
            ml_levels: None,
            ml2_passes: None,
            ml2_rounds: None,
        }),
        light_tail: None,
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
    tabu_fail_tenure: 3,
    tabu_fail_mark_len: 4096,
    extra_window: 57344,
    picks_prefix: 4096,
    filt_blocks: 32,
    hub_degree: 128,
    lanes4_degree: 64,
    lanes2_degree: 32,
    ml_tail: 1,
    grp_n: 8,
    k_cluster: "hyperedge_clustering_50k",
    k_preferences: "compute_node_preferences_50k",
    k_assignments: "execute_node_assignments_50k",
    k_edge_flags: "precompute_edge_flags_wpe_50k",
    edge_flags_large_list: true,
    k_edge_flags_incr: None,
    k_expand: None,
    k_moves: "compute_refinement_moves_warp8_50k",
    k_balance: "balance_final_50k",
    k_connectivity: "compute_connectivity_50k",
    k_reduce_conn: "reduce_connectivity_sum_50k",
    k_swap_gains: "compute_swap_gains_extended_50k",
    k_choose_elite: "choose_elite_per_hyperedge_50k",
    k_assign_elite: "assign_from_elite_votes_50k",
};

pub static P_100K: Track = Track {
    ml2: Some(Ml2Defaults {
        passes: 4,
        levels: 1,
        cmax: 4,
        rounds: 3,
        seeds: 1000,
        steps: 2000,
        cycles: 1,
        cycles_last: 0,
    }),
    d: Defaults {
        short_budget_trims: Some(ShortBudgetTrims {
            max_rounds: 5000,
            ils_iterations: Some(0),
            post_ils_polish: Some(0),
            swap_scale: Some(0),
            ml_levels: Some(1),
        }),
        grp_ils: None,
        round_cap: Some(5500),
        explicit_permille: Some(619),
        long_budget_trims: Some(LongBudgetTrims {
            min_rounds: 5000,
            ils_iterations: Some(0),
            skip: Some(80),
            post_ils_polish: Some(0),
            gc: None,
            ml_levels: None,
            ml2_passes: Some(3),
            ml2_rounds: Some(2),
        }),
        light_tail: Some(LightTail {
            fm_rounds: Some(2),
            ml_fine: Some(0),
            jet_rounds: Some(2),
            cyc_rounds: Some(0),
        }),
        effort: [
            effort(3150, 3, 20, 30, 32),
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
    tabu_fail_tenure: 4,
    tabu_fail_mark_len: 4096,
    extra_window: 61440,
    picks_prefix: 8192,
    filt_blocks: 48,
    hub_degree: 128,
    lanes4_degree: 128,
    lanes2_degree: 128,
    ml_tail: 1,
    grp_n: 0,
    k_cluster: "hyperedge_clustering_100k",
    k_preferences: "compute_node_preferences_100k",
    k_assignments: "execute_node_assignments_100k",
    k_edge_flags: "precompute_edge_flags_100k",
    edge_flags_large_list: false,
    k_edge_flags_incr: Some("precompute_edge_flags_hlist_100k"),
    k_expand: Some("expand_node_hedges_100k"),
    k_moves: "compute_refinement_moves_warp8_100k",
    k_balance: "balance_final_100k",
    k_connectivity: "compute_connectivity_100k",
    k_reduce_conn: "reduce_connectivity_sum_100k",
    k_swap_gains: "compute_swap_gains_extended_100k",
    k_choose_elite: "choose_elite_per_hyperedge_100k",
    k_assign_elite: "assign_from_elite_votes_100k",
};

pub static P_200K: Track = Track {
    ml2: Some(Ml2Defaults {
        passes: 3,
        levels: 1,
        cmax: 4,
        rounds: 3,
        seeds: 1000,
        steps: 2000,
        cycles: 1,
        cycles_last: 0,
    }),
    d: Defaults {
        short_budget_trims: None,
        grp_ils: None,
        round_cap: Some(6400),
        explicit_permille: Some(722),
        long_budget_trims: Some(LongBudgetTrims {
            min_rounds: 5000,
            ils_iterations: Some(0),
            skip: None,
            post_ils_polish: Some(0),
            gc: Some(1),
            ml_levels: Some(1),
            ml2_passes: None,
            ml2_rounds: None,
        }),
        light_tail: Some(LightTail {
            fm_rounds: Some(2),
            ml_fine: Some(0),
            jet_rounds: Some(2),
            cyc_rounds: Some(0),
        }),
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
    tabu_fail_tenure: 4,
    tabu_fail_mark_len: 4096,
    extra_window: 65536,
    picks_prefix: 8192,
    filt_blocks: 60,
    hub_degree: 128,
    lanes4_degree: 128,
    lanes2_degree: 128,
    ml_tail: 1,
    grp_n: 8,
    k_cluster: "hyperedge_clustering_200k",
    k_preferences: "compute_node_preferences_200k",
    k_assignments: "execute_node_assignments_200k",
    k_edge_flags: "precompute_edge_flags_200k",
    edge_flags_large_list: false,
    k_edge_flags_incr: Some("precompute_edge_flags_hlist_200k"),
    k_expand: Some("expand_node_hedges_200k"),
    k_moves: "compute_refinement_moves_warp8_200k",
    k_balance: "balance_final_200k",
    k_connectivity: "compute_connectivity_200k",
    k_reduce_conn: "reduce_connectivity_sum_200k",
    k_swap_gains: "compute_swap_gains_extended_200k",
    k_choose_elite: "choose_elite_per_hyperedge_200k",
    k_assign_elite: "assign_from_elite_votes_200k",
};
