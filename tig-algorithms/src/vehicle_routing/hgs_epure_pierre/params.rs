use super::problem::Problem;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const DEFAULT_EXPLORATION_LEVEL: usize = 4;

const SIZE_DEFAULTS: [(usize, &str); 5] = [
    (
        600,
        r#"{"band_bar":204939,"band_low":9000,"band_high":2025,"band_level":3,"band_lloyd":2,"band_sub_it_permille":4800,"band_phases":20,"band_stall":5,"post_high":2025,"post_low":9000,"post_phases":32,"post_stall":8,"post_sub_it_permille":19200,"triage_low":9000,"triage_permille":391,"aux_level":210332,"aux_offset":0,"aux_flag":true}"#,
    ),
    (
        700,
        r#"{"bar_factor":0.0,"bar_pos_only":true,"band_bar":196255,"band_low":4000,"band_high":4000,"band_level":4,"inh_min_level":3,"band_lloyd":2,"band_sub_it_permille":4800,"band_phases":20,"band_stall":5,"allow_swap3":true,"granularity":20,"granularity2":30,"swapstar_capa_filter":0.5,"max_credit_deterioration":20,"decomp_nb_phases":16,"sub_it_permille":4000,"pair_mode":3,"triage_permille":380,"triage_low":10632,"triage_fork":1,"triage_boost_low":1000000,"triage_boost_high":1000000,"triage_gate":1,"triage_gate_low":1763,"triage_gate_high":237,"triage_reseed":1,"post_low":2340,"low_post_entry":340,"low_stop":196792,"post_high":3660,"post_phases":19,"post_stall":6,"post_sub_it_permille":9600,"aux_level":205915,"aux_offset":0,"aux_flag":true}"#,
    ),
    (
        800,
        r#"{"pen_warm":2,"band_bar":191725,"band_low":4000,"band_high":4577,"band_level":3,"band_lloyd":2,"band_sub_it_permille":4800,"band_phases":20,"band_stall":5,"triage_permille":379,"triage_low":9000,"triage_fork":1,"triage_boost_low":1000000,"triage_boost_high":1000000,"triage_gate":1,"triage_gate_low":1347,"triage_gate_high":653,"triage_reseed":1,"post_low":1295,"post_high":4577,"post_phases":9,"post_stall":4,"post_sub_it_permille":19200,"aux_level":201207,"aux_offset":0,"aux_flag":true}"#,
    ),
    (
        900,
        r#"{"decomp_target_size":200,"decomp_nb_phases":10,"l4_permille":200,"pen_warm":2,"cluster_lloyd":2,"band_bar":191403,"band_low":5300,"low_entry":4000,"low_stop":187927,"band_high":0,"band_level":3,"band_lloyd":2,"band_sub_it_permille":9600,"band_phases":12,"band_stall":3,"triage_permille":200,"triage_low":14800,"triage_boost_low":1000000,"triage_boost_high":1000000,"triage_fork":1,"triage_gate":1,"triage_gate_low":5776,"triage_gate_high":-3776,"triage_reseed":1,"triage_band_phases":12,"triage_post_phases":9,"triage_sub_it_permille":6400,"triage_extra_phases":4,"inh_k":4,"allow_swap3":true,"granularity":30,"granularity2":40,"max_credit_deterioration":20,"post_high":0,"post_low":4000,"post_phases":9,"post_stall":4,"post_sub_it_permille":19200,"swapstar_capa_filter":0.5,"aux_level":192795,"aux_offset":0,"aux_flag":true}"#,
    ),
    (
        1000,
        r#"{"pen_warm":2,"cluster_lloyd":1,"decomp_nb_phases":11,"band_bar":196090,"band_low":6000,"band_high":0,"band_level":3,"band_lloyd":2,"band_sub_it_permille":4800,"band_phases":5,"band_stall":2,"repair_permille":500,"allow_swap3":true,"granularity":30,"granularity2":40,"swapstar_capa_filter":0.5,"max_credit_deterioration":20,"triage_permille":100,"triage_low":20000,"triage_fork":1,"triage_boost_low":1000000,"triage_boost_high":-7471,"triage_gate":1,"triage_gate_low":7590,"triage_band_phases":2,"triage_post_phases":0,"triage_gate_high":-4615,"triage_sub_it_permille":12800,"triage_extra_phases":8,"triage_reseed":1,"bar_factor":0.5,"post_high":0,"post_low":6000,"post_phases":32,"post_stall":4,"post_sub_it_permille":19200,"aux_level":201767,"aux_offset":0,"aux_flag":true}"#,
    ),
];

pub(super) fn effective_hyperparameters(
    nb_nodes: usize,
    user: &Option<Map<String, Value>>,
) -> Option<Map<String, Value>> {
    let mut effective = SIZE_DEFAULTS
        .iter()
        .find(|(size, _)| *size == nb_nodes)
        .and_then(|(_, json)| serde_json::from_str::<Map<String, Value>>(json).ok())
        .unwrap_or_default();
    if let Some(user) = user {
        effective.extend(user.clone());
    }
    if effective.is_empty() {
        None
    } else {
        Some(effective)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub(super) struct Params {
    pub(super) exploration_level: usize,
    pub(super) allow_swap3: bool,
    pub(super) granularity: usize,
    pub(super) granularity2: usize,
    pub(super) swapstar_capa_filter: f64,
    #[serde(default)]
    pub(super) pair_mode: usize,
    pub(super) penalty_tw: usize,
    pub(super) penalty_capa: usize,
    #[serde(default)]
    pub(super) pen_warm: usize,
    pub(super) target_ratio: f64,
    pub(super) max_it_noimprov: usize,
    pub(super) max_it_total: usize,
    pub(super) nb_it_adapt_penalties: usize,
    pub(super) nb_it_compression: usize,
    pub(super) mu: usize,
    pub(super) mu_start: usize,
    pub(super) lambda: usize,
    pub(super) nb_close: usize,
    pub(super) nb_elite: usize,
    pub(super) crossover_srex_percent: usize,
    #[serde(default)]
    pub(super) repair_permille: usize,
    pub(super) max_cli_srex: usize,
    pub(super) max_credit_deterioration: usize,
    pub(super) selection_weight_feasible: usize,
    pub(super) factor_split: f64,
    pub(super) decomp_target_size: usize,
    pub(super) decomp_nb_phases: usize,
    #[serde(default)]
    pub(super) cluster_lloyd: usize,
    #[serde(default)]
    pub(super) band_bar: i64,
    #[serde(default)]
    pub(super) band_low: i64,
    #[serde(default)]
    pub(super) band_high: i64,
    #[serde(default)]
    pub(super) band_phases: usize,
    #[serde(default)]
    pub(super) band_level: usize,
    #[serde(default)]
    pub(super) band_lloyd: usize,
    #[serde(default)]
    pub(super) band_sub_it_permille: usize,
    #[serde(default)]
    pub(super) band_stall: usize,
    #[serde(default)]
    pub(super) post_low: i64,
    #[serde(default)]
    pub(super) post_high: i64,
    #[serde(default)]
    pub(super) post_phases: usize,
    #[serde(default)]
    pub(super) post_stall: usize,
    #[serde(default)]
    pub(super) post_sub_it_permille: usize,
    #[serde(default, deserialize_with = "de_nonnegative_i64")]
    pub(super) low_entry: i64,
    #[serde(default, deserialize_with = "de_nonnegative_i64")]
    pub(super) low_post_entry: i64,
    #[serde(default, deserialize_with = "de_nonnegative_i64")]
    pub(super) low_stop: i64,
    #[serde(default)]
    pub(super) triage_permille: usize,
    #[serde(default)]
    pub(super) triage_low: i64,
    #[serde(default)]
    pub(super) triage_boost_low: i64,
    #[serde(default)]
    pub(super) triage_boost_high: i64,
    #[serde(default)]
    pub(super) triage_sub_it_permille: usize,
    #[serde(default)]
    pub(super) triage_extra_phases: usize,
    #[serde(default, deserialize_with = "de_count")]
    pub(super) triage_band_phases: Option<usize>,
    #[serde(default, deserialize_with = "de_count")]
    pub(super) triage_post_phases: Option<usize>,
    #[serde(default)]
    pub(super) triage_fork: usize,
    #[serde(default)]
    pub(super) triage_gate: usize,
    #[serde(default)]
    pub(super) triage_gate_low: i64,
    #[serde(default)]
    pub(super) triage_gate_high: i64,
    #[serde(default)]
    pub(super) triage_reseed: usize,
    #[serde(default = "one_thousand")]
    pub(super) l2_permille: usize,
    #[serde(default = "one_thousand")]
    pub(super) l3_permille: usize,
    #[serde(default = "six_hundred")]
    pub(super) l4_permille: usize,
    #[serde(default = "thirty_two_hundred")]
    pub(super) sub_it_permille: usize,
    #[serde(default = "eight")]
    pub(super) inh_k: usize,
    #[serde(default = "four")]
    pub(super) inh_min_level: usize,
    #[serde(default = "default_sub_it_l4_permille")]
    pub(super) sub_it_l4_permille: Option<usize>,
    #[serde(default)]
    pub(super) bar_factor: f64,
    #[serde(default)]
    pub(super) bar_pos_only: bool,
    #[serde(default)]
    pub(super) aux_flag: bool,
    #[serde(default)]
    pub(super) aux_level: i64,
    #[serde(default)]
    pub(super) aux_offset: i64,
    #[serde(default)]
    pub(super) aux_permille: usize,
}

fn de_count<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<usize>, D::Error> {
    Ok(Value::deserialize(deserializer)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok()))
}

fn de_nonnegative_i64<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<i64, D::Error> {
    Ok(Value::deserialize(deserializer)?.as_i64().unwrap_or(0).max(0))
}

fn one_thousand() -> usize {
    1000
}

fn four() -> usize {
    4
}

fn six_hundred() -> usize {
    600
}

fn thirty_two_hundred() -> usize {
    3200
}

fn eight() -> usize {
    8
}

fn default_sub_it_l4_permille() -> Option<usize> {
    Some(1400)
}

struct Preset {
    allow_swap3: bool,
    granularity: usize,
    granularity2: usize,
    max_it_noimprov: usize,
    max_it_total: usize,
    nb_it_adapt_penalties: usize,
    nb_it_compression: usize,
    mu: usize,
    mu_start: usize,
    lambda: usize,
    nb_close: usize,
    nb_elite: usize,
    max_cli_srex: usize,
    decomp_target_size: usize,
    decomp_nb_phases: usize,
}

#[rustfmt::skip]
const PRESETS_BY_EXPLORATION_LEVEL: [Preset; 7] = [
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:    0, max_it_total:     0,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  2, mu_start:  1, lambda:  1,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:    0, max_it_total:     0,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  2, mu_start: 20, lambda:  1,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:   40, max_it_total:   150,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  3, mu_start:  6, lambda:  3,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 20, granularity2: 30, max_it_noimprov:  100, max_it_total:   400,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  7, mu_start: 10, lambda:  3,
             nb_close: 2, nb_elite: 2, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 20, granularity2: 30, max_it_noimprov:  200, max_it_total:  2000,
             nb_it_adapt_penalties: 50, nb_it_compression:  75, mu: 15, mu_start: 20, lambda:  5,
             nb_close: 2, nb_elite: 3, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 25, granularity2: 35, max_it_noimprov: 2000, max_it_total: 20000,
             nb_it_adapt_penalties: 50, nb_it_compression: 100, mu: 25, mu_start: 50, lambda: 40,
             nb_close: 5, nb_elite: 4, max_cli_srex: 75, decomp_target_size: 500, decomp_nb_phases: 3 },
    Preset { allow_swap3: false, granularity: 25, granularity2: 35, max_it_noimprov: 5000, max_it_total: 50000,
             nb_it_adapt_penalties: 50, nb_it_compression: 150, mu: 25, mu_start: 50, lambda: 40,
             nb_close: 5, nb_elite: 4, max_cli_srex: 75, decomp_target_size: 500, decomp_nb_phases: 8 },
];

struct TrackProfile {
    nb_nodes: usize,
    allow_swap3: bool,
    granularity: usize,
    granularity2: usize,
    swapstar_capa_filter: f64,
    max_credit_deterioration: usize,
    decomp_target_size: usize,
    decomp_nb_phases: usize,
    l4_permille: usize,
}

#[rustfmt::skip]
const TRACK_PROFILES: [TrackProfile; 5] = [
    TrackProfile { nb_nodes:  600, allow_swap3: true, granularity: 35, granularity2: 45,
                   swapstar_capa_filter: 0.6, max_credit_deterioration: 25,
                   decomp_target_size: 200, decomp_nb_phases:  9, l4_permille: 600 },
    TrackProfile { nb_nodes:  700, allow_swap3: true, granularity: 30, granularity2: 45,
                   swapstar_capa_filter: 0.4, max_credit_deterioration: 30,
                   decomp_target_size: 233, decomp_nb_phases:  8, l4_permille: 600 },
    TrackProfile { nb_nodes:  800, allow_swap3: true, granularity: 25, granularity2: 35,
                   swapstar_capa_filter: 0.3, max_credit_deterioration: 10,
                   decomp_target_size: 200, decomp_nb_phases: 11, l4_permille: 600 },
    TrackProfile { nb_nodes:  900, allow_swap3: true, granularity: 30, granularity2: 40,
                   swapstar_capa_filter: 0.5, max_credit_deterioration: 20,
                   decomp_target_size: 150, decomp_nb_phases: 12, l4_permille: 400 },
    TrackProfile { nb_nodes: 1000, allow_swap3: true, granularity: 30, granularity2: 40,
                   swapstar_capa_filter: 0.5, max_credit_deterioration: 20,
                   decomp_target_size: 200, decomp_nb_phases: 12, l4_permille: 600 },
];

impl Params {
    fn preset(exploration_level: usize, data: &Problem) -> Self {
        let max_dist = data.distance_matrix.iter().copied().max().unwrap_or(1) as f64;
        let max_demand = data
            .node_data
            .iter()
            .skip(1)
            .map(|nd| nd.demand)
            .max()
            .unwrap_or(1) as f64;
        let penalty_capa = (max_dist / max_demand).clamp(1.0, 500.0).round() as usize;

        let preset = match PRESETS_BY_EXPLORATION_LEVEL.get(exploration_level) {
            Some(preset) => preset,
            None => return Self::defaults(data),
        };

        Self {
            exploration_level,
            allow_swap3: preset.allow_swap3,
            granularity: preset.granularity,
            granularity2: preset.granularity2,
            swapstar_capa_filter: 0.3,
            pair_mode: 0,
            penalty_tw: 10,
            penalty_capa,
            pen_warm: 0,
            target_ratio: 0.4,
            max_it_noimprov: preset.max_it_noimprov,
            max_it_total: preset.max_it_total,
            nb_it_adapt_penalties: preset.nb_it_adapt_penalties,
            nb_it_compression: preset.nb_it_compression,
            mu: preset.mu,
            mu_start: preset.mu_start,
            lambda: preset.lambda,
            nb_close: preset.nb_close,
            nb_elite: preset.nb_elite,
            crossover_srex_percent: 75,
            repair_permille: 0,
            max_cli_srex: preset.max_cli_srex,
            max_credit_deterioration: 10,
            selection_weight_feasible: 2,
            factor_split: 1.3,
            decomp_target_size: preset.decomp_target_size,
            decomp_nb_phases: preset.decomp_nb_phases,
            cluster_lloyd: 0,
            band_bar: 0,
            band_low: 0,
            band_high: 0,
            band_phases: 0,
            band_level: 0,
            band_lloyd: 0,
            band_sub_it_permille: 0,
            band_stall: 0,
            post_low: 0,
            post_high: 0,
            post_phases: 0,
            post_stall: 0,
            post_sub_it_permille: 0,
            low_entry: 0,
            low_post_entry: 0,
            low_stop: 0,
            triage_permille: 0,
            triage_low: 0,
            triage_boost_low: 0,
            triage_boost_high: 0,
            triage_sub_it_permille: 0,
            triage_extra_phases: 0,
            triage_band_phases: None,
            triage_post_phases: None,
            triage_fork: 0,
            triage_gate: 0,
            triage_gate_low: 0,
            triage_gate_high: 0,
            triage_reseed: 0,
            l2_permille: one_thousand(),
            l3_permille: one_thousand(),
            l4_permille: six_hundred(),
            sub_it_permille: thirty_two_hundred(),
            inh_k: eight(),
            inh_min_level: four(),
            sub_it_l4_permille: default_sub_it_l4_permille(),
            bar_factor: 0.0,
            bar_pos_only: false,
            aux_flag: false,
            aux_level: 0,
            aux_offset: 0,
            aux_permille: 0,
        }
    }

    fn defaults(data: &Problem) -> Self {
        let mut params = Self::preset(DEFAULT_EXPLORATION_LEVEL, data);

        if let Some(profile) = TRACK_PROFILES
            .iter()
            .find(|profile| profile.nb_nodes == data.nb_nodes)
        {
            params.allow_swap3 = profile.allow_swap3;
            params.granularity = profile.granularity;
            params.granularity2 = profile.granularity2;
            params.swapstar_capa_filter = profile.swapstar_capa_filter;
            params.max_credit_deterioration = profile.max_credit_deterioration;
            params.decomp_target_size = profile.decomp_target_size;
            params.decomp_nb_phases = profile.decomp_nb_phases;
            params.l4_permille = profile.l4_permille;
        }

        params
    }

    fn with_overrides(
        base: Self,
        hyperparameters: &Option<Map<String, Value>>,
        data: &Problem,
    ) -> Self {
        let map = match hyperparameters {
            Some(map) if !map.is_empty() => map,
            _ => return base,
        };
        let mut merged_params = serde_json::to_value(base).expect("Params serializable");
        if let Value::Object(ref mut obj) = merged_params {
            for (k, v) in map {
                if k != "exploration_level" {
                    obj.insert(k.clone(), v.clone());
                }
            }
            for (key, default) in [
                ("inh_k", Value::from(eight())),
                ("inh_min_level", Value::from(four())),
                (
                    "sub_it_l4_permille",
                    Value::from(default_sub_it_l4_permille()),
                ),
            ] {
                let valid = match obj.get(key) {
                    Some(Value::Null) => key == "sub_it_l4_permille",
                    Some(value) => value.as_u64().map_or(false, |u| usize::try_from(u).is_ok()),
                    None => false,
                };
                if !valid {
                    obj.insert(key.to_string(), default);
                }
            }
        }
        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(data))
    }

    fn with_scoped_overrides(
        mut base: Self,
        hyperparameters: &Option<Map<String, Value>>,
        scope: &str,
    ) -> Self {
        let map = match hyperparameters
            .as_ref()
            .and_then(|map| map.get(scope))
            .and_then(Value::as_object)
        {
            Some(map) if !map.is_empty() => map,
            _ => return base,
        };
        let mut merged = serde_json::to_value(base).expect("Params serializable");
        for (key, value) in map {
            if key == "exploration_level" {
                continue;
            }
            if let Some(field) = merged.get_mut(key) {
                let previous = field.clone();
                *field = value.clone();
                match serde_json::from_value(merged.clone()) {
                    Ok(params) => base = params,
                    Err(_) => merged[key] = previous,
                }
            }
        }
        base
    }

    pub(super) fn initialize(hyperparameters: &Option<Map<String, Value>>, data: &Problem) -> Self {
        let requested_level = requested_exploration_level(hyperparameters);

        let base_params = match requested_level {
            Some(level) if level != DEFAULT_EXPLORATION_LEVEL => Self::preset(level, data),
            _ => Self::defaults(data),
        };

        let params = Self::with_overrides(base_params, hyperparameters, data);
        Self::with_scoped_overrides(params, hyperparameters, "master")
    }

    pub(super) fn initialize_at_level(
        hyperparameters: &Option<Map<String, Value>>,
        exploration_level: usize,
        data: &Problem,
    ) -> Self {
        let mut params =
            Self::with_overrides(Self::preset(exploration_level, data), hyperparameters, data);
        let scope = match exploration_level {
            1 => Some("sub_l1"),
            2 => Some("sub_l2"),
            3 => Some("sub_l3"),
            4 => Some("sub_l4"),
            _ => None,
        };
        if let Some(scope) = scope {
            params = Self::with_scoped_overrides(params, hyperparameters, scope);
        }
        params
    }
}

fn requested_exploration_level(hyperparameters: &Option<Map<String, Value>>) -> Option<usize> {
    hyperparameters
        .as_ref()
        .and_then(|map| map.get("exploration_level"))
        .and_then(|value| match value {
            Value::Number(n) => n.as_u64().map(|u| u as usize),
            Value::String(s) => s.parse::<usize>().ok(),
            _ => None,
        })
}
