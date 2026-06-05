use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Config {
    pub exploration_level: usize,
    pub allow_swap3: bool,
    pub granularity: usize,
    pub granularity2: usize,
    pub penalty_tw: usize,
    pub penalty_capa: usize,
    pub target_ratio: f64,
    pub max_it_noimprov: usize,
    pub max_it_total: usize,
    pub nb_it_adapt_penalties: usize,
    pub nb_it_traces: usize,
    pub mu: usize,
    pub mu_start: usize,
    pub lambda: usize,
    pub nb_close: usize,
    pub nb_elite: usize,
}

impl Config {
    fn preset(exploration_level: usize, nb_nodes: usize, fleet_size: usize) -> Self {
        let p = if nb_nodes <= 700 {
            20
        } else if nb_nodes <= 1000 {
            30
        } else if nb_nodes <= 1200 {
            50
        } else if nb_nodes <= 1500 {
            80
        } else if nb_nodes <= 2000 {
            150
        } else if nb_nodes <= 3000 {
            200
        } else {
            500
        };

        // Dynamic iteration budget based on scenario complexity
        let complexity = (nb_nodes as f64) * (fleet_size as f64).sqrt();
        let base_iters = if complexity < 1000.0 {
            100
        } else if complexity < 5000.0 {
            500
        } else if complexity < 20000.0 {
            1000
        } else {
            5000
        };

        match exploration_level {
            0 => Self {
                exploration_level: 0,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 0,
                max_it_total: 0,
                nb_it_adapt_penalties: 100,
                nb_it_traces: 100,
                mu: 2,
                mu_start: 1,
                lambda: 1,
                nb_close: 1,
                nb_elite: 1,
            },
            1 => Self {
                exploration_level: 0,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 0,
                max_it_total: 0,
                nb_it_adapt_penalties: 100,
                nb_it_traces: 100,
                mu: 2,
                mu_start: 5,
                lambda: 1,
                nb_close: 1,
                nb_elite: 1,
            },
            2 => Self {
                exploration_level: 1,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 10,
                max_it_total: base_iters / 5,
                nb_it_adapt_penalties: 100,
                nb_it_traces: 100,
                mu: 3,
                mu_start: 6,
                lambda: 3,
                nb_close: 1,
                nb_elite: 1,
            },
            3 => Self {
                exploration_level: 2,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 100,
                max_it_total: base_iters,
                nb_it_adapt_penalties: 20,
                nb_it_traces: 20,
                mu: 5,
                mu_start: 10,
                lambda: 5,
                nb_close: 2,
                nb_elite: 2,
            },
            4 => Self {
                exploration_level: 3,
                allow_swap3: false,
                granularity: 30,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 500,
                max_it_total: base_iters * 5,
                nb_it_adapt_penalties: 20,
                nb_it_traces: 100,
                mu: 10,
                mu_start: 20,
                lambda: 10,
                nb_close: 2,
                nb_elite: 3,
            },
            _ => Self::preset(4, nb_nodes, fleet_size),
        }
    }

    pub fn defaults(nb_nodes: usize, fleet_size: usize) -> Self {
        Self::preset(0, nb_nodes, fleet_size)
    }

    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, nb_nodes: usize, fleet_size: usize) -> Self {
        let mut base_params = Self::defaults(nb_nodes, fleet_size);

        if let Some(v) = hyperparameters.as_ref().and_then(|m| m.get("exploration_level")) {
            match v {
                Value::Number(n) => {
                    if let Some(u) = n.as_u64() {
                        base_params = Self::preset(u as usize, nb_nodes, fleet_size);
                    }
                }
                Value::String(s) => {
                    if let Ok(u) = s.parse::<usize>() {
                        base_params = Self::preset(u, nb_nodes, fleet_size);
                    }
                }
                _ => {}
            }
        }

        let mut merged_params = serde_json::to_value(base_params).expect("Config serializable");
        if let (Value::Object(ref mut obj), Some(map)) = (&mut merged_params, hyperparameters) {
            for (k, v) in map {
                if k == "exploration_level" {
                    continue;
                }
                obj.insert(k.clone(), v.clone());
            }
        }

        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(nb_nodes, fleet_size))
    }

    pub fn select_algorithm(nb_nodes: usize, fleet_size: usize, time_window_tightness: f64, demand_variance: f64) -> Self {
        // Classify scenario into complexity bins
        let complexity = (nb_nodes as f64) * (fleet_size as f64).sqrt();
        let time_window_factor = if time_window_tightness > 0.7 { 1.5 } else { 1.0 };
        let demand_factor = if demand_variance > 0.5 { 1.3 } else { 1.0 };
        let adjusted_complexity = complexity * time_window_factor * demand_factor;

        // Determine algorithm based on complexity profile
        if nb_nodes <= 20 {
            // Small baseline instances - use exact DP solver
            Self::preset(0, nb_nodes, fleet_size)
        } else if adjusted_complexity < 1000.0 {
            // Simple scenarios - fast heuristic
            Self::preset(1, nb_nodes, fleet_size)
        } else if adjusted_complexity < 5000.0 {
            // Medium complexity - balanced HGS
            Self::preset(2, nb_nodes, fleet_size)
        } else if adjusted_complexity < 20000.0 {
            // High complexity - full HGS
            Self::preset(3, nb_nodes, fleet_size)
        } else {
            // Very high complexity - deep HGS
            Self::preset(4, nb_nodes, fleet_size)
        }
    }
}