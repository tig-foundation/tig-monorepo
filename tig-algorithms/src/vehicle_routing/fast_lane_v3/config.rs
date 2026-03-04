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
    fn preset(exploration_level: usize, nb_nodes: usize) -> Self {
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
                max_it_total: 50,
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
                max_it_total: 500,
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
                max_it_total: 5_000,
                nb_it_adapt_penalties: 20,
                nb_it_traces: 100,
                mu: 10,
                mu_start: 20,
                lambda: 10,
                nb_close: 2,
                nb_elite: 3,
            },
            5 => Self {
                exploration_level: 4,
                allow_swap3: false,
                granularity: 30,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 5_000,
                max_it_total: 50_000,
                nb_it_adapt_penalties: 50,
                nb_it_traces: 200,
                mu: 12,
                mu_start: 24,
                lambda: 20,
                nb_close: 3,
                nb_elite: 4,
            },
            6 => Self {
                exploration_level: 5,
                allow_swap3: false,
                granularity: 30,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 10_000,
                max_it_total: 200_000,
                nb_it_adapt_penalties: 50,
                nb_it_traces: 500,
                mu: 25,
                mu_start: 50,
                lambda: 40,
                nb_close: 3,
                nb_elite: 8,
            },
            _ => Self::defaults(nb_nodes),
        }
    }

    pub fn defaults(nb_nodes: usize) -> Self {
        Self::preset(0, nb_nodes)
    }

    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, nb_nodes: usize) -> Self {
        let mut base_params = Self::defaults(nb_nodes);

        if let Some(v) = hyperparameters.as_ref().and_then(|m| m.get("exploration_level")) {
            match v {
                Value::Number(n) => {
                    if let Some(u) = n.as_u64() {
                        base_params = Self::preset(u as usize, nb_nodes);
                    }
                }
                Value::String(s) => {
                    if let Ok(u) = s.parse::<usize>() {
                        base_params = Self::preset(u, nb_nodes);
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

        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(nb_nodes))
    }
}
