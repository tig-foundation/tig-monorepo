use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub fn effort_rounds(level: u64) -> usize {
    match level {
        1 => 15,
        2 => 22,
        3 => 30,
        4 => 40,
        5 => 50,
        6 => 60,
        _ => 15,
    }
}

pub fn effort_strength(level: u64) -> usize {
    match level {
        1 => 3,
        2 => 3,
        3 => 3,
        4 => 4,
        5 => 4,
        6 => 4,
        _ => 3,
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Params {
    pub n_perturbation_rounds: usize,
    pub perturbation_strength_base: usize,
    pub extra_starts: usize,
    pub max_frontier_swaps_override: Option<usize>,
    pub dp_passes_multiplier: usize,
}

impl Params {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let default_effort: u64 = 1;

        let mut p = Self {
            n_perturbation_rounds: effort_rounds(default_effort),
            perturbation_strength_base: effort_strength(default_effort),
            extra_starts: 0,
            max_frontier_swaps_override: None,
            dp_passes_multiplier: 1,
        };

        if let Some(m) = h {
            if let Some(level) = m.get("effort").and_then(|v| v.as_u64()) {
                let clamped = level.clamp(1, 6);
                p.n_perturbation_rounds = effort_rounds(clamped);
                p.perturbation_strength_base = effort_strength(clamped);
            }

            if let Some(v) = m.get("n_perturbation_rounds").and_then(|v| v.as_u64()) {
                p.n_perturbation_rounds = v as usize;
            }

            if let Some(v) = m.get("perturbation_strength_base").and_then(|v| v.as_u64()) {
                p.perturbation_strength_base = v as usize;
            }

            if let Some(v) = m.get("extra_starts").and_then(|v| v.as_u64()) {
                p.extra_starts = v as usize;
            }

            if let Some(v) = m.get("max_frontier_swaps_override").and_then(|v| v.as_u64()) {
                p.max_frontier_swaps_override = Some(v as usize);
            }

            if let Some(v) = m.get("dp_passes_multiplier").and_then(|v| v.as_u64()) {
                p.dp_passes_multiplier = (v as usize).max(1);
            }
        }

        p
    }
}
