use serde_json::{Map, Value};

#[derive(Clone, Copy)]
pub struct Params {
    pub n_perturbation_rounds: usize,
    pub perturbation_strength_base: usize,
    pub extra_starts: usize,
    pub max_frontier_swaps_override: Option<usize>,
    pub dp_passes_multiplier: usize,
}

impl Params {
    pub fn initialize(_h: &Option<Map<String, Value>>) -> Self {
        Self {
            n_perturbation_rounds: 15,
            perturbation_strength_base: 3,
            extra_starts: 0,
            max_frontier_swaps_override: None,
            dp_passes_multiplier: 1,
        }
    }
}
