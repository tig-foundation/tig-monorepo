use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Params {

    /// Single parameter to balance speed vs exploration
    /// Keep between 0 (very fast using a single LS) and 6 (slower but deep exploration using HGS)
    /// Specifying this parameter preloads a different base configuration for the other parameters
    pub exploration_level: usize,

    /// Indicates is SWAP and RELOCATE MOVES involving 3 consecutive nodes instead of 2 are allowed
    /// Setting it to true creates a 30% time overhead to the algorithm but permits better solutions
    /// on very short runs. On longer runs I suggest to keep it to false.
    pub allow_swap3: bool,

    /// Indicates the number of close clients for each i during the LS
    /// Keep between 10 and nb_clients
    pub granularity: usize,

    /// Indicates the number of close clients for each i during the SWAP* moves
    /// Keep between 10 and nb_clients
    pub granularity2: usize,

    /// Initial penalty for time windows violations, adapts through the search
    /// Keep between 1 and 10000
    pub penalty_tw: usize,

    /// Initial penalty for capacity violations, adapts through the search
    /// Keep between 1 and 10000
    pub penalty_capa: usize,

    /// Target ratio of naturally feasible solutions before repair
    /// For penalty parameters adaptation in the HGS
    /// Keep in [0.1,0.9]
    pub target_ratio: f64,

    /// Termination criterion: no improvement for this many iterations
    /// Keep greater than 1
    pub max_it_noimprov: usize,

    /// Termination criterion: total number of iterations
    /// Keep greater than 1
    pub max_it_total: usize,

    /// Number of iterations between penalty adaptations
    /// Keep between 20 and 100
    pub nb_it_adapt_penalties: usize,

    /// Number of iterations between population status traces
    /// Keep between 20 and 500 -- Only impacts display but not algorithm behavior
    pub nb_it_traces: usize,

    /// Number of individuals in base population size
    /// Keep between 5 and 50
    /// Smaller sizes make convergence faster but reduce exploration
    pub mu: usize,

    /// Number of individuals generated in the initialization phase
    /// Recommended : 2 * mu
    pub mu_start: usize,

    /// Number of additional individuals in each generation
    /// Keep of a similar magnitude as mu, for example 2x mu
    /// Smaller sizes make convergence faster but reduce exploration
    pub lambda: usize,

    /// Number of closest individuals to measure diversity
    /// Keep between 1 and 5
    /// The special case of 1 would only measure diversity to the closest
    pub nb_close: usize,

    /// Number of elite individuals guaranteed to be preserved
    /// Recommended values are around mu/4 so roughly 25% of the base population is elite
    pub nb_elite: usize,
}

impl Params {

    /// Suggested parameter values for various exploration levels in {0...5}
    /// Initial penalties depend on the number of nodes in the challenge
    fn preset(exploration_level: usize, nb_nodes: usize) -> Self {

        let p = if nb_nodes <= 700 { 20 }
        else if nb_nodes <= 1000 { 30 }
        else if nb_nodes <= 1200 { 50 }
        else if nb_nodes <= 1500 { 80 }
        else if nb_nodes <= 2000 { 150 }
        else if nb_nodes <= 3000 { 200 }
        else { 500 };

        match exploration_level {
            0 => Self { // Single LS
                exploration_level: 0, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 0, max_it_total: 0,
                nb_it_adapt_penalties: 100, nb_it_traces: 100,
                mu: 2, mu_start: 1, lambda: 1, nb_close: 1, nb_elite: 1
            },
            1 => Self { // Multi-Start LS
                exploration_level: 0, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 0, max_it_total: 0,
                nb_it_adapt_penalties: 100, nb_it_traces: 100,
                mu: 2, mu_start: 5, lambda: 1, nb_close: 1, nb_elite: 1
            },
            2 => Self { // Very short HGS
                exploration_level: 1, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 10, max_it_total: 50,
                nb_it_adapt_penalties: 100, nb_it_traces: 100,
                mu: 3, mu_start: 6, lambda: 3, nb_close: 1, nb_elite: 1
            },
            3 => Self {
                exploration_level: 2, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 100, max_it_total: 500,
                nb_it_adapt_penalties: 20, nb_it_traces: 20,
                mu: 5, mu_start: 10, lambda: 5, nb_close: 2, nb_elite: 2
            },
            4 => Self {
                exploration_level: 3, allow_swap3: false,
                granularity: 30, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 500, max_it_total: 5_000,
                nb_it_adapt_penalties: 20, nb_it_traces: 100,
                mu: 10, mu_start: 20, lambda: 10, nb_close: 2, nb_elite: 3
            },
            5 => Self {
                exploration_level: 4, allow_swap3: false,
                granularity: 30, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 5_000, max_it_total: 50_000,
                nb_it_adapt_penalties: 50, nb_it_traces: 200,
                mu: 12, mu_start: 24, lambda: 20, nb_close: 3, nb_elite: 4
            },
            6 => Self { // Deep HGS
                exploration_level: 5, allow_swap3: false,
                granularity: 30, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 10_000, max_it_total: 200_000,
                nb_it_adapt_penalties: 50, nb_it_traces: 500,
                mu: 25, mu_start: 50, lambda: 40, nb_close: 3, nb_elite: 8
            },
            _ => Self::defaults(nb_nodes),
        }
    }

    /// By default, we use the 4nd fastest configuration with a single LS
    pub fn defaults(nb_nodes: usize) -> Self { Self::preset(3, nb_nodes) }

    /// Initialize with defaults()
    /// If exploration_level is provided, then load preset values for this level
    /// Then update any remaining user key
    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, nb_nodes: usize) -> Self {

        // Load base parameters
        let mut base_params = Self::defaults(nb_nodes);

        // If an exploration level has been provided, load preset config values for this level
        if let Some(v) = hyperparameters.as_ref().and_then(|m| m.get("exploration_level")) {
            match v {
                Value::Number(n) => {
                    if let Some(u) = n.as_u64() { base_params = Self::preset(u as usize, nb_nodes); }
                }
                Value::String(s) => {
                    if let Ok(u) = s.parse::<usize>() { base_params = Self::preset(u, nb_nodes); }
                }
                _ => {}
            }
        }

        // Update any additional user-provided parameter
        let mut merged_params = serde_json::to_value(base_params).expect("Params serializable");
        if let (Value::Object(ref mut obj), Some(map)) = (&mut merged_params, hyperparameters) {
            for (k, v) in map {
                if k == "exploration_level" { continue; } // already encoded in the preset
                obj.insert(k.clone(), v.clone());
            }
        }

        // Display parameters
        #[cfg(debug_assertions)]
        if let Value::Object(ref map) = merged_params {
            println!("=========== Algorithm Parameters =================");
            for (k, v) in map { println!("---- {:25} is set to {}", k, v); }
            println!("==================================================");
        }

        // 4) Convert to Params object. If anything is off, fall back to defaults
        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(nb_nodes))
    }
}

