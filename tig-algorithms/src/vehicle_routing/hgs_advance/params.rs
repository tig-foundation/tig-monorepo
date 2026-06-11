use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use super::problem::Problem;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Params {

    /// Main knob balancing speed vs exploration depth.
    /// Typical range: 0 (very fast, shallow search) to 6 (slower, deeper HGS search).
    /// Setting this value loads a preset baseline for the other parameters.
    pub exploration_level: usize,

    /// Enables SWAP/RELOCATE moves on 3 consecutive nodes (instead of 2).
    /// Usually improves short-run quality at a runtime cost.
    /// For long runs, keeping it `false` is often sufficient.
    pub allow_swap3: bool,

    /// Number of nearest clients considered per customer `i` in standard LS neighborhoods.
    /// Typical range: 10..=nb_clients.
    pub granularity: usize,

    /// Number of nearest clients considered per customer `i` for SWAP* neighborhoods.
    /// Typical range: 10..=nb_clients.
    pub granularity2: usize,

    /// Fraction kept by demand-similarity prefilter before SWAP* distance ranking.
    /// Keep in [0.0, 1.0].
    pub swapstar_capa_filter: f64,

    /// Initial time-window violation penalty (adapted during search).
    /// Typical range: 1..=10000.
    pub penalty_tw: usize,

    /// Initial capacity violation penalty (adapted during search).
    /// Typical range: 1..=10000.
    pub penalty_capa: usize,

    /// Target ratio of naturally feasible solutions (before repair),
    /// used by adaptive penalty updates.
    /// Keep in [0.1, 0.9].
    pub target_ratio: f64,

    /// Stop criterion: terminate after this many iterations without improvement.
    /// Keep > 1.
    pub max_it_noimprov: usize,

    /// Stop criterion: maximum total iterations.
    /// Keep > 1.
    pub max_it_total: usize,

    /// Iterations between penalty adaptations.
    /// Typical range: 20..=100.
    pub nb_it_adapt_penalties: usize,

    /// Iterations between population trace prints.
    /// Typical range: 20..=500. Affects display only.
    pub nb_it_traces: usize,

    /// Number of iterations between compression opportunities.
    /// Compression is attempted once feasible population conditions are met after this delay.
    /// Keep at least ~50 to allow enough consensus build-up.
    /// Smaller values can be faster but less robust.
    pub nb_it_compression: usize,

    /// Base population size.
    /// Typical range: 5..=50.
    /// Smaller values speed up convergence but reduce exploration.
    pub mu: usize,

    /// Number of individuals generated during initialization.
    /// Recommended: around `2 * mu`.
    pub mu_start: usize,

    /// Number of offspring generated per generation.
    /// Common choice: same order as `mu` (e.g., `~2 * mu`).
    /// Smaller values are faster but reduce exploration.
    pub lambda: usize,

    /// Number of nearest individuals used in diversity estimation.
    /// Typical range: 1..=5.
    /// `1` measures diversity only to the closest neighbor.
    pub nb_close: usize,

    /// Number of elite individuals preserved at survivor selection.
    /// Common setting: around `mu / 4` (~25% of base population).
    pub nb_elite: usize,

    /// Probability (in %) of using SREX instead of OX crossover.
    /// Keep in [0, 100].
    /// Higher values bias search toward route-based recombination.
    pub crossover_srex_percent: usize,

    /// Maximum number of customers inherited from parent2 in SREX.
    /// Choose between 1 and the nb_customers/2 typically
    /// Smaller values around 50 are preferable to generate solutions
    /// that are more closely related to the first parent
    pub max_cli_srex: usize,

    /// Maximum deterioration credit allowed in LS loop 1.
    /// Improving moves add (-delta) credit, capped by this value.
    /// Non-improving moves are accepted only if delta <= current credit.
    /// Larger values increase diversification.
    /// Setting this to 0 disables non-improving moves.
    pub max_credit_deterioration: usize,

    /// Selection weight multiplier for feasible individuals in binary tournament.
    /// A value `k` means each feasible individual has `k` times the sampling chance
    /// relative to an infeasible individual before comparing biased fitness.
    /// Keep >= 1.
    pub selection_weight_feasible: usize,

    /// Split pruning factor used to limit predecessor candidates by cumulative demand.
    /// If split fails, the algorithm retries with factor_split + 0.1 up to 3.0.
    /// Usually has minor impact; typical range [1.3, 2.0].
    pub factor_split: f64,

    /// Controls whether algorithm traces are printed to stdout.
    /// `false` = silent mode.
    pub display_traces: bool,

    /// Target number of clients per subproblem in reverse mode.
    pub decomp_target_size: usize,

    /// Number of decomposition phases in reverse mode.
    /// 0 disables reverse mode.
    /// K > 0 runs K phases per exploration level in the reverse-mode schedule.
    /// At each phase, the number of subproblems is:
    /// k_sub = round(nb_nodes / decomp_target_size).
    /// Reverse mode still runs when k_sub = 1 (single-subproblem phase).
    pub decomp_nb_phases: usize,

}

impl Params {
    /// Preset parameter configurations for exploration levels.
    /// Initial penalties are scaled from instance characteristics.
    pub(super) fn preset(exploration_level: usize, data: &Problem) -> Self {
        let max_dist = data.distance_matrix.iter().copied().max().unwrap_or(1) as f64;
        let max_demand = data
            .node_data
            .iter()
            .skip(1)
            .map(|nd| nd.demand)
            .max()
            .unwrap_or(1) as f64;
        let p_capa = (max_dist / max_demand).clamp(1.0, 500.0).round() as usize;

        match exploration_level {
            0 => Self { // Single LS
                exploration_level: 0, allow_swap3: true,
                granularity: 30, granularity2: 40, swapstar_capa_filter: 0.3,
                penalty_tw: 10,  penalty_capa: p_capa, target_ratio: 0.4,
                max_it_noimprov: 0, max_it_total: 0,
                nb_it_adapt_penalties: 20, nb_it_traces: 100, nb_it_compression: 75,
                mu: 2, mu_start: 1, lambda: 1, nb_close: 1, nb_elite: 1,
                crossover_srex_percent: 75,
                max_cli_srex: 50,
                max_credit_deterioration: 10,
                selection_weight_feasible: 2,
                factor_split: 1.3,
                display_traces: false,
                decomp_target_size: 250,
                decomp_nb_phases: 2,
            },
            1 => Self { // Multi-Start LS
                exploration_level: 1, allow_swap3: true,
                granularity: 30, granularity2: 40, swapstar_capa_filter: 0.3,
                penalty_tw: 10,  penalty_capa: p_capa, target_ratio: 0.4,
                max_it_noimprov: 0, max_it_total: 0,
                nb_it_adapt_penalties: 20, nb_it_traces: 100, nb_it_compression: 75,
                mu: 2, mu_start: 20, lambda: 1, nb_close: 1, nb_elite: 1,
                crossover_srex_percent: 75,
                max_cli_srex: 50,
                max_credit_deterioration: 10,
                selection_weight_feasible: 2,
                factor_split: 1.3,
                display_traces: false,
                decomp_target_size: 250,
                decomp_nb_phases: 2,
            },
            2 => Self { // Very short HGS
                exploration_level: 2, allow_swap3: true,
                granularity: 30, granularity2: 40, swapstar_capa_filter: 0.3,
                penalty_tw: 10,  penalty_capa: p_capa, target_ratio: 0.4,
                max_it_noimprov: 40, max_it_total: 150,
                nb_it_adapt_penalties: 20, nb_it_traces: 100, nb_it_compression: 75,
                mu: 3, mu_start: 6, lambda: 3, nb_close: 1, nb_elite: 1,
                crossover_srex_percent: 75,
                max_cli_srex: 50,
                max_credit_deterioration: 10,
                selection_weight_feasible: 2,
                factor_split: 1.3,
                display_traces: false,
                decomp_target_size: 250,
                decomp_nb_phases: 2,
            },
            3 => Self {
                exploration_level: 3, allow_swap3: false,
                granularity: 20, granularity2: 30, swapstar_capa_filter: 0.3,
                penalty_tw: 10,  penalty_capa: p_capa, target_ratio: 0.4,
                max_it_noimprov: 100, max_it_total: 400,
                nb_it_adapt_penalties: 20, nb_it_traces: 100, nb_it_compression: 75,
                mu: 7, mu_start: 10, lambda: 3, nb_close: 2, nb_elite: 2,
                crossover_srex_percent: 75,
                max_cli_srex: 50,
                max_credit_deterioration: 10,
                selection_weight_feasible: 2,
                factor_split: 1.3,
                display_traces: false,
                decomp_target_size: 250,
                decomp_nb_phases: 2,
            },
            4 => Self {
                exploration_level: 4, allow_swap3: false,
                granularity: 20, granularity2: 30, swapstar_capa_filter: 0.3,
                penalty_tw: 10,  penalty_capa: p_capa, target_ratio: 0.4,
                max_it_noimprov: 200, max_it_total: 2_000,
                nb_it_adapt_penalties: 50, nb_it_traces: 100, nb_it_compression: 75,
                mu: 15, mu_start: 20, lambda: 5, nb_close: 2, nb_elite: 3,
                crossover_srex_percent: 75,
                max_cli_srex: 50,
                max_credit_deterioration: 10,
                selection_weight_feasible: 2,
                factor_split: 1.3,
                display_traces: false,
                decomp_target_size: 250,
                decomp_nb_phases: 2,
            },
            5 => Self {
                exploration_level: 5, allow_swap3: false,
                granularity: 25, granularity2: 35, swapstar_capa_filter: 0.3,
                penalty_tw: 10,  penalty_capa: p_capa, target_ratio: 0.4,
                max_it_noimprov: 2_000, max_it_total: 20_000,
                nb_it_adapt_penalties: 50, nb_it_traces: 200, nb_it_compression: 100,
                mu: 25, mu_start: 50, lambda: 40, nb_close: 5, nb_elite: 4,
                crossover_srex_percent: 75,
                max_cli_srex: 75,
                max_credit_deterioration: 10,
                selection_weight_feasible: 2,
                factor_split: 1.3,
                display_traces: false,
                decomp_target_size: 500,
                decomp_nb_phases: 3,
            },
            6 => Self { // Deep HGS
                exploration_level: 6, allow_swap3: false,
                granularity: 25, granularity2: 35, swapstar_capa_filter: 0.3,
                penalty_tw: 10,  penalty_capa: p_capa, target_ratio: 0.4,
                max_it_noimprov: 5_000, max_it_total: 50_000,
                nb_it_adapt_penalties: 50, nb_it_traces: 1000, nb_it_compression: 150,
                mu: 25, mu_start: 50, lambda: 40, nb_close: 5, nb_elite: 4,
                crossover_srex_percent: 75,
                max_cli_srex: 75,
                max_credit_deterioration: 10,
                selection_weight_feasible: 2,
                factor_split: 1.3,
                display_traces: false,
                decomp_target_size: 500,
                decomp_nb_phases: 8,
            },
            _ => Self::defaults(data),
        }
    }

    /// Default parameter set.
    pub fn defaults(data: &Problem) -> Self { Self::preset(4, data) }

    /// Build parameters from defaults/preset and optional user overrides.
    /// If `exploration_level` is provided, load that preset first, then apply
    /// the remaining user-provided keys.
    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, data: &Problem) -> Self {

        // Load base parameters
        let mut base_params = Self::defaults(data);

        // If an exploration level has been provided, load preset config values for this level
        if let Some(v) = hyperparameters.as_ref().and_then(|m| m.get("exploration_level")) {
            match v {
                Value::Number(n) => {
                    if let Some(u) = n.as_u64() { base_params = Self::preset(u as usize, data); }
                }
                Value::String(s) => {
                    if let Ok(u) = s.parse::<usize>() { base_params = Self::preset(u, data); }
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
        if let Value::Object(ref map) = merged_params {
            let display_traces = map
                .get("display_traces")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if display_traces {
                println!("=========== Algorithm Parameters =================");
                for (k, v) in map { println!("---- {:25} is set to {}", k, v); }
                println!("==================================================");
            }
        }

        // 4) Convert to Params object. If anything is off, fall back to defaults
        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(data))
    }
}
