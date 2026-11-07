// Algorithm dispatcher. All track files are siblings in this source_defaults/ folder.
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::{Challenge, Solution};
#[allow(dead_code)]
mod track_600;
#[allow(dead_code)]
mod track_700;
#[allow(dead_code)]
mod track_800;
#[allow(dead_code)]
mod track_900;
#[allow(dead_code)]
mod track_1000;

// Mirror the live benchmark profiles for null hyperparameters. Explicit values
// replace individual defaults, including within nested search phases.
fn resolved_hyperparameters(
    num_nodes: usize,
    supplied: &Option<Map<String, Value>>,
) -> Option<Map<String, Value>> {
    let profile = match num_nodes {
        600 | 700 => r#"{"allow_swap3":true,"decomp_nb_phases":8,"decomp_target_size":200,"exploration_level":4,"granularity":30,"granularity2":40,"max_credit_deterioration":20,"swapstar_capa_filter":0.5}"#,
        800 => r#"{"allow_swap3":true,"decomp_nb_phases":30,"decomp_target_size":200,"exploration_level":4,"granularity":32,"granularity2":40,"max_credit_deterioration":15,"swapstar_capa_filter":0.45}"#,
        900 => r#"{"allow_swap3":true,"decomp_nb_phases":30,"decomp_target_size":200,"exploration_level":4,"granularity":40,"granularity2":50,"max_credit_deterioration":40,"swapstar_capa_filter":0.6}"#,
        1000 => r#"{"allow_swap3":true,"exploration_level":4}"#,
        _ => return supplied.clone(),
    };
    let mut merged: Map<String, Value> =
        serde_json::from_str(profile).expect("valid built-in VRPTW benchmark profile");
    if let Some(overrides) = supplied {
        for (key, value) in overrides {
            merged.insert(key.clone(), value.clone());
        }
    }
    Some(merged)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hyperparameters = resolved_hyperparameters(challenge.num_nodes, hyperparameters);
    match challenge.num_nodes {
        600 => track_600::solve_challenge(challenge, save_solution, &hyperparameters),
        700 => track_700::solve_challenge(challenge, save_solution, &hyperparameters),
        800 => track_800::solve_challenge(challenge, save_solution, &hyperparameters),
        900 => track_900::solve_challenge(challenge, save_solution, &hyperparameters),
        1000 => track_1000::solve_challenge(challenge, save_solution, &hyperparameters),
        // Preserve legacy auxiliary sizes with their original root gates.
        _ => track_600::solve_challenge(challenge, save_solution, &hyperparameters),
    }
}

pub struct Solver;
impl Solver {
    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let hyperparameters = resolved_hyperparameters(challenge.num_nodes, hyperparameters);
        match challenge.num_nodes {
            600 => track_600::Solver::solve_challenge_instance(challenge, &hyperparameters, save_solution),
            700 => track_700::Solver::solve_challenge_instance(challenge, &hyperparameters, save_solution),
            800 => track_800::Solver::solve_challenge_instance(challenge, &hyperparameters, save_solution),
            900 => track_900::Solver::solve_challenge_instance(challenge, &hyperparameters, save_solution),
            1000 => track_1000::Solver::solve_challenge_instance(challenge, &hyperparameters, save_solution),
            _ => track_600::Solver::solve_challenge_instance(challenge, &hyperparameters, save_solution),
        }
    }
}

pub fn help() {
    track_600::help();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_uses_track_profile_and_partial_overrides_take_precedence() {
        let defaults = resolved_hyperparameters(800, &None).unwrap();
        assert_eq!(defaults["decomp_nb_phases"], 30);
        assert_eq!(defaults["granularity"], 32);
        assert_eq!(defaults["exploration_level"], 4);

        let overrides: Map<String, Value> = serde_json::from_str(
            r#"{"granularity":44,"exploration_level":5}"#,
        ).unwrap();
        let merged = resolved_hyperparameters(800, &Some(overrides)).unwrap();
        assert_eq!(merged["granularity"], 44);
        assert_eq!(merged["exploration_level"], 5);
        assert_eq!(merged["decomp_nb_phases"], 30);
    }

    #[test]
    fn unsupported_sizes_keep_existing_hyperparameter_behavior() {
        assert!(resolved_hyperparameters(500, &None).is_none());
        let supplied: Map<String, Value> =
            serde_json::from_str(r#"{"exploration_level":2}"#).unwrap();
        assert_eq!(resolved_hyperparameters(500, &Some(supplied.clone())), Some(supplied));
    }
}
