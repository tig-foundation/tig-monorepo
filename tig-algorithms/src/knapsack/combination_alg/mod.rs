use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::{Challenge, Solution};

mod budget25_hybrid;
mod low_budget_alg;
mod superfast_t1000;
mod superfast_t5000;
mod v11_t39;
mod v11_t40;
mod v11_t41;
mod v11_t42;
mod v11_t43;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Route {
    Budget25Hybrid,
    LowBudget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Track {
    N1000Budget5,
    N1000Budget10,
    N5000Budget10,
    N1000Budget25,
    N5000Budget25,
}

impl Track {
    const DEFAULT: Self = Self::N5000Budget25;

    fn name(self) -> &'static str {
        match self {
            Self::N1000Budget5 => "1000_5",
            Self::N1000Budget10 => "1000_10",
            Self::N5000Budget10 => "5000_10",
            Self::N1000Budget25 => "1000_25",
            Self::N5000Budget25 => "5000_25",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "1000_5" | "1000/5" | "n_items=1000,budget=5" => Ok(Self::N1000Budget5),
            "1000_10" | "1000/10" | "n_items=1000,budget=10" => Ok(Self::N1000Budget10),
            "5000_10" | "5000/10" | "n_items=5000,budget=10" => Ok(Self::N5000Budget10),
            "1000_25" | "1000/25" | "n_items=1000,budget=25" => Ok(Self::N1000Budget25),
            "5000_25" | "5000/25" | "n_items=5000,budget=25" => Ok(Self::N5000Budget25),
            _ => Err(anyhow!(
                "unknown combination_alg track {value:?}; expected 1000_5, 1000_10, 5000_10, 1000_25, or 5000_25"
            )),
        }
    }

    fn route(self) -> Route {
        match self {
            Self::N1000Budget5 | Self::N1000Budget10 | Self::N5000Budget10 => Route::LowBudget,
            Self::N1000Budget25 | Self::N5000Budget25 => Route::Budget25Hybrid,
        }
    }

    /// Complete effective per-track settings. These materialize the selected
    /// child's native defaults plus every proven track-specific tuning.
    fn preset(self) -> Map<String, Value> {
        let value = match self {
            Self::N1000Budget5 => serde_json::json!({
                "n_crossover_gen": 13,
                "n_random_starts": 3,
                "sa_rounds": 0,
                "sa_iter": 0,
                "n_sa_members": 0,
                "ils_rounds": 60,
                "ils_restart_interval": 10,
                "perturb_base_frac": 4,
                "perturb_max_frac": 5,
                "ils_vnd_level": 0,
                "bounded_2_2_k": 10,
                "n_full_restarts": 5,
                "window_k": 200,
                "core_half_dp": 40
            }),
            Self::N1000Budget10 => serde_json::json!({
                "n_perturbation_rounds": 49,
                "perturbation_strength_base": 3,
                "extra_starts": 0,
                "dp_passes_multiplier": 1
            }),
            Self::N1000Budget25 => serde_json::json!({
                "n_lambda_values": 6400,
                "n_random_starts": 3,
                "n_crossover_gen": 26,
                "sa_rounds": 18,
                "sa_iter": 440,
                "n_sa_members": 8,
                "ils_rounds": 50,
                "ils_restart_interval": 1,
                "perturb_base_frac": 100,
                "perturb_max_frac": 28,
                "ils_vnd_level": 0,
                "bounded_2_2_k": 0,
                "n_full_restarts": 12,
                "window_k": 60,
                "core_half_dp": 12
            }),
            Self::N5000Budget10 => serde_json::json!({
                "bounded_2_2_k": 20,
                "core_half_dp": 50,
                "ils_restart_interval": 15,
                "ils_rounds": 115,
                "ils_vnd_level": 3,
                "n_crossover_gen": 16,
                "n_full_restarts": 5,
                "n_random_starts": 2,
                "n_sa_members": 0,
                "perturb_base_frac": 7,
                "perturb_max_frac": 7,
                "sa_iter": 0,
                "sa_rounds": 1,
                "window_k": 200
            }),
            Self::N5000Budget25 => serde_json::json!({
                "bounded_2_2_k": 10,
                "core_half_dp": 32,
                "ils_restart_interval": 14,
                "ils_rounds": 180,
                "ils_vnd_level": 0,
                "n_crossover_gen": 8,
                "n_full_restarts": 11,
                "n_random_starts": 2,
                "n_sa_members": 0,
                "perturb_base_frac": 6,
                "perturb_max_frac": 7,
                "sa_iter": 0,
                "sa_rounds": 1,
                "window_k": 260
            }),
        };
        let Value::Object(parameters) = value else {
            unreachable!()
        };
        parameters
    }
}

fn observed_budget_pct(challenge: &Challenge) -> u64 {
    let sum_weight: u64 = challenge.weights.iter().map(|&weight| weight as u64).sum();
    if sum_weight == 0 {
        0
    } else {
        challenge.max_weight as u64 * 100 / sum_weight
    }
}

fn detected_track(challenge: &Challenge) -> Result<Track> {
    let budget_pct = observed_budget_pct(challenge);
    match (challenge.num_items, budget_pct) {
        (1000, 3..=7) => Ok(Track::N1000Budget5),
        (1000, 8..=17) => Ok(Track::N1000Budget10),
        (5000, 8..=17) => Ok(Track::N5000Budget10),
        (1000, 18..=30) => Ok(Track::N1000Budget25),
        (5000, 18..=30) => Ok(Track::N5000Budget25),
        _ => Err(anyhow!(
            "combination_alg does not recognize n_items={}, observed budget_pct={}",
            challenge.num_items,
            budget_pct
        )),
    }
}

fn is_default_value(value: &Value) -> bool {
    value.is_null()
        || value
            .as_str()
            .is_some_and(|value| value.eq_ignore_ascii_case("default"))
}

/// Select the track preset first, then overlay only concrete secondary values.
/// Missing, null, or string `"default"` values retain the per-track preset.
fn routing_configuration(
    challenge: &Challenge,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<(Route, Option<Map<String, Value>>)> {
    let requested_track = match hyperparameters.as_ref().and_then(|map| map.get("track")) {
        None | Some(Value::Null) => Track::DEFAULT,
        Some(Value::String(value)) if value.eq_ignore_ascii_case("default") => Track::DEFAULT,
        Some(Value::String(value)) => Track::parse(value)?,
        Some(_) => {
            return Err(anyhow!(
                "combination_alg hyperparameter `track` must be a string"
            ))
        }
    };

    let actual_track = detected_track(challenge)?;
    if requested_track != actual_track {
        return Err(anyhow!(
            "combination_alg track mismatch: configured track={} but challenge is {}; benchmarkers must set `track` for the track being benchmarked",
            requested_track.name(),
            actual_track.name()
        ));
    }

    let mut child_hyperparameters = requested_track.preset();
    if let Some(parameters) = hyperparameters {
        if parameters.contains_key("route") {
            return Err(anyhow!(
                "combination_alg hyperparameter `route` was replaced by `track`"
            ));
        }
        for (name, value) in parameters {
            if name != "track" && !is_default_value(value) {
                child_hyperparameters.insert(name.clone(), value.clone());
            }
        }
    }

    let child_hyperparameters = if child_hyperparameters.is_empty() {
        None
    } else {
        Some(child_hyperparameters)
    };
    Ok((requested_track.route(), child_hyperparameters))
}

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let (route, child_hyperparameters) = routing_configuration(challenge, hyperparameters)?;
    match route {
        Route::Budget25Hybrid => {
            budget25_hybrid::solve_challenge(challenge, save, &child_hyperparameters)
        }
        Route::LowBudget => {
            low_budget_alg::solve_challenge(challenge, save, &child_hyperparameters)
        }
    }
}

pub fn help() {
    println!(
        "combination_alg\n\
         Primary hyperparameter: track = 1000_5 | 1000_10 | 5000_10 | 1000_25 | 5000_25\n\
         Full TIG track names such as n_items=1000,budget=5 are also accepted.\n\
         The default track is 5000_25. Each track selects its tuned secondary defaults.\n\
         Concrete secondary values override the selected preset; missing, null, or \"default\" values retain it."
    );
}
