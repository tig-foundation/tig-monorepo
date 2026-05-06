use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod baseline;
mod congested;
mod dense;
mod multiday;
mod capstone;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub track: String,
}

pub fn help() {
    println!("Energy arbitrage solver. Pass hyperparameter: {{\"track\":\"baseline\"}}");
    println!("Available tracks: baseline, congested, multiday, dense, capstone");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let track = hyperparameters
        .as_ref()
        .and_then(|h| h.get("track"))
        .and_then(|v| v.as_str())
        .unwrap_or("baseline");

    let policy: &dyn Fn(&Challenge, &State) -> Result<Vec<f64>> = match track {
        "baseline"  => &baseline::policy,
        "congested" => &congested::policy,
        "multiday"  => &multiday::policy,
        "dense"     => &dense::policy,
        "capstone"  => &capstone::policy,
        other => return Err(anyhow!("Unknown track: {}. Use one of: baseline, congested, multiday, dense, capstone", other)),
    };

    let solution = run_policy(challenge, policy)?;
    save_solution(&solution)
}

fn run_policy(
    challenge: &Challenge,
    policy: &dyn Fn(&Challenge, &State) -> Result<Vec<f64>>,
) -> Result<Solution> {
    use rand::{rngs::{SmallRng, StdRng}, SeedableRng, Rng};
    let mut rng = SmallRng::from_seed(StdRng::from_seed(challenge.seed).r#gen());
    let congestion = vec![false; challenge.network.num_nodes];
    let rt0 = challenge.market.generate_rt_prices(&mut rng, 0, &congestion);
    let dt = tig_challenges::energy_arbitrage::constants::DELTA_T;
    let mut state = State {
        time_step: 0,
        socs: challenge.batteries.iter().map(|b| b.soc_initial_mwh).collect(),
        rt_prices: rt0,
        exogenous_injections: challenge.exogenous_injections[0].clone(),
        action_bounds: challenge.batteries.iter().map(|b| {
            let soc = b.soc_initial_mwh;
            let headroom = (b.soc_max_mwh - soc).max(0.0);
            let available = (soc - b.soc_min_mwh).max(0.0);
            let max_charge = if b.efficiency_charge > 0.0 {
                (headroom / (b.efficiency_charge * dt)).min(b.power_charge_mw).max(0.0)
            } else { 0.0 };
            let max_discharge = if b.efficiency_discharge > 0.0 {
                (available * b.efficiency_discharge / dt).min(b.power_discharge_mw).max(0.0)
            } else { 0.0 };
            (-max_charge, max_discharge)
        }).collect(),
        total_profit: 0.0,
    };
    let mut schedule = Vec::with_capacity(challenge.num_steps);
    for _ in 0..challenge.num_steps {
        let action = policy(challenge, &state)?;
        state = challenge.take_step(&state, &action, NextRTPrices::Generate(rng.r#gen()))?;
        schedule.push(action);
    }
    Ok(Solution { schedule })
}