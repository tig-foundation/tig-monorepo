use crate::energy_arbitrage::{Challenge, State};
use anyhow::Result;
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use std::sync::{Mutex, OnceLock};

static RNG: OnceLock<Mutex<SmallRng>> = OnceLock::new();

fn rng(seed: &[u8; 32]) -> &'static Mutex<SmallRng> {
    RNG.get_or_init(|| Mutex::new(SmallRng::from_seed(StdRng::from_seed(seed.clone()).r#gen())))
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let mut rng = rng(&challenge.seed).lock().unwrap();
    let mut action: Vec<f64> = state
        .action_bounds
        .iter()
        .map(|&(min, max)| {
            if min < 0.0 {
                rng.r#gen::<f64>() * max
            } else {
                rng.r#gen::<f64>() * (max - min) + min
            }
        })
        .collect();
    for _ in 0..20 {
        let injections = challenge.compute_total_injections(state, &action);
        let flows = challenge.network.compute_flows(&injections);
        if challenge.network.verify_flows(&flows).is_err() {
            action = action
                .into_iter()
                .map(|a| if a > 0.0 { a / 2.0 } else { a * 2.0 })
                .collect();
        } else {
            break;
        }
    }
    Ok(action)
}
