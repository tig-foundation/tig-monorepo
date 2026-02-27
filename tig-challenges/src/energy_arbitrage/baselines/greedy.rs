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
    Ok(state
        .action_bounds
        .iter()
        .map(|&(min, max)| rng.gen_range(min..=max))
        .collect())
}
