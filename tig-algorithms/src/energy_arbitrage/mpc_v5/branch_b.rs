use anyhow::Result;
use tig_challenges::energy_arbitrage::{Challenge, State};

// Greedy fallback for MULTIDAY and DENSE tracks.
// Logic identical to mpc_v3 v5: RT+DA sanity check, 50% action scaling, 3-step DA window.
// No Config parameters needed — Branch B has no effort-sensitive tuning in Phase 1.

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_bat = challenge.num_batteries;
    let da = &challenge.market.day_ahead_prices;

    let actions: Vec<f64> = (0..n_bat)
        .map(|b| {
            let node = challenge.batteries[b].node;
            let rt = state.rt_prices[node];
            let da_t = da[t][node];

            let win_end = (t + 4).min(challenge.num_steps);
            let count = (win_end.saturating_sub(t + 1)) as f64;
            let da_avg = if count > 0.0 {
                ((t + 1)..win_end).map(|s| da[s][node]).sum::<f64>() / count
            } else {
                rt
            };

            let (lo, hi) = state.action_bounds[b];
            if rt < da_avg * 0.9 && da_t < da_avg * 1.05 {
                lo * 0.5
            } else if rt > da_avg * 1.1 && da_t > da_avg * 0.95 {
                hi * 0.5
            } else {
                0.0
            }
        })
        .collect();

    let inj = challenge.compute_total_injections(state, &actions);
    let flows = challenge.network.compute_flows(&inj);
    if challenge.network.verify_flows(&flows).is_ok() {
        return Ok(actions);
    }
    for &scale in &[0.5f64, 0.25, 0.0] {
        let scaled: Vec<f64> = actions.iter().map(|a| a * scale).collect();
        let inj2 = challenge.compute_total_injections(state, &scaled);
        let fl2 = challenge.network.compute_flows(&inj2);
        if challenge.network.verify_flows(&fl2).is_ok() {
            return Ok(scaled);
        }
    }
    Ok(vec![0.0; n_bat])
}
