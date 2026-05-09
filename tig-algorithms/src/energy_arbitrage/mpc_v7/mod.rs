// mpc_v7 — Spy-correction for BASELINE + mpc_v6 for all other tracks.
//
// Self-contained single-file submission: mpc_v5 and mpc_v6 logic inlined.
//
// Track routing:
//   BASELINE  (10×96, size ≤ 1000) → spy_correction (RT-aware greedy)
//   all other tracks                → mpc_v6_inlined (ADMM + SDP backward induction)
//
// Hyperparameters (JSON):
//   deviation_threshold  float  spread $/MWh vs future DA avg (default 8.0)
//   rt_blend             float  weight on RT price 0..1 (default 1.0 = pure RT)
//   force_baseline       bool   force spy_correction for any track size (testing)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::{Challenge, Solution, State};

mod spy_correction {
    use anyhow::Result;
    use tig_challenges::energy_arbitrage::{Challenge, State};

    const EPS: f64 = 1e-9;
    const MAX_FLOW_ITERS: usize = 64;
    const BSEARCH_ITERS: usize = 32;

    pub struct SpyCorrectionParams {
        pub deviation_threshold: f64,
        pub rt_blend: f64,
    }

    impl Default for SpyCorrectionParams {
        fn default() -> Self {
            SpyCorrectionParams {
                deviation_threshold: 8.0,
                rt_blend: 1.0,
            }
        }
    }

    pub fn policy(
        challenge: &Challenge,
        state: &State,
        params: &SpyCorrectionParams,
    ) -> Result<Vec<f64>> {
        let t = state.time_step;
        let n_bat = challenge.num_batteries;
        let da = &challenge.market.day_ahead_prices;
        let n = challenge.num_steps;

        let mut actions: Vec<f64> = (0..n_bat)
            .map(|b| {
                let node = challenge.batteries[b].node;
                let (lo, hi) = state.action_bounds[b];

                let da_current = da[t][node];
                let rt_current = state.rt_prices[node];

                let current_price =
                    params.rt_blend * rt_current + (1.0 - params.rt_blend) * da_current;

                let (future_sum, future_count) =
                    (t + 1..n).fold((0.0f64, 0usize), |(s, c), step| (s + da[step][node], c + 1));
                let future_avg = if future_count > 0 {
                    future_sum / future_count as f64
                } else {
                    current_price
                };

                let thr = params.deviation_threshold;
                if current_price < future_avg - thr && lo < -EPS {
                    lo
                } else if current_price > future_avg + thr && hi > EPS {
                    hi
                } else {
                    0.0
                }
            })
            .collect();

        for _ in 0..MAX_FLOW_ITERS {
            let inj = challenge.compute_total_injections(state, &actions);
            let flows = challenge.network.compute_flows(&inj);
            if challenge.network.verify_flows(&flows).is_ok() {
                return Ok(actions);
            }

            let mut worst_l = 0usize;
            let mut worst_viol = f64::NEG_INFINITY;
            let mut worst_flow = 0.0f64;
            for (l, &fl) in flows.iter().enumerate() {
                let limit = challenge.network.flow_limits[l];
                let viol = fl.abs() - limit;
                if viol > worst_viol {
                    worst_viol = viol;
                    worst_l = l;
                    worst_flow = fl;
                }
            }
            if worst_viol <= 0.0 {
                return Ok(actions);
            }

            let dir = worst_flow.signum();
            if dir.abs() <= EPS {
                break;
            }

            let mut worsening: Vec<usize> = Vec::new();
            let mut strength = 0.0f64;
            for (i, battery) in challenge.batteries.iter().enumerate() {
                let contrib = challenge.network.ptdf[worst_l][battery.node] * actions[i];
                if dir * contrib > EPS {
                    strength += dir * contrib;
                    worsening.push(i);
                }
            }
            if worsening.is_empty() || strength <= EPS {
                break;
            }
            let keep = (1.0 - worst_viol / strength).clamp(0.0, 1.0);
            if (1.0 - keep).abs() <= EPS {
                break;
            }
            for i in worsening {
                actions[i] *= keep;
            }
        }

        {
            let inj = challenge.compute_total_injections(state, &actions);
            let flows = challenge.network.compute_flows(&inj);
            if challenge.network.verify_flows(&flows).is_ok() {
                return Ok(actions);
            }
        }

        let base = actions;
        let mut lo_scale = 0.0f64;
        let mut hi_scale = 1.0f64;
        for _ in 0..BSEARCH_ITERS {
            let mid = (lo_scale + hi_scale) * 0.5;
            let scaled: Vec<f64> = base.iter().map(|u| u * mid).collect();
            let inj2 = challenge.compute_total_injections(state, &scaled);
            let fl2 = challenge.network.compute_flows(&inj2);
            if challenge.network.verify_flows(&fl2).is_ok() {
                lo_scale = mid;
            } else {
                hi_scale = mid;
            }
        }
        Ok(base.iter().map(|u| u * lo_scale).collect())
    }
}

// ─── mpc_v5 (inlined) ─────────────────────────────────────────────────────────

mod mpc_v5_inlined {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use std::cell::RefCell;
    use tig_challenges::energy_arbitrage::{constants, Challenge, Solution, State};

    mod config {
        use serde_json::{Map, Value};

        pub const FULL_MPC_THRESHOLD: usize = 3000;
        pub const SHALLOW_MPC_THRESHOLD: usize = 15000;
        pub const TERM_LOOK: usize = 12;

        // (n_cand_a, horizon_a, n_cand_c, warm_start)
        const PRESETS: [(usize, usize, usize, bool); 5] = [
            (5,  96,  3, false),  // effort=0
            (5,  96,  3, false),  // effort=1
            (5,  96,  3, false),  // effort=2: balanced, mpc_v3 identical
            (9,  96,  5, false),  // effort=3: deep quality (default)
            (17, 128, 5, true),   // effort=4: maximum
        ];

        pub struct Config {
            pub effort: usize,
            pub n_cand_a: usize,
            pub horizon: usize,
            pub n_cand_c: usize,
            pub warm_start: bool,
        }

        impl Config {
            pub fn initialize(hyperparameters: &Option<Map<String, Value>>) -> Self {
                let effort = hyperparameters
                    .as_ref()
                    .and_then(|h| h.get("effort"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v.clamp(0, 4) as usize)
                    .unwrap_or(3);
                let (n_cand_a, horizon, n_cand_c, warm_start) = PRESETS[effort];
                Config { effort, n_cand_a, horizon, n_cand_c, warm_start }
            }
        }
    }

    mod branch_baseline {
        use anyhow::Result;
        use tig_challenges::energy_arbitrage::{Challenge, State};

        const EPS: f64 = 1e-9;
        const PRICE_THRESHOLD: f64 = 5.0;
        const MAX_FLOW_ITERS: usize = 64;
        const BSEARCH_ITERS: usize = 32;

        pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
            let t = state.time_step;
            let n_bat = challenge.num_batteries;
            let da = &challenge.market.day_ahead_prices;
            let n = challenge.num_steps;

            let mut actions: Vec<f64> = (0..n_bat)
                .map(|b| {
                    let node = challenge.batteries[b].node;
                    let (lo, hi) = state.action_bounds[b];
                    let current_price = da[t][node];
                    let (future_sum, future_count) = (t + 1..n).fold((0.0f64, 0usize), |(s, c), step| {
                        (s + da[step][node], c + 1)
                    });
                    let future_avg = if future_count > 0 {
                        future_sum / future_count as f64
                    } else {
                        current_price
                    };
                    if current_price < future_avg - PRICE_THRESHOLD && lo < -EPS {
                        lo
                    } else if current_price > future_avg + PRICE_THRESHOLD && hi > EPS {
                        hi
                    } else {
                        0.0
                    }
                })
                .collect();

            for _ in 0..MAX_FLOW_ITERS {
                let inj = challenge.compute_total_injections(state, &actions);
                let flows = challenge.network.compute_flows(&inj);
                if challenge.network.verify_flows(&flows).is_ok() {
                    return Ok(actions);
                }
                let mut worst_l = 0usize;
                let mut worst_viol = f64::NEG_INFINITY;
                let mut worst_flow = 0.0f64;
                for (l, &fl) in flows.iter().enumerate() {
                    let limit = challenge.network.flow_limits[l];
                    let viol = fl.abs() - limit;
                    if viol > worst_viol {
                        worst_viol = viol;
                        worst_l = l;
                        worst_flow = fl;
                    }
                }
                if worst_viol <= 0.0 {
                    return Ok(actions);
                }
                let dir = worst_flow.signum();
                if dir.abs() <= EPS {
                    break;
                }
                let mut worsening: Vec<usize> = Vec::new();
                let mut strength = 0.0f64;
                for (i, battery) in challenge.batteries.iter().enumerate() {
                    let contrib = challenge.network.ptdf[worst_l][battery.node] * actions[i];
                    if dir * contrib > EPS {
                        strength += dir * contrib;
                        worsening.push(i);
                    }
                }
                if worsening.is_empty() || strength <= EPS {
                    break;
                }
                let keep = (1.0 - worst_viol / strength).clamp(0.0, 1.0);
                if (1.0 - keep).abs() <= EPS {
                    break;
                }
                for i in worsening {
                    actions[i] *= keep;
                }
            }
            {
                let inj = challenge.compute_total_injections(state, &actions);
                let flows = challenge.network.compute_flows(&inj);
                if challenge.network.verify_flows(&flows).is_ok() {
                    return Ok(actions);
                }
            }
            let base = actions;
            let mut lo_scale = 0.0f64;
            let mut hi_scale = 1.0f64;
            for _ in 0..BSEARCH_ITERS {
                let mid = (lo_scale + hi_scale) * 0.5;
                let scaled: Vec<f64> = base.iter().map(|u| u * mid).collect();
                let inj2 = challenge.compute_total_injections(state, &scaled);
                let fl2 = challenge.network.compute_flows(&inj2);
                if challenge.network.verify_flows(&fl2).is_ok() {
                    lo_scale = mid;
                } else {
                    hi_scale = mid;
                }
            }
            Ok(base.iter().map(|u| u * lo_scale).collect())
        }
    }

    mod branch_b {
        use anyhow::Result;
        use tig_challenges::energy_arbitrage::{Challenge, State};

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
    }

    mod branch_a {
        use anyhow::Result;
        use tig_challenges::energy_arbitrage::{Challenge, NextRTPrices, State};
        use super::config::Config;

        pub fn policy(
            challenge: &Challenge,
            state: &State,
            config: &Config,
            warm_hints: Option<&[f64]>,
        ) -> Result<Vec<f64>> {
            let t = state.time_step;
            let n_bat = challenge.num_batteries;

            let mut bat_order: Vec<usize> = (0..n_bat).collect();
            bat_order.sort_by(|&a, &b| {
                challenge.batteries[b]
                    .capacity_mwh
                    .partial_cmp(&challenge.batteries[a].capacity_mwh)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut chosen = vec![0.0f64; n_bat];
            for &b in &bat_order {
                let hint = warm_hints.map(|h| h[b]);
                chosen[b] = optimize_battery(challenge, state, b, t, config, hint);
            }

            Ok(super::apply_flow_scale(challenge, state, chosen))
        }

        fn optimize_battery(
            challenge: &Challenge,
            state: &State,
            bat: usize,
            t: usize,
            config: &Config,
            warm_hint: Option<f64>,
        ) -> f64 {
            let (lo, hi) = state.action_bounds[bat];
            let mut candidates = super::make_candidates(lo, hi, config.n_cand_a);

            if let Some(prev) = warm_hint {
                let clamped = prev.clamp(lo, hi);
                if candidates.iter().all(|&p| (p - clamped).abs() > 1e-6) {
                    candidates.push(clamped);
                }
            }

            let mut best_val = f64::NEG_INFINITY;
            let mut best_u = 0.0f64;
            for u in candidates {
                let u = u.clamp(lo, hi);
                let val = lookahead(challenge, state, bat, u, t, config);
                if val > best_val {
                    best_val = val;
                    best_u = u;
                }
            }
            best_u
        }

        fn lookahead(
            challenge: &Challenge,
            state: &State,
            bat: usize,
            u0: f64,
            t: usize,
            config: &Config,
        ) -> f64 {
            let n_bat = challenge.num_batteries;
            let da = &challenge.market.day_ahead_prices;
            let horizon = config.horizon.min(challenge.num_steps.saturating_sub(t));

            let mut sim = state.clone();
            let mut total = 0.0f64;

            for h in 0..horizon {
                let step = t + h;
                let mut action = vec![0.0f64; n_bat];
                action[bat] = if h == 0 {
                    u0
                } else {
                    da_greedy_action(challenge, &sim, bat, step)
                };
                action[bat] = action[bat].clamp(sim.action_bounds[bat].0, sim.action_bounds[bat].1);

                let step_profit = challenge.compute_profit(&sim, &action);
                let next_prices = da[(step + 1).min(da.len() - 1)].clone();

                match challenge.take_step(&sim, &action, NextRTPrices::Override(next_prices)) {
                    Ok(next_sim) => {
                        total += step_profit;
                        sim = next_sim;
                    }
                    Err(_) => return f64::NEG_INFINITY,
                }
            }

            if !sim.socs.is_empty() {
                total += super::terminal_soc_value(challenge, &sim, bat, t + horizon);
            }
            total
        }

        fn da_greedy_action(challenge: &Challenge, state: &State, bat: usize, t: usize) -> f64 {
            let node = challenge.batteries[bat].node;
            let da = &challenge.market.day_ahead_prices;
            let current = da[t][node];
            let look_end = (t + 12).min(challenge.num_steps);
            let count = (look_end.saturating_sub(t + 1)) as f64;
            let avg = if count > 0.0 {
                ((t + 1)..look_end).map(|s| da[s][node]).sum::<f64>() / count
            } else {
                current
            };
            let (lo, hi) = state.action_bounds[bat];
            if current < avg * 0.9 {
                lo * 0.5
            } else if current > avg * 1.1 {
                hi * 0.5
            } else {
                0.0
            }
        }
    }

    mod branch_c {
        use anyhow::Result;
        use tig_challenges::energy_arbitrage::{Challenge, NextRTPrices, State};
        use super::config::Config;

        pub fn policy(challenge: &Challenge, state: &State, config: &Config) -> Result<Vec<f64>> {
            let t = state.time_step;
            let n_bat = challenge.num_batteries;
            let da = &challenge.market.day_ahead_prices;

            let mut bat_order: Vec<usize> = (0..n_bat).collect();
            bat_order.sort_by(|&a, &b| {
                challenge.batteries[b]
                    .capacity_mwh
                    .partial_cmp(&challenge.batteries[a].capacity_mwh)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut da_state = state.clone();
            da_state.rt_prices = da[t].clone();
            let next_da = da[(t + 1).min(da.len() - 1)].clone();

            let mut chosen = vec![0.0f64; n_bat];
            for &b in &bat_order {
                let (lo, hi) = state.action_bounds[b];
                let candidates: Vec<f64> = if config.n_cand_c == 3 {
                    vec![lo, 0.0f64, hi]
                } else {
                    super::make_candidates(lo, hi, config.n_cand_c)
                };

                let mut best_val = f64::NEG_INFINITY;
                let mut best_u = 0.0f64;

                for u in candidates {
                    let u = u.clamp(lo, hi);
                    let mut action = vec![0.0f64; n_bat];
                    action[b] = u;
                    let da_profit = challenge.compute_profit(&da_state, &action);
                    match challenge.take_step(state, &action, NextRTPrices::Override(next_da.clone())) {
                        Ok(next_sim) => {
                            let term = if !next_sim.socs.is_empty() {
                                super::terminal_soc_value(challenge, &next_sim, b, t + 1)
                            } else {
                                0.0
                            };
                            let val = da_profit + term;
                            if val > best_val {
                                best_val = val;
                                best_u = u;
                            }
                        }
                        Err(_) => {}
                    }
                }
                chosen[b] = best_u;
            }

            Ok(super::apply_flow_scale(challenge, state, chosen))
        }
    }

    use config::{Config, FULL_MPC_THRESHOLD, SHALLOW_MPC_THRESHOLD, TERM_LOOK};
    const BASELINE_THRESHOLD: usize = 1000;

    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let config = Config::initialize(hyperparameters);
        let size = challenge.num_batteries * challenge.num_steps;

        let solution = if size <= BASELINE_THRESHOLD {
            challenge.grid_optimize(&|c, s| branch_baseline::policy(c, s))?
        } else if size <= FULL_MPC_THRESHOLD {
            if config.warm_start {
                let cache = RefCell::new(WarmStartCache::new(challenge.num_batteries));
                challenge.grid_optimize(&|c, s| {
                    let hints = cache.borrow().prev_actions.clone();
                    let result = branch_a::policy(c, s, &config, Some(&hints))?;
                    cache.borrow_mut().prev_actions = result.clone();
                    Ok(result)
                })?
            } else {
                challenge.grid_optimize(&|c, s| branch_a::policy(c, s, &config, None))?
            }
        } else if size <= SHALLOW_MPC_THRESHOLD {
            challenge.grid_optimize(&|c, s| branch_b::policy(c, s))?
        } else {
            challenge.grid_optimize(&|c, s| branch_c::policy(c, s, &config))?
        };

        save_solution(&solution)?;
        Ok(())
    }

    pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        let config = Config::initialize(&None);
        let size = challenge.num_batteries * challenge.num_steps;
        if size <= BASELINE_THRESHOLD {
            branch_baseline::policy(challenge, state)
        } else if size <= FULL_MPC_THRESHOLD {
            branch_a::policy(challenge, state, &config, None)
        } else if size <= SHALLOW_MPC_THRESHOLD {
            branch_b::policy(challenge, state)
        } else {
            branch_c::policy(challenge, state, &config)
        }
    }

    struct WarmStartCache {
        prev_actions: Vec<f64>,
    }

    impl WarmStartCache {
        fn new(n_bats: usize) -> Self {
            Self { prev_actions: vec![0.0; n_bats] }
        }
    }

    pub fn make_candidates(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        if n <= 1 || (hi - lo).abs() < 1e-9 {
            return vec![lo];
        }
        let mut pts: Vec<f64> = (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
            .collect();
        pts[0] = lo;
        pts[n - 1] = hi;
        pts
    }

    pub fn apply_flow_scale(
        challenge: &Challenge,
        state: &State,
        mut chosen: Vec<f64>,
    ) -> Vec<f64> {
        let inj = challenge.compute_total_injections(state, &chosen);
        let flows = challenge.network.compute_flows(&inj);
        if challenge.network.verify_flows(&flows).is_ok() {
            return chosen;
        }
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        for _ in 0..32 {
            let mid = (lo + hi) * 0.5;
            let scaled: Vec<f64> = chosen.iter().map(|a| a * mid).collect();
            let inj2 = challenge.compute_total_injections(state, &scaled);
            let fl2 = challenge.network.compute_flows(&inj2);
            if challenge.network.verify_flows(&fl2).is_ok() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        for a in chosen.iter_mut() {
            *a *= lo;
        }
        chosen
    }

    pub fn terminal_soc_value(
        challenge: &Challenge,
        state: &State,
        bat: usize,
        horizon_end: usize,
    ) -> f64 {
        let b = &challenge.batteries[bat];
        let available = (state.socs[bat] - b.soc_min_mwh).max(0.0);
        if available < 1e-9 {
            return 0.0;
        }
        let da = &challenge.market.day_ahead_prices;
        let end = (horizon_end + TERM_LOOK).min(challenge.num_steps);
        if end <= horizon_end {
            return 0.0;
        }
        let node = b.node;
        let count = (end - horizon_end) as f64;
        let avg_price: f64 = (horizon_end..end).map(|s| da[s][node]).sum::<f64>() / count;
        let power = (available * b.efficiency_discharge / constants::DELTA_T).min(b.power_discharge_mw);
        let revenue = power * avg_price * constants::DELTA_T;
        let tx = constants::KAPPA_TX * power * constants::DELTA_T;
        let deg_base = (power * constants::DELTA_T) / b.capacity_mwh;
        let deg = constants::KAPPA_DEG * deg_base.powf(constants::BETA_DEG);
        (revenue - tx - deg).max(0.0)
    }
}

// ─── mpc_v6 (inlined) ─────────────────────────────────────────────────────────

mod mpc_v6_inlined {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::energy_arbitrage::{constants, Challenge, Solution, State};

    // ── config ────────────────────────────────────────────────────────────────

    mod config {
        use serde_json::{Map, Value};

        pub struct TrackParams {
            pub sigma: f64,
            pub rho_jump: f64,
            pub alpha: f64,
        }

        pub fn infer_track_params(num_batteries: usize, num_steps: usize) -> TrackParams {
            let size = num_batteries * num_steps;
            match size {
                s if s <= 1000 => TrackParams { sigma: 0.10, rho_jump: 0.01, alpha: 4.0 },
                s if s <= 2000 => TrackParams { sigma: 0.15, rho_jump: 0.02, alpha: 3.5 },
                s if s <= 8000 => TrackParams { sigma: 0.20, rho_jump: 0.03, alpha: 3.0 },
                s if s <= 12000 => TrackParams { sigma: 0.25, rho_jump: 0.04, alpha: 2.7 },
                _ => TrackParams { sigma: 0.30, rho_jump: 0.05, alpha: 2.5 },
            }
        }

        pub const BASELINE_THRESHOLD: usize = 1000;
        pub const FULL_MPC_THRESHOLD: usize = 3000;
        pub const SHALLOW_MPC_THRESHOLD: usize = 15000;

        pub const PRESETS: [(usize, usize, bool); 5] = [
            (50, 5, false),   // effort 0
            (50, 5, false),   // effort 1
            (64, 9, false),   // effort 2
            (80, 9, false),   // effort 3 (default)
            (100, 17, true),  // effort 4
        ];

        pub struct Config {
            pub effort: usize,
            pub n_grid: usize,
            pub n_cand_forward: usize,
            pub warm_start: bool,
        }

        pub fn preset_for_effort(effort: usize) -> (usize, usize, bool) {
            PRESETS[effort.min(4)]
        }

        impl Config {
            pub fn initialize(hyperparameters: &Option<Map<String, Value>>) -> Self {
                let effort = hyperparameters
                    .as_ref()
                    .and_then(|h| h.get("effort"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v.clamp(0, 4) as usize)
                    .unwrap_or(3);
                let (n_grid, n_cand_forward, warm_start) = PRESETS[effort];
                Config { effort, n_grid, n_cand_forward, warm_start }
            }
        }
    }

    // ── sdp_value ─────────────────────────────────────────────────────────────

    mod sdp_value {
        pub struct NormalizedValueTable {
            pub val: Vec<Vec<f64>>,
            pub e_grid: Vec<f64>,
            pub n_grid: usize,
            pub n_steps: usize,
            pub de: f64,
        }

        impl NormalizedValueTable {
            pub fn new(n_steps: usize, n_grid: usize, e_min: f64, e_max: f64) -> Self {
                let n = n_grid.max(2);
                let de = (e_max - e_min) / (n - 1) as f64;
                let e_grid: Vec<f64> = (0..n).map(|k| e_min + de * k as f64).collect();
                let val = vec![vec![0.0f64; n]; n_steps + 1];
                Self { val, e_grid, n_grid: n, n_steps, de }
            }

            pub fn with_custom_grid(n_steps: usize, e_grid: Vec<f64>) -> Self {
                let n = e_grid.len().max(2);
                let de = if n >= 2 {
                    (e_grid[n - 1] - e_grid[0]) / (n - 1) as f64
                } else {
                    0.0
                };
                let val = vec![vec![0.0f64; n]; n_steps + 1];
                Self { val, e_grid, n_grid: n, n_steps, de }
            }

            #[inline]
            pub fn interpolate(&self, t: usize, e_norm: f64) -> f64 {
                interp_val(&self.val[t], &self.e_grid, e_norm)
            }

            #[inline]
            pub fn marginal(&self, t: usize, e_norm: f64) -> f64 {
                marginal_val(&self.val[t], &self.e_grid, e_norm)
            }
        }

        pub fn interp_val(val: &[f64], e_grid: &[f64], e: f64) -> f64 {
            let n = val.len();
            if n == 0 { return 0.0; }
            let e = e.clamp(e_grid[0], e_grid[n - 1]);
            let idx = e_grid.partition_point(|&g| g <= e).saturating_sub(1).min(n - 2);
            let lo = e_grid[idx];
            let hi = e_grid[idx + 1];
            let span = hi - lo;
            if span < 1e-14 { return val[idx]; }
            let alpha = (e - lo) / span;
            val[idx] + alpha * (val[idx + 1] - val[idx])
        }

        pub fn marginal_val(val: &[f64], e_grid: &[f64], e: f64) -> f64 {
            let n = val.len();
            if n < 2 { return 0.0; }
            let e = e.clamp(e_grid[0], e_grid[n - 1]);
            let k = e_grid.partition_point(|&g| g <= e).saturating_sub(1).min(n - 2);
            let span = e_grid[k + 1] - e_grid[k];
            if span < 1e-14 { return 0.0; }
            (val[k + 1] - val[k]) / span
        }

        pub fn enforce_concavity_val(val: &mut [f64]) {
            let n = val.len();
            if n < 3 { return; }
            let orig = val.to_vec();
            let mut hull: Vec<usize> = Vec::with_capacity(n);
            for k in 0..n {
                while hull.len() >= 2 {
                    let j = *hull.last().unwrap();
                    let i = hull[hull.len() - 2];
                    let slope_ij = (orig[j] - orig[i]) / (j - i) as f64;
                    let slope_jk = (orig[k] - orig[j]) / (k - j) as f64;
                    if slope_jk > slope_ij { hull.pop(); } else { break; }
                }
                hull.push(k);
            }
            for h in 0..hull.len().saturating_sub(1) {
                let i = hull[h];
                let j = hull[h + 1];
                let vi = orig[i];
                let vj = orig[j];
                let span = (j - i) as f64;
                for k in i..=j {
                    val[k] = vi + (vj - vi) * (k - i) as f64 / span;
                }
            }
        }
    }

    // ── sdp_backward ──────────────────────────────────────────────────────────

    mod sdp_backward {
        use tig_challenges::energy_arbitrage::constants;
        use super::sdp_value::{enforce_concavity_val, interp_val, marginal_val, NormalizedValueTable};

        const C_RATE: f64 = constants::NOMINAL_POWER / constants::NOMINAL_CAPACITY;
        const DELTA_CHARGE_MAX: f64 = constants::ETA_CHARGE * C_RATE * constants::DELTA_T;
        const DELTA_DISCHARGE_MAX: f64 = C_RATE * constants::DELTA_T / constants::ETA_DISCHARGE;

        pub struct NormalizedSDPParams {
            pub n_grid: usize,
            pub n_steps: usize,
            pub sigma: f64,
            pub rho_jump: f64,
            pub alpha: f64,
            pub e_min: f64,
            pub e_max: f64,
            pub e_grid: Vec<f64>,
        }

        impl NormalizedSDPParams {
            pub fn new(n_grid: usize, n_steps: usize, sigma: f64, rho_jump: f64, alpha: f64) -> Self {
                let e_min = constants::E_MIN_FRAC;
                let e_max = constants::E_MAX_FRAC;
                let n = n_grid.max(2);
                let de = (e_max - e_min) / (n - 1) as f64;
                let e_grid: Vec<f64> = (0..n).map(|k| e_min + de * k as f64).collect();
                Self { n_grid: n, n_steps, sigma, rho_jump, alpha, e_min, e_max, e_grid }
            }

            pub fn with_grid(e_grid: Vec<f64>, n_steps: usize, sigma: f64, rho_jump: f64, alpha: f64) -> Self {
                let n = e_grid.len().max(2);
                let e_min = e_grid.first().copied().unwrap_or(constants::E_MIN_FRAC);
                let e_max = e_grid.last().copied().unwrap_or(constants::E_MAX_FRAC);
                Self { n_grid: n, n_steps, sigma, rho_jump, alpha, e_min, e_max, e_grid }
            }
        }

        pub fn backward_induction_normalized(
            params: &NormalizedSDPParams,
            da_ref: &[f64],
            cong_prob: &[Vec<f64>],
            cong_node: usize,
        ) -> NormalizedValueTable {
            let t_steps = params.n_steps;
            let mut table = NormalizedValueTable::with_custom_grid(t_steps, params.e_grid.clone());
            let g = table.n_grid;

            for step in (0..t_steps).rev() {
                let da_t = da_ref[step].max(1e-6);

                let cong_bonus_per_step: f64 = if !cong_prob.is_empty() {
                    let row = &cong_prob[step];
                    let p_cong = if cong_node < row.len() { row[cong_node] } else { 0.0 };
                    p_cong * constants::GAMMA_PRICE * 0.39894 * C_RATE * constants::DELTA_T
                } else {
                    0.0
                };

                let rj = params.rho_jump;
                let pareto_mean = if params.alpha > 1.01 {
                    params.alpha / (params.alpha - 1.0)
                } else {
                    params.alpha / 0.01
                };

                let e_grid = table.e_grid.clone();
                let val_next: Vec<f64> = table.val[step + 1].clone();

                for k in 0..g {
                    let e = e_grid[k];
                    let soc_limited_d = (e - params.e_min) / DELTA_DISCHARGE_MAX;
                    let u_hi = soc_limited_d.min(1.0).max(0.0);
                    let soc_limited_c = (params.e_max - e) / DELTA_CHARGE_MAX;
                    let u_lo_abs = soc_limited_c.min(1.0).max(0.0);

                    let e_c = (e + u_lo_abs * DELTA_CHARGE_MAX).min(params.e_max);
                    let e_d = (e - u_hi * DELTA_DISCHARGE_MAX).max(params.e_min);

                    let v_c_next = interp_val(&val_next, &e_grid, e_c);
                    let v_i_next = val_next[k];
                    let v_d_next = interp_val(&val_next, &e_grid, e_d);

                    let mv_c = marginal_val(&val_next, &e_grid, e_c);
                    let mv_d = marginal_val(&val_next, &e_grid, e_d);

                    let profit_unit = da_t * C_RATE * constants::DELTA_T;

                    let (phi_c_val, phi_c_pdf, phi_d_val, phi_d_pdf) = if params.sigma < 1e-9 {
                        let thresh_c = mv_c * constants::ETA_CHARGE - constants::KAPPA_TX;
                        let thresh_d = mv_d / constants::ETA_DISCHARGE + constants::KAPPA_TX;
                        let p_c = if da_t <= thresh_c { 1.0 } else { 0.0 };
                        let p_d = if da_t >= thresh_d { 1.0 } else { 0.0 };
                        (p_c, 0.0f64, 1.0 - p_d, 0.0f64)
                    } else {
                        let xi_c = ((mv_c * constants::ETA_CHARGE - constants::KAPPA_TX) / da_t - 1.0) / params.sigma;
                        let xi_d = ((mv_d / constants::ETA_DISCHARGE + constants::KAPPA_TX) / da_t - 1.0) / params.sigma;
                        (hart_phi(xi_c), normal_pdf(xi_c), hart_phi(xi_d), normal_pdf(xi_d))
                    };

                    let p_c = phi_c_val;
                    let p_d = 1.0 - phi_d_val;
                    let p_i = (phi_d_val - phi_c_val).max(0.0);

                    let charge_revenue = -u_lo_abs * profit_unit * (p_c - params.sigma * phi_c_pdf);
                    let charge_tx = -constants::KAPPA_TX * u_lo_abs * C_RATE * constants::DELTA_T * p_c;
                    let charge_deg = -constants::KAPPA_DEG * (u_lo_abs * C_RATE * constants::DELTA_T).powf(constants::BETA_DEG) * p_c;

                    let discharge_revenue = u_hi * profit_unit * (p_d + params.sigma * phi_d_pdf);
                    let discharge_tx = -constants::KAPPA_TX * u_hi * C_RATE * constants::DELTA_T * p_d;
                    let discharge_deg = -constants::KAPPA_DEG * (u_hi * C_RATE * constants::DELTA_T).powf(constants::BETA_DEG) * p_d;

                    let cong_bonus = u_hi * cong_bonus_per_step;

                    let v_gauss = charge_revenue + charge_tx + charge_deg
                        + p_c * v_c_next + p_i * v_i_next
                        + discharge_revenue + discharge_tx + discharge_deg
                        + cong_bonus + p_d * v_d_next;

                    let lambda_jump = da_t * (1.0 + pareto_mean);
                    let jump_revenue = u_hi * lambda_jump * C_RATE * constants::DELTA_T;
                    let jump_tx = -constants::KAPPA_TX * u_hi * C_RATE * constants::DELTA_T;
                    let jump_deg = -constants::KAPPA_DEG * (u_hi * C_RATE * constants::DELTA_T).powf(constants::BETA_DEG);
                    let v_jump = jump_revenue + jump_tx + jump_deg + v_d_next;

                    table.val[step][k] = (1.0 - rj) * v_gauss + rj * v_jump;
                }

                enforce_concavity_val(&mut table.val[step]);
            }

            table
        }

        pub fn precompute_congestion(
            exogenous_injections: &[Vec<f64>],
            ptdf: &[Vec<f64>],
            flow_limits: &[f64],
            tau_cong: f64,
        ) -> Vec<Vec<f64>> {
            let t_steps = exogenous_injections.len();
            let n_lines = ptdf.len();
            if n_lines == 0 || t_steps == 0 { return vec![]; }
            let mut result = vec![vec![0.0f64; n_lines]; t_steps];
            for t in 0..t_steps {
                let inj = &exogenous_injections[t];
                for l in 0..n_lines {
                    let ptdf_row = &ptdf[l];
                    let flow: f64 = ptdf_row.iter().zip(inj.iter()).map(|(&p, &x)| p * x).sum();
                    let lim = flow_limits[l];
                    if lim > 1e-9 {
                        let ratio = flow.abs() / (tau_cong * lim);
                        result[t][l] = ratio.powi(10).min(1.0);
                    }
                }
            }
            result
        }

        pub fn split_slack_bus(battery_nodes: &[usize], ptdf: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>) {
            let n_lines = ptdf.len();
            let mut slack = Vec::new();
            let mut net = Vec::new();
            for (i, &node) in battery_nodes.iter().enumerate() {
                let on_slack = n_lines == 0
                    || ptdf.iter().all(|row| node >= row.len() || row[node].abs() < 1e-9);
                if on_slack { slack.push(i); } else { net.push(i); }
            }
            (slack, net)
        }

        const SQRT_2PI_INV: f64 = 0.3989422804014327;

        pub fn hart_phi(x: f64) -> f64 {
            if x > 8.0 { return 1.0; }
            if x < -8.0 { return 0.0; }
            let neg = x < 0.0;
            let z = if neg { -x } else { x };
            const P: f64 = 0.2316419;
            const A: [f64; 5] = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429];
            let t = 1.0 / (1.0 + P * z);
            let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
            let tail = SQRT_2PI_INV * (-0.5 * z * z).exp() * poly;
            let result = 1.0 - tail;
            if neg { 1.0 - result } else { result }
        }

        #[inline]
        pub fn normal_pdf(x: f64) -> f64 {
            SQRT_2PI_INV * (-0.5 * x * x).exp()
        }

        #[inline]
        pub fn phi(x: f64) -> f64 {
            hart_phi(x)
        }
    }

    // ── hyperparams ───────────────────────────────────────────────────────────

    mod hyperparams {
        use serde_json::{Map, Value};

        #[derive(Clone, Debug, PartialEq)]
        pub enum WarmStartStrategy { None, Primal, Dual, Both }

        #[derive(Clone, Debug, PartialEq)]
        pub enum MultistartVariant { DaProxy, ZeroInit, SmoothInit }

        #[derive(Clone, Debug, PartialEq)]
        pub enum SelectStrategy { Default, Best, First }

        #[derive(Clone, Debug, PartialEq)]
        pub enum SocGridStrategy { Uniform, Cosine, Adaptive }

        pub struct Mpc6Hyperparams {
            pub effort: u8,
            pub track_threshold_small: usize,
            pub track_threshold_medium: usize,
            pub track_threshold_large: usize,
            pub force_branch: Option<String>,
            pub rho_init: f64,
            pub rho_adaptive: bool,
            pub max_iter: usize,
            pub eps_abs: f64,
            pub eps_rel: f64,
            pub warm_start_strategy: WarmStartStrategy,
            pub soc_grid_resolution: usize,
            pub soc_grid_strategy: SocGridStrategy,
            pub use_qp: bool,
            pub qp_solver_max_iter: usize,
            pub multistart_variant: MultistartVariant,
            pub multistart_count: usize,
            pub multistart_select: SelectStrategy,
            pub(super) n_cand_forward: usize,
        }

        impl Mpc6Hyperparams {
            pub fn from_hyperparameter_map(hyperparameters: &Option<Map<String, Value>>) -> Self {
                let h = match hyperparameters.as_ref() {
                    Some(h) => h,
                    None => return Self::default_for_effort(3),
                };
                let effort = h.get("effort").and_then(|v| v.as_u64())
                    .map(|v| v.clamp(0, 4) as u8).unwrap_or(3);
                let (preset_n_grid, preset_n_cands, _) = super::config::preset_for_effort(effort as usize);
                let soc_grid_resolution = h.get("soc_grid_resolution").and_then(|v| v.as_u64())
                    .map(|v| v.clamp(8, 512) as usize).unwrap_or(preset_n_grid);
                let rho_init = h.get("rho_init").and_then(|v| v.as_f64()).filter(|&v| v > 0.0).unwrap_or(50.0);
                let rho_adaptive = h.get("rho_adaptive").and_then(|v| v.as_bool()).unwrap_or(true);
                let max_iter = h.get("max_iter").and_then(|v| v.as_u64()).map(|v| v.max(1) as usize).unwrap_or(30);
                let eps_abs = h.get("eps_abs").and_then(|v| v.as_f64()).filter(|&v| v > 0.0).unwrap_or(0.1);
                let eps_rel = h.get("eps_rel").and_then(|v| v.as_f64()).filter(|&v| v > 0.0).unwrap_or(0.001);
                let warm_start_strategy = h.get("warm_start_strategy").and_then(|v| v.as_str())
                    .map(|s| match s {
                        "None" => WarmStartStrategy::None,
                        "Primal" => WarmStartStrategy::Primal,
                        "Dual" => WarmStartStrategy::Dual,
                        _ => WarmStartStrategy::Both,
                    }).unwrap_or(WarmStartStrategy::Both);
                let soc_grid_strategy = h.get("soc_grid_strategy").and_then(|v| v.as_str())
                    .map(|s| match s {
                        "Cosine" => SocGridStrategy::Cosine,
                        "Adaptive" => SocGridStrategy::Adaptive,
                        _ => SocGridStrategy::Uniform,
                    }).unwrap_or(SocGridStrategy::Uniform);
                let use_qp = h.get("use_qp").and_then(|v| v.as_bool()).unwrap_or(false);
                let qp_solver_max_iter = h.get("qp_solver_max_iter").and_then(|v| v.as_u64())
                    .map(|v| v.max(1) as usize).unwrap_or(200);
                let track_threshold_small = h.get("track_threshold_small").and_then(|v| v.as_u64())
                    .map(|v| v as usize).unwrap_or(1000);
                let track_threshold_medium = h.get("track_threshold_medium").and_then(|v| v.as_u64())
                    .map(|v| v as usize).unwrap_or(7680);
                let track_threshold_large = h.get("track_threshold_large").and_then(|v| v.as_u64())
                    .map(|v| v as usize).unwrap_or(19200);
                let force_branch = h.get("force_branch").and_then(|v| v.as_str()).map(|s| s.to_string());
                let multistart_variant = h.get("multistart_variant").and_then(|v| v.as_str())
                    .map(|s| match s {
                        "ZeroInit" => MultistartVariant::ZeroInit,
                        "SmoothInit" => MultistartVariant::SmoothInit,
                        _ => MultistartVariant::DaProxy,
                    }).unwrap_or(MultistartVariant::DaProxy);
                let multistart_count = h.get("multistart_count").and_then(|v| v.as_u64())
                    .map(|v| v.clamp(1, 3) as usize).unwrap_or(1);
                let multistart_select = h.get("multistart_select").and_then(|v| v.as_str())
                    .map(|s| match s {
                        "Best" => SelectStrategy::Best,
                        "First" => SelectStrategy::First,
                        _ => SelectStrategy::Default,
                    }).unwrap_or(SelectStrategy::Default);
                Mpc6Hyperparams {
                    effort, track_threshold_small, track_threshold_medium, track_threshold_large,
                    force_branch, rho_init, rho_adaptive, max_iter, eps_abs, eps_rel,
                    warm_start_strategy, soc_grid_resolution, soc_grid_strategy,
                    use_qp, qp_solver_max_iter, multistart_variant, multistart_count,
                    multistart_select, n_cand_forward: preset_n_cands,
                }
            }

            fn default_for_effort(effort: u8) -> Self {
                let (preset_n_grid, preset_n_cands, _) = super::config::preset_for_effort(effort as usize);
                Mpc6Hyperparams {
                    effort, track_threshold_small: 1000, track_threshold_medium: 7680,
                    track_threshold_large: 19200, force_branch: None,
                    rho_init: 50.0, rho_adaptive: true, max_iter: 30,
                    eps_abs: 0.1, eps_rel: 0.001, warm_start_strategy: WarmStartStrategy::Both,
                    soc_grid_resolution: preset_n_grid, soc_grid_strategy: SocGridStrategy::Uniform,
                    use_qp: false, qp_solver_max_iter: 200,
                    multistart_variant: MultistartVariant::DaProxy,
                    multistart_count: 1, multistart_select: SelectStrategy::Default,
                    n_cand_forward: preset_n_cands,
                }
            }

            pub fn build_soc_grid(&self, e_min: f64, e_max: f64) -> Vec<f64> {
                let n = self.soc_grid_resolution.max(2);
                match self.soc_grid_strategy {
                    SocGridStrategy::Uniform | SocGridStrategy::Adaptive => {
                        let de = (e_max - e_min) / (n - 1) as f64;
                        (0..n).map(|k| e_min + de * k as f64).collect()
                    }
                    SocGridStrategy::Cosine => {
                        let half = (e_min + e_max) / 2.0;
                        let span = (e_max - e_min) / 2.0;
                        (0..n).map(|k| {
                            let theta = std::f64::consts::PI * k as f64 / (n - 1) as f64;
                            half - span * theta.cos()
                        }).collect()
                    }
                }
            }

            pub fn to_admm_params(&self) -> super::path_admm::ADMMParams {
                super::path_admm::ADMMParams {
                    rho_init: self.rho_init,
                    rho_adaptive: self.rho_adaptive,
                    max_iters: self.max_iter,
                    eps_abs: self.eps_abs,
                    n_u_cands: self.n_cand_forward,
                    ..super::path_admm::ADMMParams::default()
                }
            }

            #[cfg(feature = "clarabel_solver")]
            pub fn to_clarabel_params(&self) -> super::path_clarabel::ClarabelParams {
                super::path_clarabel::ClarabelParams {
                    use_qp: self.use_qp,
                    max_iter: self.qp_solver_max_iter.min(u32::MAX as usize) as u32,
                    ..super::path_clarabel::ClarabelParams::default()
                }
            }
        }
    }

    // ── path_admm PLACEHOLDER ─────────────────────────────────────────────────
    // Will be filled in next edit

    mod path_admm {
        use super::sdp_value::NormalizedValueTable;
        use std::time::Instant;
        use tig_challenges::energy_arbitrage::{constants, Battery, Challenge};

        const DT: f64 = constants::DELTA_T;
        const ETA_C: f64 = constants::ETA_CHARGE;
        const ETA_D: f64 = constants::ETA_DISCHARGE;
        const K_TX: f64 = constants::KAPPA_TX;
        const K_DEG: f64 = constants::KAPPA_DEG;
        const NOM_CAP: f64 = constants::NOMINAL_CAPACITY;

        #[derive(Clone)]
        pub struct ADMMParams {
            pub rho_init: f64,
            pub rho_min: f64,
            pub rho_max: f64,
            pub mu_resid: f64,
            pub tau_incr: f64,
            pub tau_decr: f64,
            pub max_iters: usize,
            pub eps_abs: f64,
            pub n_u_cands: usize,
            pub rho_adaptive: bool,
        }

        impl Default for ADMMParams {
            fn default() -> Self {
                ADMMParams {
                    rho_init: 50.0, rho_min: 0.1, rho_max: 50_000.0,
                    mu_resid: 10.0, tau_incr: 2.0, tau_decr: 2.0,
                    max_iters: 30, eps_abs: 0.1, n_u_cands: 17, rho_adaptive: true,
                }
            }
        }

        pub struct ADMMState {
            pub u: Vec<f64>,
            pub z: Vec<f64>,
            pub mu: Vec<f64>,
            pub rho: f64,
        }

        #[derive(Debug)]
        pub struct ADMMResult {
            pub iters: usize,
            pub primal_resid: f64,
            pub dual_resid: f64,
            pub converged: bool,
        }

        impl ADMMState {
            pub fn cold_start(n_bats: usize, ext_flow: &[f64], rho_init: f64) -> Self {
                ADMMState {
                    u: vec![0.0; n_bats],
                    z: ext_flow.to_vec(),
                    mu: vec![0.0; ext_flow.len()],
                    rho: rho_init,
                }
            }
        }

        pub fn compute_ext_flow(ptdf: &[Vec<f64>], exog: &[f64]) -> Vec<f64> {
            ptdf.iter().map(|row| {
                row.iter().zip(exog.iter()).map(|(&p, &x)| p * x).sum::<f64>()
            }).collect()
        }

        #[inline]
        fn apply_action_soc(action: f64, soc: f64, bat: &Battery) -> f64 {
            let c = (-action).max(0.0);
            let d = action.max(0.0);
            let new_soc = soc + bat.efficiency_charge * c * DT - d * DT / bat.efficiency_discharge;
            new_soc.clamp(bat.soc_min_mwh, bat.soc_max_mwh)
        }

        #[inline]
        fn action_bounds_from_soc(bat: &Battery, soc: f64) -> (f64, f64) {
            let headroom = (bat.soc_max_mwh - soc).max(0.0);
            let available = (soc - bat.soc_min_mwh).max(0.0);
            let max_charge = if bat.efficiency_charge > 1e-9 {
                (headroom / (bat.efficiency_charge * DT)).min(bat.power_charge_mw).max(0.0)
            } else { 0.0 };
            let max_discharge = (available * bat.efficiency_discharge / DT).min(bat.power_discharge_mw).max(0.0);
            (-max_charge, max_discharge)
        }

        #[inline]
        fn battery_aug_obj(
            u: f64, soc: f64, rt_price: f64, bat: &Battery,
            vtable: &NormalizedValueTable, t_next: usize, val_scale: f64, eff_shift: f64, rho_a: f64,
        ) -> f64 {
            let cap = bat.capacity_mwh;
            let new_soc = apply_action_soc(u, soc, bat);
            let e_norm = (new_soc / cap).clamp(constants::E_MIN_FRAC, constants::E_MAX_FRAC);
            let v_cont = vtable.interpolate(t_next, e_norm) * val_scale;
            let abs_u = u.abs();
            u * rt_price * DT - K_TX * abs_u * DT - K_DEG * (abs_u * DT / cap).powi(2)
                + v_cont + eff_shift * u - 0.5 * rho_a * u * u
        }

        fn solve_battery_subproblem(
            soc: f64, rt_price: f64, u_lo: f64, u_hi: f64, bat: &Battery,
            vtable: &NormalizedValueTable, t_next: usize, val_scale: f64,
            eff_shift: f64, rho_a: f64, n_cands: usize,
        ) -> f64 {
            let cap = bat.capacity_mwh;
            let obj = |u: f64| battery_aug_obj(u, soc, rt_price, bat, vtable, t_next, val_scale, eff_shift, rho_a);
            let u_zero = 0.0f64.clamp(u_lo, u_hi);
            let mut best_u = u_zero;
            let mut best_f = obj(u_zero);
            let n = n_cands.max(3);
            for k in 0..n {
                let alpha = k as f64 / (n - 1) as f64;
                let u = u_lo + alpha * (u_hi - u_lo);
                let f = obj(u);
                if f > best_f { best_f = f; best_u = u; }
            }
            let denom_deg = 2.0 * K_DEG * (DT / cap).powi(2);
            for &(region_lo, region_hi, is_discharge) in &[
                (u_lo, 0.0f64.min(u_hi), false),
                (0.0f64.max(u_lo), u_hi, true),
            ] {
                if region_hi - region_lo < 1e-12 { continue; }
                let u_mid = (region_lo + region_hi) * 0.5;
                let soc_mid = apply_action_soc(u_mid, soc, bat);
                let e_mid = (soc_mid / cap).clamp(constants::E_MIN_FRAC, constants::E_MAX_FRAC);
                let dsoc_du = if is_discharge { -DT / ETA_D } else { -ETA_C * DT };
                let mv = vtable.marginal(t_next, e_mid);
                let dv_du = mv * val_scale * (dsoc_du / cap);
                let tx_sign = if is_discharge { 1.0 } else { -1.0 };
                let lin = rt_price * DT + eff_shift + dv_du - tx_sign * K_TX * DT;
                let denom = rho_a + denom_deg;
                let u_star = if denom > 1e-12 {
                    (lin / denom).clamp(region_lo, region_hi)
                } else if lin >= 0.0 { region_hi } else { region_lo };
                let f = obj(u_star);
                if f > best_f { best_f = f; best_u = u_star; }
            }
            best_u
        }

        pub fn admm_iteration(
            state: &mut ADMMState,
            socs: &[f64], rt_prices: &[f64], bounds: &[(f64, f64)], ext_flow: &[f64],
            batteries: &[Battery], bat_ptdf_col: &[Vec<f64>], bat_a: &[f64],
            flow_limits: &[f64], vtable: &NormalizedValueTable, t_next: usize,
            val_scales: &[f64], params: &ADMMParams,
        ) -> (f64, f64) {
            let n = batteries.len();
            let n_lines = flow_limits.len();
            let rho = state.rho;
            let mut cur_flow: Vec<f64> = ext_flow.to_vec();
            for i in 0..n {
                let u_i = state.u[i];
                for (l, p) in bat_ptdf_col[i].iter().enumerate() { cur_flow[l] += p * u_i; }
            }
            for i in 0..n {
                let p_col = &bat_ptdf_col[i];
                let (r_i, shadow_i): (f64, f64) = (0..n_lines).fold((0.0, 0.0), |(r, s), l| {
                    let res = cur_flow[l] - state.z[l];
                    (r + p_col[l] * res, s + state.mu[l] * p_col[l])
                });
                let rho_a = rho * bat_a[i];
                let eff_shift = rho_a * state.u[i] - rho * r_i - shadow_i;
                let rt_price = rt_prices.get(batteries[i].node).copied().unwrap_or(0.0);
                let u_new = solve_battery_subproblem(
                    socs[i], rt_price, bounds[i].0, bounds[i].1, &batteries[i],
                    vtable, t_next, val_scales[i], eff_shift, rho_a, params.n_u_cands,
                );
                let delta = u_new - state.u[i];
                for (l, p) in p_col.iter().enumerate() { cur_flow[l] += p * delta; }
                state.u[i] = u_new;
            }
            let new_flow = cur_flow;
            let z_prev: Vec<f64> = state.z.clone();
            for l in 0..n_lines {
                let limit = flow_limits[l];
                state.z[l] = (new_flow[l] + state.mu[l] / rho).clamp(-limit, limit);
            }
            for l in 0..n_lines { state.mu[l] += rho * (new_flow[l] - state.z[l]); }
            let primal_resid = (0..n_lines).map(|l| { let r = new_flow[l] - state.z[l]; r * r }).sum::<f64>().sqrt();
            let dual_resid = rho * (0..n_lines).map(|l| { let d = state.z[l] - z_prev[l]; d * d }).sum::<f64>().sqrt();
            if params.rho_adaptive {
                if primal_resid > params.mu_resid * dual_resid {
                    state.rho = (state.rho * params.tau_incr).min(params.rho_max);
                } else if dual_resid > params.mu_resid * primal_resid && primal_resid > 1.0 {
                    state.rho = (state.rho / params.tau_decr).max(params.rho_min);
                }
            }
            (primal_resid, dual_resid)
        }

        pub fn solve_step_admm(
            state: &mut ADMMState,
            socs: &[f64], rt_prices: &[f64], bounds: &[(f64, f64)], ext_flow: &[f64],
            batteries: &[Battery], bat_ptdf_col: &[Vec<f64>], bat_a: &[f64],
            flow_limits: &[f64], vtable: &NormalizedValueTable, t_next: usize,
            val_scales: &[f64], params: &ADMMParams,
        ) -> ADMMResult {
            let mut pr = f64::MAX;
            let mut dr = f64::MAX;
            let mut iters = 0;
            for _ in 0..params.max_iters {
                let (p, d) = admm_iteration(
                    state, socs, rt_prices, bounds, ext_flow, batteries,
                    bat_ptdf_col, bat_a, flow_limits, vtable, t_next, val_scales, params,
                );
                pr = p; dr = d; iters += 1;
                if pr < params.eps_abs && dr < params.eps_abs { break; }
            }
            ADMMResult { iters, primal_resid: pr, dual_resid: dr, converged: pr < params.eps_abs && dr < params.eps_abs }
        }

        pub fn solve_via_admm(
            challenge: &Challenge,
            vtable: &NormalizedValueTable,
            da_ref: &[f64],
            params: &ADMMParams,
        ) -> (Vec<Vec<f64>>, Vec<usize>, f64) {
            let t0 = Instant::now();
            let n = challenge.num_batteries;
            let t_max = challenge.num_steps;
            let bats = &challenge.batteries;
            let ptdf = &challenge.network.ptdf;
            let flow_limits = &challenge.network.flow_limits;
            let n_lines = ptdf.len();
            let bat_ptdf_col: Vec<Vec<f64>> = (0..n).map(|i| {
                let node = bats[i].node;
                (0..n_lines).map(|l| ptdf[l].get(node).copied().unwrap_or(0.0)).collect()
            }).collect();
            let bat_a: Vec<f64> = bat_ptdf_col.iter().map(|col| col.iter().map(|&p| p * p).sum()).collect();
            let mut socs: Vec<f64> = bats.iter().map(|b| b.soc_initial_mwh).collect();
            let da = &challenge.market.day_ahead_prices;
            let ext0 = compute_ext_flow(ptdf, &challenge.exogenous_injections[0]);
            let mut st = ADMMState::cold_start(n, &ext0, params.rho_init);
            let mut schedule = Vec::with_capacity(t_max);
            let mut iter_counts = Vec::with_capacity(t_max);
            for t in 0..t_max {
                let ext_flow = compute_ext_flow(ptdf, &challenge.exogenous_injections[t]);
                let t_next = (t + 1).min(vtable.n_steps);
                let bounds: Vec<(f64, f64)> = (0..n).map(|i| action_bounds_from_soc(&bats[i], socs[i])).collect();
                let da_t = &da[t];
                let da_next = if t + 1 < t_max { &da[t + 1] } else { &da[t_max - 1] };
                let da_ref_next = da_ref.get(t_next.min(t_max.saturating_sub(1))).copied().unwrap_or(50.0).max(1e-6);
                let val_scales: Vec<f64> = (0..n).map(|i| {
                    let cap = bats[i].capacity_mwh;
                    let da_node = da_next.get(bats[i].node).copied().unwrap_or(da_ref_next);
                    (cap / NOM_CAP) * (da_node / da_ref_next)
                }).collect();
                let result = solve_step_admm(
                    &mut st, &socs, da_t, &bounds, &ext_flow, bats,
                    &bat_ptdf_col, &bat_a, flow_limits, vtable, t_next, &val_scales, params,
                );
                let actions: Vec<f64> = st.u.iter().enumerate().map(|(i, &u)| u.clamp(bounds[i].0, bounds[i].1)).collect();
                iter_counts.push(result.iters);
                for i in 0..n { socs[i] = apply_action_soc(actions[i], socs[i], &bats[i]); }
                schedule.push(actions);
            }
            (schedule, iter_counts, t0.elapsed().as_secs_f64() * 1000.0)
        }

        pub fn solve_via_admm_smooth_init(
            challenge: &Challenge,
            vtable: &NormalizedValueTable,
            da_ref: &[f64],
            params: &ADMMParams,
        ) -> (Vec<Vec<f64>>, Vec<usize>, f64) {
            let t0 = Instant::now();
            let n = challenge.num_batteries;
            let t_max = challenge.num_steps;
            let bats = &challenge.batteries;
            let ptdf = &challenge.network.ptdf;
            let flow_limits = &challenge.network.flow_limits;
            let n_lines = ptdf.len();
            let bat_ptdf_col: Vec<Vec<f64>> = (0..n).map(|i| {
                let node = bats[i].node;
                (0..n_lines).map(|l| ptdf[l].get(node).copied().unwrap_or(0.0)).collect()
            }).collect();
            let bat_a: Vec<f64> = bat_ptdf_col.iter().map(|col| col.iter().map(|&p| p * p).sum()).collect();
            let mut socs: Vec<f64> = bats.iter().map(|b| b.soc_initial_mwh).collect();
            let da = &challenge.market.day_ahead_prices;
            let ext0 = compute_ext_flow(ptdf, &challenge.exogenous_injections[0]);
            let mut st = ADMMState::cold_start(n, &ext0, params.rho_init);
            let mut schedule = Vec::with_capacity(t_max);
            let mut iter_counts = Vec::with_capacity(t_max);
            for t in 0..t_max {
                let ext_flow = compute_ext_flow(ptdf, &challenge.exogenous_injections[t]);
                let t_next = (t + 1).min(vtable.n_steps);
                let bounds: Vec<(f64, f64)> = (0..n).map(|i| action_bounds_from_soc(&bats[i], socs[i])).collect();
                for i in 0..n { st.u[i] = (bounds[i].0 + bounds[i].1) * 0.5; }
                let da_t = &da[t];
                let da_next = if t + 1 < t_max { &da[t + 1] } else { &da[t_max - 1] };
                let da_ref_next = da_ref.get(t_next.min(t_max.saturating_sub(1))).copied().unwrap_or(50.0).max(1e-6);
                let val_scales: Vec<f64> = (0..n).map(|i| {
                    let cap = bats[i].capacity_mwh;
                    let da_node = da_next.get(bats[i].node).copied().unwrap_or(da_ref_next);
                    (cap / NOM_CAP) * (da_node / da_ref_next)
                }).collect();
                let result = solve_step_admm(
                    &mut st, &socs, da_t, &bounds, &ext_flow, bats,
                    &bat_ptdf_col, &bat_a, flow_limits, vtable, t_next, &val_scales, params,
                );
                let actions: Vec<f64> = st.u.iter().enumerate().map(|(i, &u)| u.clamp(bounds[i].0, bounds[i].1)).collect();
                iter_counts.push(result.iters);
                for i in 0..n { socs[i] = apply_action_soc(actions[i], socs[i], &bats[i]); }
                schedule.push(actions);
            }
            (schedule, iter_counts, t0.elapsed().as_secs_f64() * 1000.0)
        }

        pub fn solve_via_admm_warmed(
            challenge: &Challenge,
            vtable: &NormalizedValueTable,
            da_ref: &[f64],
            params: &ADMMParams,
            primal_init: &[Vec<f64>],
        ) -> (Vec<Vec<f64>>, Vec<usize>, f64) {
            let t0 = Instant::now();
            let n = challenge.num_batteries;
            let t_max = challenge.num_steps;
            let bats = &challenge.batteries;
            let ptdf = &challenge.network.ptdf;
            let flow_limits = &challenge.network.flow_limits;
            let n_lines = ptdf.len();
            let bat_ptdf_col: Vec<Vec<f64>> = (0..n).map(|i| {
                let node = bats[i].node;
                (0..n_lines).map(|l| ptdf[l].get(node).copied().unwrap_or(0.0)).collect()
            }).collect();
            let bat_a: Vec<f64> = bat_ptdf_col.iter().map(|col| col.iter().map(|&p| p * p).sum()).collect();
            let mut socs: Vec<f64> = bats.iter().map(|b| b.soc_initial_mwh).collect();
            let da = &challenge.market.day_ahead_prices;
            let ext0 = compute_ext_flow(ptdf, &challenge.exogenous_injections[0]);
            let mut st = ADMMState::cold_start(n, &ext0, params.rho_init);
            let mut schedule = Vec::with_capacity(t_max);
            let mut iter_counts = Vec::with_capacity(t_max);
            for t in 0..t_max {
                let ext_flow = compute_ext_flow(ptdf, &challenge.exogenous_injections[t]);
                let t_next = (t + 1).min(vtable.n_steps);
                let bounds: Vec<(f64, f64)> = (0..n).map(|i| action_bounds_from_soc(&bats[i], socs[i])).collect();
                if let Some(init_t) = primal_init.get(t) {
                    for i in 0..n {
                        st.u[i] = init_t.get(i).copied().unwrap_or(0.0).clamp(bounds[i].0, bounds[i].1);
                    }
                }
                let da_t = &da[t];
                let da_next = if t + 1 < t_max { &da[t + 1] } else { &da[t_max - 1] };
                let da_ref_next = da_ref.get(t_next.min(t_max.saturating_sub(1))).copied().unwrap_or(50.0).max(1e-6);
                let val_scales: Vec<f64> = (0..n).map(|i| {
                    let cap = bats[i].capacity_mwh;
                    let da_node = da_next.get(bats[i].node).copied().unwrap_or(da_ref_next);
                    (cap / NOM_CAP) * (da_node / da_ref_next)
                }).collect();
                let result = solve_step_admm(
                    &mut st, &socs, da_t, &bounds, &ext_flow, bats,
                    &bat_ptdf_col, &bat_a, flow_limits, vtable, t_next, &val_scales, params,
                );
                let actions: Vec<f64> = st.u.iter().enumerate().map(|(i, &u)| u.clamp(bounds[i].0, bounds[i].1)).collect();
                iter_counts.push(result.iters);
                for i in 0..n { socs[i] = apply_action_soc(actions[i], socs[i], &bats[i]); }
                schedule.push(actions);
            }
            (schedule, iter_counts, t0.elapsed().as_secs_f64() * 1000.0)
        }
    }

    // ── path_lagrangian ───────────────────────────────────────────────────────

    mod path_lagrangian {
        use super::path_admm::compute_ext_flow;
        use super::sdp_value::NormalizedValueTable;
        use std::time::Instant;
        use tig_challenges::energy_arbitrage::{constants, Battery, Challenge};

        const DT: f64 = constants::DELTA_T;
        const ETA_C: f64 = constants::ETA_CHARGE;
        const ETA_D: f64 = constants::ETA_DISCHARGE;
        const K_TX: f64 = constants::KAPPA_TX;
        const K_DEG: f64 = constants::KAPPA_DEG;
        const NOM_CAP: f64 = constants::NOMINAL_CAPACITY;

        #[derive(Clone)]
        pub struct LagrangianParams {
            pub max_iter: usize,
            pub mu_init: f64,
            pub mu_step: f64,
            pub mu_step_decay: f64,
            pub eps_violation: f64,
            pub n_u_cands: usize,
        }

        impl Default for LagrangianParams {
            fn default() -> Self {
                LagrangianParams { max_iter: 50, mu_init: 1.0, mu_step: 0.5, mu_step_decay: 0.95, eps_violation: 0.1, n_u_cands: 17 }
            }
        }

        pub struct LagrangianState {
            pub u: Vec<f64>,
            pub mu: Vec<f64>,
            pub iteration_count: usize,
        }

        #[derive(Debug)]
        pub struct LagrangianResult {
            pub iters: usize,
            pub max_violation: f64,
            pub converged: bool,
        }

        impl LagrangianState {
            pub fn cold_start(n_bats: usize, n_lines: usize, mu_init: f64) -> Self {
                LagrangianState { u: vec![0.0; n_bats], mu: vec![mu_init; n_lines], iteration_count: 0 }
            }
        }

        pub struct LagrangianBatteryData {
            pub abs_ptdf_col: Vec<Vec<f64>>,
            pub signed_ptdf_col: Vec<Vec<f64>>,
        }

        impl LagrangianBatteryData {
            pub fn new(batteries: &[Battery], ptdf: &[Vec<f64>]) -> Self {
                let n_lines = ptdf.len();
                let mut abs_ptdf_col = Vec::with_capacity(batteries.len());
                let mut signed_ptdf_col = Vec::with_capacity(batteries.len());
                for b in batteries {
                    let col: Vec<f64> = (0..n_lines).map(|l| ptdf[l].get(b.node).copied().unwrap_or(0.0)).collect();
                    abs_ptdf_col.push(col.iter().map(|&p| p.abs()).collect());
                    signed_ptdf_col.push(col);
                }
                LagrangianBatteryData { abs_ptdf_col, signed_ptdf_col }
            }
        }

        #[inline]
        fn apply_action_soc(action: f64, soc: f64, bat: &Battery) -> f64 {
            let c = (-action).max(0.0);
            let d = action.max(0.0);
            (soc + bat.efficiency_charge * c * DT - d * DT / bat.efficiency_discharge)
                .clamp(bat.soc_min_mwh, bat.soc_max_mwh)
        }

        #[inline]
        fn action_bounds_from_soc(bat: &Battery, soc: f64) -> (f64, f64) {
            let headroom = (bat.soc_max_mwh - soc).max(0.0);
            let available = (soc - bat.soc_min_mwh).max(0.0);
            let max_charge = if bat.efficiency_charge > 1e-9 {
                (headroom / (bat.efficiency_charge * DT)).min(bat.power_charge_mw).max(0.0)
            } else { 0.0 };
            let max_discharge = (available * bat.efficiency_discharge / DT).min(bat.power_discharge_mw).max(0.0);
            (-max_charge, max_discharge)
        }

        #[inline]
        fn lagrangian_battery_obj(
            u: f64, soc: f64, rt_price: f64, bat: &Battery,
            vtable: &NormalizedValueTable, t_next: usize, val_scale: f64, cong_rate: f64,
        ) -> f64 {
            let cap = bat.capacity_mwh;
            let new_soc = apply_action_soc(u, soc, bat);
            let e_norm = (new_soc / cap).clamp(constants::E_MIN_FRAC, constants::E_MAX_FRAC);
            let v_cont = vtable.interpolate(t_next, e_norm) * val_scale;
            let abs_u = u.abs();
            u * rt_price * DT - K_TX * abs_u * DT - K_DEG * (abs_u * DT / cap).powi(2) + v_cont - cong_rate * abs_u
        }

        fn solve_battery_lagrangian(
            soc: f64, rt_price: f64, u_lo: f64, u_hi: f64, bat: &Battery,
            vtable: &NormalizedValueTable, t_next: usize, val_scale: f64, cong_rate: f64, n_cands: usize,
        ) -> f64 {
            let cap = bat.capacity_mwh;
            let obj = |u: f64| lagrangian_battery_obj(u, soc, rt_price, bat, vtable, t_next, val_scale, cong_rate);
            let u_zero = 0.0f64.clamp(u_lo, u_hi);
            let mut best_u = u_zero;
            let mut best_f = obj(u_zero);
            let n = n_cands.max(3);
            for k in 0..n {
                let alpha = k as f64 / (n - 1) as f64;
                let u = u_lo + alpha * (u_hi - u_lo);
                let f = obj(u);
                if f > best_f { best_f = f; best_u = u; }
            }
            let denom_deg = 2.0 * K_DEG * (DT / cap).powi(2);
            for &(region_lo, region_hi, is_discharge) in &[
                (u_lo, 0.0f64.min(u_hi), false),
                (0.0f64.max(u_lo), u_hi, true),
            ] {
                if region_hi - region_lo < 1e-12 { continue; }
                let u_mid = (region_lo + region_hi) * 0.5;
                let soc_mid = apply_action_soc(u_mid, soc, bat);
                let e_mid = (soc_mid / cap).clamp(constants::E_MIN_FRAC, constants::E_MAX_FRAC);
                let dsoc_du = if is_discharge { -DT / ETA_D } else { -ETA_C * DT };
                let mv = vtable.marginal(t_next, e_mid);
                let dv_du = mv * val_scale * (dsoc_du / cap);
                let tx_sign: f64 = if is_discharge { 1.0 } else { -1.0 };
                let lin = rt_price * DT + dv_du - tx_sign * (K_TX * DT + cong_rate);
                let u_star = if denom_deg > 1e-12 {
                    (lin / denom_deg).clamp(region_lo, region_hi)
                } else if lin >= 0.0 { region_hi } else { region_lo };
                let f = obj(u_star);
                if f > best_f { best_f = f; best_u = u_star; }
            }
            best_u
        }

        pub fn lagrangian_iteration(
            state: &mut LagrangianState,
            socs: &[f64], rt_prices: &[f64], bounds: &[(f64, f64)], ext_flow: &[f64],
            batteries: &[Battery], bat_data: &LagrangianBatteryData,
            flow_limits: &[f64], vtable: &NormalizedValueTable, t_next: usize,
            val_scales: &[f64], params: &LagrangianParams, mu_step: &mut f64,
        ) -> f64 {
            let n = batteries.len();
            let n_lines = flow_limits.len();
            for i in 0..n {
                let cong_rate: f64 = bat_data.abs_ptdf_col[i].iter().zip(state.mu.iter()).map(|(&p, &m)| p * m).sum();
                let rt_price = rt_prices.get(batteries[i].node).copied().unwrap_or(0.0);
                state.u[i] = solve_battery_lagrangian(
                    socs[i], rt_price, bounds[i].0, bounds[i].1, &batteries[i],
                    vtable, t_next, val_scales[i], cong_rate, params.n_u_cands,
                );
            }
            let mut total_flow = ext_flow.to_vec();
            for i in 0..n {
                let u_i = state.u[i];
                for (l, &p) in bat_data.signed_ptdf_col[i].iter().enumerate() { total_flow[l] += p * u_i; }
            }
            let mut max_viol = 0.0f64;
            for l in 0..n_lines {
                let viol = (total_flow[l].abs() - flow_limits[l]).max(0.0);
                if viol > max_viol { max_viol = viol; }
                state.mu[l] += *mu_step * viol;
            }
            *mu_step *= params.mu_step_decay;
            max_viol
        }

        pub fn solve_step_lagrangian(
            state: &mut LagrangianState,
            socs: &[f64], rt_prices: &[f64], bounds: &[(f64, f64)], ext_flow: &[f64],
            batteries: &[Battery], bat_data: &LagrangianBatteryData,
            flow_limits: &[f64], vtable: &NormalizedValueTable, t_next: usize,
            val_scales: &[f64], params: &LagrangianParams,
        ) -> LagrangianResult {
            let mut mu_step = params.mu_step;
            let mut max_viol = f64::MAX;
            let mut iters = 0;
            for _ in 0..params.max_iter {
                max_viol = lagrangian_iteration(
                    state, socs, rt_prices, bounds, ext_flow, batteries, bat_data,
                    flow_limits, vtable, t_next, val_scales, params, &mut mu_step,
                );
                iters += 1;
                if max_viol < params.eps_violation { break; }
            }
            state.iteration_count += iters;
            LagrangianResult { iters, max_violation: max_viol, converged: max_viol < params.eps_violation }
        }

        pub fn solve_via_lagrangian(
            challenge: &Challenge,
            vtable: &NormalizedValueTable,
            da_ref: &[f64],
            params: &LagrangianParams,
        ) -> (Vec<Vec<f64>>, Vec<usize>, f64) {
            let t0 = Instant::now();
            let n = challenge.num_batteries;
            let t_max = challenge.num_steps;
            let bats = &challenge.batteries;
            let ptdf = &challenge.network.ptdf;
            let flow_limits = &challenge.network.flow_limits;
            let n_lines = ptdf.len();
            let bat_data = LagrangianBatteryData::new(bats, ptdf);
            let mut socs: Vec<f64> = bats.iter().map(|b| b.soc_initial_mwh).collect();
            let da = &challenge.market.day_ahead_prices;
            let mut st = LagrangianState::cold_start(n, n_lines, params.mu_init);
            let mut schedule = Vec::with_capacity(t_max);
            let mut iter_counts = Vec::with_capacity(t_max);
            for t in 0..t_max {
                let ext_flow = compute_ext_flow(ptdf, &challenge.exogenous_injections[t]);
                let t_next = (t + 1).min(vtable.n_steps);
                let bounds: Vec<(f64, f64)> = (0..n).map(|i| action_bounds_from_soc(&bats[i], socs[i])).collect();
                let da_next = if t + 1 < t_max { &da[t + 1] } else { &da[t_max - 1] };
                let da_ref_next = da_ref.get(t_next.min(t_max.saturating_sub(1))).copied().unwrap_or(50.0).max(1e-6);
                let val_scales: Vec<f64> = (0..n).map(|i| {
                    let cap = bats[i].capacity_mwh;
                    let da_node = da_next.get(bats[i].node).copied().unwrap_or(da_ref_next);
                    (cap / NOM_CAP) * (da_node / da_ref_next)
                }).collect();
                let result = solve_step_lagrangian(
                    &mut st, &socs, &da[t], &bounds, &ext_flow, bats, &bat_data,
                    flow_limits, vtable, t_next, &val_scales, params,
                );
                let actions: Vec<f64> = st.u.iter().enumerate().map(|(i, &u)| u.clamp(bounds[i].0, bounds[i].1)).collect();
                iter_counts.push(result.iters);
                for i in 0..n { socs[i] = apply_action_soc(actions[i], socs[i], &bats[i]); }
                schedule.push(actions);
            }
            (schedule, iter_counts, t0.elapsed().as_secs_f64() * 1000.0)
        }
    }

    // ── path_clarabel (feature-gated) ─────────────────────────────────────────

    #[cfg(feature = "clarabel_solver")]
    mod path_clarabel {
        use super::sdp_value::NormalizedValueTable;
        use clarabel::algebra::CscMatrix;
        use clarabel::solver::{
            DefaultSettings, DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
        };
        use std::time::Instant;
        use tig_challenges::energy_arbitrage::{constants, Battery, Challenge};

        const DT: f64 = constants::DELTA_T;
        const ETA_C: f64 = constants::ETA_CHARGE;
        const ETA_D: f64 = constants::ETA_DISCHARGE;
        const K_TX: f64 = constants::KAPPA_TX;
        const K_DEG: f64 = constants::KAPPA_DEG;
        const NOM_CAP: f64 = constants::NOMINAL_CAPACITY;

        #[derive(Clone)]
        pub struct ClarabelParams {
            pub use_qp: bool,
            pub max_iter: u32,
            pub tol_gap_abs: f64,
            pub tol_gap_rel: f64,
            pub tol_feas: f64,
            pub time_limit_s: f64,
        }

        impl Default for ClarabelParams {
            fn default() -> Self {
                ClarabelParams { use_qp: false, max_iter: 50, tol_gap_abs: 1e-5, tol_gap_rel: 1e-5, tol_feas: 1e-5, time_limit_s: 2.0 }
            }
        }

        pub struct ClarabelStepResult {
            pub u: Vec<f64>,
            pub status: SolverStatus,
            pub obj_val: f64,
        }

        #[inline]
        fn apply_action_soc(action: f64, soc: f64, bat: &Battery) -> f64 {
            let c = (-action).max(0.0);
            let d = action.max(0.0);
            (soc + bat.efficiency_charge * c * DT - d * DT / bat.efficiency_discharge)
                .clamp(bat.soc_min_mwh, bat.soc_max_mwh)
        }

        #[inline]
        fn action_bounds_from_soc(bat: &Battery, soc: f64) -> (f64, f64) {
            let headroom = (bat.soc_max_mwh - soc).max(0.0);
            let available = (soc - bat.soc_min_mwh).max(0.0);
            let max_charge = if bat.efficiency_charge > 1e-9 {
                (headroom / (bat.efficiency_charge * DT)).min(bat.power_charge_mw).max(0.0)
            } else { 0.0 };
            let max_discharge = (available * bat.efficiency_discharge / DT).min(bat.power_discharge_mw).max(0.0);
            (-max_charge, max_discharge)
        }

        pub fn precompute_bat_ptdf(batteries: &[Battery], ptdf: &[Vec<f64>]) -> Vec<Vec<(usize, f64)>> {
            (0..batteries.len()).map(|i| {
                let node = batteries[i].node;
                (0..ptdf.len()).filter_map(|l| {
                    let p = ptdf[l].get(node).copied().unwrap_or(0.0);
                    if p.abs() > 1e-15 { Some((l, p)) } else { None }
                }).collect()
            }).collect()
        }

        fn sort_col_entries(rows: &mut [usize], vals: &mut [f64]) {
            let n = rows.len();
            for i in 1..n {
                let kr = rows[i]; let kv = vals[i]; let mut j = i;
                while j > 0 && rows[j - 1] > kr { rows[j] = rows[j - 1]; vals[j] = vals[j - 1]; j -= 1; }
                rows[j] = kr; vals[j] = kv;
            }
        }

        pub fn solve_step_clarabel(
            socs: &[f64], rt_prices_by_node: &[f64], bounds: &[(f64, f64)], ext_flow: &[f64],
            batteries: &[Battery], bat_ptdf: &[Vec<(usize, f64)>],
            flow_limits: &[f64], vtable: &NormalizedValueTable, t_next: usize,
            val_scales: &[f64], params: &ClarabelParams,
        ) -> ClarabelStepResult {
            let n = batteries.len();
            let n_lines = flow_limits.len();
            let n_vars = 2 * n;
            let n_con = 4 * n + 2 * n_lines;
            let mut q = vec![0.0f64; n_vars];
            for i in 0..n {
                let cap = batteries[i].capacity_mwh.max(1e-9);
                let rt_i = rt_prices_by_node.get(batteries[i].node).copied().unwrap_or(0.0);
                let e_norm = (socs[i] / cap).clamp(constants::E_MIN_FRAC, constants::E_MAX_FRAC);
                let mv = vtable.marginal(t_next, e_norm) * val_scales[i];
                q[i] = -(rt_i - K_TX) * DT + mv * DT / (ETA_D * cap);
                q[n + i] = (rt_i + K_TX) * DT - mv * ETA_C * DT / cap;
            }
            let p_csc = if params.use_qp {
                let mut colptr = Vec::with_capacity(n_vars + 1);
                let mut rowval = Vec::with_capacity(n_vars);
                let mut nzval = Vec::with_capacity(n_vars);
                colptr.push(0usize);
                for i in 0..n {
                    let pii = 2.0 * K_DEG * (DT / batteries[i].capacity_mwh.max(1e-9)).powi(2);
                    rowval.push(i); nzval.push(pii); colptr.push(rowval.len());
                }
                for i in 0..n {
                    let pii = 2.0 * K_DEG * (DT / batteries[i].capacity_mwh.max(1e-9)).powi(2);
                    rowval.push(n + i); nzval.push(pii); colptr.push(rowval.len());
                }
                CscMatrix::new(n_vars, n_vars, colptr, rowval, nzval)
            } else {
                CscMatrix::new(n_vars, n_vars, vec![0usize; n_vars + 1], vec![], vec![])
            };
            let mut b = vec![0.0f64; n_con];
            for i in 0..n {
                let (lo, hi) = bounds[i];
                b[i] = hi.max(0.0); b[n + i] = (-lo).max(0.0);
            }
            for l in 0..n_lines {
                b[4 * n + l] = flow_limits[l] - ext_flow[l];
                b[4 * n + n_lines + l] = flow_limits[l] + ext_flow[l];
            }
            let mut colptr = vec![0usize; n_vars + 1];
            let mut rowval: Vec<usize> = Vec::new();
            let mut nzval: Vec<f64> = Vec::new();
            for i in 0..n {
                let cs = rowval.len();
                rowval.push(i); nzval.push(1.0);
                rowval.push(2 * n + i); nzval.push(-1.0);
                for &(l, p) in &bat_ptdf[i] {
                    rowval.push(4 * n + l); nzval.push(p);
                    rowval.push(4 * n + n_lines + l); nzval.push(-p);
                }
                let ce = rowval.len();
                sort_col_entries(&mut rowval[cs..ce], &mut nzval[cs..ce]);
                colptr[i + 1] = ce;
            }
            for i in 0..n {
                let cs = rowval.len();
                rowval.push(n + i); nzval.push(1.0);
                rowval.push(3 * n + i); nzval.push(-1.0);
                for &(l, p) in &bat_ptdf[i] {
                    rowval.push(4 * n + l); nzval.push(-p);
                    rowval.push(4 * n + n_lines + l); nzval.push(p);
                }
                let ce = rowval.len();
                sort_col_entries(&mut rowval[cs..ce], &mut nzval[cs..ce]);
                colptr[n + i + 1] = ce;
            }
            let a_csc = CscMatrix::new(n_con, n_vars, colptr, rowval, nzval);
            let cones = vec![SupportedConeT::NonnegativeConeT(n_con)];
            let settings = DefaultSettingsBuilder::<f64>::default()
                .max_iter(params.max_iter)
                .tol_gap_abs(params.tol_gap_abs)
                .tol_gap_rel(params.tol_gap_rel)
                .tol_feas(params.tol_feas)
                .time_limit(params.time_limit_s)
                .verbose(false)
                .build()
                .unwrap_or_else(|_| { let mut s = DefaultSettings::default(); s.verbose = false; s });
            let mut solver = DefaultSolver::new(&p_csc, &q, &a_csc, &b, &cones, settings);
            solver.solve();
            let status = solver.solution.status;
            let obj_val = solver.solution.obj_val;
            let u = match status {
                SolverStatus::Solved | SolverStatus::AlmostSolved => {
                    let x = &solver.solution.x;
                    (0..n).map(|i| {
                        let d = x.get(i).copied().unwrap_or(0.0).max(0.0);
                        let c = x.get(n + i).copied().unwrap_or(0.0).max(0.0);
                        (d - c).clamp(bounds[i].0, bounds[i].1)
                    }).collect()
                }
                _ => vec![0.0f64; n],
            };
            ClarabelStepResult { u, status, obj_val }
        }

        pub fn solve_via_clarabel(
            challenge: &Challenge,
            vtable: &NormalizedValueTable,
            da_ref: &[f64],
            params: &ClarabelParams,
        ) -> (Vec<Vec<f64>>, f64) {
            let t0 = Instant::now();
            let n = challenge.num_batteries;
            let t_max = challenge.num_steps;
            let bats = &challenge.batteries;
            let ptdf = &challenge.network.ptdf;
            let flow_limits = &challenge.network.flow_limits;
            let da = &challenge.market.day_ahead_prices;
            let bat_ptdf = precompute_bat_ptdf(bats, ptdf);
            let mut socs: Vec<f64> = bats.iter().map(|b| b.soc_initial_mwh).collect();
            let mut schedule = Vec::with_capacity(t_max);
            for t in 0..t_max {
                let t_next = (t + 1).min(vtable.n_steps);
                let bounds: Vec<(f64, f64)> = (0..n).map(|i| action_bounds_from_soc(&bats[i], socs[i])).collect();
                let da_next = if t + 1 < t_max { &da[t + 1] } else { &da[t_max - 1] };
                let da_ref_next = da_ref.get(t_next.min(t_max.saturating_sub(1))).copied().unwrap_or(50.0).max(1e-6);
                let val_scales: Vec<f64> = (0..n).map(|i| {
                    let cap = bats[i].capacity_mwh;
                    let da_node = da_next.get(bats[i].node).copied().unwrap_or(da_ref_next);
                    (cap / NOM_CAP) * (da_node / da_ref_next)
                }).collect();
                let ext_flow: Vec<f64> = ptdf.iter().map(|row| {
                    row.iter().zip(challenge.exogenous_injections[t].iter()).map(|(&p, &x)| p * x).sum::<f64>()
                }).collect();
                let result = solve_step_clarabel(
                    &socs, &da[t], &bounds, &ext_flow, bats, &bat_ptdf,
                    flow_limits, vtable, t_next, &val_scales, params,
                );
                for i in 0..n { socs[i] = apply_action_soc(result.u[i], socs[i], &bats[i]); }
                schedule.push(result.u);
            }
            (schedule, t0.elapsed().as_secs_f64() * 1000.0)
        }
    }

    // ── mpc_v6 module body (from mod.rs) ─────────────────────────────────────

    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let hp = hyperparams::Mpc6Hyperparams::from_hyperparameter_map(hyperparameters);

        if let Some(ref branch) = hp.force_branch {
            return match branch.as_str() {
                "mpc_v5" => super::mpc_v5_inlined::solve_challenge(challenge, save_solution, hyperparameters),
                _ => mpc_v6_admm_solve(challenge, save_solution, &hp),
            };
        }

        match (challenge.num_batteries, challenge.num_steps) {
            (40, _) | (60, _) => {
                #[cfg(feature = "clarabel_solver")]
                if hp.use_qp {
                    return mpc_v6_clarabel_solve(challenge, save_solution, &hp);
                }
                mpc_v6_admm_solve(challenge, save_solution, &hp)
            }
            _ => super::mpc_v5_inlined::solve_challenge(challenge, save_solution, hyperparameters),
        }
    }

    fn mpc_v6_admm_solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hp: &hyperparams::Mpc6Hyperparams,
    ) -> Result<()> {
        let v5_schedule = simulate_mpc_v5_da_proxy(challenge);
        save_solution(&Solution { schedule: v5_schedule.clone() })?;

        let da_ref = build_da_ref(challenge);
        let (vtable, cong_prob) = build_sdp_vtable(challenge, hp, &da_ref);
        let _ = cong_prob;

        let admm_params = hp.to_admm_params();
        let schedule = run_multistart_admm(challenge, &vtable, &da_ref, &admm_params, &v5_schedule, hp);
        replay_schedule(challenge, save_solution, schedule)
    }

    fn run_admm_single_variant(
        challenge: &Challenge,
        vtable: &sdp_value::NormalizedValueTable,
        da_ref: &[f64],
        admm_params: &path_admm::ADMMParams,
        v5_schedule: &[Vec<f64>],
        variant: &hyperparams::MultistartVariant,
    ) -> Vec<Vec<f64>> {
        use hyperparams::MultistartVariant;
        match variant {
            MultistartVariant::DaProxy =>
                path_admm::solve_via_admm_warmed(challenge, vtable, da_ref, admm_params, v5_schedule).0,
            MultistartVariant::ZeroInit =>
                path_admm::solve_via_admm(challenge, vtable, da_ref, admm_params).0,
            MultistartVariant::SmoothInit =>
                path_admm::solve_via_admm_smooth_init(challenge, vtable, da_ref, admm_params).0,
        }
    }

    fn schedule_proxy_profit(challenge: &Challenge, schedule: &[Vec<f64>], da_ref: &[f64]) -> f64 {
        let dt = constants::DELTA_T;
        let k_tx = constants::KAPPA_TX;
        let bats = &challenge.batteries;
        let da = &challenge.market.day_ahead_prices;
        schedule.iter().enumerate().map(|(t, actions)| {
            let da_t = &da[t];
            let da_fallback = da_ref.get(t).copied().unwrap_or(50.0);
            actions.iter().enumerate().map(|(i, &u)| {
                let price = da_t.get(bats.get(i).map(|b| b.node).unwrap_or(0)).copied().unwrap_or(da_fallback);
                u * price * dt - k_tx * u.abs() * dt
            }).sum::<f64>()
        }).sum()
    }

    fn run_multistart_admm(
        challenge: &Challenge,
        vtable: &sdp_value::NormalizedValueTable,
        da_ref: &[f64],
        admm_params: &path_admm::ADMMParams,
        v5_schedule: &[Vec<f64>],
        hp: &hyperparams::Mpc6Hyperparams,
    ) -> Vec<Vec<f64>> {
        use hyperparams::{MultistartVariant, SelectStrategy};
        let count = hp.multistart_count.clamp(1, 3);
        if count == 1 || hp.multistart_select == SelectStrategy::Default {
            return run_admm_single_variant(challenge, vtable, da_ref, admm_params, v5_schedule, &hp.multistart_variant);
        }
        let all_variants = [MultistartVariant::DaProxy, MultistartVariant::ZeroInit, MultistartVariant::SmoothInit];
        match hp.multistart_select {
            SelectStrategy::Best => {
                let mut best_sched = run_admm_single_variant(challenge, vtable, da_ref, admm_params, v5_schedule, &all_variants[0]);
                let mut best_profit = schedule_proxy_profit(challenge, &best_sched, da_ref);
                for variant in all_variants[1..].iter().take(count.saturating_sub(1)) {
                    let sched = run_admm_single_variant(challenge, vtable, da_ref, admm_params, v5_schedule, variant);
                    let profit = schedule_proxy_profit(challenge, &sched, da_ref);
                    if profit > best_profit { best_profit = profit; best_sched = sched; }
                }
                best_sched
            }
            SelectStrategy::First => {
                for variant in all_variants.iter().take(count) {
                    let sched = run_admm_single_variant(challenge, vtable, da_ref, admm_params, v5_schedule, variant);
                    if schedule_proxy_profit(challenge, &sched, da_ref) > 0.0 { return sched; }
                }
                run_admm_single_variant(challenge, vtable, da_ref, admm_params, v5_schedule, &MultistartVariant::DaProxy)
            }
            SelectStrategy::Default => run_admm_single_variant(
                challenge, vtable, da_ref, admm_params, v5_schedule, &hp.multistart_variant,
            ),
        }
    }

    #[cfg(feature = "clarabel_solver")]
    fn mpc_v6_clarabel_solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hp: &hyperparams::Mpc6Hyperparams,
    ) -> Result<()> {
        let v5_schedule = simulate_mpc_v5_da_proxy(challenge);
        save_solution(&Solution { schedule: v5_schedule })?;
        let da_ref = build_da_ref(challenge);
        let (vtable, _cong_prob) = build_sdp_vtable(challenge, hp, &da_ref);
        let clarabel_params = hp.to_clarabel_params();
        let (schedule, _elapsed) = path_clarabel::solve_via_clarabel(challenge, &vtable, &da_ref, &clarabel_params);
        replay_schedule(challenge, save_solution, schedule)
    }

    pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
        super::mpc_v5_inlined::policy(challenge, state)
    }

    fn build_da_ref(challenge: &Challenge) -> Vec<f64> {
        let da = &challenge.market.day_ahead_prices;
        (0..challenge.num_steps).map(|t| {
            let row = &da[t];
            if row.is_empty() { 50.0 } else { row.iter().sum::<f64>() / row.len() as f64 }
        }).collect()
    }

    fn build_sdp_vtable(
        challenge: &Challenge,
        hp: &hyperparams::Mpc6Hyperparams,
        da_ref: &[f64],
    ) -> (sdp_value::NormalizedValueTable, Vec<Vec<f64>>) {
        let tp = config::infer_track_params(challenge.num_batteries, challenge.num_steps);
        let e_grid = hp.build_soc_grid(constants::E_MIN_FRAC, constants::E_MAX_FRAC);
        let sdp_params = sdp_backward::NormalizedSDPParams::with_grid(
            e_grid, challenge.num_steps, tp.sigma, tp.rho_jump, tp.alpha,
        );
        let cong_prob = sdp_backward::precompute_congestion(
            &challenge.exogenous_injections,
            &challenge.network.ptdf,
            &challenge.network.flow_limits,
            constants::TAU_CONG,
        );
        let vtable = sdp_backward::backward_induction_normalized(&sdp_params, da_ref, &cong_prob, 0);
        (vtable, cong_prob)
    }

    fn replay_schedule(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        schedule: Vec<Vec<f64>>,
    ) -> Result<()> {
        let schedule: Vec<Vec<f64>> = schedule.into_iter()
            .map(|row| row.into_iter().map(|a| if a.is_finite() { a } else { 0.0 }).collect())
            .collect();
        let solution = challenge.grid_optimize(&|c, state| {
            let raw = &schedule[state.time_step];
            let clamped: Vec<f64> = raw.iter().enumerate()
                .map(|(i, &a)| a.clamp(state.action_bounds[i].0, state.action_bounds[i].1))
                .collect();
            Ok(apply_flow_scale(c, state, clamped))
        })?;
        save_solution(&solution)?;
        Ok(())
    }

    fn simulate_mpc_v5_da_proxy(challenge: &Challenge) -> Vec<Vec<f64>> {
        let n = challenge.num_batteries;
        let t_max = challenge.num_steps;
        let bats = &challenge.batteries;
        let dt = constants::DELTA_T;
        let mut socs: Vec<f64> = bats.iter().map(|b| b.soc_initial_mwh).collect();
        let mut schedule = Vec::with_capacity(t_max);
        for t in 0..t_max {
            let action_bounds: Vec<(f64, f64)> = bats.iter().enumerate().map(|(i, b)| {
                let soc = socs[i];
                let headroom = (b.soc_max_mwh - soc).max(0.0);
                let available = (soc - b.soc_min_mwh).max(0.0);
                let max_charge = if b.efficiency_charge > 0.0 {
                    (headroom / (b.efficiency_charge * dt)).min(b.power_charge_mw).max(0.0)
                } else { 0.0 };
                let max_discharge = if b.efficiency_discharge > 0.0 {
                    (available * b.efficiency_discharge / dt).min(b.power_discharge_mw).max(0.0)
                } else { 0.0 };
                (-max_charge, max_discharge)
            }).collect();
            let state = State {
                time_step: t,
                socs: socs.clone(),
                rt_prices: challenge.market.day_ahead_prices[t].clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds,
                total_profit: 0.0,
            };
            let actions = super::mpc_v5_inlined::policy(challenge, &state).unwrap_or_else(|_| vec![0.0; n]);
            for i in 0..n { socs[i] = bats[i].apply_action_to_soc(actions[i], socs[i]); }
            schedule.push(actions);
        }
        schedule
    }

    pub(super) fn apply_flow_scale(challenge: &Challenge, state: &State, mut chosen: Vec<f64>) -> Vec<f64> {
        let inj = challenge.compute_total_injections(state, &chosen);
        let flows = challenge.network.compute_flows(&inj);
        if challenge.network.verify_flows(&flows).is_ok() { return chosen; }
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        for _ in 0..32 {
            let mid = (lo + hi) * 0.5;
            let scaled: Vec<f64> = chosen.iter().map(|a| a * mid).collect();
            let inj2 = challenge.compute_total_injections(state, &scaled);
            let fl2 = challenge.network.compute_flows(&inj2);
            if challenge.network.verify_flows(&fl2).is_ok() { lo = mid; } else { hi = mid; }
        }
        for a in chosen.iter_mut() { *a *= lo; }
        chosen
    }
}

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!(
        r#"mpc_v7 — Spy-correction for BASELINE + mpc_v6 for all other tracks.
  BASELINE  (10×96, size ≤ 1000) → spy_correction (RT-aware greedy)
  MULTIDAY / DENSE               → mpc_v6 ADMM
  CONGESTED / CAPSTONE           → mpc_v5 branches (via mpc_v6)
Hyperparameters:
  deviation_threshold  float  spread $/MWh vs future DA avg (default 8.0)
  rt_blend             float  0..1 weight on RT price (default 1.0 = pure RT)
  force_baseline       bool   apply spy_correction to any track size (testing)"#
    );
}

const BASELINE_THRESHOLD: usize = 1000;

fn parse_spy_params(
    hyperparameters: &Option<Map<String, Value>>,
) -> spy_correction::SpyCorrectionParams {
    let default = spy_correction::SpyCorrectionParams::default();
    let h = match hyperparameters.as_ref() {
        Some(h) => h,
        None => return default,
    };
    let deviation_threshold = h
        .get("deviation_threshold")
        .and_then(|v| v.as_f64())
        .filter(|&v| v > 0.0)
        .unwrap_or(default.deviation_threshold);
    let rt_blend = h
        .get("rt_blend")
        .and_then(|v| v.as_f64())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(default.rt_blend);
    spy_correction::SpyCorrectionParams {
        deviation_threshold,
        rt_blend,
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let size = challenge.num_batteries * challenge.num_steps;

    let force_baseline = hyperparameters
        .as_ref()
        .and_then(|h| h.get("force_baseline"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if size <= BASELINE_THRESHOLD || force_baseline {
        let params = parse_spy_params(hyperparameters);
        let solution = challenge.grid_optimize(&|c, s| spy_correction::policy(c, s, &params))?;
        return save_solution(&solution);
    }

    mpc_v6_inlined::solve_challenge(challenge, save_solution, hyperparameters)
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let size = challenge.num_batteries * challenge.num_steps;
    if size <= BASELINE_THRESHOLD {
        spy_correction::policy(
            challenge,
            state,
            &spy_correction::SpyCorrectionParams::default(),
        )
    } else {
        mpc_v6_inlined::policy(challenge, state)
    }
}
