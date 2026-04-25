use anyhow::{anyhow, Result};
use std::cell::RefCell;
use tig_challenges::energy_arbitrage::{Challenge, State};

const MAX_FLOW_ADJUST_ITERS: usize = 128;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 48;
const EPS: f64 = 1e-12;

thread_local! {
    static CACHE: RefCell<Option<EpisodeCache>> = RefCell::new(None);
}

struct EpisodeCache {
    seed: [u8; 32],
    rt_efficiency: Vec<f64>,
    suffix_min: Vec<Vec<f64>>,
    suffix_median: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug)]
struct F64Ord(f64);

impl PartialEq for F64Ord {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for F64Ord {}

impl PartialOrd for F64Ord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for F64Ord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn get_or_init_cache(challenge: &Challenge) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {    
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if cache.as_ref().map_or(true, |e| e.seed != challenge.seed) {
            let num_steps = challenge.num_steps;
            let num_nodes = challenge.network.num_nodes;
            let da_prices = &challenge.market.day_ahead_prices;

            let rt_efficiency: Vec<f64> = challenge
                .batteries
                .iter()
                .map(|b| b.efficiency_charge * b.efficiency_discharge)
                .collect();

            let mut suffix_min = vec![vec![f64::INFINITY; num_steps + 1]; num_nodes];
            for node in 0..num_nodes {
                for t in (0..num_steps).rev() {
                    suffix_min[node][t] = da_prices[t][node].min(suffix_min[node][t + 1]);
                }
            }
            
            let mut suffix_median = vec![vec![0.0f64; num_steps + 1]; num_nodes];
            for node in 0..num_nodes {
                let mut lower: std::collections::BinaryHeap<F64Ord> = std::collections::BinaryHeap::new();
                let mut upper: std::collections::BinaryHeap<std::cmp::Reverse<F64Ord>> =
                    std::collections::BinaryHeap::new();

                for t in (0..num_steps).rev() {
                    let x = da_prices[t][node];
                    if lower
                        .peek()
                        .map_or(true, |m| x.total_cmp(&m.0) != std::cmp::Ordering::Greater)
                    {
                        lower.push(F64Ord(x));
                    } else {
                        upper.push(std::cmp::Reverse(F64Ord(x)));
                    }

                    if lower.len() > upper.len() + 1 {
                        if let Some(v) = lower.pop() {
                            upper.push(std::cmp::Reverse(v));
                        }
                    } else if upper.len() > lower.len() {
                        if let Some(std::cmp::Reverse(v)) = upper.pop() {
                            lower.push(v);
                        }
                    }

                    suffix_median[node][t] = lower.peek().map(|v| v.0).unwrap_or(0.0);
                }
            }

            *cache = Some(EpisodeCache {
                seed: challenge.seed,
                rt_efficiency: rt_efficiency.clone(),
                suffix_min: suffix_min.clone(),
                suffix_median: suffix_median.clone(),
            });
            (rt_efficiency, suffix_min, suffix_median)
        } else {
            let e = cache.as_ref().unwrap();
            (
                e.rt_efficiency.clone(),
                e.suffix_min.clone(),
                e.suffix_median.clone(),
            )
        }
    })
}

struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    (0..challenge.network.num_lines)
        .map(|l| {
            (0..challenge.network.num_nodes)
                .map(|k| challenge.network.ptdf[l][k] * injections[k])
                .sum::<f64>()
        })
        .collect()
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation > tig_challenges::energy_arbitrage::constants::EPS_FLOW * limit {
            let candidate = Violation { line: l, flow, amount: violation };
            match best {
                Some(ref current) if candidate.amount <= current.amount => {}
                _ => best = Some(candidate),
            }
        }
    }
    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, &flows).is_none()
}

fn soften_most_violated_line(
    challenge: &Challenge,
    violation: &Violation,
    action: &mut [f64],
    flows: &mut [f64],
) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    let ptdf = &challenge.network.ptdf;

    let mut worsening_strength = 0.0;
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let a = action[i];
        let signed_contribution = signed_direction * ptdf[line][battery.node] * a;
        if signed_contribution > EPS {
            worsening_strength += signed_contribution;
        }
    }

    if worsening_strength <= EPS {
        return false;
    }

    let relief_needed = violation.amount;
    let scale = if relief_needed >= worsening_strength {
        0.0
    } else {
        (1.0 - relief_needed / worsening_strength).clamp(0.0, 1.0)
    };

    let num_lines = challenge.network.num_lines;

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let old_a = action[i];
        let signed_contribution = signed_direction * ptdf[line][node] * old_a;
        if signed_contribution > EPS {
            let new_a = old_a * scale;
            let delta = new_a - old_a;
            action[i] = new_a;

            if delta.abs() > EPS {
                for l in 0..num_lines {
                    flows[l] += ptdf[l][node] * delta;
                }
            }
        }
    }

    true
}

fn enforce_flow_feasibility(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Result<Vec<f64>> {
    let mut flows = compute_flows(challenge, state, &action);

    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let Some(violation) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !soften_most_violated_line(challenge, &violation, &mut action, &mut flows) {
            break;
        }
    }
    if most_violated_line(challenge, &flows).is_none() {
        return Ok(action);
    }

    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!("Grid infeasible even with zero battery actions"));
    }

    let base = action;
    let flows_base = flows;
    let violation = most_violated_line(challenge, &flows_base)
        .ok_or_else(|| anyhow!("Action infeasible but no violated line found"))?;

    let line = violation.line;
    let signed_direction = violation.flow.signum();

    let mut contribs: Vec<(usize, f64)> = Vec::new();
    if signed_direction.abs() > EPS {
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let signed_contribution = signed_direction * challenge.network.ptdf[line][battery.node] * base[i];
            if signed_contribution > EPS {
                contribs.push((i, signed_contribution));
            }
        }
    }
    contribs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut in_subset = vec![false; base.len()];
    let mut test = base.clone();
    let mut subset_found = false;

    for &(i, _) in &contribs {
        in_subset[i] = true;
        test[i] = 0.0;
        if is_flow_feasible(challenge, state, &test) {
            subset_found = true;
            break;
        }
    }

    if !subset_found {
        for f in &mut in_subset {
            *f = true;
        }
        test.fill(0.0);
    }

    if !is_flow_feasible(challenge, state, &test) {
        for f in &mut in_subset {
            *f = true;
        }
        test.fill(0.0);
    }

    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        let mut scaled = base.clone();
        for (i, flag) in in_subset.iter().enumerate() {
            if *flag {
                scaled[i] = mid * scaled[i];
            }
        }
        if is_flow_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }

    let mut out = base;
    for (i, flag) in in_subset.iter().enumerate() {
        if *flag {
            out[i] *= low;
        }
    }
    Ok(out)
}

fn compute_battery_schedule(
    challenge: &Challenge,
    battery_idx: usize,
    current_soc: f64,
    t: usize,
    rt_efficiency: f64,
    suffix_min: &[f64],
    suffix_median: &[f64],
) -> Vec<f64> {
    let battery = &challenge.batteries[battery_idx];
    let node = battery.node;
    let num_steps = challenge.num_steps;
    let da_prices = &challenge.market.day_ahead_prices;

    let delta_t = tig_challenges::energy_arbitrage::constants::DELTA_T;
    let kappa_deg = tig_challenges::energy_arbitrage::constants::KAPPA_DEG;
    let beta_deg = tig_challenges::energy_arbitrage::constants::BETA_DEG;

    let mut schedule = vec![0.0f64; num_steps];
    let mut soc = current_soc;

    for s in t..num_steps {
        let p_now = da_prices[s][node];

        let available = soc - battery.soc_min_mwh;
        let headroom = battery.soc_max_mwh - soc;

        let cand_discharge_mw = if available > EPS && battery.efficiency_discharge > EPS {
            (available * battery.efficiency_discharge / delta_t).min(battery.power_discharge_mw)
        } else {
            0.0
        };

        let cand_charge_mw = if headroom > EPS && battery.efficiency_charge > EPS {
            (headroom / (delta_t * battery.efficiency_charge)).min(battery.power_charge_mw)
        } else {
            0.0
        };

        let deg_cost = |mw: f64| -> f64 {
            if mw.abs() <= EPS || battery.capacity_mwh <= EPS {
                0.0
            } else {
                kappa_deg * (mw.abs() * delta_t / battery.capacity_mwh).powf(beta_deg)
            }
        };

        if s + 1 >= num_steps {
            let discharge_value = if p_now > EPS && cand_discharge_mw > EPS {
                p_now * cand_discharge_mw * delta_t - deg_cost(cand_discharge_mw)
            } else {
                0.0
            };

            let charge_value = if p_now < -EPS && cand_charge_mw > EPS {
                (-p_now) * cand_charge_mw * delta_t - deg_cost(cand_charge_mw)
            } else {
                0.0
            };

            if discharge_value > charge_value && discharge_value > 0.0 {
                schedule[s] = cand_discharge_mw;
                soc -= cand_discharge_mw * delta_t / battery.efficiency_discharge;
            } else if charge_value > 0.0 {
                schedule[s] = -cand_charge_mw;
                soc += cand_charge_mw * delta_t * battery.efficiency_charge;
            }
            continue;
        }

        let p_future_min = suffix_min[s + 1];
        let p_future_sell = suffix_median[s + 1];

        let charge_value = if cand_charge_mw > EPS {
            let margin = p_future_sell * rt_efficiency - p_now;
            margin * cand_charge_mw * delta_t - deg_cost(cand_charge_mw)
        } else {
            0.0
        };

        let discharge_value = if cand_discharge_mw > EPS && rt_efficiency > EPS {
            let margin = p_now - p_future_min / rt_efficiency;
            margin * cand_discharge_mw * delta_t - deg_cost(cand_discharge_mw)
        } else {
            0.0
        };

        if charge_value > 0.0 || discharge_value > 0.0 {
            let choose_discharge = discharge_value > charge_value;

            if choose_discharge && discharge_value > 0.0 {
                schedule[s] = cand_discharge_mw;
                soc -= cand_discharge_mw * delta_t / battery.efficiency_discharge;
            } else if charge_value > 0.0 {
                schedule[s] = -cand_charge_mw;
                soc += cand_charge_mw * delta_t * battery.efficiency_charge;
            }
        }
    }

    schedule
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let (rt_efficiency, suffix_min, suffix_median) = get_or_init_cache(challenge);

    let delta_t = tig_challenges::energy_arbitrage::constants::DELTA_T;
    let kappa_deg = tig_challenges::energy_arbitrage::constants::KAPPA_DEG;
    let beta_deg = tig_challenges::energy_arbitrage::constants::BETA_DEG;

    let zero_action = vec![0.0f64; challenge.num_batteries];
    let base_flows = compute_flows(challenge, state, &zero_action);

    let mut node_shadow = vec![0.0f64; challenge.network.num_nodes];
    for (l, &flow) in base_flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        if limit <= EPS {
            continue;
        }
        let stress = flow.abs() / limit;
        let mag = stress / ((1.0 - stress).abs() + EPS);
        let signed_shadow = flow.signum() * mag;
        if signed_shadow.abs() <= EPS {
            continue;
        }
        for k in 0..challenge.network.num_nodes {
            node_shadow[k] += challenge.network.ptdf[l][k] * signed_shadow;
        }
    }

    let mut action = vec![0.0f64; challenge.num_batteries];

    for i in 0..challenge.num_batteries {
        let (min_bound, max_bound) = state.action_bounds[i];
        let battery = &challenge.batteries[i];
        let node = battery.node;

        let schedule = compute_battery_schedule(
            challenge,
            i,
            state.socs[i],
            t,
            rt_efficiency[i],
            &suffix_min[node],
            &suffix_median[node],
        );
        let da_action = schedule[t].clamp(min_bound, max_bound);

        let da_price = challenge.market.day_ahead_prices[t][node];
        let rt_price = state.rt_prices[node];

        let net_profit = if da_action > EPS {
            let deg_cost = kappa_deg * (da_action.abs() * delta_t / battery.capacity_mwh).powf(beta_deg);
            rt_price * da_action * battery.efficiency_discharge * delta_t - deg_cost
        } else if da_action < -EPS {
            if da_price > EPS && rt_price > da_price * 2.0 {
                f64::NEG_INFINITY
            } else {
                0.0
            }
        } else {
            0.0
        };

        let final_action = if da_action > EPS && net_profit < 0.0 {
            0.0f64.clamp(min_bound, max_bound)
        } else if da_action < -EPS && net_profit == f64::NEG_INFINITY {
            0.0f64.clamp(min_bound, max_bound)
        } else {
            da_action
        };

        let corrected_action = if da_price > EPS {
            let price_ratio = rt_price / da_price;
            if final_action > EPS && price_ratio > 1.0 {
                (final_action * price_ratio.sqrt()).clamp(min_bound, max_bound)
            } else if final_action < -EPS && price_ratio < 1.0 {
                (final_action * (1.0 / price_ratio).sqrt()).clamp(min_bound, max_bound)
            } else {
                final_action
            }
        } else {
            final_action
        };

        let shadow = node_shadow[node];
        let congestion_aware_action = if shadow * corrected_action > EPS {
            let scale = 1.0 / (1.0 + shadow.abs());
            (corrected_action * scale).clamp(min_bound, max_bound)
        } else {
            corrected_action
        };

        action[i] = congestion_aware_action;
    }

    enforce_flow_feasibility(challenge, state, action)
}