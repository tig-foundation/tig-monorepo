// linepulse: energy_arbitrage solver, congestion-priced stochastic DP with exact step LPs.
//
// 1. Per-battery value tables: backward Bellman recursion over a state-of-charge grid,
//    taking the expectation over the public real-time price model (Gauss-Hermite nodes
//    for the Gaussian noise, conditional-mean bins for the Pareto jumps).
// 2. Congestion pricing: roll the fleet forward on day-ahead prices, solving the step LP
//    at every step; line dual prices x PTDF give a per-battery price shift; rebuild the
//    tables on shifted prices; repeat as a damped fixed point.
// 3. Fleet-aggregate correction: a DP of the whole fleet as one battery limited by the
//    network's deliverable move corrects each battery's marginal value of energy.
// 4. Decisions: an exact LP per step (piecewise-linear table values, every line limit,
//    constraint generation); over the last few steps, one LP plans all remaining steps.
// Only causal information is used (day-ahead prices, the public RT model parameters and
// the realised current state).
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

const S: usize = 25;              // Forecast SOC grid
const K: usize = 13;             // Candidate power grid
const K_FULL_MAX_M: usize = 40;
const K_LARGE: usize = 9;
const DT: f64 = 0.25;             // hours
const TX: f64 = 0.25;             // $/MWh
const EPS: f64 = 1e-12;
// P2: value tables are the expectation over the public RT model
// RT = DA*(1 + sigma*xi) + jump, jump = DA*X w.p. rho, X ~ Pareto(alpha) on [1, inf).
const SDP: bool = true;
// P1: future power in the tables is scaled by deliverability theta in [0,1]
// (share of a simultaneous fleet move the battery's lines can carry);
// effective power = P * (DELIVER_FLOOR + (1 - DELIVER_FLOOR) * theta). 1.0 = off.
const DELIVER_FLOOR: f64 = 1.0;
// Capstone-size fleets (m >= END_DRAIN_MIN_FLEET) before the end-drain window.
const DELIVER_FLOOR_CAP: f64 = 0.65;
// Multiday-size fleets (21..=40).
const DELIVER_FLOOR_MID: f64 = 0.8;
// Deliverability floor used in the last END_DRAIN_STEPS steps of the tables.
const END_DRAIN_FLOOR: f64 = 0.6;
// G2: terminal salvage price on stored energy, as a fraction of the last-24-step mean DA.
const SALVAGE: f64 = 0.2;
const SALVAGE_STEPS: usize = 24;
const SALVAGE_MIN_FLEET: usize = 60;
// G7: exact multi-step LP over the last ENDGAME_STEPS steps (fleets up to ENDGAME_MAX_M).
const ENDGAME_STEPS: usize = 6;
const ENDGAME_MAX_M: usize = 40;
const ENDGAME_HINTS: bool = true;
// Long-step (bound-flipping) dual ratio test in both simplex codes.
const EG_BFRT: bool = true;
const ENDGAME_STEPS_LARGE: usize = 4;
const END_DRAIN_STEPS: usize = 48;
// ...on fleets of at least this many batteries (capstone strands energy; dense does not).
const END_DRAIN_MIN_FLEET: usize = 100;
// Step LP constraint generation: line sides added per round (most overloaded first).
const LP_ROWS_PER_ROUND: usize = 8;
// E4: rebuild the tables once on DA prices net of per-battery congestion prices
// (step-LP line duals x PTDF along a DA-price rollout).
const CONGESTION_TABLES: bool = true;
// Half-width (steps) of the moving average applied to the congestion prices.
const CONGESTION_SMOOTH: usize = 4;
// Rollouts in the damped fixed-point iteration for the congestion prices.
const CONGESTION_ROUNDS: usize = 4;
// Rollout step LPs use every ROLLOUT_GRID_STRIDE-th SOC breakpoint of V (coarser, cheaper).
const ROLLOUT_GRID_STRIDE: usize = 8;
// The decision step LP uses every MAIN_GRID_STRIDE-th SOC breakpoint of V.
const MAIN_GRID_STRIDE: usize = 4;
const CAP_MAIN_STRIDE: usize = 8;
// Seed each rollout step LP with the previous round's positive-dual rows at that step.
const ROLLOUT_HINTS: bool = true;
// ...and the decision LP with the last rollout's rows at that step.
const DECISION_HINTS_ON: bool = true;
// F1: weight of the fleet-aggregate marginal-value correction in the step LP (0 = off).
const FLEET_CORRECTION: f64 = 0.05;
const MOVE_EVERY: usize = 8;
const DENSE_ROUNDS: usize = 3;
const CAP_ROUNDS: usize = 3;
// Rollout SOC-breakpoint stride on dense-size fleets (cheaper rollouts there).
const DENSE_ROLLOUT_STRIDE: usize = 16;
// Per-node table reuse across batteries with identical normalised parameters.
const TABLE_REUSE: bool = true;
// Candidate power levels in the intermediate fixed-point tables (rollouts only); the
// decision tables keep k_eff(m).
const INTERMEDIATE_K: usize = 5;
thread_local! {
    static K_COARSE: std::cell::Cell<Option<usize>> = std::cell::Cell::new(None);
}
// 5-point Gauss-Hermite rule for N(0,1): nodes 0, +-sqrt(5 -+ sqrt(10)).
const GH_Z: [f64; 5] = [0.0, 1.355626, -1.355626, 2.856970, -2.856970];
const GH_W: [f64; 5] = [0.533333, 0.222076, 0.222076, 0.011257, 0.011257];
// Pareto jump discretised by conditional means over these U-quantile bins.
const JUMP_BINS: [f64; 4] = [0.0, 0.9, 0.99, 1.0];

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // No hyperparameters: the operating point is fixed by the constants above.
}

pub fn help() {
    println!("linepulse: per-battery stochastic DP value tables over the public RT price model,");
    println!("re-priced for network congestion by a damped fixed point of step-LP rollouts,");
    println!("with a fleet-aggregate value correction. Each step solves an exact LP over all");
    println!("line limits (a multi-step LP over the last steps). No hyperparameters.");
    println!("Fuel: at most 16.8 Gfuel per instance in testing (budget 500 Gfuel); no tuning is needed.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let _ = hyperparameters;
    // Per-instance caches start empty, so results do not depend on run order.
    reset_caches();
    let planner = Planner::new(challenge);
    let solution = challenge.grid_optimize(&|ch, state| planner.act(ch, state))?;
    save_solution(&solution)?;
    Ok(())
}

/// The full policy: forecast tables, then (at the first decision, when the initial
/// state is first seen) congestion-priced tables and the fleet correction.
/// No realised future real-time price is accessed.
pub struct Planner {
    values: Vec<f64>,
    net: Net,
    refined: std::cell::RefCell<Option<Vec<f64>>>,
}

impl Planner {
    pub fn new(ch: &Challenge) -> Self {
        K_COARSE.with(|c| c.set(Some(INTERMEDIATE_K)));
        let values = compile_tables(ch);
        K_COARSE.with(|c| c.set(None));
        Planner { values, net: Net::new(ch), refined: std::cell::RefCell::new(None) }
    }

    /// E4: roll the fleet forward on DA prices with the step LP from `state`, then
    /// rebuild the tables on congestion-adjusted prices (and the F1 aggregate).
    pub fn prepare(&self, ch: &Challenge, state: &State) {
        if !CONGESTION_TABLES || self.refined.borrow().is_some() { return; }
        // Damped fixed point: pi <- (pi + rollout prices on tables(pi)) / 2.
        let mut hints = vec![Vec::new(); ch.num_steps];
        let mut pi = congestion_prices(ch, state, &self.values, &self.net, &mut hints);
        // Dense-size fleets (60 <= m < 100) dominate seconds: one round fewer there.
        let m = ch.batteries.len();
        let rounds = if (60..END_DRAIN_MIN_FLEET).contains(&m) { DENSE_ROUNDS } else if m >= END_DRAIN_MIN_FLEET { CAP_ROUNDS } else { CONGESTION_ROUNDS };
        for r in 1..rounds {
            K_COARSE.with(|c| c.set(Some(INTERMEDIATE_K)));
            let tk = compile_tables_shift(ch, &pi);
            K_COARSE.with(|c| c.set(None));
            let pk = if r + 1 == rounds && m <= 40 {
                // G6: two sigma-perturbed full-horizon rollouts (small fleets).
                let mut acc = vec![0.0; pi.len()];
                for sd in 1..=2u64 {
                    let q = congestion_prices_seeded(ch, state, &tk, &self.net, &mut hints, sd, 0);
                    for (a, b) in acc.iter_mut().zip(q.iter()) { *a += 0.5 * b; }
                }
                acc
            } else if r + 1 == rounds && m >= END_DRAIN_MIN_FLEET {
                // G19 run 8: RT sampled over the last 48 steps (multiday/capstone).
                congestion_prices_seeded(ch, state, &tk, &self.net, &mut hints, 1, 48)
            } else { congestion_prices(ch, state, &tk, &self.net, &mut hints) };
            for (a, b) in pi.iter_mut().zip(pk.iter()) { let d = if (60..END_DRAIN_MIN_FLEET).contains(&m) { 0.8 } else { 0.5 }; *a = (1.0 - d) * *a + d * b; }
        }
        if DECISION_HINTS_ON { DECISION_HINTS.with(|d| *d.borrow_mut() = hints.clone()); }
        if FLEET_CORRECTION > 0.0 { AGG.with(|a| *a.borrow_mut() = Some(aggregate_table(ch, &pi))); }
        *self.refined.borrow_mut() = Some(compile_tables_shift(ch, &pi));
    }

    pub fn act(&self, ch: &Challenge, state: &State) -> Result<Vec<f64>> {
        self.prepare(ch, state);
        match self.refined.borrow().as_ref() {
            Some(v) => decide(ch, state, v, &self.net),
            None => decide(ch, state, &self.values, &self.net),
        }
    }

    /// The tables the decisions use (for diagnostics).
    pub fn tables(&self) -> Vec<f64> { self.refined.borrow().clone().unwrap_or_else(|| self.values.clone()) }
}

fn value_idx(bi: usize, t: usize, si: usize, horizon: usize) -> usize {
    (bi * (horizon + 1) + t) * S + si
}

fn interpolate(values: &[f64], bi: usize, t: usize, soc: f64, capacity: f64, horizon: usize) -> f64 {
    let low = capacity * 0.10;
    let high = capacity * 0.90;
    let w = ((soc.clamp(low, high) - low) / (high - low) * (S - 1) as f64)
        .clamp(0.0, (S - 1) as f64);
    let j = (w.floor() as usize).min(S - 2);
    let z = w - j as f64;
    let left = values[value_idx(bi, t, j, horizon)];
    let right = values[value_idx(bi, t, j + 1, horizon)];
    left * (1.0 - z) + right * z
}

fn next_soc(soc: f64, action: f64, eta_charge: f64, eta_discharge: f64) -> f64 {
    if action >= 0.0 { soc - DT * action / eta_discharge }
    else { soc - DT * action * eta_charge }
}

/// Candidate power grid size per fleet: small/mid fleets use the full K, large fleets 9.
fn k_eff(m: usize) -> usize { if m <= 20 { K } else if m <= K_FULL_MAX_M { 9 } else if m < END_DRAIN_MIN_FLEET { 7 } else { K_LARGE } }

fn action_at(k: usize, power_charge: f64, power_discharge: f64, kk: usize) -> f64 {
    -power_charge + k as f64 * (power_charge + power_discharge) / (kk - 1) as f64
}

fn pnl(action: f64, price: f64, cap: f64) -> f64 {
    let amp = action.abs();
    action * price * DT - TX * amp * DT - (amp * DT / cap).powi(2)
}

/// Price scenarios (multiplier of DA, weight) for the RT distribution.
fn price_scenarios(ch: &Challenge) -> Vec<(f64, f64)> {
    if !SDP { return vec![(1.0, 1.0)]; }
    let sigma = ch.market.params.volatility;
    let rho = ch.market.params.jump_probability;
    let alpha = ch.market.params.tail_index;
    let mut out: Vec<(f64, f64)> = GH_Z.iter().zip(GH_W.iter())
        .map(|(&z, &w)| (1.0 + sigma * z, (1.0 - rho) * w)).collect();
    // E[X | U in [a,b]] with X = (1-U)^(-1/alpha); preserves the jump mean for alpha > 1.
    let e = 1.0 - 1.0 / alpha;
    for win in JUMP_BINS.windows(2) {
        let (a, b) = (win[0], win[1]);
        let mean = ((1.0 - a).powf(e) - (1.0 - b).powf(e)) / (e * (b - a));
        out.push((1.0 + mean, rho * (b - a)));
    }
    out
}

/// Per battery and step: (charge, discharge) deliverability in [0,1].
/// F1: value table [t][S grid over lo..hi] of the fleet as one aggregate battery.
/// Capacity and power are summed; power is scaled by the fleet-wide corridor room
/// (the share of a full simultaneous fleet move the lines carry). Price is the
/// capacity-weighted DA minus congestion shift; the same RT scenarios as the tables.
/// Returns (table, lo, hi).
fn aggregate_table(ch: &Challenge, pi: &[f64]) -> (Vec<f64>, f64, f64) {
    let h = ch.num_steps;
    let m = ch.batteries.len();
    let bats = &ch.batteries;
    let cap: f64 = bats.iter().map(|b| b.capacity_mwh).sum();
    let lo: f64 = bats.iter().map(|b| b.soc_min_mwh).sum();
    let hi: f64 = bats.iter().map(|b| b.soc_max_mwh).sum();
    let ec = bats.iter().map(|b| b.efficiency_charge * b.capacity_mwh).sum::<f64>() / cap;
    let ed = bats.iter().map(|b| b.efficiency_discharge * b.capacity_mwh).sum::<f64>() / cap;
    let (pc0, pd0): (f64, f64) = (bats.iter().map(|b| b.power_charge_mw).sum(), bats.iter().map(|b| b.power_discharge_mw).sum());
    let ptdf = &ch.network.ptdf;
    let limits = &ch.network.flow_limits;
    let lcount = limits.len();
    let (mut d_dis, mut d_chg) = (vec![0.0; lcount], vec![0.0; lcount]);
    for l in 0..lcount {
        for b in bats { d_dis[l] += ptdf[l][b.node] * b.power_discharge_mw; d_chg[l] -= ptdf[l][b.node] * b.power_charge_mw; }
    }
    let room = |f: f64, d: f64, lim: f64| -> f64 {
        if d.abs() < 1e-9 { 1.0 } else if d > 0.0 { ((lim - f) / d).clamp(0.0, 1.0) } else { ((lim + f) / -d).clamp(0.0, 1.0) }
    };
    let scen = price_scenarios(ch);
    let mut v = vec![0.0; (h + 1) * S];
    let grid = |si: usize| lo + (hi - lo) * si as f64 / (S - 1) as f64;
    if SALVAGE > 0.0 && m >= SALVAGE_MIN_FLEET {
        let w0 = h.saturating_sub(SALVAGE_STEPS);
        let tail = (w0..h).map(|t| bats.iter().map(|b| ch.market.day_ahead_prices[t][b.node] * b.capacity_mwh).sum::<f64>() / cap).sum::<f64>() / (h - w0).max(1) as f64;
        for si in 0..S { v[h * S + si] = SALVAGE * tail * (grid(si) - lo); }
    }
    let interp = |v: &[f64], t: usize, x: f64| -> f64 {
        let w = ((x.clamp(lo, hi) - lo) / (hi - lo) * (S - 1) as f64).clamp(0.0, (S - 1) as f64);
        let j = (w.floor() as usize).min(S - 2);
        let z = w - j as f64;
        v[t * S + j] * (1.0 - z) + v[t * S + j + 1] * z
    };
    let _ = (&d_dis, &d_chg, room, pc0, pd0);
    // Largest fleet net discharge / charge the lines carry, every MOVE_EVERY steps
    // (exogenous flows change slowly), held in between.
    let mut moves = vec![(0.0, 0.0); h];
    for t in 0..h {
        moves[t] = if t % MOVE_EVERY == 0 {
            let base = ch.network.compute_flows(&ch.exogenous_injections[t]);
            (fleet_max_move(ch, &base, 1.0), fleet_max_move(ch, &base, -1.0))
        } else { moves[t - 1] };
    }
    for t in (0..h).rev() {
        let (pd, pc) = moves[t];
        let mut da = 0.0;
        for (i, b) in bats.iter().enumerate() {
            let shift = if pi.is_empty() { 0.0 } else { pi[i * h + t] / DT };
            da += (ch.market.day_ahead_prices[t][b.node] - shift) * b.capacity_mwh;
        }
        da /= cap;
        let _ = m;
        for si in 0..S {
            let soc = grid(si);
            let mut cand: Vec<(f64, f64)> = Vec::with_capacity(K);
            let kk = k_eff(m);
            for k in 0..kk {
                let a = action_at(k, pc, pd, kk);
                let s2 = next_soc(soc, a, ec, ed);
                if s2 < lo - 1e-9 || s2 > hi + 1e-9 { continue; }
                cand.push((a, pnl(a, 0.0, cap) + interp(&v, t + 1, s2)));
            }
            let mut e = 0.0;
            for &(mult, w) in &scen {
                let price = da * mult;
                let best = cand.iter().map(|&(a, c)| a * price * DT + c).fold(f64::NEG_INFINITY, f64::max);
                e += w * if best.is_finite() { best } else { 0.0 };
            }
            v[t * S + si] = e;
        }
    }
    (v, lo, hi)
}

/// Max of sum_i u_i (dir = +1, discharge) or of -sum_i u_i (dir = -1, charge) over
/// power bounds and line limits, by constraint generation on the bounded tableau.
fn fleet_max_move(ch: &Challenge, base: &[f64], dir: f64) -> f64 {
    let m = ch.batteries.len();
    let limits = &ch.network.flow_limits;
    if base.iter().zip(limits).any(|(f, l)| f.abs() > l - 1e-7) { return 0.0; }
    let ub: Vec<f64> = ch.batteries.iter().map(|b| if dir > 0.0 { b.power_discharge_mw } else { b.power_charge_mw }).collect();
    let mut lp = Tableau::new(&vec![1.0; m], &ub, 2 * limits.len());
    let coef = |l: usize, i: usize| dir * ch.network.ptdf[l][ch.batteries[i].node];
    let mut added = vec![[false; 2]; limits.len()];
    for _ in 0..200 {
        let x = lp.x();
        let mut grew = false;
        for l in 0..limits.len() {
            let f = base[l] + (0..m).map(|i| coef(l, i) * x[i]).sum::<f64>();
            for (side, sg) in [(0usize, 1.0_f64), (1, -1.0)] {
                if sg * f > limits[l] - 1e-7 + 1e-9 && !added[l][side] {
                    added[l][side] = true; grew = true;
                    let a: Vec<f64> = (0..m).map(|i| sg * coef(l, i)).collect();
                    lp.add_row(&a, limits[l] - 1e-7 - sg * base[l]);
                }
            }
        }
        if !grew { return x.iter().sum(); }
        if !lp.dual() || !lp.primal() { return 0.0; }
    }
    0.0
}

thread_local! {
    // F1: (aggregate table, lo, hi) once the congestion prices are final.
    static AGG: std::cell::RefCell<Option<(Vec<f64>, f64, f64)>> = std::cell::RefCell::new(None);
}

thread_local! {
    // (num_steps, fleet size, total capacity) -> deliverability; one instance per thread.
    static DELIVER_CACHE: std::cell::RefCell<Option<((usize, usize, u64), std::rc::Rc<Vec<(f64, f64)>>)>> = std::cell::RefCell::new(None);
}

fn deliverability(ch: &Challenge) -> std::rc::Rc<Vec<(f64, f64)>> {
    let (h, m) = (ch.num_steps, ch.batteries.len());
    // Only the whole-horizon floor or the capstone-size end drain read it.
    if DELIVER_FLOOR >= 1.0 && m > 40 && (END_DRAIN_FLOOR >= 1.0 || m < END_DRAIN_MIN_FLEET) { return std::rc::Rc::new(vec![(1.0, 1.0); m * h]); }
    let key = (h, m, ch.batteries.iter().map(|b| b.capacity_mwh).sum::<f64>().to_bits());
    if let Some(v) = DELIVER_CACHE.with(|c| c.borrow().as_ref().filter(|(k, _)| *k == key).map(|(_, v)| v.clone())) { return v; }
    let v = std::rc::Rc::new(deliverability_compute(ch));
    DELIVER_CACHE.with(|c| *c.borrow_mut() = Some((key, v.clone())));
    v
}

fn deliverability_compute(ch: &Challenge) -> Vec<(f64, f64)> {
    let h = ch.num_steps;
    let m = ch.batteries.len();
    if DELIVER_FLOOR >= 1.0 && END_DRAIN_FLOOR >= 1.0 { return vec![(1.0, 1.0); m * h]; }
    let ptdf = &ch.network.ptdf;
    let limits = &ch.network.flow_limits;
    let lcount = limits.len();
    // Flow change if the whole fleet discharges (charges) at full power.
    let mut d_dis = vec![0.0; lcount];
    let mut d_chg = vec![0.0; lcount];
    for l in 0..lcount {
        for b in &ch.batteries {
            d_dis[l] += ptdf[l][b.node] * b.power_discharge_mw;
            d_chg[l] -= ptdf[l][b.node] * b.power_charge_mw;
        }
    }
    let room = |f: f64, d: f64, lim: f64| -> f64 {
        if d.abs() < 1e-9 { 1.0 }
        else if d > 0.0 { ((lim - f) / d).clamp(0.0, 1.0) }
        else { ((lim + f) / -d).clamp(0.0, 1.0) }
    };
    let mut out = vec![(1.0, 1.0); m * h];
    let reaches: Vec<f64> = ch.batteries.iter().map(|b| (0..lcount).map(|l| ptdf[l][b.node].abs()).fold(0.0, f64::max)).collect();
    for t in 0..h {
        let base = ch.network.compute_flows(&ch.exogenous_injections[t]);
        let th_dis: Vec<f64> = (0..lcount).map(|l| room(base[l], d_dis[l], limits[l])).collect();
        let th_chg: Vec<f64> = (0..lcount).map(|l| room(base[l], d_chg[l], limits[l])).collect();
        for (bi, b) in ch.batteries.iter().enumerate() {
            let reach = reaches[bi];
            let (mut tc, mut td) = (1.0_f64, 1.0_f64);
            for l in 0..lcount {
                if ptdf[l][b.node].abs() < 0.1 * reach { continue; }
                tc = tc.min(th_chg[l]);
                td = td.min(th_dis[l]);
            }
            out[bi * h + t] = (tc, td);
        }
    }
    out
}

fn compile_tables(ch: &Challenge) -> Vec<f64> { compile_tables_shift(ch, &[]) }

/// Congestion price ($ per MW of injection, per step) for each battery and step
/// along a rollout that uses DA prices as the RT forecast.
/// `hints[t]` carries each step's positive-dual line sides from one rollout round to the next.
fn congestion_prices(ch: &Challenge, st0: &State, values: &[f64], net: &Net, hints: &mut [Vec<(usize, usize)>]) -> Vec<f64> {
    congestion_prices_seeded(ch, st0, values, net, hints, 0, 0)
}

/// Deterministic standard normal from (seed, t, k) (splitmix64 + Box-Muller).
fn hash_normal(seed: u64, t: usize, k: usize) -> f64 {
    let mix = |mut z: u64| { z = z.wrapping_add(0x9E3779B97F4A7C15); z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) };
    let a = mix(seed.wrapping_mul(0x100000001B3) ^ ((t as u64) << 20) ^ k as u64);
    let b = mix(a);
    let u1 = ((a >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((b >> 11) as f64) / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
fn hash_uniform(seed: u64, t: usize, k: usize) -> f64 { 1.0 / (1.0 + (-1.702 * hash_normal(seed, t, k)).exp()) }

/// seed 0: rollout on DA. seed > 0 with tail 0: iid sigma noise over the whole horizon (G6).
/// seed > 0 with tail L: RT-model sample (common factor, noise, jumps) over the last L steps only.
fn congestion_prices_seeded(ch: &Challenge, st0: &State, values: &[f64], net: &Net, hints: &mut [Vec<(usize, usize)>], seed: u64, tail: usize) -> Vec<f64> {
    let h = ch.num_steps;
    let m = ch.batteries.len();
    let mut pi = vec![0.0; m * h];
    let mut st = st0.clone();
    for t in st0.time_step..h {
        let hint: &[(usize, usize)] = if ROLLOUT_HINTS { &hints[t] } else { &[] };
        let stride = if m >= 60 { DENSE_ROLLOUT_STRIDE } else { ROLLOUT_GRID_STRIDE };
        let (mut u, p, tight) = lp_step_duals(ch, &st, values, net, stride, hint)
            .unwrap_or((vec![0.0; m], vec![0.0; m], Vec::new()));
        hints[t] = tight;
        for i in 0..m { pi[i * h + t] = p[i]; }
        if t + 1 >= h { break; }
        let mut next = ch.market.day_ahead_prices[t + 1].clone();
        if seed > 0 {
            let p = &ch.market.params;
            if tail == 0 {
                for (k, x) in next.iter_mut().enumerate() { *x *= (1.0 + p.volatility * hash_normal(seed, t + 1, k)).max(0.05); }
            } else if t + 1 + tail >= h {
                let zc = hash_normal(seed, t + 1, usize::MAX >> 1);
                for (k, x) in next.iter_mut().enumerate() {
                    let da = *x;
                    *x = da * (1.0 + p.volatility * (0.836660 * zc + 0.547723 * hash_normal(seed, t + 1, k))).max(0.05);
                    if hash_uniform(seed ^ 0xA5A5, t + 1, k) < p.jump_probability {
                        *x += da * (1.0 - hash_uniform(seed ^ 0x5A5A, t + 1, k).min(0.999)).powf(-1.0 / p.tail_index);
                    }
                }
            }
        }
        let mut ok = false;
        for _ in 0..4 {
            if let Ok(s2) = ch.take_step(&st, &u, NextRTPrices::Override(next.clone())) { st = s2; ok = true; break; }
            for x in &mut u { *x *= 0.999; }
        }
        if !ok {
            match ch.take_step(&st, &vec![0.0; m], NextRTPrices::Override(next)) { Ok(s2) => st = s2, Err(_) => break }
        }
    }
    // Single-step duals jump between LP vertices; smooth each battery's series.
    let w = CONGESTION_SMOOTH as isize;
    let mut out = vec![0.0; m * h];
    for i in 0..m {
        for t in 0..h as isize {
            let (a, b) = ((t - w).max(0), (t + w).min(h as isize - 1));
            out[i * h + t as usize] = (a..=b).map(|k| pi[i * h + k as usize]).sum::<f64>() / (b - a + 1) as f64;
        }
    }
    out
}

fn compile_tables_shift(ch: &Challenge, pi: &[f64]) -> Vec<f64> {
    let h = ch.num_steps;
    let m = ch.batteries.len();
    let scen = price_scenarios(ch);
    let deliver = deliverability(ch);
    let mut values = vec![0.0; m * (h + 1) * S];
    let mut acts = [0.0_f64; K];
    let mut cont = [0.0_f64; K];
    // Per (SOC, candidate) transition: action, friction, V-grid cell and weight.
    // They depend on t only through (pc, pd), so they are rebuilt only when those change.
    let mut tr = vec![(0.0_f64, 0.0_f64, 0usize, 0.0_f64); S * K];
    let mut tr_n = [0usize; S];
    // Batteries at one node with the same power/capacity ratios and efficiencies see the
    // same prices, deliverability and congestion shift: reuse the first one's table scaled
    // by capacity (exact except the small quadratic friction term).
    let q = |x: f64| (x * 1e6).round() as i64;
    let class = |b: &Battery| (b.node, q(b.power_charge_mw / b.capacity_mwh), q(b.power_discharge_mw / b.capacity_mwh), q(b.efficiency_charge), q(b.efficiency_discharge), q(b.soc_min_mwh / b.capacity_mwh), q(b.soc_max_mwh / b.capacity_mwh));
    let reps: Vec<usize> = (0..m).map(|i| if TABLE_REUSE { (0..i).find(|&j| class(&ch.batteries[j]) == class(&ch.batteries[i])).unwrap_or(i) } else { i }).collect();
    for (bi, battery) in ch.batteries.iter().enumerate() {
        if reps[bi] != bi && (pi.is_empty() || (0..h).all(|t| pi[bi * h + t] == pi[reps[bi] * h + t])) {
            let r = reps[bi];
            let ratio = battery.capacity_mwh / ch.batteries[r].capacity_mwh;
            for t in 0..=h { for si in 0..S { values[value_idx(bi, t, si, h)] = ratio * values[value_idx(r, t, si, h)]; } }
            continue;
        }
        let cap = battery.capacity_mwh;
        let lo = battery.soc_min_mwh;
        let hi = battery.soc_max_mwh;
        let (glo, ghi) = (cap * 0.10, cap * 0.90);
        let mut key = (f64::NAN, f64::NAN);
        if SALVAGE > 0.0 && m >= SALVAGE_MIN_FLEET {
            let w0 = h.saturating_sub(SALVAGE_STEPS);
            let tail = (w0..h).map(|t| ch.market.day_ahead_prices[t][battery.node]).sum::<f64>() / (h - w0).max(1) as f64;
            for si in 0..S {
                let soc = glo + (ghi - glo) * si as f64 / (S - 1) as f64;
                values[value_idx(bi, h, si, h)] = SALVAGE * tail * (soc - lo).max(0.0);
            }
        }
        for t in (0..h).rev() {
            let da = ch.market.day_ahead_prices[t][battery.node];
            let shift = if pi.is_empty() { 0.0 } else { pi[bi * h + t] / DT };
            let (tc, td) = deliver[bi * h + t];
            // Only the end-of-horizon drain is scaled by deliverability: there the whole
            // fleet sells at once through the binding lines and energy is stranded.
            let floor = if m >= END_DRAIN_MIN_FLEET && t + END_DRAIN_STEPS >= h { END_DRAIN_FLOOR } else if m >= END_DRAIN_MIN_FLEET { DELIVER_FLOOR_CAP } else if m <= 40 { DELIVER_FLOOR_MID } else { DELIVER_FLOOR };
            let pc = battery.power_charge_mw * (floor + (1.0 - floor) * tc);
            let pd = battery.power_discharge_mw * (floor + (1.0 - floor) * td);
            if (pc, pd) != key {
                key = (pc, pd);
                for si in 0..S {
                    let soc = lo + (hi - lo) * si as f64 / (S - 1) as f64;
                    let mut n = 0;
                    let kk = K_COARSE.with(|c| c.get()).unwrap_or_else(|| k_eff(m));
                    for k in 0..kk {
                        let a = action_at(k, pc, pd, kk);
                        let s2 = next_soc(soc, a, battery.efficiency_charge, battery.efficiency_discharge);
                        if s2 < lo - 1e-9 || s2 > hi + 1e-9 { continue; }
                        // Same arithmetic as interpolate().
                        let w = ((s2.clamp(glo, ghi) - glo) / (ghi - glo) * (S - 1) as f64).clamp(0.0, (S - 1) as f64);
                        let j = (w.floor() as usize).min(S - 2);
                        let c = si * K + n;
                        tr[c] = (a, pnl(a, 0.0, cap), j, w - j as f64);
                        n += 1;
                    }
                    tr_n[si] = n;
                }
            }
            let next = value_idx(bi, t + 1, 0, h);
            let vn: Vec<f64> = values[next..next + S].to_vec();
            for si in 0..S {
                // Price-independent part of each candidate: friction + continuation.
                let n = tr_n[si];
                for ((ak, ck), &(a, p, j, z)) in acts.iter_mut().zip(cont.iter_mut()).zip(&tr[si * K..si * K + n]) {
                    *ak = a;
                    *ck = p + (vn[j] * (1.0 - z) + vn[j + 1] * z);
                }
                // RT is observed before acting, so the max sits inside the expectation.
                let mut expected = 0.0;
                for &(mult, w) in &scen {
                    let price = da * mult - shift;
                    let mut best = f64::NEG_INFINITY;
                    for (&a, &c) in acts[..n].iter().zip(&cont[..n]) {
                        best = best.max(a * price * DT + c);
                    }
                    expected += w * if best.is_finite() { best } else { 0.0 };
                }
                values[value_idx(bi, t, si, h)] = expected;
            }
        }
    }
    values
}

fn choose_requests(ch: &Challenge, state: &State, values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let m = ch.batteries.len();
    let t = state.time_step;
    let h = ch.num_steps;
    let mut target = vec![0.0; m];
    let mut benefit = vec![0.0; m];
    for (bi, battery) in ch.batteries.iter().enumerate() {
        let cap = battery.capacity_mwh;
        let soc = state.socs[bi];
        let price = state.rt_prices[battery.node];
        let (alo, ahi) = state.action_bounds[bi];
        let baseline = interpolate(values, bi, t + 1, soc, cap, h);
        let mut best = f64::NEG_INFINITY;
        let mut best_action = 0.0;
        let kk = k_eff(m);
        for k in 0..kk {
            let a = action_at(k, battery.power_charge_mw, battery.power_discharge_mw, kk);
            if a < alo - 1e-9 || a > ahi + 1e-9 { continue; }
            let s2 = next_soc(soc, a, battery.efficiency_charge, battery.efficiency_discharge);
            let q = pnl(a, price, cap) + interpolate(values, bi, t + 1, s2, cap, h);
            if q > best { best = q; best_action = a; }
        }
        if best.is_finite() {
            target[bi] = best_action.clamp(alo, ahi);
            benefit[bi] = (best - baseline).max(0.0);
        }
    }
    (target, benefit)
}

fn descending_order(scores: &[f64]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]).then(i.cmp(&j)));
    ids
}

fn greedy(a: &[f64], b: &[f64], base: &[f64], limits: &[f64], order: &[usize]) -> Vec<f64> {
    let m = b.len();
    let lcount = limits.len();
    let mut x = vec![0.0; m];
    let mut flow = base.to_vec();
    for &i in order {
        if b[i] <= 0.0 { continue; }
        let mut room: f64 = 1.0;
        for l in 0..lcount {
            let direction = a[l * m + i];
            if direction > EPS {
                room = room.min((limits[l] - flow[l]) / direction);
            } else if direction < -EPS {
                room = room.min((-limits[l] - flow[l]) / direction);
            }
        }
        let frac = room.clamp(0.0, 1.0) * 0.999999999;
        if frac > 0.0 {
            x[i] = frac;
            for l in 0..lcount { flow[l] += a[l * m + i] * frac; }
        }
    }
    x
}
fn merit(a: &[f64], b: &[f64], base: &[f64], limits: &[f64], amps: &[f64]) -> Vec<f64> {
    let score: Vec<f64> = b.iter().zip(amps.iter()).map(|(&v, &amp)| v / amp.max(1e-9)).collect();
    greedy(a, b, base, limits, &descending_order(&score))
}


/// Instance-constant network data: battery PTDF rows (pm[l*m + i] = ptdf[l][node_i]).
struct Net {
    pm: Vec<f64>,
}

impl Net {
    fn new(ch: &Challenge) -> Self {
        let m = ch.batteries.len();
        let lcount = ch.network.flow_limits.len();
        let mut pm = vec![0.0; lcount * m];
        for l in 0..lcount {
            for (i, b) in ch.batteries.iter().enumerate() { pm[l * m + i] = ch.network.ptdf[l][b.node]; }
        }
        Net { pm }
    }
}

/// Exact step LP: per-battery segment variables over V's SOC grid (degradation,
/// <= ~$1, dropped), |line flow| <= limit. Returns the LP action, or None when
/// the zero action is already infeasible.
fn lp_step(ch: &Challenge, st: &State, values: &[f64], net: &Net) -> Option<Vec<f64>> {
    let hint = DECISION_HINTS.with(|d| d.borrow().get(st.time_step).cloned().unwrap_or_default());
    let stride = if ch.batteries.len() >= 60 { CAP_MAIN_STRIDE } else { MAIN_GRID_STRIDE };
    lp_step_duals(ch, st, values, net, stride, &hint).map(|(u, _, _)| u)
}

thread_local! {
    // Positive-dual line sides per step from the last congestion rollout: seeds the
    // decision LP at the same step (valid rows, so the optimum is unchanged).
    static DECISION_HINTS: std::cell::RefCell<Vec<Vec<(usize, usize)>>> = std::cell::RefCell::new(Vec::new());
}

/// lp_step plus each battery's congestion price: sum over generated line sides of
/// dual x signed PTDF ($ per MW of injection this step).
/// `hint`: line sides (line, side 0 = +, 1 = -) to add before generation starts; valid
/// rows, so the optimum is unchanged. Also returns the sides with a positive dual.
fn lp_step_duals(ch: &Challenge, st: &State, values: &[f64], net: &Net, stride: usize, hint: &[(usize, usize)])
    -> Option<(Vec<f64>, Vec<f64>, Vec<(usize, usize)>)> {
    let m = ch.batteries.len();
    let t = st.time_step;
    let h = ch.num_steps;
    let lcount = ch.network.flow_limits.len();
    let base = ch.network.compute_flows(&st.exogenous_injections);
    let (mut c, mut ub, mut owner) = (Vec::new(), Vec::new(), Vec::new());
    // F1: common correction delta = aggregate marginal value at the fleet SOC minus
    // the capacity-weighted mean of the batteries' own marginal values.
    let delta = AGG.with(|a| a.borrow().as_ref().map(|(av, alo, ahi)| {
        let tot: f64 = st.socs.iter().sum();
        let span = (ahi - alo) / (S - 1) as f64;
        let ai = |x: f64| -> f64 {
            let w = ((x.clamp(*alo, *ahi) - alo) / (ahi - alo) * (S - 1) as f64).clamp(0.0, (S - 1) as f64);
            let j = (w.floor() as usize).min(S - 2);
            let z = w - j as f64;
            av[(t + 1) * S + j] * (1.0 - z) + av[(t + 1) * S + j + 1] * z
        };
        let lam = (ai(tot + span) - ai(tot - span)) / (2.0 * span);
        let (mut num, mut capsum) = (0.0, 0.0);
        for (i, b) in ch.batteries.iter().enumerate() {
            let e = 0.02 * b.capacity_mwh;
            let sl = (interpolate(values, i, t + 1, st.socs[i] + e, b.capacity_mwh, h) - interpolate(values, i, t + 1, st.socs[i] - e, b.capacity_mwh, h)) / (2.0 * e);
            num += sl * b.capacity_mwh; capsum += b.capacity_mwh;
        }
        FLEET_CORRECTION * (lam - num / capsum)
    })).unwrap_or(0.0);
    // Reused buffers: grid points ascending, then s0 -> grid points -> send with V cached.
    let mut grid: Vec<f64> = Vec::with_capacity(S);
    let mut pts: Vec<f64> = Vec::with_capacity(S + 2);
    let mut vs: Vec<f64> = Vec::with_capacity(S + 2);
    for (i, b) in ch.batteries.iter().enumerate() {
        let cap = b.capacity_mwh;
        let (lo, hi) = st.action_bounds[i];
        let s0 = st.socs[i];
        let p = st.rt_prices[b.node];
        let (gl, gh) = (cap * 0.10, cap * 0.90);
        grid.clear();
        grid.extend((0..S).step_by(stride).map(|k| gl + (gh - gl) * k as f64 / (S - 1) as f64));
        let v = |x: f64| interpolate(values, i, t + 1, x, cap, h);
        for (dir, amax) in [(1.0_f64, hi.max(0.0)), (-1.0_f64, (-lo).max(0.0))] {
            if amax <= 1e-9 { continue; }
            let send = next_soc(s0, dir * amax, b.efficiency_charge, b.efficiency_discharge);
            pts.clear();
            pts.push(s0);
            let keep = |g: f64| (g - s0) * (send - s0) > 0.0 && (g - send) * (s0 - send) > 0.0;
            // Charging moves SOC up (ascending grid order); discharging moves it down.
            if dir < 0.0 { for &g in grid.iter() { if keep(g) { pts.push(g); } } }
            else { for &g in grid.iter().rev() { if keep(g) { pts.push(g); } } }
            pts.push(send);
            vs.clear();
            vs.extend(pts.iter().map(|&x| v(x)));
            for q in 0..pts.len() - 1 {
                let w = [pts[q], pts[q + 1]];
                let ds = (w[1] - w[0]).abs();
                if ds < 1e-12 { continue; }
                let mw = if dir > 0.0 { ds * b.efficiency_discharge / DT } else { ds / (DT * b.efficiency_charge) };
                c.push(dir * p * DT - TX * DT + (vs[q + 1] - vs[q] + delta * (w[1] - w[0])) / mw);
                ub.push(mw);
                owner.push((i, dir));
            }
        }
    }
    let n = c.len();
    // Constraint generation: solve with the line sides added so far, then add the
    // LP_ROWS_PER_ROUND most overloaded (relative to limit), repeat. Only ~15% of
    // overloaded sides bind at the optimum, so the tableau stays small.
    for l in 0..lcount {
        if base[l].abs() > ch.network.flow_limits[l] - 1e-7 { return None; }
    }
    // Rows are appended to the optimal tableau and repaired by dual simplex, so
    // no round re-solves from scratch.
    let mut lp = Tableau::new(&c, &ub, 2 * lcount);
    let mut added = vec![[false; 2]; lcount];
    let mut meta: Vec<(usize, f64)> = Vec::new();
    for &(l, side) in hint {
        if l >= lcount || added[l][side] { continue; }
        let sg = if side == 0 { 1.0 } else { -1.0 };
        let row = &net.pm[l * m..(l + 1) * m];
        added[l][side] = true;
        meta.push((l, sg));
        let a: Vec<f64> = owner.iter().map(|&(i, d)| sg * d * row[i]).collect();
        lp.add_row(&a, ch.network.flow_limits[l] - 1e-7 - sg * base[l]);
    }
    if !hint.is_empty() && (!lp.dual() || !lp.primal()) { return None; }
    let mut x = lp.x();
    let mut converged = false;
    for _round in 0..200 {
        let mut over: Vec<(f64, usize, usize)> = Vec::new();
        // Collapse segments to per-battery actions once; each line is then an m-term dot.
        let mut ub_act = vec![0.0; m];
        for k in 0..n { ub_act[owner[k].0] += owner[k].1 * x[k]; }
        for l in 0..lcount {
            let lim = ch.network.flow_limits[l] - 1e-7;
            let row = &net.pm[l * m..(l + 1) * m];
            let f = base[l] + row.iter().zip(&ub_act).map(|(r, u)| r * u).sum::<f64>();
            for (side, sg) in [(0usize, 1.0_f64), (1, -1.0)] {
                if sg * f > lim + 1e-9 && !added[l][side] { over.push(((sg * f - lim) / lim, l, side)); }
            }
        }
        if over.is_empty() { converged = true; break; }
        over.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, l, side) in over.iter().take(LP_ROWS_PER_ROUND) {
            let sg = if side == 0 { 1.0 } else { -1.0 };
            let row = &net.pm[l * m..(l + 1) * m];
            added[l][side] = true;
            meta.push((l, sg));
            let a: Vec<f64> = owner.iter().map(|&(i, d)| sg * d * row[i]).collect();
            lp.add_row(&a, ch.network.flow_limits[l] - 1e-7 - sg * base[l]);
        }
        if !lp.dual() || !lp.primal() { return None; }
        x = lp.x();
    }
    let y = lp.duals();
    if !converged { return None; }
    let mut u = vec![0.0; m];
    for k in 0..n { u[owner[k].0] += owner[k].1 * x[k]; }
    for i in 0..m { u[i] = u[i].clamp(st.action_bounds[i].0, st.action_bounds[i].1); }
    let mut pi = vec![0.0; m];
    let mut tight = Vec::new();
    for (r, &(l, sg)) in meta.iter().enumerate() {
        if y[r] <= 0.0 { continue; }
        tight.push((l, if sg > 0.0 { 0 } else { 1 }));
        let row = &net.pm[l * m..(l + 1) * m];
        for i in 0..m { pi[i] += y[r] * sg * row[i]; }
    }
    Some((u, pi, tight))
}

/// Maximise c.x subject to appended rows a.x <= b (b >= 0, so x = 0 is feasible)
/// and 0 <= x <= ub: dense bounded-variable tableau with one slack per row.
/// Starts optimal with no rows; after add_row, dual() restores primal feasibility
/// and primal() cleans up.
struct Tableau {
    n: usize,
    nt: usize,
    tab: Vec<Vec<f64>>,
    d: Vec<f64>,
    ubt: Vec<f64>,
    basis: Vec<usize>,
    pos: Vec<usize>,
    at_up: Vec<bool>,
    beta: Vec<f64>,
}

impl Tableau {
    const EPS: f64 = 1e-9;

    fn new(c: &[f64], ub: &[f64], max_rows: usize) -> Self {
        let n = c.len();
        let cap = n + max_rows;
        let mut d = c.to_vec(); d.resize(cap, 0.0);
        let mut ubt = ub.to_vec(); ubt.resize(cap, f64::INFINITY);
        let at_up = (0..cap).map(|j| j < n && d[j] > 0.0).collect();
        Tableau { n, nt: n, tab: Vec::new(), d, ubt, basis: Vec::new(), pos: vec![usize::MAX; cap], at_up, beta: Vec::new() }
    }

    fn value(&self, j: usize) -> f64 {
        if self.pos[j] != usize::MAX { self.beta[self.pos[j]] } else if self.at_up[j] { self.ubt[j] } else { 0.0 }
    }

    fn x(&self) -> Vec<f64> {
        // Nonbasic at their bound in one pass; basic values written through the basis.
        let mut x: Vec<f64> = self.at_up[..self.n].iter().zip(&self.ubt[..self.n]).map(|(&up, &u)| if up { u } else { 0.0 }).collect();
        for (&bv, &b) in self.basis.iter().zip(&self.beta) { if bv < self.n { x[bv] = b; } }
        x
    }

    /// Row duals: minus each slack's reduced cost.
    fn duals(&self) -> Vec<f64> { (0..self.nt - self.n).map(|i| (-self.d[self.n + i]).max(0.0)).collect() }

    fn add_row(&mut self, a: &[f64], b: f64) {
        let s = self.nt;
        let cap = self.d.len();
        let mut row = vec![0.0; cap];
        row[..self.n].copy_from_slice(a);
        row[s] = 1.0;
        let xv = self.x();
        let slack = b - a[..self.n].iter().zip(&xv).map(|(aj, xj)| aj * xj).sum::<f64>();
        for (i, &bv) in self.basis.iter().enumerate() {
            let f = row[bv];
            if f.abs() < 1e-15 { continue; }
            for (rj, sj) in row[..s].iter_mut().zip(&self.tab[i][..s]) { *rj -= f * sj; }
        }
        self.tab.push(row);
        self.basis.push(s);
        self.pos[s] = self.basis.len() - 1;
        self.beta.push(slack);
        self.nt += 1;
    }

    /// Basic `e` enters at row `leave`, moving by dir * theta; the leaving variable
    /// goes to its upper bound if `leave_up`.
    fn pivot(&mut self, e: usize, leave: usize, dir: f64, theta: f64, leave_up: bool) {
        let nt = self.nt;
        let entering_value = self.value(e) + dir * theta;
        for (b, r) in self.beta.iter_mut().zip(&self.tab) { *b -= r[e] * dir * theta; }
        let lv = self.basis[leave];
        self.pos[lv] = usize::MAX; self.at_up[lv] = leave_up;
        self.basis[leave] = e; self.pos[e] = leave; self.at_up[e] = false;
        self.beta[leave] = entering_value;
        let piv = self.tab[leave][e];
        for v in self.tab[leave][..nt].iter_mut() { *v /= piv; }
        let prow: Vec<f64> = self.tab[leave][..nt].to_vec();
        for (i, r) in self.tab.iter_mut().enumerate() {
            if i == leave { continue; }
            let f = r[e];
            if f.abs() < 1e-15 { continue; }
            for (rj, pj) in r[..nt].iter_mut().zip(&prow) { *rj -= f * pj; }
        }
        let f = self.d[e];
        for (dj, pj) in self.d[..nt].iter_mut().zip(&prow) { *dj -= f * pj; }
    }

    /// Bounded primal simplex (Dantzig) from a primal-feasible basis.
    fn primal(&mut self) -> bool {
        let eps = Self::EPS;
        for _ in 0..20000 {
            let (mut e, mut best) = (usize::MAX, eps);
            for j in 0..self.nt {
                if self.pos[j] != usize::MAX { continue; }
                let gain = if self.at_up[j] { -self.d[j] } else { self.d[j] };
                if gain > best { best = gain; e = j; }
            }
            if e == usize::MAX { return true; }
            let dir = if self.at_up[e] { -1.0 } else { 1.0 };
            let mut theta = self.ubt[e];
            let (mut leave, mut leave_up) = (usize::MAX, false);
            for i in 0..self.beta.len() {
                let a = self.tab[i][e] * dir;
                let bv = self.basis[i];
                if a > eps {
                    let lim = self.beta[i] / a;
                    if lim < theta { theta = lim; leave = i; leave_up = false; }
                } else if a < -eps && self.ubt[bv].is_finite() {
                    let lim = (self.ubt[bv] - self.beta[i]) / (-a);
                    if lim < theta { theta = lim; leave = i; leave_up = true; }
                }
            }
            if !theta.is_finite() { return false; }
            if leave == usize::MAX {
                for i in 0..self.beta.len() { self.beta[i] -= self.tab[i][e] * dir * theta; }
                self.at_up[e] = !self.at_up[e];
                continue;
            }
            self.pivot(e, leave, dir, theta, leave_up);
        }
        false
    }

    /// Bounded dual simplex from a dual-feasible basis: drive every basic variable
    /// back inside its bounds.
    fn dual(&mut self) -> bool {
        let eps = Self::EPS;
        for _ in 0..20000 {
            let (mut leave, mut worst, mut to_up) = (usize::MAX, 1e-9, false);
            for i in 0..self.beta.len() {
                let (b, u) = (self.beta[i], self.ubt[self.basis[i]]);
                if -b > worst { worst = -b; leave = i; to_up = false; }
                if b - u > worst { worst = b - u; leave = i; to_up = true; }
            }
            if leave == usize::MAX { return true; }
            let lrow = &self.tab[leave];
            // Long-step (bound-flipping) ratio test, as in the endgame RevLP.
            let mut cands: Vec<(f64, f64, usize)> = Vec::new();
            for j in 0..self.nt {
                if self.pos[j] != usize::MAX { continue; }
                let dir = if self.at_up[j] { -1.0 } else { 1.0 };
                let a = lrow[j] * dir;
                // x_B moves by -a * theta: it must rise to 0 or fall to its upper bound.
                if (!to_up && a < -eps) || (to_up && a > eps) {
                    let g = (-dir * self.d[j]).max(0.0);
                    cands.push((g / a.abs(), a.abs(), j));
                }
            }
            if cands.is_empty() { return false; }
            cands.sort_by(|x, y| x.0.total_cmp(&y.0).then(y.1.total_cmp(&x.1)).then(x.2.cmp(&y.2)));
            let mut slope = worst;
            let mut e = usize::MAX;
            let mut flips: Vec<usize> = Vec::new();
            for &(_, aa, j) in &cands {
                let u = self.ubt[j];
                if u.is_finite() && slope - aa * u > 1e-9 { slope -= aa * u; flips.push(j); } else { e = j; break; }
            }
            if e == usize::MAX { e = flips.pop().unwrap(); }
            for &j in &flips {
                let dj = if self.at_up[j] { -1.0 } else { 1.0 };
                let u = self.ubt[j];
                for (b, r) in self.beta.iter_mut().zip(&self.tab) { *b -= r[j] * dj * u; }
                self.at_up[j] = !self.at_up[j];
            }
            let dir = if self.at_up[e] { -1.0 } else { 1.0 };
            let target = if to_up { self.ubt[self.basis[leave]] } else { 0.0 };
            let theta = (self.beta[leave] - target) / (self.tab[leave][e] * dir);
            self.pivot(e, leave, dir, theta, to_up);
        }
        false
    }
}

/// G7: the remaining horizon as one LP. Variables: discharge and charge per battery
/// per remaining step; rows: cumulative SOC bounds and per-step line limits, added by
/// constraint generation. Step 0 is priced at the observed RT, later steps at
/// DA x the scenario-mean multiplier; stored energy is worth 0 at the end. Returns
/// the first step's action.
/// Revised bounded simplex for the endgame: same pivot rules as Tableau, but keeps
/// B^-1 (column-major) and sparse rows, computing only the pivot row and column.
struct RevLP {
    n: usize,
    nt: usize,
    d: Vec<f64>,
    ubt: Vec<f64>,
    basis: Vec<usize>,
    pos: Vec<usize>,
    at_up: Vec<bool>,
    beta: Vec<f64>,
    rows: Vec<Vec<(usize, f64)>>,
    cols: Vec<Vec<(usize, f64)>>,
    bc: Vec<Vec<f64>>,
}

impl RevLP {
    const EPS: f64 = 1e-9;

    fn new(c: &[f64], ub: &[f64], max_rows: usize) -> Self {
        let n = c.len();
        let cap = n + max_rows;
        let mut d = c.to_vec(); d.resize(cap, 0.0);
        let mut ubt = ub.to_vec(); ubt.resize(cap, f64::INFINITY);
        let at_up = (0..cap).map(|j| j < n && d[j] > 0.0).collect();
        RevLP { n, nt: n, d, ubt, basis: Vec::new(), pos: vec![usize::MAX; cap], at_up, beta: Vec::new(),
                rows: Vec::new(), cols: vec![Vec::new(); n], bc: Vec::new() }
    }

    fn value(&self, j: usize) -> f64 {
        if self.pos[j] != usize::MAX { self.beta[self.pos[j]] } else if self.at_up[j] { self.ubt[j] } else { 0.0 }
    }

    fn x(&self) -> Vec<f64> {
        let mut x: Vec<f64> = self.at_up[..self.n].iter().zip(&self.ubt[..self.n]).map(|(&up, &u)| if up { u } else { 0.0 }).collect();
        for (&bv, &b) in self.basis.iter().zip(&self.beta) { if bv < self.n { x[bv] = b; } }
        x
    }

    fn add_row(&mut self, a: &[f64], b: f64) {
        let r = self.beta.len();
        let s = self.nt;
        let xv = self.x();
        let slack = b - a[..self.n].iter().zip(&xv).map(|(aj, xj)| aj * xj).sum::<f64>();
        let mut sp = Vec::new();
        for j in 0..self.n { if a[j] != 0.0 { sp.push((j, a[j])); self.cols[j].push((r, a[j])); } }
        self.rows.push(sp);
        // New row of B^-1: -(a_B B^-1); new column: e_r.
        let q: Vec<f64> = self.basis.iter().map(|&bv| if bv < self.n { a[bv] } else { 0.0 }).collect();
        for col in self.bc.iter_mut() {
            let v: f64 = q.iter().zip(col.iter()).map(|(qi, ci)| qi * ci).sum();
            col.push(-v);
        }
        let mut e = vec![0.0; r + 1]; e[r] = 1.0;
        self.bc.push(e);
        self.basis.push(s);
        self.pos[s] = r;
        self.beta.push(slack);
        self.nt += 1;
    }

    /// B^-1 A_j.
    fn column(&self, j: usize) -> Vec<f64> {
        let rr = self.beta.len();
        if j >= self.n { return self.bc[j - self.n].clone(); }
        let mut w = vec![0.0; rr];
        for &(r, v) in &self.cols[j] { for (wi, ci) in w.iter_mut().zip(&self.bc[r]) { *wi += v * ci; } }
        w
    }

    /// Row `leave` of B^-1 [A I] over all nt columns.
    fn row_alpha(&self, leave: usize) -> Vec<f64> {
        let mut al = vec![0.0; self.nt];
        for (r, col) in self.bc.iter().enumerate() {
            let rho = col[leave];
            al[self.n + r] = rho;
            if rho == 0.0 { continue; }
            for &(j, v) in &self.rows[r] { al[j] += rho * v; }
        }
        al
    }

    fn pivot(&mut self, e: usize, leave: usize, dir: f64, theta: f64, leave_up: bool, w: &[f64], al: &[f64]) {
        let nt = self.nt;
        let entering_value = self.value(e) + dir * theta;
        for (b, wi) in self.beta.iter_mut().zip(w) { *b -= wi * dir * theta; }
        let lv = self.basis[leave];
        self.pos[lv] = usize::MAX; self.at_up[lv] = leave_up;
        self.basis[leave] = e; self.pos[e] = leave; self.at_up[e] = false;
        self.beta[leave] = entering_value;
        let piv = w[leave];
        for col in self.bc.iter_mut() {
            let xl = col[leave] / piv;
            if xl != 0.0 { for (ci, wi) in col.iter_mut().zip(w) { *ci -= wi * xl; } }
            col[leave] = xl;
        }
        let f = self.d[e] / al[e];
        for (dj, aj) in self.d[..nt].iter_mut().zip(al) { *dj -= f * aj; }
    }

    fn primal(&mut self) -> bool {
        let eps = Self::EPS;
        for _ in 0..20000 {
            let (mut e, mut best) = (usize::MAX, eps);
            for j in 0..self.nt {
                if self.pos[j] != usize::MAX { continue; }
                let gain = if self.at_up[j] { -self.d[j] } else { self.d[j] };
                if gain > best { best = gain; e = j; }
            }
            if e == usize::MAX { return true; }
            let dir = if self.at_up[e] { -1.0 } else { 1.0 };
            let w = self.column(e);
            let mut theta = self.ubt[e];
            let (mut leave, mut leave_up) = (usize::MAX, false);
            for i in 0..self.beta.len() {
                let a = w[i] * dir;
                let bv = self.basis[i];
                if a > eps {
                    let lim = self.beta[i] / a;
                    if lim < theta { theta = lim; leave = i; leave_up = false; }
                } else if a < -eps && self.ubt[bv].is_finite() {
                    let lim = (self.ubt[bv] - self.beta[i]) / (-a);
                    if lim < theta { theta = lim; leave = i; leave_up = true; }
                }
            }
            if !theta.is_finite() { return false; }
            if leave == usize::MAX {
                for i in 0..self.beta.len() { self.beta[i] -= w[i] * dir * theta; }
                self.at_up[e] = !self.at_up[e];
                continue;
            }
            let al = self.row_alpha(leave);
            self.pivot(e, leave, dir, theta, leave_up, &w, &al);
        }
        false
    }

    fn dual(&mut self) -> bool {
        let eps = Self::EPS;
        for _ in 0..20000 {
            let (mut leave, mut worst, mut to_up) = (usize::MAX, 1e-9, false);
            for i in 0..self.beta.len() {
                let (b, u) = (self.beta[i], self.ubt[self.basis[i]]);
                if -b > worst { worst = -b; leave = i; to_up = false; }
                if b - u > worst { worst = b - u; leave = i; to_up = true; }
            }
            if leave == usize::MAX { return true; }
            let lrow = self.row_alpha(leave);
            // Bound-flipping (long-step) ratio test: walk the breakpoints in ratio order,
            // flipping boxed candidates while the leaving row stays infeasible.
            let mut cands: Vec<(f64, f64, usize)> = Vec::new();
            for j in 0..self.nt {
                if self.pos[j] != usize::MAX { continue; }
                let dir = if self.at_up[j] { -1.0 } else { 1.0 };
                let a = lrow[j] * dir;
                if (!to_up && a < -eps) || (to_up && a > eps) {
                    let g = (-dir * self.d[j]).max(0.0);
                    cands.push((g / a.abs(), a.abs(), j));
                }
            }
            if cands.is_empty() { return false; }
            cands.sort_by(|x, y| x.0.total_cmp(&y.0).then(y.1.total_cmp(&x.1)).then(x.2.cmp(&y.2)));
            let mut slope = worst;
            let mut e = usize::MAX;
            let mut flips: Vec<usize> = Vec::new();
            for &(_, aa, j) in &cands {
                let u = self.ubt[j];
                if EG_BFRT && u.is_finite() && slope - aa * u > 1e-9 { slope -= aa * u; flips.push(j); } else { e = j; break; }
            }
            if e == usize::MAX { e = flips.pop().unwrap(); }
            for &j in &flips {
                let dj = if self.at_up[j] { -1.0 } else { 1.0 };
                let wj = self.column(j);
                let u = self.ubt[j];
                for (b, wi) in self.beta.iter_mut().zip(&wj) { *b -= wi * dj * u; }
                self.at_up[j] = !self.at_up[j];
            }
            let dir = if self.at_up[e] { -1.0 } else { 1.0 };
            let target = if to_up { self.ubt[self.basis[leave]] } else { 0.0 };
            let w = self.column(e);
            let theta = (self.beta[leave] - target) / (w[leave] * dir);
            self.pivot(e, leave, dir, theta, to_up, &w, &lrow);
        }
        false
    }
}

fn endgame_lp(ch: &Challenge, st: &State, net: &Net) -> Option<Vec<f64>> {
    let m = ch.batteries.len();
    let t0 = st.time_step;
    let h = ch.num_steps;
    let len = h - t0;
    let lcount = ch.network.flow_limits.len();
    let mean_mult: f64 = price_scenarios(ch).iter().map(|&(x, w)| x * w).sum::<f64>()
        / price_scenarios(ch).iter().map(|&(_, w)| w).sum::<f64>();
    let bases: Vec<Vec<f64>> = (0..len).map(|k| if k == 0 { ch.network.compute_flows(&st.exogenous_injections) }
        else { ch.network.compute_flows(&ch.exogenous_injections[t0 + k]) }).collect();
    for b in &bases { for l in 0..lcount { if b[l].abs() > ch.network.flow_limits[l] - 1e-7 { return None; } } }
    // var index: ((k * m) + i) * 2 + {0: discharge, 1: charge}
    let n = len * m * 2;
    let (mut c, mut ub) = (vec![0.0; n], vec![0.0; n]);
    for k in 0..len {
        for (i, b) in ch.batteries.iter().enumerate() {
            let p = if k == 0 { st.rt_prices[b.node] } else { ch.market.day_ahead_prices[t0 + k][b.node] * mean_mult };
            let j = (k * m + i) * 2;
            c[j] = p * DT - TX * DT;
            c[j + 1] = -p * DT - TX * DT;
            if k == 0 { ub[j] = st.action_bounds[i].1.max(0.0); ub[j + 1] = (-st.action_bounds[i].0).max(0.0); }
            else { ub[j] = b.power_discharge_mw; ub[j + 1] = b.power_charge_mw; }
        }
    }
    // Candidate rows: (kind, battery or line, step, side).
    let soc_rows = m * len * 2;
    let mut lp = RevLP::new(&c, &ub, soc_rows + 2 * lcount * len);
    // Added rows: flags per (kind, battery/line, step, side) plus insertion order (deterministic).
    let mut flag_soc = vec![[false; 2]; m * len];
    let mut flag_line = vec![[false; 2]; lcount * len];
    let mut added_list: Vec<(u8, usize, usize, usize)> = Vec::new();
    let mut insert = |kind: u8, a_: usize, k: usize, side: usize, fs: &mut Vec<[bool; 2]>, fl: &mut Vec<[bool; 2]>| -> bool {
        let f = if kind == 0 { &mut fs[a_ * len + k][side] } else { &mut fl[a_ * len + k][side] };
        if *f { return false; }
        *f = true; added_list.push((kind, a_, k, side)); true
    };
    let mk = |kind: u8, a_: usize, k: usize, side: usize| -> (Vec<f64>, f64) {
        let mut a = vec![0.0; n];
        let sg = if side == 0 { 1.0 } else { -1.0 };
        let rhs;
        if kind == 0 {
            let (i, b) = (a_, &ch.batteries[a_]);
            for kk in 0..=k {
                let j = (kk * m + i) * 2;
                a[j] = -sg * DT / b.efficiency_discharge;
                a[j + 1] = sg * DT * b.efficiency_charge;
            }
            rhs = if side == 0 { b.soc_max_mwh - st.socs[i] } else { st.socs[i] - b.soc_min_mwh };
        } else {
            let l = a_;
            let row = &net.pm[l * m..(l + 1) * m];
            for i in 0..m { let j = (k * m + i) * 2; a[j] = sg * row[i]; a[j + 1] = -sg * row[i]; }
            rhs = ch.network.flow_limits[l] - 1e-7 - sg * bases[k][l];
        }
        (a, rhs.max(0.0))
    };
    // Seed with the rows the previous step's endgame added (absolute steps); they are
    // valid constraints of this problem too, so the optimum is unchanged.
    if ENDGAME_HINTS {
        let prev = EG_ROWS.with(|r| r.borrow().clone());
        let mut any = false;
        for (kind, a_, abs, side) in prev {
            if abs < t0 || abs >= h { continue; }
            let k = abs - t0;
            if insert(kind, a_, k, side, &mut flag_soc, &mut flag_line) { let (a, r) = mk(kind, a_, k, side); lp.add_row(&a, r); any = true; }
        }
        if any && (!lp.dual() || !lp.primal()) { return None; }
    }
    for _round in 0..400 {
        let x = lp.x();
        let mut over: Vec<(f64, u8, usize, usize, usize)> = Vec::new();
        // SOC after step k (k = 0..len-1) for each battery.
        for (i, b) in ch.batteries.iter().enumerate() {
            let mut s = st.socs[i];
            for k in 0..len {
                let j = (k * m + i) * 2;
                s += (b.efficiency_charge * x[j + 1] - x[j] / b.efficiency_discharge) * DT;
                if s > b.soc_max_mwh + 1e-7 && !flag_soc[i * len + k][0] { over.push(((s - b.soc_max_mwh) / b.capacity_mwh, 0, i, k, 0)); }
                if s < b.soc_min_mwh - 1e-7 && !flag_soc[i * len + k][1] { over.push(((b.soc_min_mwh - s) / b.capacity_mwh, 0, i, k, 1)); }
            }
        }
        for k in 0..len {
            for l in 0..lcount {
                let row = &net.pm[l * m..(l + 1) * m];
                let mut f = bases[k][l];
                for i in 0..m { let j = (k * m + i) * 2; f += row[i] * (x[j] - x[j + 1]); }
                let lim = ch.network.flow_limits[l] - 1e-7;
                for (side, sg) in [(0usize, 1.0_f64), (1, -1.0)] {
                    if sg * f > lim + 1e-9 && !flag_line[l * len + k][side] { over.push(((sg * f - lim) / lim, 1, l, k, side)); }
                }
            }
        }
        if over.is_empty() {
            if ENDGAME_HINTS {
                // Seed the next step with the rows binding here only (slack nonbasic or at zero).
                let keep: Vec<(u8, usize, usize, usize)> = added_list.iter().enumerate()
                    .filter(|&(r, _)| { let p = lp.pos[lp.n + r]; p == usize::MAX || lp.beta[p] <= 1e-7 })
                    .map(|(_, &(kind, a_, k, side))| (kind, a_, k + t0, side)).collect();
                EG_ROWS.with(|r| *r.borrow_mut() = keep);
            }
            let mut u = vec![0.0; m];
            for i in 0..m { u[i] = (x[2 * i] - x[2 * i + 1]).clamp(st.action_bounds[i].0, st.action_bounds[i].1); }
            return Some(u);
        }
        over.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, kind, a_, k, side) in over.iter().take(LP_ROWS_PER_ROUND) {
            insert(kind, a_, k, side, &mut flag_soc, &mut flag_line);
            let (a, r) = mk(kind, a_, k, side);
            lp.add_row(&a, r);
        }
        if !lp.dual() || !lp.primal() { return None; }
    }
    None
}

thread_local! {
    // Endgame rows (kind, battery/line, absolute step, side) from the previous step.
    static EG_ROWS: std::cell::RefCell<Vec<(u8, usize, usize, usize)>> = std::cell::RefCell::new(Vec::new());
}

fn decide(ch: &Challenge, state: &State, values: &[f64], net: &Net) -> Result<Vec<f64>> {
    // Exact step LP, or the multi-step endgame LP over the last steps. The merit
    // allocator below is the fallback when no LP answer exists (base flows already
    // at a limit) or the LP action stays infeasible after scaling back.
    let m0 = ch.batteries.len();
    let egs = if m0 <= 20 { 6 } else if m0 <= ENDGAME_MAX_M { ENDGAME_STEPS } else { ENDGAME_STEPS_LARGE };
    let eg = if egs > 0 && state.time_step + egs >= ch.num_steps { endgame_lp(ch, state, net) } else { None };
    let lp = if eg.is_some() { eg } else { lp_step(ch, state, values, net) };
    let limits = &ch.network.flow_limits;
    let within = |act: &[f64]| -> bool {
        let flows = ch.network.compute_flows(&ch.compute_total_injections(state, act));
        flows.iter().zip(limits.iter()).all(|(&f, &lim)| f.abs() <= lim + 1e-9)
    };
    if let Some(u) = &lp {
        let mut action = u.clone();
        for _ in 0..4 {
            if within(&action) { return Ok(action); }
            for v in &mut action { *v *= 0.999; }
        }
    }
    FALLBACK_STEPS.with(|c| c.set(c.get() + 1));
    // Merit allocator: each battery requests its one-step Bellman action; requests are
    // granted in order of value per MW until a line binds.
    let (target, benefit) = choose_requests(ch, state, values);
    let m = target.len();
    let lcount = limits.len();
    let base = ch.network.compute_flows(&state.exogenous_injections);
    let amps: Vec<f64> = target.iter().map(|u| u.abs()).collect();
    let mut a = vec![0.0; lcount * m];
    for l in 0..lcount {
        for i in 0..m { a[l * m + i] = ch.network.ptdf[l][ch.batteries[i].node] * target[i]; }
    }
    let x = merit(&a, &benefit, &base, limits, &amps);
    let greedy_action: Vec<f64> = target.iter().zip(x.iter()).map(|(&u, &f)| u * f).collect();
    let eval_act = |act: &[f64]| -> f64 {
        let mut total = 0.0_f64;
        for (i, battery) in ch.batteries.iter().enumerate() {
            let soc = state.socs[i];
            let price = state.rt_prices[battery.node];
            let cap = battery.capacity_mwh;
            let s2 = next_soc(soc, act[i], battery.efficiency_charge, battery.efficiency_discharge);
            total += pnl(act[i], price, cap) + interpolate(values, i, state.time_step + 1, s2, cap, ch.num_steps);
        }
        total
    };
    let mut action = greedy_action;
    if let Some(u) = &lp { if eval_act(u) > eval_act(&action) { action = u.clone(); } }
    for _ in 0..4 {
        if within(&action) { return Ok(action); }
        for u in &mut action { *u *= 0.999; }
    }
    Ok(vec![0.0; m])
}

thread_local! {
    // Decision steps that used the merit fallback (diagnostics only).
    pub static FALLBACK_STEPS: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

/// Clear every per-instance thread-local cache.
fn reset_caches() {
    AGG.with(|a| *a.borrow_mut() = None);
    DELIVER_CACHE.with(|c| *c.borrow_mut() = None);
    DECISION_HINTS.with(|d| d.borrow_mut().clear());
    EG_ROWS.with(|r| r.borrow_mut().clear());
    FALLBACK_STEPS.with(|c| c.set(0));
    K_COARSE.with(|c| c.set(None));
}
