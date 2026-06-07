use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

use tig_challenges::energy_arbitrage::*;
use anyhow::Result;
use serde_json::{Map, Value};
use std::time::Instant;

pub fn solve_challenge(
  challenge: &Challenge,
  save_solution: &dyn Fn(&Solution) -> Result<()>,
  hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
  let deadline = Instant::now() + std::time::Duration::from_secs(27);

  let num_batteries = challenge.num_batteries;
  let num_steps = challenge.num_steps;
  let num_nodes = challenge.network.num_nodes;
  let num_lines = challenge.network.num_lines;
  let dt: f64 = 0.25;
  let kappa_tx: f64 = 0.25;
  let kappa_deg: f64 = 1.0;
  let beta_deg: f64 = 2.0;
  let slack = challenge.network.slack_bus;

  let (soc_levels, action_grid, asca_iters) = if num_batteries <= 15 {
    (401, 80, 25)
  } else if num_batteries <= 30 {
    (201, 40, 20)
  } else {
    (101, 25, 15)
  };

  let lmp_threshold: f64 = 0.65;
  let lmp_scale: f64 = 1.0;
  let base_premium: f64 = 20.0 * lmp_scale;
  let lmp_premiums: Vec<Vec<f64>> = (0..num_steps)
    .map(|t| {
      let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
      let mut premiums = vec![0.0_f64; num_nodes];
      for l in 0..num_lines {
        let limit = challenge.network.flow_limits[l];
        if limit <= 1e-6 { continue; }
        let ratio = f_exo[l].abs() / limit;
        if ratio > lmp_threshold {
          let proba = ((ratio - lmp_threshold) / (1.0 - lmp_threshold).max(1e-6)).min(1.0).max(0.0);
          let premium = base_premium * proba;
          let sign_f = f_exo[l].signum();
          for n in 0..num_nodes {
            if n != slack {
              premiums[n] += -challenge.network.ptdf[l][n] * sign_f * premium;
            }
          }
        }
      }
      premiums
    })
    .collect();

  let ptdf_b: Vec<Vec<f64>> = challenge.batteries.iter().map(|bat| {
    let n = bat.node;
    (0..num_lines)
      .map(|l| challenge.network.ptdf[l][n] - challenge.network.ptdf[l][slack])
      .collect()
  }).collect();

  let b_lines: Vec<Vec<(usize, f64)>> = ptdf_b.iter().map(|row| {
    row.iter().enumerate()
      .filter(|&(_, &p)| p.abs() > 1e-9)
      .map(|(l, &p)| (l, p))
      .collect()
  }).collect();

  let mut congestion_penalty_d = vec![vec![0.0_f64; num_batteries]; num_steps];
  let mut congestion_penalty_c = vec![vec![0.0_f64; num_batteries]; num_steps];

  for t in 0..num_steps {
    let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
    for b in 0..num_batteries {
      let n = challenge.batteries[b].node;
      let p_ref = challenge.market.day_ahead_prices[t][n].abs().max(1.0);
      for &(l, p) in &b_lines[b] {
        let limit = challenge.network.flow_limits[l];
        if limit <= 1e-6 { continue; }
        let headroom = (limit - f_exo[l].abs()).max(0.0);
        let tightness = 1.0 - headroom / limit;
        if tightness <= 1e-4 { continue; }
        let sign_flow = f_exo[l].signum();
        let p_sign = p.signum();
        let worsens_d = if p_sign == sign_flow { tightness } else { -tightness * 0.3 };
        let worsens_c = if p_sign == -sign_flow { tightness } else { -tightness * 0.3 };
        congestion_penalty_d[t][b] += worsens_d * tightness * p_ref * 0.12;
        congestion_penalty_c[t][b] += worsens_c * tightness * p_ref * 0.12;
      }
    }
  }

  let mut dp = vec![vec![vec![0.0_f64; soc_levels]; num_steps + 1]; num_batteries];

  for b in 0..num_batteries {
    let bat = &challenge.batteries[b];
    let n = bat.node;
    let soc_min = bat.soc_min_mwh;
    let soc_max = bat.soc_max_mwh;
    let soc_span = (soc_max - soc_min).max(1e-9);

    for t in (0..num_steps).rev() {
      let p_da = challenge.market.day_ahead_prices[t][n];
      let extra = lmp_premiums[t][n];
      let cong_d = congestion_penalty_d[t][b];
      let cong_c = congestion_penalty_c[t][b];
      let p_sell = p_da + extra - cong_d;
      let p_buy = p_da + extra + cong_c;
      let eta_c = bat.efficiency_charge;
      let eta_d = bat.efficiency_discharge;
      let cap = bat.capacity_mwh.max(1e-9);

      for i in 0..soc_levels {
        let soc = soc_min + soc_span * (i as f64) / ((soc_levels - 1) as f64);

        let charge_limit = if eta_c > 0.0 { (soc_max - soc) / (eta_c * dt) } else { 0.0 };
        let discharge_limit = if eta_d > 0.0 { (soc - soc_min) * eta_d / dt } else { 0.0 };
        let u_lo = -(bat.power_charge_mw.min(charge_limit.max(0.0)));
        let u_hi = bat.power_discharge_mw.min(discharge_limit.max(0.0));
        let u_hi = u_hi.max(u_lo);

        let span = u_hi - u_lo;
        let mut max_val = f64::NEG_INFINITY;

        for j in 0..=action_grid {
          let u = if span > 0.0 { u_lo + span * (j as f64) / (action_grid as f64) } else { u_lo };
          let price = if u > 0.0 { p_sell } else { p_buy };
          let abs_u = u.abs();
          let revenue = u * price * dt;
          let tx = kappa_tx * abs_u * dt;
          let deg_base = (abs_u * dt) / cap;
          let deg = kappa_deg * deg_base * deg_base;
          let profit = revenue - tx - deg;

          let next_soc = if u < 0.0 {
            (soc + eta_c * (-u) * dt).min(soc_max)
          } else {
            (soc - u / eta_d.max(1e-9) * dt).max(soc_min)
          };

          let idx_f = (next_soc - soc_min) / soc_span * ((soc_levels - 1) as f64);
          let idx0 = (idx_f.floor() as isize).max(0) as usize;
          let idx0c = idx0.min(soc_levels - 1);
          let idx1c = (idx0 + 1).min(soc_levels - 1);
          let frac = (idx_f - idx0 as f64).max(0.0).min(1.0);
          let v_next = dp[b][t + 1][idx0c] * (1.0 - frac) + dp[b][t + 1][idx1c] * frac;

          let val = profit + v_next;
          if val > max_val { max_val = val; }
        }
        dp[b][t][i] = max_val;
      }
    }
  }

  let dp_value = |b: usize, t: usize, soc: f64| -> f64 {
    let bat = &challenge.batteries[b];
    let soc_min = bat.soc_min_mwh;
    let soc_max = bat.soc_max_mwh;
    let soc_span = (soc_max - soc_min).max(1e-9);
    let idx_f = (soc - soc_min) / soc_span * ((soc_levels - 1) as f64);
    let idx0 = (idx_f.floor() as isize).max(0) as usize;
    let idx0c = idx0.min(soc_levels - 1);
    let idx1c = (idx0 + 1).min(soc_levels - 1);
    let frac = (idx_f - idx0 as f64).max(0.0).min(1.0);
    dp[b][t][idx0c] * (1.0 - frac) + dp[b][t][idx1c] * frac
  };

  let epsilon_soc: f64 = 0.5_f64;
  let compute_shadows = |b: usize, t: usize, soc: f64| -> (f64, f64) {
    let bat = &challenge.batteries[b];
    let v_curr = dp_value(b, t, soc);
    let soc_hi = (soc + epsilon_soc).min(bat.soc_max_mwh);
    let soc_lo = (soc - epsilon_soc).max(bat.soc_min_mwh);
    let v_hi = dp_value(b, t, soc_hi);
    let v_lo = dp_value(b, t, soc_lo);
    let span_hi = soc_hi - soc;
    let span_lo = soc - soc_lo;
    let shadow_d = if span_hi > 1e-12 { (v_hi - v_curr) / span_hi } else { 0.0 };
    let shadow_c = if span_lo > 1e-12 { (v_curr - v_lo) / span_lo } else { 0.0 };
    (shadow_c, shadow_d)
  };

  let soc_dispatch_bias = |b: usize, soc: f64, soc_min: f64, soc_max: f64, t: usize| -> f64 {
    let battery = &challenge.batteries[b];
    let soc_range = soc_max - soc_min;
    if soc_range <= 1e-9 {
      return 0.0;
    }
    let soc_frac = ((soc - soc_min) / soc_range).clamp(0.0, 1.0);
    let horizon = 96usize.min(challenge.num_steps.saturating_sub(t));
    if horizon <= 1 {
      return 0.0;
    }
    let end = (t + horizon).min(challenge.num_steps);
    let node = battery.node;
    let mut future_peak = f64::NEG_INFINITY;
    let mut future_trough = f64::INFINITY;
    for tt in t..end {
      let p = challenge.market.day_ahead_prices[tt][node];
      if p > future_peak { future_peak = p; }
      if p < future_trough { future_trough = p; }
    }
    if !future_peak.is_finite() || !future_trough.is_finite() {
      return 0.0;
    }
    let current_price = challenge.market.day_ahead_prices[t][node];
    let future_center = 0.5 * (future_peak + future_trough);
    let future_spread = (future_peak - future_trough).max(1e-6);
    let price_bias = ((current_price - future_center) / future_spread).clamp(-1.0, 1.0);
    let target_soc = (0.55 - 0.25 * price_bias).clamp(0.20, 0.80);
    let soc_gap = soc_frac - target_soc;
    let remaining_weight = (horizon as f64 / 96.0).clamp(0.0, 1.0);
    let bias_scale = future_peak
      * battery.efficiency_charge
      * battery.efficiency_discharge
      * remaining_weight
      * 0.08;
    bias_scale * soc_gap
  };

  let compute_unconstrained = |b: usize, t: usize, rt_prices: &[f64], socs: &[f64]| -> f64 {
    let bat = &challenge.batteries[b];
    let n = bat.node;
    let rt_p = rt_prices[n];
    let eta_c = bat.efficiency_charge;
    let eta_d = bat.efficiency_discharge;
    let cap = bat.capacity_mwh.max(1e-9);
    let (shadow_c, shadow_d) = compute_shadows(b, t, socs[b]);
    let soc_bias = soc_dispatch_bias(b, socs[b], bat.soc_min_mwh, bat.soc_max_mwh, t);

    let marginal_d = rt_p + shadow_d / eta_d + soc_bias - kappa_tx;
    let u_discharge = if marginal_d > 0.0 {
      (marginal_d * cap * cap / (2.0_f64 * kappa_deg * dt)).min(bat.power_discharge_mw)
    } else { 0.0_f64 };

    let marginal_c = shadow_c * eta_c - rt_p + soc_bias - kappa_tx;
    let u_charge_abs = if marginal_c > 0.0 {
      (marginal_c * cap * cap / (2.0_f64 * kappa_deg * dt * eta_c)).min(bat.power_charge_mw)
    } else { 0.0_f64 };

    if u_discharge * rt_p * dt > u_charge_abs * rt_p * dt {
      u_discharge
    } else {
      -u_charge_abs
    }
  };

  let eval_profit = |b: usize, u: f64, rt_prices: &[f64], socs: &[f64], t: usize| -> f64 {
    let bat = &challenge.batteries[b];
    let n = bat.node;
    let rt_p = rt_prices[n];
    let abs_u = u.abs();
    let revenue = u * rt_p * dt;
    let tx = kappa_tx * abs_u * dt;
    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
    let deg = kappa_deg * deg_base * deg_base;
    let immediate = revenue - tx - deg;

    let next_soc = if u < 0.0 {
      (socs[b] + bat.efficiency_charge * (-u) * dt).min(bat.soc_max_mwh)
    } else {
      (socs[b] - u / bat.efficiency_discharge.max(1e-9) * dt).max(bat.soc_min_mwh)
    };

    let t_next = (t + 1).min(num_steps);
    immediate + dp_value(b, t_next, next_soc)
  };

  let ternary = |f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64, iters: usize| -> (f64, f64) {
    if lo >= hi { return (lo, f(lo)); }
    for _ in 0..iters {
      let m1 = lo + (hi - lo) / 3.0_f64;
      let m2 = hi - (hi - lo) / 3.0_f64;
      if f(m1) < f(m2) { lo = m1; } else { hi = m2; }
    }
    let u = 0.5_f64 * (lo + hi);
    (u, f(u))
  };

  let solution = challenge.grid_optimize(&|ch: &Challenge, state: &State| {
    if Instant::now() > deadline {
      return Ok(vec![0.0_f64; num_batteries]);
    }

    let t = state.time_step;
    let rt = &state.rt_prices;
    let socs = &state.socs;

    let zero_action = vec![0.0_f64; num_batteries];
    let inj_base = ch.compute_total_injections(state, &zero_action);
    let flows_base = ch.network.compute_flows(&inj_base);

    let unconstrained: Vec<f64> = (0..num_batteries)
      .map(|b| compute_unconstrained(b, t + 1, rt, socs))
      .collect();

    // Clip unconstrained to action bounds
    let clipped: Vec<f64> = (0..num_batteries)
      .map(|b| {
        let (lo, hi) = state.action_bounds[b];
        unconstrained[b].max(lo).min(hi)
      })
      .collect();

    // Compute profit density for ordering
    let profit_density: Vec<(usize, f64)> = (0..num_batteries)
      .map(|b| {
        let u = clipped[b];
        if u.abs() < 1e-9 { return (b, 0.0_f64); }
        let v_curr = eval_profit(b, u, rt, socs, t);
        let v_zero = eval_profit(b, 0.0_f64, rt, socs, t);
        let denom = u.abs().max(1.0_f64);
        (b, (v_curr - v_zero).max(0.0) / denom)
      })
      .collect();

    // === Greedy initialization ===
    let mut order_greedy: Vec<usize> = profit_density.iter()
      .filter(|&&(_, d)| d > 0.0)
      .map(|&(b, _)| b)
      .collect();
    order_greedy.sort_by(|&a, &b| {
      let da = profit_density[a].1;
      let db = profit_density[b].1;
      db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
    // Append zero-density batteries
    for &(b, d) in &profit_density {
      if d <= 0.0 && !order_greedy.contains(&b) {
        order_greedy.push(b);
      }
    }

    // === Zero initialization ===
    let init_zero: Vec<f64> = vec![0.0_f64; num_batteries];

    // === Reversed-priority initialization ===
    let mut order_rev: Vec<usize> = order_greedy.clone();
    order_rev.reverse();

    // === Greedy dispatch with flow-room accounting ===
    let dispatch_greedy = |order: &[usize]| -> Vec<f64> {
      let mut actions = vec![0.0_f64; num_batteries];
      let mut line_flows: Vec<f64> = flows_base.clone();

      for &b in order {
        let u_want = clipped[b];
        if u_want.abs() < 1e-9 { continue; }

        let mut u_max_feasible = u_want;
        for &(l, p) in &b_lines[b] {
          if p.abs() < 1e-9 { continue; }
          let limit = ch.network.flow_limits[l];
          let f_other = line_flows[l];
          let b1 = (-limit - f_other) / p;
          let b2 = (limit - f_other) / p;
          let (u_lo_line, u_hi_line) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
          if u_want > 0.0 {
            u_max_feasible = u_max_feasible.min(u_hi_line);
          } else {
            u_max_feasible = u_max_feasible.max(u_lo_line);
          };
        }

        let (lo, hi) = state.action_bounds[b];
        u_max_feasible = u_max_feasible.max(lo).min(hi);

        let final_u = if u_want > 0.0 { u_max_feasible.max(0.0_f64) } else { u_max_feasible.min(0.0_f64) };
        actions[b] = final_u;

        for &(l, p) in &b_lines[b] {
          line_flows[l] += p * final_u;
        }
      }
      actions
    };

    let init_greedy = dispatch_greedy(&order_greedy);
    let init_reversed = dispatch_greedy(&order_rev);

    // Evaluate candidates from greedy dispatch
    let total_profit = |actions: &[f64]| -> f64 {
      let mut total = 0.0_f64;
      for b in 0..num_batteries {
        total += eval_profit(b, actions[b], rt, socs, t);
      }
      total
    };

    let pg = total_profit(&init_greedy);
    let pr = total_profit(&init_reversed);
    let pz = total_profit(&init_zero);

    let mut best_init = if pg >= pr && pg >= pz {
      init_greedy
    } else if pr >= pz {
      init_reversed
    } else {
      init_zero
    };

    // === Joint Projected Gradient Descent refinement ===
    // Optimizes all batteries simultaneously, avoiding coordinate descent ordering bias
    let mut actions = best_init.clone();
    let mut flows: Vec<f64> = flows_base.clone();
    for b in 0..num_batteries {
      for &(l, p) in &b_lines[b] {
        flows[l] += p * actions[b];
      }
    }

    let pgd_iters = asca_iters * 2; // more iters since joint optimization converges faster
    let mut pgd_step = 0.5_f64;

    for _iter in 0..pgd_iters {
      // Compute gradient (marginal profit) for each battery
      let mut gradient = vec![0.0_f64; num_batteries];
      for b in 0..num_batteries {
        let eps = 0.1_f64;
        let (lo_b, hi_b) = state.action_bounds[b];
        let v_base = eval_profit(b, actions[b], rt, socs, t);

        let u_up = (actions[b] + eps).min(hi_b);
        let v_up = eval_profit(b, u_up, rt, socs, t);

        let u_dn = (actions[b] - eps).max(lo_b);
        let v_dn = eval_profit(b, u_dn, rt, socs, t);

        if u_up > actions[b] && u_dn < actions[b] {
          gradient[b] = (v_up - v_dn) / (u_up - u_dn);
        } else if u_up > actions[b] {
          gradient[b] = (v_up - v_base) / (u_up - actions[b]);
        } else if u_dn < actions[b] {
          gradient[b] = (v_base - v_dn) / (actions[b] - u_dn);
        }
      }

      // Gradient step
      let mut proposed = vec![0.0_f64; num_batteries];
      for b in 0..num_batteries {
        proposed[b] = (actions[b] + pgd_step * gradient[b])
          .max(state.action_bounds[b].0)
          .min(state.action_bounds[b].1);
      }

      // Project onto flow constraints: iteratively fix violated lines
      let mut proj = proposed.clone();
      for _proj_iter in 0..12 {
        let mut test_flows = flows_base.clone();
        for b in 0..num_batteries {
          for &(l, p) in &b_lines[b] {
            test_flows[l] += p * proj[b];
          }
        }

        let mut any_violated = false;
        for l in 0..num_lines {
          let limit = ch.network.flow_limits[l];
          let flow = test_flows[l];
          if flow.abs() > limit + 1e-9 {
            any_violated = true;
            let target = if flow > 0.0 { limit } else { -limit };
            let excess = flow - target;
            let ptdf_norm_sq: f64 = (0..num_batteries)
              .map(|b| {
                let p = challenge.network.ptdf[l][challenge.batteries[b].node]
                  - challenge.network.ptdf[l][slack];
                p * p
              })
              .sum();
            if ptdf_norm_sq > 1e-15 {
              let lambda = excess / ptdf_norm_sq;
              for b in 0..num_batteries {
                let p = challenge.network.ptdf[l][challenge.batteries[b].node]
                  - challenge.network.ptdf[l][slack];
                proj[b] -= lambda * p;
              }
            }
          }
        }
        // Re-clip to box after flow projection
        for b in 0..num_batteries {
          proj[b] = proj[b].max(state.action_bounds[b].0).min(state.action_bounds[b].1);
        }
        if !any_violated { break; }
      }

      // Verify feasibility of projected point
      let mut check_flows = flows_base.clone();
      for b in 0..num_batteries {
        for &(l, p) in &b_lines[b] {
          check_flows[l] += p * proj[b];
        }
      }
      let feasible = (0..num_lines).all(|l| check_flows[l].abs() <= ch.network.flow_limits[l] + 1e-6);

      let new_profit = total_profit(&proj);
      let old_profit = total_profit(&actions);

      if feasible && new_profit > old_profit {
        // Accept and update flows
        for b in 0..num_batteries {
          let delta = proj[b] - actions[b];
          for &(l, p) in &b_lines[b] {
            flows[l] += p * delta;
          }
        }
        actions = proj;
        pgd_step = (pgd_step * 1.2).min(2.0); // increase step size on success
      } else {
        pgd_step *= 0.5; // decrease step size on failure
        if pgd_step < 1e-6 { break; }
      }
    }

    // === Final coordinate descent sweep (ASCA) for fine-tuning ===
    let cd_sweeps = if asca_iters >= 6 { 3 } else { asca_iters / 2 };
    for _sweep in 0..cd_sweeps {
      for b in 0..num_batteries {
        let (mut u_lo, mut u_hi) = state.action_bounds[b];

        for &(l, p) in &b_lines[b] {
          if p.abs() < 1e-9 { continue; }
          let limit = ch.network.flow_limits[l];
          let f_other = flows[l] - p * actions[b];
          let b1 = (-limit - f_other) / p;
          let b2 = (limit - f_other) / p;
          let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
          if lo > u_lo { u_lo = lo; }
          if hi < u_hi { u_hi = hi; }
        }

        if u_lo > u_hi { u_lo = actions[b]; u_hi = actions[b]; }
        u_lo = u_lo.min(actions[b]);
        u_hi = u_hi.max(actions[b]);

        let mut best_u = actions[b];
        let mut best_v = eval_profit(b, best_u, rt, socs, t);

        let v0 = eval_profit(b, 0.0_f64, rt, socs, t);
        if u_lo <= 0.0 && 0.0 <= u_hi && v0 > best_v { best_v = v0; best_u = 0.0_f64; }

        if u_lo < 0.0 {
          let lo = u_lo;
          let hi = 0.0_f64.min(u_hi);
          if lo < hi {
            let (u, v) = ternary(&|u| eval_profit(b, u, rt, socs, t), lo, hi, 12);
            if v > best_v { best_v = v; best_u = u; }
          }
        }

        if u_hi > 0.0 {
          let lo = 0.0_f64.max(u_lo);
          let hi = u_hi;
          if lo < hi {
            let (u, v) = ternary(&|u| eval_profit(b, u, rt, socs, t), lo, hi, 12);
            if v > best_v { best_v = v; best_u = u; }
          }
        }

        let delta = best_u - actions[b];
        if delta.abs() > 1e-6 {
          actions[b] = best_u;
          for &(l, p) in &b_lines[b] { flows[l] += p * delta; }
        }
      }
    }

    let mut feasible_actions = actions;

    // Final safety check: verify flows
    let inj = ch.compute_total_injections(state, &feasible_actions);
    let final_flows = ch.network.compute_flows(&inj);
    if ch.network.verify_flows(&final_flows).is_err() {
      let mut beta = 1.0_f64;
      for l in 0..num_lines {
        let limit = ch.network.flow_limits[l];
        let total = final_flows[l].abs();
        if total > limit && limit > 0.0 {
          let candidate = limit / total;
          if candidate < beta { beta = candidate; }
        }
      }
      beta = beta * 0.98;
      for b in 0..num_batteries {
        feasible_actions[b] *= beta;
        let (lo, hi) = state.action_bounds[b];
        feasible_actions[b] = feasible_actions[b].max(lo).min(hi);
      }
      let inj2 = ch.compute_total_injections(state, &feasible_actions);
      let flows2 = ch.network.compute_flows(&inj2);
      if ch.network.verify_flows(&flows2).is_err() {
        return Ok(vec![0.0_f64; num_batteries]);
      }
    }

    Ok(feasible_actions)
  })?;

  save_solution(&solution)?;
  Ok(())
}

pub fn policy(_challenge: &Challenge, _state: &State) -> Result<Vec<f64>> {
  Ok(vec![0.0_f64; 1])
}

pub fn help() {
}