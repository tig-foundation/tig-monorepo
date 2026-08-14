# Energy Arbitrage

## What You Are Solving

You control a fleet of batteries placed across an electrical grid. At each 15-minute time step, your policy decides how much to charge or discharge each battery. You earn money by buying cheap energy and selling it when prices are high. Your total profit must beat a baseline policy to score positively.

The core difficulty is the combination of:
1. **Unknown future prices** — real-time prices are stochastic and revealed one step at a time.
2. **Network flow constraints** — battery actions affect line flows; violating a line limit makes the step invalid and terminates the rollout.
3. **Battery physics** — efficiency losses, SOC limits, and power limits couple decisions across time.


## Required Function Signature

```rust
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    // TODO: implement your policy here
    Err(anyhow!("Not implemented"))
}
```

`grid_optimize` runs the full episode by calling your policy once per time step. **It may only be called once per challenge instance.** If your policy returns `Err`, the rollout terminates immediately and the solution is invalid.


## Key Types

### `Challenge` — static problem data (available throughout the episode)

```
challenge.num_steps          : usize        — total time steps H (96 or 192)
challenge.num_batteries      : usize        — number of batteries m
challenge.network            : Network      — grid topology, PTDFs, flow limits
challenge.batteries          : Vec<Battery> — battery specs (see below)
challenge.exogenous_injections: Vec<Vec<f64>> — [H][n] pre-generated nodal injections (MW)
challenge.market.day_ahead_prices: Vec<Vec<f64>> — [H][n] day-ahead prices ($/MWh), fully known
```

### `State` — dynamic information revealed each step

```
state.time_step     : usize           — current step index (0-based)
state.socs          : Vec<f64>        — current state-of-charge per battery (MWh)
state.rt_prices     : Vec<f64>        — real-time nodal prices THIS step ($/MWh)
state.exogenous_injections: Vec<f64>  — exogenous nodal injections THIS step (MW)
state.action_bounds : Vec<(f64, f64)> — (u_min, u_max) per battery (MW)
state.total_profit  : f64             — cumulative profit so far ($)
```

### `Battery`

```
battery.node              : usize  — grid node where this battery is located
battery.capacity_mwh      : f64    — energy capacity Ē_b (MWh)
battery.power_charge_mw   : f64    — max charge power P̄^c_b (MW)
battery.power_discharge_mw: f64    — max discharge power P̄^d_b (MW)
battery.efficiency_charge : f64    — η^c = 0.95
battery.efficiency_discharge: f64  — η^d = 0.95
battery.soc_min_mwh       : f64    — E^min = 0.10 × Ē_b
battery.soc_max_mwh       : f64    — E^max = 0.90 × Ē_b
battery.soc_initial_mwh   : f64    — E_0 = 0.50 × Ē_b
```


## Actions

Your policy returns `Vec<f64>` of length `num_batteries`. Each element is a **signed power** value in MW:

- **Negative** → charge (battery draws from the grid)
- **Positive** → discharge (battery injects into the grid)
- **Zero** → idle

**Critical:** every action must satisfy `action_bounds[b].0 <= action[b] <= action_bounds[b].1`. Violating this causes an error and terminates the rollout. The bounds are pre-computed for you in `state.action_bounds` based on current SOCs and battery limits — just respect them.


## Profit Formula

Per step, for each battery $b$ with action $u_b$:

```
revenue   = u_b × rt_prices[battery.node] × 0.25        ($)
tx_cost   = 0.25 × |u_b| × 0.25                          ($)   [κ_tx = 0.25 $/MWh]
deg_cost  = 1.0 × (|u_b| × 0.25 / battery.capacity_mwh)²  ($)   [κ_deg=1.0, β=2.0]
profit_b  = revenue - tx_cost - deg_cost
```

Total step profit = sum over all batteries. Idle batteries (u=0) contribute nothing. The degradation cost scales with cycle depth relative to capacity, so it is small for modest actions.


## Network Constraint — The Hard Constraint

After computing total nodal injections (exogenous + battery actions, with node 0 as slack), line flows are:

```
flow[l] = sum over k of: network.ptdf[l][k] × injection[k]
```

Every line must satisfy `|flow[l]| <= network.flow_limits[l]`. **If any line is violated, the step fails and the episode ends with an error.** This is the most common cause of invalid solutions.

**How to stay feasible:**
- Use `challenge.compute_total_injections(state, &action)` to get the injection vector.
- Use `challenge.network.compute_flows(&injections)` to get line flows.
- Use `challenge.network.verify_flows(&flows)` to check feasibility before returning an action.
- Scale actions down if needed — returning zeros is always feasible (exogenous injections are pre-scaled to leave headroom).


## SOC Update (for your own planning)

After an action is applied, the SOC updates as:

```
charge_amount    = max(-u_b, 0.0) × η^c × Δt      [MWh stored]
discharge_amount = max(u_b,  0.0) / η^d × Δt      [MWh consumed from store]
new_soc = clamp(soc + charge_amount - discharge_amount, E^min, E^max)
```

Use `challenge.batteries[b].apply_action_to_soc(action, soc)` to compute this.


## Look-ahead Planning with `take_step`

For planning (e.g., simulating future scenarios, dynamic programming), use:

```rust
challenge.take_step(&state, &action, NextRTPrices::Override(your_price_forecast))
```

This validates the action and returns the next `State` without touching the hidden commitment chain used by `grid_optimize`. You can call this as many times as you like for offline planning.

RT prices are **policy-independent** — the actual prices in the real rollout are fully determined by the challenge seed before the episode starts. This means you can simulate the real rollout exactly if you know the prices at future steps. However, future RT prices are not directly accessible; you can forecast them using day-ahead prices as a guide.


## Scoring

```
quality = (total_profit - baseline_profit) / (baseline_profit + 1e-6)
```

The baseline is the better of:
1. A **greedy DA policy**: charge when current DA price is below a 3-hour look-ahead average minus a threshold; discharge when above.
2. A **conservative policy**: do nothing (zero actions every step).

Quality > 0 means you beat the baseline. The score is a fixed-point integer with 6 decimal places.


## Scenario Parameters

The `Track` passed to `generate_instance` determines the scenario. All scenarios use the same API.

| Track | Nodes | Lines | Batteries | Steps | Flow limits | Price volatility |
|-------|-------|-------|-----------|-------|-------------|-----------------|
| BASELINE | 20 | 30 | 10 | 96 | nominal | low |
| CONGESTED | 40 | 60 | 20 | 96 | tight (×0.80) | medium |
| MULTIDAY | 80 | 120 | 40 | 192 | tighter (×0.60) | medium-high |
| DENSE | 100 | 200 | 60 | 192 | tight (×0.50) | high |
| CAPSTONE | 150 | 300 | 100 | 192 | very tight (×0.40) | high |

In harder scenarios, flow constraints become more binding and price spikes are more frequent (higher jump probability, heavier Pareto tails). Fleet heterogeneity also increases — batteries vary widely in size.


## Practical Tips

- **Day-ahead prices are your best forecast.** `challenge.market.day_ahead_prices[t][node]` gives the DA price at each node and step. RT prices are noisy deviations around DA prices.
- **Check flow feasibility before returning actions.** A single flow violation terminates the rollout. When in doubt, scale actions toward zero.
- **Node 0 is the slack bus.** Its injection is set automatically to balance the system; you don't control it directly.
- **Battery heterogeneity matters.** In harder scenarios batteries span a 3× size range. Larger batteries have lower degradation cost per unit energy but the same efficiency losses.
- **The action bounds already account for SOC and power limits.** You do not need to recompute them — just stay within `state.action_bounds[b]`.
- **Zeros are always safe.** `vec![0.0; num_batteries]` is a valid action at every step if you need a fallback.
