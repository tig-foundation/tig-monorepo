# CLAUDE.md — TIG Monorepo (Energy Arbitrage iterate-and-optimize loop)

This repo is scoped — in this setup — to **iterative, automated improvement of a single algorithm for the `energy_arbitrage` challenge**. When the user triggers the loop (see *Trigger*), Claude repeatedly: proposes a child algorithm that improves on the current best, exposes its hyperparameters, grid-searches those hyperparameters in Docker on one track, and promotes the child if it beats the parent.

Only `energy_arbitrage` is in scope. Do not create or test algorithms for any other challenge.

---

## About the challenge

`energy_arbitrage` is a sequential decision problem, not a one-shot solver:

- An algorithm must implement `policy(challenge, state) -> Vec<f64>` and invoke `challenge.grid_optimize(&policy)` once inside `solve_challenge`. `grid_optimize` drives the policy one 15-minute step at a time across the full episode.
- Each step the policy returns a signed MW action per battery (negative = charge, positive = discharge). The action must lie in `state.action_bounds[b]` — those bounds already account for SOC and power limits.
- **Top failure mode: line-flow violations.** Total nodal injections (exogenous + battery actions) produce line flows via `challenge.network.compute_flows(...)`. If any `|flow[l]| > network.flow_limits[l]`, the whole rollout ends with an invalid solution. Use `challenge.network.verify_flows(...)` to check before returning an action, and scale toward zero if infeasible. Zero actions are always feasible.
- **Quality = (total_profit − baseline_profit) / (baseline_profit + 1e-6).** Baseline is the better of (a) a greedy DA-vs-3hr-lookahead threshold policy or (b) do-nothing. Quality > 0 means the algorithm beats both.
- Day-ahead prices (`challenge.market.day_ahead_prices`) are fully known; RT prices are revealed one step at a time. RT prices are policy-independent for a given seed, which means `challenge.take_step(&state, &action, NextRTPrices::Override(forecast))` can be used for offline look-ahead planning without touching the real rollout.

Full spec: `tig-challenges/src/energy_arbitrage/README.md`. Read it before proposing a child.

Existing algorithm(s) live in `tig-algorithms/src/energy_arbitrage/`. The current canonical seed is `energy_solver_1` — the universal cross-track starting point used by every per-track iteration loop. (Historically the seed was `energy_solverv4`; that lineage was iterated to v19 then renamed to `energy_solver_1` after a cleanup.)

---

## Compute budget — leverage proactive planning

`test_algorithm` enforces a fuel cap (500B max, 100B default). On the iteration tracks here, the threshold-style baseline runs at roughly **~3 ms per `policy(...)` call** — far below the practical budget. We explicitly allow children to use **up to ~10 seconds per nonce**, i.e., **~50 ms per `policy(...)` call on a 192-step episode** (s=multiday) and **~100 ms per call on a 96-step one** (s=baseline / s=congested). On larger tracks (s=dense, s=capstone) the same per-nonce budget gives proportionally less per-step time, but still ~30+ ms.

This unlocks **proactive planning** designs that the reactive threshold policy can't compete with on long-episode / volatile / congested tracks:

- **Forward-simulation MPC**: at each step, use `challenge.take_step(&state, &action, NextRTPrices::Override(da_forecast))` to simulate K steps ahead per candidate action, score by simulated total profit, pick the best. The challenge spec explicitly hints at this — offline lookahead doesn't pollute the real rollout because RT prices are policy-independent for a given seed. Reasonable budget: 5–20 candidates × 10–48 step horizon.
- **Per-battery dynamic programming**: backward DP with finer SOC grids (`soc_levels` 41–81), longer horizons (`dp_horizon` up to 96 on s=multiday), and finer action discretization (`action_levels` 11–21). Earlier `v13` failed at coarse settings on s=congested — finer settings on long-episode tracks should fare better.
- **Monte-carlo scenario sampling**: 50–100 future RT scenarios per step (DA + lognormal noise), compute expected profit per candidate.
- **Joint per-step QP**: solve actions across all batteries simultaneously with flow constraints baked in (instead of per-battery threshold + post-hoc softening).

When designing compute-heavy children, keep the **reactive threshold policy as a strong inner / tail / fallback policy** — it's well-tuned (s=multiday lineage tunes thresholds, lookahead, end-window, slope, profit-weight, and DA-trend) and cheap. A planning layer should pick *which action to play now*; the threshold policy can score *what happens after*. Compute-heavy designs should beat the threshold baseline on tracks where planning matters most (s=multiday with 192-step episodes, s=dense / s=capstone with tight flow regimes), and at minimum match it where it's already near-optimal (s=baseline).

Practical guardrails:
- Per-step budget is shared across all batteries — don't let an inner DP/MPC explode quadratically with `num_batteries`.
- Walk-clock target is ~5 s/nonce in normal operation; ~10 s leaves margin. If a grid point hits the fuel cap, log it.
- `take_step` with `NextRTPrices::Override` is the canonical "rollout the next K steps under a forecast" primitive — preferred over hand-rolled SOC propagation.

---

## Trigger

The user will type one of:

- `iterate on energy_arbitrage/<algorithm> [track=<track_id>]` — start a fresh run. Example: `iterate on energy_arbitrage/energy_solver_1 track=s=congested`. If `track=` is omitted, default to **`s=baseline`**. The state is loaded from `.tig-iterate/state.<track>.json` (one file per track).
- `continue iterating [track=<track_id>]` — resume from the persisted state for that track. If `track=` is omitted and only one track has unfinished work, use it; otherwise ask.
- `stop iterating` — exit the loop cleanly and print the current best for the active track.

On any of these, Claude enters the loop described in *The Loop*. Between iterations Claude prints a one-line status (`iter N: <child> scored X vs parent <parent> Y → promoted/kept`) and then starts the next iteration without waiting. The user interrupts to stop, or types `stop iterating`.

Available tracks (pick **one per loop run** and stick with it — each loop is single-track; per-track state files keep them independent):

| `track=` | Scenario |
|---|---|
| `s=baseline` | 20 nodes, 10 batteries, 96 steps, low volatility — correctness regime |
| `s=congested` | 40 nodes, 20 batteries, tight flow limits, medium volatility |
| `s=multiday` | 80 nodes, 40 batteries, 192 steps, medium-high volatility |
| `s=dense` | 100 nodes, 60 batteries, very tight flow limits, high volatility |
| `s=capstone` | 150 nodes, 100 batteries, ×0.40 flow limits, high volatility |

---

## Persistent state

Each track has its own state file at `.tig-iterate/state.<track>.json` (e.g., `state.s=baseline.json`). Schema:

```json
{
  "challenge": "energy_arbitrage",
  "track": "s=baseline",
  "stem": "energy_solver_baseline",
  "seed_algorithm": "energy_solver_1",
  "next_version": 1,
  "seed": "iterate-v1",
  "nonces": 100,
  "grid_points": 12,
  "current_best": {
    "name": "energy_solver_1",
    "hyperparameters": {"charge_threshold": 8.0, "discharge_threshold": 6.0},
    "score": 0
  },
  "history": [
    {"event": "track_init", "note": "..."},
    {"iter": 1, "child": "energy_solver_baseline_1", "best_hyperparameters": {"discharge_threshold": 5.0}, "score": 2400000, "promoted": true, "parent": "energy_solver_1"}
  ]
}
```

- `score` is `avg_quality` from `test_algorithm` (6-digit integer = 6-decimal ratio).
- `seed` is a fixed string (`iterate-v1`) so re-tests are deterministic and comparable across iterations.
- `seed_algorithm` is the universal cross-track starting point (typically `energy_solver_1`). It is NEVER renamed or deleted — every track's loop falls back to it as the canonical ancestor.
- `stem` is the **per-track child stem**, conventionally `energy_solver_<track>` (e.g., `energy_solver_baseline`, `energy_solver_congested`). Children are named `<stem>_<N>` using the per-track `next_version` counter starting at 1.
- `track` is fixed for the whole loop run. Scores are per-track — never aggregate across tracks.
- `nonces=100` and `grid_points=12` are the user-specified defaults — keep them fixed unless the user overrides.
- **Per-track initialization**: when creating or restarting a track, set `current_best = {name: <seed_algorithm>, hyperparameters: <known-good or empty>, score: 0}`. The seed algorithm is *not* tested. Its score is 0 by fiat, so the first child that produces any valid solutions automatically promotes.
- Legacy single-state runs (pre-cleanup) are preserved in `.tig-iterate/state.legacy.json` for reference.

Also append a one-line summary per iteration to `.tig-iterate/log.md` for quick scanning. The log is shared across tracks; section it by track.

---

## Docker execution model

All `build_algorithm` / `test_algorithm` calls happen inside the `energy_arbitrage` `dev` container. Use one long-lived container per session — per-command `docker run` is too slow when 300 nonces/iter.

```bash
# at loop start (check `docker ps --filter name=tig-dev-energy_arbitrage` first; reuse if running)
docker run -d --name tig-dev-energy_arbitrage \
  -v $(pwd):/app -w /app \
  ghcr.io/tig-foundation/tig-monorepo/energy_arbitrage/dev:latest \
  sleep infinity

# per build / test
docker exec tig-dev-energy_arbitrage build_algorithm <name>
docker exec tig-dev-energy_arbitrage test_algorithm <name> <track> <hyperparams_json> \
  --nonces 25 --seed iterate-v1 --workers $(nproc)
```

On `stop iterating` (or if the user asks), `docker rm -f tig-dev-energy_arbitrage`. Don't tear it down between iterations.

---

## The loop

Each iteration:

### 1. Load parent
Read `current_best` from state. The parent always has a recorded score: either from the last iteration that promoted it, or `0` on the very first iteration (the seed algorithm is never tested — see *First-run initialization* above).

### 2. Propose a child
Read the current parent's source (`tig-algorithms/src/energy_arbitrage/<parent>.rs` or `.../<parent>/mod.rs`). Write a child at `tig-algorithms/src/energy_arbitrage/<stem>_<N>.rs` where `N = state.next_version`; increment `state.next_version` after writing. Register it in `tig-algorithms/src/energy_arbitrage/mod.rs` with `pub mod <stem>_<N>;`.

The child must:
- Make a concrete change aimed at improvement. For this challenge, useful axes to explore include: threshold levels for charge/discharge vs. DA forecast, look-ahead window size, DA-vs-RT blending weights, SOC target curves, flow-aware action scaling, per-battery heterogeneity handling, scenario-based branching on congestion, etc. Document the change in a one-line comment at the top of the file.
- **Preserve the required structure**: `solve_challenge` calls `challenge.grid_optimize(&policy)` exactly once and then `save_solution(&solution)`. The real work lives in `policy(challenge, state) -> Result<Vec<f64>>`.
- **Stay flow-feasible.** Every action returned from `policy` must pass `challenge.network.verify_flows(...)`. If unsure, project toward zero. A single infeasible step fails the whole nonce.
- **Expose tunable hyperparameters** via the `Hyperparameters` struct — even if the parent had none, introduce at least one. Prefer 1–2 hyperparameters (keeps the grid dense at 12 points); 3 is the ceiling.
- **Document ranges in `help()`** in a machine-readable block that Claude can parse back in step 4. Use this exact format:
  ```rust
  pub fn help() {
      println!("HYPERPARAMETERS:");
      println!("  discharge_threshold: float range=[1.0, 10.0] default=4.0  # $/MWh above 3-hr DA mean");
      println!("  lookahead_steps:     int   range=[4, 24]     default=12   # 15-min steps");
      // ...any additional human-readable notes below...
  }
  ```
  `range=[lo, hi]` is inclusive. `int` → integer steps; `float` → linear steps (use `scale=log` for log-spaced ranges when values span orders of magnitude).
- Follow the rules from `template.rs`: no `#[cfg(test)]`, no tests, call `save_solution` (just once, after `grid_optimize` returns), seed `SmallRng` from `challenge.seed`, use `seeded_hasher` for `HashMap`/`HashSet`.

### 3. Build the child
```
docker exec tig-dev-energy_arbitrage build_algorithm <stem>_<N>
```
If build fails, read the error, fix the child (do **not** touch the parent), rebuild. After 3 consecutive build failures on the same child, abandon it — revert the `mod.rs` entry, delete the file, and generate a different child from the same parent. (Do not increment `next_version` on abandon; reuse the slot for the replacement.)

### 4. Grid-search hyperparameters (12 points)
Parse the `help()` ranges (re-read the file you just wrote — don't trust memory). Build a grid of ≈12 points:

- 1 hyperparameter → 12 evenly-spaced points.
- 2 hyperparameters → 4×3 or 3×4 grid.
- 3 hyperparameters → 2×2×3 grid.
- Rounding: aim for ≤12 points total; fewer is fine if the grid divides unevenly — never exceed 12.
- Integers: round and deduplicate.
- Always include the declared `default` as one of the points (add it, then drop the nearest spaced point if that overshoots 12).

For each grid point, run:
```
docker exec tig-dev-energy_arbitrage test_algorithm <stem>_<N> <track> '<json>' \
  --nonces 100 --seed iterate-v1 --workers $(nproc)
```
Parse the final `avg_quality: N` from stdout. If `test_algorithm` reports any invalid solutions, that grid point scores `0` (do not pass `--ignore-invalid`; an invalid run usually means a flow violation, which is a real regression). If the process crashes or times out, same — score `0` and continue.

The child's score is the **max** `avg_quality` across the 12 points; record the winning hyperparameter dict alongside.

### 5. Compare and promote
- If `child.score > parent.score` → **promote and delete the parent — but never delete the universal seed `seed_algorithm` (e.g., `energy_solver_1`)**:
  1. Set `current_best = {name: <stem>_<N>, hyperparameters: <winning>, score: <child.score>}` in the active track's state file.
  2. If parent is **not** the universal `seed_algorithm`: delete the defeated parent's source, remove its `pub mod ...;` line from `mod.rs`, and delete its built artifact.
  3. If parent **is** the universal `seed_algorithm`: leave it intact (other tracks' loops also depend on it).
- Otherwise → keep the parent. Leave the losing child's source and `mod.rs` entry in place (cheap, and it documents what was tried).

Append an entry to `history` and a line to `.tig-iterate/log.md` in both cases.

### 6. Loop
Go to step 1. No prompt between iterations — the single user command kicks off the whole process; the loop runs until the user interrupts or types `stop iterating`.

---

## Rules Claude must not break

- **Work only on `energy_arbitrage`.** Never touch `tig-algorithms/src/<other_challenge>/` or spawn containers for other challenges.
- **Never modify an in-flight algorithm file.** Only add new `<stem>_<N>` children, or delete a defeated parent on promotion. Never hand-edit an existing `.rs` to "try something" — every change is a new child so the grid-search result is attributable.
- **Never delete or modify the universal seed (`energy_solver_1`).** Every track's loop falls back to it as the canonical ancestor. A losing child can be deleted; the seed cannot.
- **Never change the seed, nonces, grid size, or track mid-run** — it invalidates all prior comparisons for that track. If the user changes any of them, reset that track's `current_best.score` to `0` and start over from the current best algorithm. Other tracks' state files are unaffected.
- **Never mix tracks within one loop run.** Each loop run uses exactly one `state.<track>.json` file and one track. Scores from different tracks are not comparable.
- **Never pass `--ignore-invalid`.** An invalid solution is a real regression — the grid point scores 0.
- **Never submit anything.** Submissions cost 10 TIG and are final; this loop is local-only.
- Don't add cleanup/refactor changes beyond the child's specific improvement — each iteration should have exactly one testable hypothesis.

---

## Quick repo orientation (for the proposal step)

- `tig-algorithms/src/energy_arbitrage/template.rs` — canonical shape: `Hyperparameters`, `help()`, `solve_challenge` (which calls `challenge.grid_optimize(&policy)`).
- `tig-algorithms/src/energy_arbitrage/mod.rs` — add `pub mod <name>;` here.
- `tig-algorithms/src/energy_arbitrage/energy_solver_1.rs` — universal seed, used as the starting parent for every per-track iteration loop. Read this before proposing the first child on any track.
- `tig-challenges/src/energy_arbitrage/README.md` — full challenge spec (scoring, signatures, network/battery types, tips).
- `tig-challenges/src/energy_arbitrage/mod.rs` — `Challenge`, `State`, `grid_optimize`, `take_step`, and the `Track { s: Scenario }` definition.
- `tig-challenges/src/energy_arbitrage/scenarios.rs` — the five scenario configs (BASELINE/CONGESTED/MULTIDAY/DENSE/CAPSTONE).
- `tig-challenges/src/energy_arbitrage/baselines/` — reference policies (including the greedy DA baseline used in scoring); useful for guardrails and sanity checks.
- `scripts/` — `build_algorithm`, `test_algorithm`, `list_algorithms`, `download_algorithm`, `help_algorithm` (all on `PATH` inside the dev container; `CHALLENGE=energy_arbitrage` is pre-set).
- `tig-algorithms/lib/energy_arbitrage/<arch>/<name>.so` — build output. Presence = built.
- Max fuel per instance is 500 billion; `test_algorithm` defaults to 100B which is fine for the loop. If a grid point is hitting the fuel cap, note it in the log — the child likely needs to move expensive precomputation out of the per-step `policy` call.
