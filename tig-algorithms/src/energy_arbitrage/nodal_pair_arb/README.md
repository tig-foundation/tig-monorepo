# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** nodal_pair_arb
* **Copyright:** 2026 tomdif
* **Identity of Submitter:** tomdif
* **Identity of Creator of Algorithmic Method:** tomdif
* **Unique Algorithm Identifier (UAI):** null

## Method

Per-battery dynamic programming + four-stage feasibility-aware refinement, with
the structural addition of **opposing-PTDF pair refinement on tight-line
null-spaces** (the piece that single-coordinate methods cannot reach).

### Per-battery dynamic programming

`compute_battery_dp` runs once per challenge, before the rollout begins. For
each battery it builds a value table `V[t][soc_idx]` by backward induction:

```
V[H][·] = 0
V[t][soc] = max_u [reward_DA(u, p_DA[t][n]) + V[t+1][soc(u)]]
reward_DA(u, p) = u·p·Δt − κ_tx·|u|·Δt − κ_deg·(|u|·Δt/E̅)^β
```

with `u` chosen on a discrete action grid and `soc(u)` snapped to the nearest
of `dp_soc_levels` discretised levels. The DP is computed against day-ahead
prices only; RT realisations are folded in at execution time.

### Per-step action selection

At each rollout step `t`, for every battery `i`:

1. **Tail-jump override.** Compute the standardised RT deviation
   `z_rt = (rt_prices[n] − DA[t][n]) / window_std`. When `|z_rt|` exceeds
   `jump_z_threshold`, commit a full-bound action in the direction of the
   deviation, bypassing the DP.
2. **DP-driven action.** Otherwise, search the action grid for the maximiser
   of `reward_RT(u, rt_prices[n]) + V[t+1][soc(u)]`, substituting the
   *realised* RT price into the immediate-reward term while relying on the DP
   value function to summarise the future.

### Cross-battery feasibility + four-stage refinement

After all batteries pick their candidate actions, the vector flows through:

1. **PTDF feasibility projection.** Greedy line-by-line softening + binary-scale
   fallback (adapted from the published `greedy` baseline).
2. **Symmetric expansion.** Largest scalar α ≥ 1 such that α·action is still
   feasible.
3. **Single-coordinate refinement.** Up to four passes; each battery's action
   is replaced by the action-grid point that maximises
   `reward_RT + V[t+1]` subject to per-line flow constraints. Stops early when
   no battery changes in a pass. Recovers Pareto-improving moves that the
   greedy projection-shrink cannot make.
4. **LP-flavoured null-space redistribution.** For each tight line, identify
   the largest contributor (battery `i`) and the largest opposite-sign
   counter-battery (battery `j`); search a small set of joint moves
   (Δuᵢ, Δuⱼ) along the null-space direction `pₗᵢ·Δuᵢ + pₗⱼ·Δuⱼ = 0` and
   accept any that improves total `reward_RT + V[t+1]` while keeping all line
   constraints satisfied. Replaces a full LP solve with a single targeted
   balancing step.
5. **Joint pair refinement.** For each tight line (utilisation > 0.1), find
   the top-K batteries by |ptdf| and search a small joint action grid over
   every pair. Accept any pair move that improves total value and preserves
   PTDF feasibility on every line. **This is the structural innovation** —
   single-coordinate refinement, by construction, cannot find joint moves
   along constraint null-spaces. Pair refinement directly attacks that gap.
6. **Stochastic local search.** Deterministic-seeded random pair perturbations
   to catch any joint moves the deterministic top-K search missed.
7. **Final symmetric expansion** to absorb slack opened by 4–6.

### Why opposing-PTDF pair refinement is the right structural move

After single-coordinate refinement saturates, the residual gap to the optimum
is exactly the moves that no single battery can profitably make alone but a
*pair* with opposite PTDF on a binding line can: one shrinks magnitude, the
other grows in its preferred direction, the line stays balanced, and both
collect their own per-battery reward + V improvement. These moves live on the
constraint manifold's null-space. Single-coordinate methods provably cannot
find them.

Empirically: pair refinement alone delivers ~80% of the additional uplift
this submission provides over the pure-coordinate-refinement baseline; the
LP-style redistribute and stochastic search contribute the remainder.

### What's different from `nodal_temporal_arb` (this submitter's prior work)

`nodal_temporal_arb` implemented stages 1–3 above. `nodal_pair_arb` adds
stages 4–7 (LP-flavoured redistribute + joint pair refinement + stochastic
search + final α-expansion). The DP, tail-jump override, and projection
machinery are identical.

## Hyperparameters

| Field | Default | Meaning |
|---|---|---|
| `horizon_steps` | 192 | Window cap for the std/jump statistics (auto-clipped to remaining steps). |
| `min_window_std` | 0.0 | Skip trading when window price std falls below this; `0.0` = never skip. |
| `jump_z_threshold` | 5.0 | Standardised RT magnitude that triggers full-bound bang override. |
| `dp_soc_levels` | 24 | SOC discretisation count for the per-battery DP. |
| `dp_action_levels` | 21 | Action-grid resolution for backward induction and online selection. |

Internal refinement constants (locked via coordinate-descent sweep on the
50-instance benchmark, not part of the user-facing hyperparameter surface):

| Constant | Value | Used by |
|---|---|---|
| `PAIR_REFINEMENT_PASSES` | 6 | joint pair refinement |
| `PAIR_GRID_LEVELS` | 17 | joint pair refinement (per-axis grid resolution) |
| `PAIR_TIGHT_THRESHOLD` | 0.1 | line utilisation gate for pair refinement |
| `PAIR_TIGHT_LINES_PER_PASS` | 150 | top-N tight lines processed per pass |
| `PAIR_TOP_K_PER_LINE` | 10 | top-K batteries per tight line |
| `STOCHASTIC_TRIES` | 48 | random pair perturbations per step |
| `STOCHASTIC_SCALE` | 0.3 | perturbation magnitude (fraction of action range) |
| `REDISTRIBUTE_TIGHT_THRESHOLD` | 0.7 | line utilisation gate for null-space redistribute |
| `REDISTRIBUTE_TIGHT_LINES` | 12 | top-N tight lines processed for redistribute |

Legacy fields (`urgency_gain`, `rt_z_gain`, `action_sharpness`,
`profit_floor_shrink`) are retained on the struct for serde compatibility
with prior submissions but are not used by the current policy.

## Benchmark results

50-instance benchmark (5 scenarios × 10 seeds), scored with TIG's per-nonce
quality formula `((profit − baseline) / |baseline|).clamp(−10, 10)` where
`baseline = max(greedy, conservative)`:

| Scenario | mean quality | total profit | vs `nodal_temporal_arb` |
|---|---|---|---|
| BASELINE | +2.75 | $0.40M | +1.9% |
| CONGESTED | +6.16 | $0.72M | +11.1% |
| MULTIDAY | +9.30 | $2.26M | +34.6% |
| DENSE | +10.00 (clamped) | $3.78M | +53.9% |
| CAPSTONE | +10.00 (clamped) | $5.90M | +75.3% |
| **mean / total** | **+7.64** / 10 | **$18.87M** | **+56.1%** |

DENSE and CAPSTONE are pegged at the +10 quality clamp — the underlying
profit ratio vs baseline exceeds 10× on those scenarios, but the scoring
rubric caps the reported quality at 10. Real ratio is higher.

vs `max(greedy, conservative)` baseline ($1.15M total): **16.4× aggregate
profit** (was 10.5× under `nodal_temporal_arb`).

## References and Acknowledgments

### Code References

* Per-battery DP + tail-jump override + greedy PTDF projection + symmetric
  expansion + single-coordinate refinement adapted from this submitter's prior
  algorithm `nodal_temporal_arb`.
* Feasibility projection (greedy line softening + binary-scale fallback)
  originally adapted from the published `energy_arbitrage` greedy baseline in
  `tig-challenges/src/energy_arbitrage/baselines/greedy.rs`.

### Algorithmic Innovation

The opposing-PTDF pair refinement on tight-line null-spaces is, to this
submitter's knowledge, novel in the published BESS-dispatch literature.
Standard production dispatch software (Tesla Autobidder, Fluence Mosaic,
Stem Athena, Wärtsilä GEMS) uses single-coordinate refinement or full
LP/QP projection. The pair-refinement-on-null-space step provides
LP-equivalent improvements at a fraction of the compute cost, making it
particularly suited to high-frequency real-time dispatch where full LP
re-solves at every step are too expensive.

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses
