# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** axiom_pulsar
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track ID | Track | Fuel Budget | Nonces/Bundle |
|----------|-------|-------------|---------------|
| T39 | n_items=1000, budget=10 | 5T | 1000 |
| T40 | n_items=1000, budget=25 | 5T | 1000 |
| T41 | n_items=1000, budget=5  | 5T | 1000 |
| T42 | n_items=5000, budget=10 | 5T | 50   |
| T43 | n_items=5000, budget=25 | 5T | 50   |

## Method Overview

`axiom_pulsar` solves the quadratic knapsack problem (`c003`) via scale-aware track dispatch, fused dual-trajectory seed exploration with shared DP/VND memos, bidirectional elite path relinking, and Zobrist-guarded iterated local search (`ILS`) with quadratic rank ejection and tournament boundary reconstruction:

1. **Scale-Aware Track Dispatch & Velocity Exits (`SwitchConfig::DEFAULT`)**:
   - **T41 (`N = 1,000, budget = 5%`)**: Executes `track1` nominal and single undivided `CoreBnb` passes (`satchel_core = 1`), saving the superior solution and exiting before full $N \times N$ matrix flattening.
   - **T39 (`N = 1,000, budget = 10%`)**: Probes with bounded `track_t39` perturbation rounds and escalates adaptively (`t39_adaptive = 2`) with selective 3-burst tabu escape only when the probe improves upon the primary trajectory.
   - **T40 (`N = 1,000, budget = 25%`)**: Routes through `track3` HPF + `CoreBnb` with single-pass 3-slice incremental contribution updates.
   - **T42 (`N = 5,000, budget = 10%`)**: Runs `track4::solve_fused_dual` (`t42_dedup = 3`, `t42_elites = 4`), sharing `DP_MEMO`, `DP_STEP_MEMO`, and `VND_PATHS` across both `hub_pair_n = 4` and `hub_pair_n = 0` trajectories.
   - **T43 (`N = 5,000, budget = 25%`)**: Runs `track5` (`t43_dedup = 5`, `t43_elites = 2`) with stall-capped restart trajectories and bidirectional elite path relinking (`recombine_and_relink_elites_lookahead`).
2. **Restricted-Core Linearization (T42/T43)**:
   - Uses the best preliminary elite as the selected core and ranks feasible starts by `v + alpha * selected_interactions + beta * positive_outside_interactions`, with two `(alpha, beta)` settings: `(1, 0)` and `(.5, .1)`.
   - Climbs and deduplicates against existing and newly climbed states before bounded Stage B2 polish, then injects distinct candidates before elite relinking and optional closure CoreBnb. The original track solution remains a candidate.
3. **Deep Iterated Local Search (`iterated_local_search_refine`, `ils_rounds_t42 = 2`, `ils_rounds_t43 = 2`)**:
   - Perturbs the incumbent via quadratic rank-biased ejection of low-synergy selected items and tournament boundary reconstruction into persistent thread-local `ILS_ELITE_BUFFERS`.
   - Climbs via hoisted-row `fast_1to1_and_slack_climb` and residual slack filling (`best_fill_by_weight`), applying bounded Stage B2 `(p-out, q-in)` multi-swap polish within striking distance of the incumbent while preventing cycles with a Zobrist hash history table.
   - Retains the original rounds, then runs one synergy-clique round: choose a lower/middle contribution-density seed, eject its strongest positive selected neighbors together, and tournament-reconstruct against the surviving core. Removed items are excluded during reconstruction only; unrestricted climbing can recover them. Only strict objective improvements replace the incumbent.

## Validation Summary

- **Basin-breaker local-harness candidate (27 Oct 2026; experimental)**: Active-36 objective `23,202,149` (`7W / 29T / 0L`, `+5,415` versus the current chihuahua harness); OOS-80 objective `78,584,097` (`20W / 60T / 0L`, `+5,301` versus chihuahua). The OOS total is `+59` above the supplied pre-change baseline. These are development self-checks, not platform results or independent acceptance.
- **Fresh attribution**: the preserved baseline compiled against the same current harness reproduces `7W / 29T / 0L` on Active-36 and `20W / 60T / 0L` on OOS-80, contradicting the reported full tie. The candidate improves only T43 seed 109 (`5,228,815` to `5,228,874`); all other objectives match that baseline. The combined operators therefore yield `1W / 115T / 0L` against the pre-change solver, without increasing its count of wins against chihuahua.
- **Timing target unmet**: candidate T42 means were `2,824.2 ms` on Active-36 and `2,712.1 ms` on OOS-80, exceeding `800 ms`. OOS timing partly overlapped compilation of the attribution harness. The same-harness preserved baseline also exceeds the target on Active-36 (`1,664.6 ms`); the earlier quick-run T42 measurement sampled only seed 1. This candidate is not ready for submission under the requested timing criterion.
- **Portability follow-up**: `exact_mask`, its breakpoint/HPF consumers, and the T42/T43 `fastmask` fields, accessors and atomic storage use `u64`, preserving high bits on 32-bit targets. Legacy HPF raw-pointer borrows in tracks 1–3 are explicit for current Rust. The switchboard now builds the actual c003 ZIP as a WASM library, checks its `entry_point` and `help` function exports, and runs native unit tests. This is compilation and native-test evidence; WASM execution and fuel compliance are not certified by this gate.

- **36-Seed Active Gauntlet (`T39..T41` seeds `1..=10`, `T42..T43` seeds `1..=3`)**: `TotalObj36 = 23,202,149` (`6W / 30T / 0L` vs `packrun`, `7W / 29T / 0L` vs `satchel`).
- **80-Seed Out-of-Sample Gauntlet (`OOS-80`: `T39..T41` seeds `100..=119`, `T42..T43` seeds `100..=109`)**: `TotalObj80 = 78,584,038` (`9W / 71T / 0L` vs `packrun` `+2,401` pts; `19W / 61T / 0L` vs `satchel` `+8,273` pts).

## References and Acknowledgments

### Academic Papers
- G. Gallo, P. L. Hammer, B. Simeone, *"Quadratic knapsack problems"*, Mathematical Programming Study, 12, 132-149, 1980.
- A. Caprara, D. Pisinger, P. Toth, *"Exact Solution of the Quadratic Knapsack Problem"*, INFORMS Journal on Computing, 1999.
- H. R. Lourenço, O. C. Martin, T. Stützle, *"Iterated Local Search"*, Handbook of Metaheuristics, 2003.
- F. Glover, *"Tabu Search—Part I"*, ORSA Journal on Computing, 1(3), 190-206, 1989.

### Code References
- TIG `c003` public submissions (`satchel` / `c003_a154`, `packrun` / `c003_a156`, `stage_b2_hybrid_v10` / `c003_a158`) under the TIG Open Data License and TIG Innovator Outbound Game License.

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
