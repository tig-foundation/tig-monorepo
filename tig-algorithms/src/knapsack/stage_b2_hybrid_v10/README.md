# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** stage_b2_hybrid_v10
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track ID | Track | Fuel Budget | Nonces/Bundle |
|----------|-------|-------------|---------------|
| T39 | n_items=1000, budget=10 | 5T | 500 |
| T40 | n_items=1000, budget=25 | 5T | 500 |
| T41 | n_items=1000, budget=5  | 5T | 500 |
| T42 | n_items=5000, budget=10 | 5T | 25  |
| T43 | n_items=5000, budget=25 | 5T | 25  |

## Algorithm Summary & Scale-Aware Hybrid Architecture (`TIG-EVO-10`)

`stage_b2_hybrid_v10` solves the TIG `c003` Quadratic Knapsack Problem via a self-contained hybrid metaheuristic combining Core-Window Dynamic Programming, Unified $(p\text{-out}, q\text{-in})$ ($1 \le p, q \le 3$) Stage B2 Multi-Swap Polish with $O(N)$ incremental contribution maintenance (`delta[k]`), Cross-Basin Path Relinking, Boundary-Window Tabu Escape, and progressive `save_solution` checkpointing:

1. **Default `N = 1,000` Fast-Polish Single-Winner + Runner-Up Relink Path (`FP3`, `solve_fast_polish_single_relink`):**
   - Collects candidate restart basins from the track-specialized Core-DP + ILS solvers (`track_t41` for `budget <= 7%`, `track_t39` for `budget <= 17%`, `track_t40` for `budget > 17%`) and immediately checkpoints the winning candidate via `save_solution`.
   - Restricts initial Unified $(p\text{-out}, q\text{-in})$ ($1 \le p, q \le 3$) Stage B2 boundary-window polish (`76 / 60 / 28 / 48`) strictly to the **single winning candidate** (`cand_states[0]`).
   - Executes lightweight core-intersection lock and bidirectional Path Relinking against the top distinct runner-up candidate already produced during the track run (`top_pair`), followed by `3` boundary-window Tabu escape bursts (`24` steps) and a final Stage B2 polish pass.
   - **Full Hybrid Fallback (`solve_full_hybrid_challenge`):** Retains the multi-basin standalone `TIG-EVO-10a` (`power_w` in `[0.80, 0.72, 0.90, 1.00]`) + top-6 candidate polish architecture as automatic fallback (and explicit `"mode": "full_hybrid"` override).
2. **Unified $(p\text{-out}, q\text{-in})$ ($1 \le p, q \le 3$) Stage B2 Boundary-Windowed Multi-Swap Polish (`polish_stage_b2_hybrid`):**
   - Restricts higher-order swap neighborhoods to a narrow boundary window (`sel_k` lowest marginal-density selected items $\cup$ `unsel_k` highest marginal-density unselected items, plus top-`slack_k` residual slack items).
   - Evaluates exact quadratic gains in $O(1)$ using pre-cached `pair_unsel` (`(u1, u2, w, gain)`) and `pair_sel` (`(r1, r2, w, loss)`) tables with exact bilinear cross-interaction corrections across `1-for-1`, `1-for-2`, `2-for-1`, `2-for-2`, `1-for-3`, `3-for-1`, `2-for-3`, and `3-for-2` moves.
3. **`N <= 600` & `N = 5,000` Scale Routing:**
   - On $N \le 600$, combines `solve_core_dp_b2_standalone_elites` with `base_evo9`, path-relinks across elites, and executes `5` boundary-window Tabu escape bursts (`28` steps).
   - On $N = 5,000$ (`T42` `10%`, `T43` `25%` capacity tracks), routes directly to `track_t42` (`budget_pct <= 17`) or `track_t43` (`budget_pct > 17`) with live `save_solution` streaming, then applies bounded `48 / 40 / 18 / 32` Stage B2 multi-swap polish.

## Head-to-Head SOTA Benchmark Telemetry (`--release`)

| Track | Instance Scale & Budget | Seeds | `knap_quality_opt_v11` Mean (`ms`) | `stage_b2_hybrid_v10` Mean (`ms`, `% Time`) | Mean Delta (`+%`) | `W / T / L` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **T39** | `N = 1,000, budget = 10%` | `50` | `151,186.66` (`528.08 ms` avg) | **`151,202.40`** (`578.28 ms` avg, `+9.51%`) | **`+15.74` (`+0.0104%`)** | **`16 / 34 / 0`** |
| **T40** | `N = 1,000, budget = 25%` | `50` | `323,872.58` (`528.08 ms` avg) | **`323,892.56`** (`578.28 ms` avg, `+9.51%`) | **`+19.98` (`+0.0062%`)** | **`23 / 27 / 0`** |
| **T41** | `N = 1,000, budget = 5%` | `20` | `68,975.80` (`311.89 ms`) | **`68,979.30`** (`334.14 ms`, `+7.13%`) | **`+3.50` (`+0.0051%`)** | **`2 / 18 / 0`** |
| **T42** | `N = 5,000, budget = 10%` | `5` | `1,830,537.60` (`1,780.62 ms`) | **`1,830,582.60`** (`1,915.42 ms`, `+7.57%`) | **`+45.00` (`+0.0025%`)** | **`2 / 3 / 0`** |
| **T43** | `N = 5,000, budget = 25%` | `5` | `4,543,761.40` (`2,921.04 ms`) | **`4,543,870.80`** (`3,093.32 ms`, `+5.90%`) | **`+109.40` (`+0.0024%`)** | **`1 / 4 / 0`** |
| **N=500** | `N = 500, 10% & 25%` (vs `knap_supreme`) | `200` | `62,914.51` (`0.87 ms`) | **`63,989.65`** (`179.94 ms`) | **`+1,075.14` (`+1.7089%`)** | **`199 / 1 / 0`** |

## References and Acknowledgments

### 1. Academic Papers
- Gallo, G., Hammer, P. L., & Simeone, B. (1980), *"Quadratic knapsack problems"*, Mathematical Programming Study, 12, 132-149.
- Glover, F. (1989), *"Tabu Search—Part I"*, ORSA Journal on Computing, 1(3), 190-206.
- Pisinger, D. (2007), *"The quadratic knapsack problem—a survey"*, Discrete Applied Mathematics, 155(5), 623-648.

### 2. Code References
- TIG baseline, `knap_supreme`, and `knap_quality_opt_v11` (TIG Open Data License / TIG Innovator Outbound Game License)

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
