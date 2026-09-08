# Iterate log — energy_arbitrage

## track=s=baseline (nonces=25)
iter 1: energy_solverv4_v5 scored 2152478 vs parent energy_solverv4 0 → promoted (best={discharge_threshold:4.0, lookahead_steps:24})
iter 2: energy_solverv4_v6 scored 2080679 vs parent energy_solverv4_v5 2152478 → kept (best={band_pct:0.30, lookahead_steps:48})
iter 3: energy_solverv4_v7 scored 2326526 vs parent energy_solverv4_v5 2152478 → promoted (best={charge_threshold:6.0, discharge_threshold:4.0, lookahead_steps:48})
iter 4: energy_solverv4_v8 scored 2202251 vs parent energy_solverv4_v7 2326526 → kept (best={base_threshold:5.0, soc_bias:0.0, lookahead_steps:64}; SOC bias hurt, asymmetric thresholds beat symmetric)
iter 5: energy_solverv4_v9 scored 2029485 vs parent energy_solverv4_v7 2326526 → kept (best={charge_spread_frac:0.30, discharge_spread_frac:0.10, lookahead_steps:48}; spread-relative thresholds underperformed fixed $/MWh)
iter 6: energy_solverv4_v10 scored 2384460 vs parent energy_solverv4_v7 2326526 → promoted (best={charge_threshold:8.0, discharge_threshold:6.0, slope_factor:2.0}; median ref allows higher thresholds than mean)
iter 7: energy_solverv4_v11 scored 2384460 vs parent energy_solverv4_v10 2384460 → kept (tie; lookahead 48 confirmed optimal for median ref)
iter 8: energy_solverv4_v12 scored 2388567 vs parent energy_solverv4_v10 2384460 → promoted (best={end_window:24, end_factor:0.0}; forced last-step discharge captures residual SOC value)
iter 9: energy_solverv4_v13 scored 2076453 vs parent energy_solverv4_v12 2388567 → kept (best={dp_horizon:24, soc_levels:41, action_levels:11}; DP-with-DA-proxy fundamentally undervalues SOC vs threshold strategies that wait for RT spikes)

## track=s=congested (nonces=100) — user-directed switch 2026-04-28
iter 10: TRACK CHANGE — track=s=congested, nonces=100, parent reset to energy_solverv4_v14 (off-loop winner on s=baseline at default hp), score reset to 0 per CLAUDE.md.
iter 11: energy_solverv4_v18 scored 2376849 vs parent energy_solverv4_v14 0 → promoted (best={profit_weight_power:2.545}; profit-weighted line softening; v14-equiv at p=0 scored 2,340,322, best p=2.545 → +36k/+1.5%; v14 deleted)
iter 12: energy_solverv4_v19 scored 2390789 vs parent energy_solverv4_v18 2376849 → promoted (best={charge_threshold:8.0, discharge_threshold:5.0}; lower discharge_th wins on s=congested medium-volatility; +14k/+0.6%; v18 deleted)
iter 13: energy_solverv4_v20 scored 2390789 vs parent energy_solverv4_v19 2390789 → kept (best={end_window:24, end_factor:0.0}; v14 defaults confirmed optimal on s=congested; ew=0 -12k, ew=48 -14k; partial liquidation always lost)
iter 14: energy_solverv4_v21 scored 2390789 vs parent energy_solverv4_v19 2390789 → kept (best={counter_help_factor:0.0}; counter-help mechanism never triggers — all 12 points identical; v19 soften loop already resolves all violations on s=congested)
iter 15: energy_solverv4_v22 scored 2390789 vs parent energy_solverv4_v19 2390789 → kept (best={lookahead_steps:48, slope_factor:2.0}; v14 defaults confirmed optimal; ls=24 -15%, ls=72/96 -2-3%, sf=4.0 always worse)
iter 16: energy_solverv4_v23 scored 2390789 vs parent energy_solverv4_v19 2390789 → kept (best={profit_weight_power:2.545}; finer grid [2.2, 2.9] confirms 2.545 winner; sharp peak with noisy adjacents)
iter 17: energy_solverv4_v24 scored 2390789 vs parent energy_solverv4_v19 2390789 → kept (best={quantile_offset:0.0}; quantile-shifted reference; monotonic decrease from median to extremes; median decisively optimal)
iter 18: energy_solverv4_v25 scored 2390789 vs parent energy_solverv4_v19 2390789 → kept (best={mean_blend:0.0}; pure median wins, pure mean -19k; 6 consecutive non-improvements — v19 plateau)

## Cleanup + multi-track restructure (2026-04-28)
- User deleted v6/v8/v9/v11/v12/v13/v14-v17/v19-v25 except v19, then renamed energy_solverv4_v19 → energy_solver_1.
- Pre-rename s=baseline iter history (iters 1-9, v5 → v7 → v10 → v12) and s=congested iter history (iters 10-18, v14 → v18 → v19) preserved in `.tig-iterate/state.legacy.json`.
- Per-track state files created: `state.s=baseline.json`, `state.s=congested.json`, `state.s=multiday.json`, `state.s=dense.json`, `state.s=capstone.json`. Each with current_best=energy_solver_1 score=0 (first child auto-promotes per CLAUDE.md).
- Per-track child stem: `energy_solver_<track>` (e.g., `energy_solver_baseline_1`, `energy_solver_congested_1`). Each track has independent next_version counter.
- Off-loop sanity test: energy_solver_1 (charge=8, discharge=6) scored 2,755,198 on s=baseline @ 100 nonces (vs v14 default 2,755,198 — tied within noise; profit-weighted soften dormant on loose-flow track).

## track=s=multiday (nonces=100)
iter 1: energy_solver_multiday_1 scored 1293096 vs parent energy_solver_1 0 → promoted (best={end_window:48, end_factor:0.0}; doubling end_window from 24→48 matches 2× episode length; ew=48 +47k vs ew=24, ew=96 catastrophic -285k; seed preserved)
iter 2: energy_solver_multiday_2 scored 1391150 vs parent energy_solver_multiday_1 1293096 → promoted (best={charge_threshold:10.0, discharge_threshold:4.0}; medium-high vol prefers higher ct + lower dt; +98k/+7.6%; result at grid corner; multiday_1 deleted)
iter 3: energy_solver_multiday_3 scored 1401292 vs parent energy_solver_multiday_2 1391150 → promoted (best={charge_threshold:10.0, discharge_threshold:2.0}; extended grid; ct=10 peak confirmed (12/14/16 worse), dt=2 wins +10k; multiday_2 deleted)
iter 4: energy_solver_multiday_4 scored 1587908 vs parent energy_solver_multiday_3 1401292 → promoted (best={lookahead_steps:72}; HUGE +187k/+13.3%; longer episode loves longer lookahead; unimodal peak at 72=37.5% of episode; ls=24 catastrophic -573k; multiday_3 deleted)
iter 5: energy_solver_multiday_5 scored 1614849 vs parent energy_solver_multiday_4 1587908 → promoted (best={slope_factor:1.273}; +27k/+1.7%; lower slope wins; peak near 1.27 (vs v14 default 2.0); multiday_4 deleted)
iter 6: energy_solver_multiday_6 scored 1614849 vs parent energy_solver_multiday_5 1614849 → kept (best={profit_weight_power:2.545}; p=2.545 still wins on s=multiday — same as s=congested; p=0 uniform soften -57k; mechanism still helps but already optimal)
iter 7: energy_solver_multiday_7 scored 1614884 vs parent energy_solver_multiday_5 1614849 → promoted (best={charge_threshold:8.0, discharge_threshold:1.0}; +35 within noise but strictly > per spec; threshold surface flat near optimum at new operating point; multiday_5 deleted)
iter 8: energy_solver_multiday_8 scored 1641038 vs parent energy_solver_multiday_7 1614884 → promoted (best={end_window:66}; finer ew search 36-69 step 3 reveals peak at 66=34% of episode; +26k/+1.6%; coarse grid in iter 1 missed it; multiday_7 deleted)
iter 9: energy_solver_multiday_9 scored 1664953 vs parent energy_solver_multiday_8 1641038 → promoted (best={lookahead_steps:66}; finer ls search [60, 82] step 2; peak at 66 matches end_window; +24k/+1.5%; multiday_8 deleted)
iter 10: energy_solver_multiday_10 scored 1664953 vs parent energy_solver_multiday_9 1664953 → kept (best={charge_threshold:8.0, discharge_threshold:1.0}; thresholds at new operating point; sharp peak at ct=8/dt=1.0 stays; no headroom)
iter 11: energy_solver_multiday_11 scored 1664953 vs parent energy_solver_multiday_9 1664953 → kept (best={soc_bias:0.0}; STRUCTURAL: SOC-aware threshold tilt; b=0 wins, all bias values hurt; mechanism rejected — bounds already encode SOC well)
iter 12: energy_solver_multiday_12 scored 1737895 vs parent energy_solver_multiday_9 1664953 → promoted (best={trend_factor:0.6364}; STRUCTURAL: DA-trend-aware threshold shift wins big +73k/+4.4%; rising DA → wait, falling → act; multiday_9 deleted)
iter 13: energy_solver_multiday_13 scored 1737895 vs parent energy_solver_multiday_12 1737895 → kept (best={charge_threshold:8.0, discharge_threshold:1.0}; thresholds at trend-aware operating point; same peak as multiday_12; no shift)
iter 14: energy_solver_multiday_14 scored 1737895 vs parent energy_solver_multiday_12 1737895 → kept (best={vol_factor:0.0}; STRUCTURAL: DA-vol-aware threshold scaling; monotonic decrease, vf=0 wins; volatility carries no signal beyond trend; mechanism rejected)

