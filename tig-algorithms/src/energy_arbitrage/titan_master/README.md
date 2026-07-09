# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** titan_master
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Bellman, R., *"Dynamic Programming"*, Princeton University Press, 1957
- Bertsekas, D.P., *"Dynamic Programming and Optimal Control"*, Athena Scientific, 2017
- Boyd, S. and Vandenberghe, L., *"Convex Optimization"*, Cambridge University Press, 2004
- Dantzig, G.B. and Wolfe, P., *"Decomposition Principle for Linear Programs"*, Operations Research, 1960
- Mohsenian-Rad, H., *"Optimal Bidding, Scheduling, and Deployment of Battery Systems in California Day-Ahead Energy Market"*, IEEE Transactions on Power Systems, 2016

### 2. Code References

`titan_master` is inspired from titan_v6 with performance improvements and achieving higher qualities across all track

## Hyperparameter Guide

Hyperparameters are overridable through the TIG JSON map, but they are track-dependent and many only matter when a specific feature is enabled. In practice, the defaults in `mod.rs` are the intended starting point and most tuning should focus on just a few runtime / quality knobs per track.

### Baseline (`s=baseline`, up to 15 batteries)

- Main knobs: `soc_levels`, `action_grid`, `asca_iters`, `lp_refine_sweeps`, `cg_iters`.
- Main feature toggles: `use_lp`, `use_sdp`, `use_cg`.
- Advanced options like `use_tree_search`, `use_soc_ref_track`, `use_dfl_select`, and `use_pce_affine_recourse` exist, but they are secondary rather than the normal first tuning targets.

### Congested (`s=congested`, up to 30 batteries)

- Main knobs: `network_derating`, `lp_total_pivots`, `dw_total_pivot_budget`, `lns_lp_pivots_total`.
- Main feature toggles: `use_lp`, `use_dw`, `use_lns`, `use_kkt`, `use_primal_refine`.
- Congestion shaping is mainly controlled through `anticipate_lmp`, `lmp_threshold`, `lmp_premium_scale`, and the PTDF-tracking settings.

### Multiday (`s=multiday`, up to 50 batteries)

- Main knobs: `grad_outer_iters`, `proj_max_iters`, `coord_polish_passes`, `lookahead_horizon`, `fuel_budget`.
- Main feature toggles: `use_rolling_horizon`, `use_joint_pair_polish`, `use_joint_triplet_polish`.
- Optional extras such as `use_admm_solver`, `use_admm_polish`, `use_ejection_chain`, and `use_scvc` are there for more experimental variants.

### Dense (`s=dense`, up to 80 batteries)

- Main knobs: `grad_outer_iters`, `proj_max_iters`, `grad_ls_iters`, `joint_pair_budget`.
- Main feature toggles: `use_momentum`, `use_bb_clamps`, `use_ptdf_ct`, `use_composite_wv`, `use_joint_pair_polish`.
- Resolution knobs like `dp_soc_levels`, `dp_action_levels`, and `policy_action_levels` are available, but they are expensive and not usually the first place to tune.

### Capstone (`s=capstone`, up to 150 batteries)

- Main knobs: `grad_outer_iters`, `proj_max_iters`, `coord_polish_passes`, `num_seeds`, `fuel_budget`.
- Main feature toggles: `use_joint_pair_polish`, `use_ptdf_ct`, `use_composite_wv`.
- Additional optional paths such as `use_lp_dispatch`, `use_dual_dispatch`, `use_mpc_lookahead`, `use_sqdp`, `use_coupling_cut`, and `use_aggregate_reg` are present for larger-search variants.

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
