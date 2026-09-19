# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** wasserwert
* **Copyright:** 2026 Kernelsmith
* **Identity of Submitter:** Kernelsmith
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

`wasserwert` ("water value") is a rolling-horizon dispatch policy for the battery fleet built from
four pieces, all deterministic and CPU-only:

1. **Per-battery water value by backward dynamic programming** over the planning prices:
   the marginal value of stored energy `dV/dSoC(t, soc)` replaces fixed price thresholds. On the
   192-step tracks (multiday, dense, capstone) the expectation over the real-time price is taken
   on a **deterministic quadrature of the public price law**: Gauss-Hermite nodes for the Gaussian
   shock (3 or 5 points) times {no jump, conditional means of the Pareto jump on two tail slices},
   with exact weights (9 or 15 weighted scenarios). On the 96-step tracks (baseline, congested) the
   DP averages K = 128 real-time paths sampled from the public price law with our own seeds.
2. **Network-aware multi-period planning.** An outer loop DP <-> dual prices pulls the line
   shadow prices back into the planning price signal as an LMP-like effective nodal price
   `da_eff_i(t) = da_i(t) - sum_l PTDF[l][i] * nu_l(t)` (plan_rounds = 3 on every track);
   a plan-value gate guarantees that uncongested grids never regress.
3. **Exact per-step fleet LP (the main lever, `lp_exact = 2`).** Each step solves a small
   bounded LP over piecewise-linear concavised water-value slopes (`lp_seg = 4` segments per
   battery and direction) subject to the active line limits, by row generation: lines are
   added only when violated, the simplex is bounded (`max_iter`), and a failure to converge
   falls back to the previous subgradient dual loop. The exact duals are fed back into the
   planning loop (`nu_scale = 4`).
4. **Feasibility backstop.** Every action is clamped to the challenge's `action_bounds` and
   passed through a flow check that uses the challenge's own PTDF summation order; if a line
   is still violated, a value-weighted softening / bisection makes it feasible. The returned
   action is valid on every code path.

Hyperparameters are optional JSON overrides; the per-track defaults (selected by
`num_batteries`) are the measured best configuration. On the two 96-step tracks (baseline,
congested) the defaults additionally use `sdp_k = 128` sampled real-time paths, `plan_rounds = 8`
and a scenario gate `gate_k = 20` (+1.1 % / +0.45 % officially, fuel 47 / 113 billion on nonce 0).

Measured on 2026-09-19 with the official `tig-runtime` / `tig-verifier` (fuel-instrumented `.so`
built with `build_so` + the LLVM fuel plugin, monorepo `a4db8c5`), hyperparameters `{}` = defaults,
fresh instances, 0 invalid. Against the previous version of this algorithm (which differs only in
the planning scenarios on the 192-step tracks), 25 instances per track, pairwise on identical
instances:

| Track | avg quality | higher / lower than previous version | fuel vs. previous |
|---|---:|---:|---:|
| multiday | 3,487,780 | 12 / 12 | 0.74x |
| dense    | 2,652,362 | 22 / 3  | 1.17x |
| capstone | 4,075,143 | 25 / 0  | 0.95x |

baseline / congested are unchanged (bit-identical solutions). Maximum fuel over all measured runs:
788.6 billion (15.8 % of the 5e12 budget). Qualities depend on the seeds and are not directly
comparable to network bundle qualities. `dp_cache` (precomputed DP transitions) is on by default
and returns bit-identical solutions to `dp_cache=0`.

Hyperparameters `sdp_q` (Gauss-Hermite points: 0 = sampled SDP, 1, 3 or 5; 2 and 4 round up) and
`sdp_qj` (jump slices 0-3) override the quadrature; while `sdp_q > 0` an explicit `sdp_k` is not used.

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
