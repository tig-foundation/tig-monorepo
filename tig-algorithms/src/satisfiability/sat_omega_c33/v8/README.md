# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_tailwalk_v8
* **Copyright:** 2026 zhenglcc; upstream copyrights remain with their respective owners, including NVX, ChervovNikita, Fred and testing
* **Identity of Submitter:** zhenglcc
* **Identity of Creator of Algorithmic Method:** zhenglcc
* **Unique Algorithm Identifier (UAI):** null

## Method and Target Tracks

This is a derivative/composite continuation of `sat_tailwalk_v7` and `sat_tailwalk_v6`.
The creator field and null UAI are retained from the existing series. They do not
claim authorship of the constituent upstream methods. No new survey-propagation
or stochastic-local-search method is claimed.

| Track | Primary route and implementation changes |
|---|---|
| `n_vars=5000,ratio=4267` | Cached a115/a125-derived R5 first; independent IV search, conditional single-residual rescue and AS tail search follow on failure. SAT-only helper variants avoid unnecessary state maintenance. |
| `n_vars=7500,ratio=4267` | Adaptive segmented local search inherited from v6, with cache helpers, weighted checkpoint rescue and compile-time specialization. |
| `n_vars=10000,ratio=4267` | a112/a125-derived cached local search with a complementary rescue path; single-clause sampling is applied only in the rescue phase. |
| `n_vars=100000,ratio=4150` | Adapted a130 route with first-critical-position reuse and SP denominator simplification; the previous T4 route remains available on failure. |
| `n_vars=100000,ratio=4200` | Adapted a130 route with maintained SAT-status return and SP denominator simplification; the earlier SP portfolio and local-search fallback remain available on failure. |

Alternate search paths use complete-SAT acceptance checks where required by the
existing composition. The T5 primary path retains its initial/partial saves.
Implementation improvements are described separately from the upstream methods.

## Parameter Tuning (Optional)

Pass hyperparameters as a JSON object. Use `{}` or omit them to keep the defaults.
The main controls are:

| Parameter | Applicable route | How to adjust |
|---|---|---|
| `max_fuel_high` | T1–T3 | Increase to allow more local-search effort; decrease to shorten the search. |
| `max_fuel_low` | T4/T5 | Increase to allow more local-search effort; decrease to shorten the search. |
| `check_interval` | T3 and the T4/T5 primary routes | Use a positive integer. Smaller values check progress more frequently; larger values reduce check frequency. |
| `base_prob`, `max_prob` | T4 primary route | Adjust the starting random-walk probability and its adaptive upper limit. Higher values favor exploration; lower values favor greedy moves. |
| `stagnation_limit`, `perturbation_flips` | T4/T5 primary routes | A lower stagnation threshold triggers perturbations sooner. A larger flip count strengthens perturbations, subject to the route's adaptive limits. |
| `spr_on` | T4/T5 primary routes | Set `true` or `false` to enable or disable reinforced-SP initialization. |
| `sp_on` | T5 primary route | Set `true` or `false` to enable or disable the SP decimation phase. |

To retain the optimized T1 composite, use defaults or provide only `max_fuel_high`.
Other explicit parameter combinations select the inherited a115 compatibility route.
Internal fuel parameters estimate search effort and may apply to several phases;
they do not set the total runtime allowance. The official runtime fuel limit still applies.

The code runs in the official TIG environment and requires no custom runtime,
Slave or external files. All Rust sources are included, with top-level modules declared in `mod.rs`.
Related implementations are grouped in `a115.rs`, `a130.rs`, `engine_iv.rs`,
`track1.rs` and `engine_b.rs`; shared kernels and the other tracks remain separate.

## References and Acknowledgments

### Academic Papers

The following papers underpin mechanisms used directly or inherited through the
upstream implementations. The notes identify their role in this composite solver.

1. H. Fu, W. Zhang, G. Wu, Y. Xu and J. Liu, *[Improving stochastic local search for uniform k-SAT by generating appropriate initial assignment](https://doi.org/10.1111/coin.12438)*, Computational Intelligence 37(4), 1706–1744 (2021). The T1 AS tail adapts the occurrence-based allocation strategy for its first initial assignment; subsequent restarts retain the inherited R5 behavior.
2. Adrian Balint and Uwe Schöning, *[Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break](https://link.springer.com/chapter/10.1007/978-3-642-31612-8_3)*, SAT 2012, LNCS 7317, 16–29. The inherited local-search engines use probSAT-style break-based probability selection, including exponential weighting in T3 and polynomial weighting in retained fallback engines.
3. Alfredo Braunstein, Marc Mézard and Riccardo Zecchina, *[Survey propagation: an algorithm for satisfiability](https://arxiv.org/abs/cs/0212002)*, Random Structures & Algorithms 27, 201–226 (2005). This provides the basis for the inherited SP message updates and decimation components, including the T5 SP phase.
4. Joel Chavas, Cyril Furtlehner, Marc Mézard and Riccardo Zecchina, *[Survey-propagation decimation through distributed local computations](https://arxiv.org/abs/cond-mat/0512002)*, Journal of Statistical Mechanics: Theory and Experiment (2005), P11016; DOI: [10.1088/1742-5468/2005/11/P11016](https://doi.org/10.1088/1742-5468/2005/11/P11016). The reinforced-SP initialization inherited through a123/a118 and a130 uses this method family, with upstream adaptations to the reinforcement schedule, stopping rules and local-search hand-off.

These are mechanism-level references; this composite adapts and combines them
with the code sources below. Its scan reuse, SAT-status maintenance and arithmetic
simplifications are implementation changes.

### Code References

The links pin the upstream source revisions. Existing method authorship and notices are retained.

- [flipwave / c001_a130](https://github.com/tig-foundation/tig-monorepo/tree/c7c44b1510faeffb84f11031d86d225bbe397945/tig-algorithms/src/satisfiability/flipwave), copyright 2026 NVX: primary T4/T5 routes, preprocessing, reinforced survey propagation and local search. This version adds the T4 critical-position reuse, T5 maintained SAT-status return and SP denominator simplifications.
- `sat_tailwalk_v7` / c001_a128: immediate development baseline; this v8 source carries forward its composite structure and upstream acknowledgments.


| Public source | Author / upstream attribution | Reused contribution |
|---|---|---|
| [sat_tailwalk_v6 / c001_a114](https://github.com/tig-foundation/tig-monorepo/tree/6a4da2d7da002bea33e8f3b8ec3fe794761e1d41/tig-algorithms/src/satisfiability/sat_tailwalk_v6) | zhenglcc | Base dispatcher, preprocessing, Track2 lineage and retained fallback routes. |
| [sat_exact / c001_a125](https://github.com/tig-foundation/tig-monorepo/tree/b3c056ad909a65243314cbae24347cc4d0e9bdd6/tig-algorithms/src/satisfiability/sat_exact) | ChervovNikita | Vendored/adapted cache, range, exact division/roulette and clause-order helpers, Track1 cached kernel, Track2 fixed-restart reference and Track3/4 cached implementations. Its Track2 lineage itself credits v6. |
| [sat_hybrid_v3 / c001_a115](https://github.com/tig-foundation/tig-monorepo/tree/84125f66ca7b8a96430a635983aa97a31e913acc/tig-algorithms/src/satisfiability/sat_hybrid_v3) | NVX | Track1 preprocessing, restart defaults and compatibility fallback. |
| [sat_imp_giveup2 / c001_a112](https://github.com/tig-foundation/tig-monorepo/tree/4a0404680950bb11f97a0f1c8e19c4747bd16e9a/tig-algorithms/src/satisfiability/sat_imp_giveup2) | ChervovNikita | Track3 lineage, Track2 give-up reference and Track4 ancestry through a125. |
| [sat_hybrid_v4 / c001_a118](https://github.com/tig-foundation/tig-monorepo/tree/d2f974477490a028c5e411ffc9c303315dbc16d2/tig-algorithms/src/satisfiability/sat_hybrid_v4) | NVX | Track1 complementary engine and portfolio/SP ancestry. |
| [sat_hybrid_sp128 / c001_a123](https://github.com/tig-foundation/tig-monorepo/tree/c681d549498d6ddb35dd66861d11da29ff4b9e67/tig-algorithms/src/satisfiability/sat_hybrid_sp128) | Fred; donor copyright NVX | Track5 reinforced-SP seed, early hand-off and local-search route. |
| [sat_imp_v5 / c001_a119](https://github.com/tig-foundation/tig-monorepo/tree/6fe5e28f5597936c9b3300920bc724519c4ac606/tig-algorithms/src/satisfiability/sat_imp_v5) | testing | Scoring and fused-scan changes inherited by the retained v6-derived Track4/5 paths. |
| [sat_hybrid / c001_a109](https://github.com/tig-foundation/tig-monorepo/tree/c8fe46d5ad5574ae6eb17ed5d6c3e829da01eb27/tig-algorithms/src/satisfiability/sat_hybrid) | NVX | Indirect engine ancestry through the hybrid and a125 derivatives. |

Earlier ancestry acknowledged by v6 is also retained:

* [sat_imp_v4](https://github.com/tig-foundation/tig-monorepo/tree/b8a95b0d9723ac2e12957830d22dd08293fd76bc/tig-algorithms/src/satisfiability/sat_imp_v4)
* [sat_vanguard_v9](https://github.com/tig-foundation/tig-monorepo/tree/6717cd2e740e664f22da06089fbc252b666ce4cb/tig-algorithms/src/satisfiability/sat_vanguard_v9)

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
