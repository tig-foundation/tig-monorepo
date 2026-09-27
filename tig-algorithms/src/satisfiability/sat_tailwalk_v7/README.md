# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_tailwalk_v7
* **Copyright:** 2026 zhenglcc; upstream copyrights remain with their respective owners, including NVX, ChervovNikita, Fred and testing
* **Identity of Submitter:** zhenglcc
* **Identity of Creator of Algorithmic Method:** zhenglcc
* **Unique Algorithm Identifier (UAI):** null

This is a derivative/composite continuation of `sat_tailwalk_v6`, with code and
methods reused from the public submissions listed below. The creator field is
retained from `sat_tailwalk_v6`; it does not claim authorship of the constituent
upstream methods. No new survey-propagation or stochastic-local-search method
is claimed.

## Method and Target Tracks

| Track | Route |
|---|---|
| `n_vars=5000,ratio=4267` | Independent `sat_imp_v4`-derived complement, followed when needed by the a115-derived fallback; cached exact selection kernels adapted from a125. |
| `n_vars=7500,ratio=4267` | v6 adaptive segmented local search, bounded checkpoint rescue and a125-derived cache/range helpers. |
| `n_vars=10000,ratio=4267` | a112/a125-derived cached local search, with a separate majority-initialized rescue after a sufficiently close failed search. |
| `n_vars=100000,ratio=4150` | Cached local search with the retained v6-derived selection policy, including its a119-derived scoring. |
| `n_vars=100000,ratio=4200` | a123 reinforced survey-propagation seeding and local search; the retained v6-derived solver is an independent fallback. |

Alternate portfolio paths are accepted only after checking a complete SAT
assignment. A failed alternate path keeps the fallback's original RNG seed.

The main Track5 change comes from reusing a123's SP/local-search route and its
early hand-off criterion. This submission adds integration with the v6
fallback and implementation-level loop work; that upstream algorithmic
contribution is not presented as our invention.

## Implementation Changes

* Track1: phase-batched tail control and exact three-literal selection, while retaining the original random-word consumption and fallback behavior.
* Track1: reconstruct the assignment from the existing cache only at tail snapshots and final output, instead of writing it on every tail flip.
* Track2: move repeated controller checks to exact event boundaries, preserving the original observation order, and retain the bounded checkpoint rescue.
* Track3: give the primary route a larger default budget while keeping the majority rescue at its original budget.
* Track4: remove appearance/age state that the selected policy does not read; preserve its move selection.
* Track5: retain the a123 early SP hand-off, the v6 fallback and exact three-edge loop specialization already validated before this window.

These source changes run with the unmodified official TIG environment. They
require no custom Runtime, Slave, TLS interposition or local Runtime update.

## Hyperparameters

Use `{}` or `null` for all five standard tracks. The release defaults are the
defaults used in the accompanying tests.

Here `null` means an absent hyperparameter object in the Slave request. For a
direct `tig-runtime` invocation, omit `--hyperparameters`; its CLI does not
accept the literal JSON `null` as a parameter map.

Track3 uses a primary default of `180e9` and a majority-rescue default of `125e9`; an explicit `max_fuel_high` continues to configure both.

The inherited route-specific parameters remain available for compatibility.
The optional `t4_fuel_b` key controls the Track1 complement budget despite its
historical name. Non-default parameters were not tuned in the release
comparison; they can alter quality and cost.

## Validation

Frozen independent inputs, official unmodified Slave/Runtime/Verifier 0.0.8, amd64, external fuel `5e12`. a125 uses its published `180e9` recommendation on Track3 and `{}` elsewhere.

| Track | Selected source solves | a125 solves | Selected mean (s) | a125 mean (s) |
|---|---:|---:|---:|---:|
| 1 | 16/64 | 14/64 | 64.824 | 58.614 |
| 2 | 5/64 | 5/64 | 92.994 | 106.815 |
| 3 | 4/32 | 4/32 | 101.270 | 93.049 |
| 4 | 64/64 | 64/64 | 4.648 | 4.978 |
| 5 | 64/64 | 64/64 | 5.935 | 37.617 |

These A/B measurements belong to the frozen per-track candidate binaries selected for this release. The separately rebuilt, renamed and flattened release artifact is checked against their complete solution vectors; see the accompanying release-validation evidence.

The reported results are finite-sample measurements on the specified host and
inputs, not a guarantee of identical success rates or speedups on future
instances or other processors. Failed inputs are included in time averages.

## References and Acknowledgments

The links pin the source revisions used during development.

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

The reinforced-SP route retains the academic attribution carried by its
upstream implementation, including Chavas, Furtlehner, Mezard and Zecchina,
J. Stat. Mech. (2005) P11016. See the pinned a123/a118 sources for that method's
provenance and original implementation notes.

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
