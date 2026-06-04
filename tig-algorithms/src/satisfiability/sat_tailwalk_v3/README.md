# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_tailwalk_v3
* **Copyright:** 2026 zhenglcc
* **Identity of Submitter:** zhenglcc
* **Identity of Creator of Algorithmic Method:** zhenglcc
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Selman, B., Kautz, H.A., Cohen, B., *"Noise Strategies for Improving Local Search"*, AAAI 1994
- Selman, B., Kautz, H.A., *"Domain-Independent Extensions to GSAT: Solving Large Structured Satisfiability Problems"*, IJCAI 1993

### 2. Code References
- TIG Foundation - public satisfiability algorithms, including `sat_vanguard_v4`
  and related WalkSAT-style submissions: https://github.com/tig-foundation/tig-monorepo

  Referenced for the official TIG satisfiability interface, C001 challenge
  behavior, baseline comparison, and dense-track implementation lessons. The
  implementation here uses its own data layout, scoring sidecars, route
  defaults, and target-walk code.

### 3. Other
- Development assistance: algorithm research and implementation were carried
  out by zhenglcc with Codex using GPT-5.5 as an AI coding and research
  assistant.

## Additional Notes

`sat_tailwalk_v3` is a track-routed tail-aware WalkSAT-style solver for TIG C001
satisfiability. It uses one codebase with broad CPU-family defaults selected by
`hw_profile`.

The first optimization layer is data layout and scoring:

- normalized clauses remove duplicate literals and tautologies;
- clause literals and offsets are stored as flat arrays;
- positive and negative variable occurrences use `u32` CSR;
- `make_score[v]` and `break_score[v]` are maintained incrementally;
- adaptive weights bump currently unsatisfied clauses and feed weighted make/break
  sidecars.

For the current dense C001 track, the solver enables a target fast path by
default. That path uses a focused local-search loop with occurrence-biased
initialization, direct occurrence arrays, exact unsatisfied-clause tracking,
randomized zero-break selection, weighted break fallback, short stagnation
kicks, and bounded reinitialization. The heavier make/break/weight path remains
available for non-target tracks or when `target_fast_path=false`.

This package uses explicit track-aware strategies within C001 rather than behavior tied
to a fixed nonce or worker count:

- `n5000_r4267`, `n7500_r4267`, `n10000_r4267`,
  `n100000_r4150`, and `n100000_r4200` are classified from public challenge
  shape (`num_variables` and clause count) before dispatch. Unknown shapes keep
  the stable fallback route instead of entering a hard-coded track branch.
- `n7500_r4267` and `n10000_r4267` use the high-density target route. Their
  default initialization and restart behavior includes a deterministic
  `seed_key` bucket portfolio for broader coverage across nonce ranges.
- `n100000_r4150` uses the existing mid-density target route and remained a
  lightweight route for the easier high-variable-density case.
- `n100000_r4200` uses the mid-density target route. For the extended no-tail
  setting, it includes a deterministic b128 `seed_key` salt fallback
  for selected buckets.

SIMD is currently a runtime label/helper only. The hot flip loop remains scalar
because typical occurrence lists are short; AVX2/AVX-512 should only be used for
future linear rebuild or cleanup paths after benchmark evidence.

Hyperparameter guidance:

The intended release path is to run with no supplied hyperparameters. The
validated comparison runs used `null` hyperparameters, which lets the solver use
its built-in track-specific defaults and deterministic `seed_key` bucket
portfolio.

The optional controls below are retained for controlled benchmark experiments
and debugging. They can materially change either quality or elapsed time, so they
should not be treated as routine submission settings.

- `hw_profile`: optional CPU-family override (`auto`, `zen4`, `zen5`, `zen5c`,
  `generic_avx512`, `generic`). Omit it or use `auto` for normal runs. Forcing a
  mismatched profile may change timing and can indirectly change quality on
  time-limited tracks.
- `target_max_fuel`: high-impact runtime budget for the C001 target fast path.
  Lower values can lose quality by stopping search too early. Higher values can
  recover late solves on hard nonces, but may spend more time with no quality
  gain.
- `init_noise`, `target_nad`, `target_base_prob`, and `max_prob`: high-impact
  search-shape controls. They affect initialization bias and random-walk
  behavior, so they can change the solved nonce set substantially. Leaving them
  unset is important because the default route includes per-track and per-seed
  safeguards.
- `restart_interval`, `stagnation_limit`, `perturbation_flips`, and
  `check_interval`: convergence and tail-search controls. Aggressive restart or
  kick settings can improve time on some runs but can also interrupt late
  convergence and reduce quality.
- `target_tail_cut_fuel`, `target_tail_cut_unsat_threshold`, and
  `target_tail_cut_best_unsat_threshold`: experimental mid-route tail cutoffs.
  They may reduce time when the remaining tail is unlikely to solve, but overly
  aggressive values can drop quality.
- `base_prob`, `make_mult`, `break_mult`, `weight_update_interval`,
  `max_clause_weight`, `clause_pick_samples`, `phase_restart_prob`,
  `phase_noise_divisor`, `age_shift`, and `age_cap`: fallback or secondary
  local-search tuning controls. They can affect quality and time, but are mainly
  intended for controlled experiments rather than normal release use.
- `disable_make_score`, `disable_clause_weights`, and `verify_invariants`:
  debugging controls. Disabling scoring or clause weights can significantly hurt
  quality; invariant verification adds overhead.

For historical controlled `n100000_r4200` experiments, an extended no-tail
setting was tested as:

`{"hw_profile":"zen4","target_max_fuel":450000000000,"target_tail_cut_fuel":0}`

This is not the default release setting. The default release setting remains no
supplied hyperparameters.

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
