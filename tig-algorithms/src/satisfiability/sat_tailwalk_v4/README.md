# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_tailwalk_v4
* **Copyright:** 2026 zhenglcc
* **Identity of Submitter:** zhenglcc
* **Identity of Creator of Algorithmic Method:** zhenglcc
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Selman, B., Kautz, H.A., Cohen, B., *"Noise Strategies for Improving Local Search"*, AAAI 1994
- Selman, B., Kautz, H.A., *"Domain-Independent Extensions to GSAT: Solving Large Structured Satisfiability Problems"*, IJCAI 1993

These papers are cited for the local-search SAT background used by the
algorithm family: randomized flips, noise control, and restart-style escape from
stagnation. No paper source code was copied.

### 2. Code References
- TIG Foundation - public satisfiability algorithms, including `sat_vanguard_v4`
  and related WalkSAT-style submissions: https://github.com/tig-foundation/tig-monorepo

  Referenced for the official TIG satisfiability interface, C001 challenge
  behavior, public baseline comparison, and dense-track implementation lessons.
  This submission uses its own clause layout, scoring state, route defaults, and
  target-walk implementation.

### 3. Other
- Development assistance: algorithm research and implementation were carried
  out by zhenglcc with Codex using GPT-5 as an AI coding and research
  assistant.

## Additional Notes

`sat_tailwalk_v4` is a track-routed tail-aware WalkSAT-style solver for TIG C001
satisfiability. It uses one codebase with broad CPU-family defaults selected by
`hw_profile`.

The main implementation features are:

- track dispatch for the active C001 shapes: `n5000_r4267`, `n7500_r4267`,
  `n10000_r4267`, `n100000_r4150`, and `n100000_r4200`;
- a target fast path with compact clause/occurrence storage, incremental
  scoring state, adaptive unsatisfied-clause weighting, and bounded
  reinitialization;
- deterministic seed-bucket route selection for selected dense-track tails,
  avoiding behavior tied to a fixed worker count.

The v4 update focuses on runtime-quality balance for the current target tracks:

- `n7500_r4267` uses tighter built-in phase and route fuel defaults to reduce
  tail runtime while preserving the stronger target-search route.
- `n100000_r4150` uses a capped default low-route fuel budget to keep the easy
  high-variable case fast without relying on external HP fields.
- Other track routes are retained from the accepted v3 code path unless their
  existing dispatch naturally selects the shared target fast path.

Hyperparameter guidance:

The intended release path is to run with no supplied hyperparameters. The
default path lets the solver use its built-in track-specific defaults and
deterministic `seed_key` bucket portfolio.

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
- `max_fuel_high` and `max_fuel_low` are used by some public TIG SAT algorithms
  but are not the fuel controls for this submission.
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

The default release setting remains no supplied hyperparameters.

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
