# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_tailwalk
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

`sat_tailwalk` is a tail-aware WalkSAT-style solver for TIG C001
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

SIMD is currently a runtime label/helper only. The hot flip loop remains scalar
because typical occurrence lists are short; AVX2/AVX-512 should only be used for
future linear rebuild or cleanup paths after benchmark evidence.

Recommended hyperparameters:

Use the built-in route defaults. The normal submission surface is only the CPU
profile selector:

- Zen 4 class CPU: `{"hw_profile":"zen4"}`
- Zen 5 class CPU: `{"hw_profile":"zen5"}`
- Zen 5c class CPU: `{"hw_profile":"zen5c"}`

If `hw_profile` is omitted or set to `auto`, the solver detects the CPU profile
and selects defaults internally.

Key optional controls for controlled benchmark runs:

- `target_max_fuel`: C001 target fast-path internal fuel budget. Leave unset for
  route-specific defaults.
- `target_base_prob`: C001 target fast-path random-walk probability override.
  Leave unset for route-specific defaults.
- `max_prob`: upper bound for adaptive random-walk probability.
- `check_interval`: progress-check interval used by the target fast path.
- `stagnation_limit`: number of non-improving checks before a kick or restart.
- `perturbation_flips`: kick size after stagnation on the target dense routes.
- `target_nad`: occurrence-bias threshold for target initialization.
- `init_noise`: random override probability during biased initialization.

Other accepted controls are intended for debugging or controlled experiments,
not routine submission use.

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
