# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_exact
* **Copyright:** 2026 ChervovNikita
* **Identity of Submitter:** ChervovNikita
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track | Fuel Budget | Nonces/Bundle |
|---|---|---|
| n_vars=5000,ratio=4267 | 5T | 100 |
| n_vars=7500,ratio=4267 | 5T | 100 |
| n_vars=10000,ratio=4267 | 5T | 100 |
| n_vars=100000,ratio=4150 | 5T | 100 |
| n_vars=100000,ratio=4200 | 5T | 100 |

## Hyperparameters

**Send `{}` on four of five tracks.** Every route carries its own tuned defaults.

| Track | Hyperparameters to send |
|---|---|
| n_vars=5000,ratio=4267 | `{}` |
| n_vars=7500,ratio=4267 | `{}` |
| **n_vars=10000,ratio=4267** | **`{"max_fuel_high": 180000000000}`** |
| n_vars=100000,ratio=4150 | `{}` |
| n_vars=100000,ratio=4200 | `{}` |

`n_vars=10000` is the one track where an explicit value matters. Its route
defaults to `max_fuel_high = 125e9`, which places the give-up deadline at 93% of
the flip budget and effectively disables the mechanism. `180e9` is the on-chain
attested majority for this track and is the value every measurement used.

Accepted keys on all routes: `base_prob`, `max_prob`, `check_interval`,
`stagnation_limit`, `perturbation_flips`, `max_fuel_high`, `max_fuel_low`.
Only the two fuel keys are live levers.

## Measured Quality

Quality is binary per nonce, so the bundle score is the solve count. 32 nonces,
seed `base1`, start 0, one batch per track.

| Track | Solves | Nonces run |
|---|---:|---:|
| n_vars=5000,ratio=4267 | 7 | 32 |
| n_vars=7500,ratio=4267 | 5 | 32 |
| n_vars=10000,ratio=4267 | 1 | 32 |
| n_vars=100000,ratio=4150 | 32 | 32 |
| n_vars=100000,ratio=4200 | 32 | 32 |

Solve sets are identical to the per-track best-of baseline and to the previous
version on every track. Determinism verified: re-running the same nonces
reproduces per-nonce identical `(nonce, quality)`. See `PROOF.md` for the cost
comparison against the fastest rival on each track.

## What this algorithm is

A per-track composite. Each track is routed to the solver that measured best on
that track, and that solver's hot kernel is replaced with an output-preserving
faster implementation.

| Track | Routed to | Optimised file |
|---|---|---|
| n=5000 | `sat_hybrid` engine_b | `hybrid_engine_b.rs` |
| n=7500 | `sat_tailwalk_v6` track2 | `tw6_track2.rs` |
| n=10000 | `sat_imp_giveup2` track3 | `ours_track3.rs` |
| r4150 | `sat_imp_giveup2` track4 | `ours_track4.rs` |
| r4200 | `sat_hybrid` engine_b | `hybrid_engine_b.rs` |

Supporting modules: `exact_coin`, `exact_div`, `explicit_roulette`,
`raw_roulette`, `clause_order`, `signed_order`, `paged_flip_ranges`,
`compact_ranges`, `one_load_ranges`, `paged_cache`, `subset_cache`,
`paged_order`, `paged_small_order`, `paged_small_signed`, `phase_div`,
`short_mask`, `large_buffer`, `list_events`.

Only code reachable from this composite's five routes is shipped. The vendored
dispatchers keep their original track detection; routes the composite never
calls return an error instead of linking unused engines.

The directory is single-level, matching every shipped TIG algorithm, and fits
TIG's 20-file submission limit: 8 `.rs` files. Vendored solver files are mapped
with `#[path]` attributes and the supporting modules above are defined inline in
`mod.rs` as `mod name { ... }` blocks, so every module path is unchanged.

## Changes vs submission003 (`sat_exact` v005)

Further output-preserving kernel work on n=5000/r4200 (`hybrid_engine_b.rs`),
n=7500 (`tw6_track2.rs`), n=10000 (`ours_track3.rs`) and r4150
(`ours_track4.rs`), plus thirteen new supporting modules. Measured 1.408x faster
than v005 over the four tracks with an established magnitude, at identical solve
sets on all five.

## References and Acknowledgments

### 1. Academic Papers
- N/A

### 2. Code References
Vendors and optimises the following TIG code submissions:
- `satisfiability/sat_hybrid` (c001_a109)
- `satisfiability/sat_tailwalk_v6` (c001_a114)
- `satisfiability/sat_imp_giveup2` (c001_a112), itself derived from
  `satisfiability/sat_imp_v4` (c001_a098), which carries no Unique Algorithm
  Identifier and embodies no registered Advance Submission.

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
