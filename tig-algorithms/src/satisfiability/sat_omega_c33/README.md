# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_omega_c33
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

`sat_omega_c33` routes each `c001` track directly by `(num_variables, clauses.len())`:
- `5000 / 21335` (T1): extended bitwise probSAT (`run_near_fast_minlen`, 392M cap, calibrated early giveup checkpoints) with gated `v8::c20_t1_rescue` and `v8::c16_jump_one` (5B fuel).
- `7500 / 32002` (T2): 4-attempt restart bitwise probSAT with near-miss capture and uniform tail continuation, backed by `v8::c26_t2_rescue` and `v8::c16_jump_one` (12B fuel).
- `10000 / 42670` (T3): multi-stage gated narrow-clause SLS (`v8::c32_gated_t3`) with `v8::c16_jump_one` (12B fuel).
- `100000 / 415000` (T4) and `100000 / 420000` (T5): `tekinon::c32_solve_large` (AVX2 SP + bitwise SLS) with bounded `v8::n3_bounded_rescue` fallback.
Every candidate solution is verified against all clauses before saving, and any unsolved instance emits a `vec![false; num_variables]` fallback solution to ensure valid length for `tig-verifier`.

### Code References
- TIG `c001` public submissions (`tekinon` / `c001_a135` by tilk, `sat_tailwalk_v8` / `c001_a137`, `sat_tailwalk_v7` / `c001_a128`) under the TIG Open Data License and TIG Innovator Outbound Game License.

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
