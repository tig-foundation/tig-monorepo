# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** survey_relay
* **Copyright:** 2026 Fred; donor copyrights remain with ChervovNikita (sat_exact), NVX (sat_hybrid_v4) and zhenglcc (sat_tailwalk_v7)
* **Identity of Submitter:** Fred
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## What this is

`survey_relay` is a per-track composite. On each c001 track it runs, unmodified, the engine
of the active code that is strongest on that track. No new algorithmic method is claimed.

| Track | Engine | From | Files |
|---|---|---|---|
| n_vars=100000, ratio=4200 | reinforced survey-propagation seed with early hand-off, then engine_b local search, with a faster, output-identical SP kernel | `sat_exact_spfast` (Fred; = `c001_a127` with the faster kernel) | `sp128_mod.rs`, `sp128_engine_b.rs`, `sp128_engine_spr.rs`: identical to `sat_exact_spfast` |
| n_vars=100000, ratio=4150 | `sat_imp_giveup2` track 4 | `c001_a125` sat_exact (ChervovNikita) | `ours_mod.rs`, `ours_track3.rs`, `ours_track4.rs`: identical to a125, except one attribute (below) |
| n_vars=5000, ratio=4267 | engine_iv, then engine_ag on a verified failure | `c001_a123` sat_hybrid_sp128 (Fred) | `hsp_engine_iv.rs`, `hsp_engine_ag.rs`, `hsp_engine_e.rs` |
| n_vars=7500, ratio=4267 | engine_d, then engine_ag on a verified failure | `c001_a123` | `hsp_engine_d.rs`, `hsp_engine_ag.rs`, `hsp_engine_e.rs` |
| n_vars=10000, ratio=4267 | engine_d, then perturbed engine_d on a verified failure | `c001_a123` | `hsp_engine_d.rs` |

* `hsp_engine_{d,e,ag,iv}.rs` are byte-identical to a123's `engine_{d,e,ag,iv}.rs`, attached by
  `#[path]` at the same place in the module tree as in a123.
* `mod.rs`: the first 5,581 lines (the helper modules of a125) are byte-identical to a125.
  `solve_challenge` and its helpers (`merge_hp`, `u`, `get_u64`, `with_fuel`) are a123's own,
  unchanged: same five arms in the same order, same bodies, same baked default budgets. Only the
  two `n_vars=100000` arms differ: they call the vendored 4150 and 4200 routes instead of a123's
  engine_b.
* `ours_mod.rs` carries one added attribute, `#[inline(never)]` on its `solve_challenge`, on the
  same line. In a125 (and in sat_exact_spfast) that function has two call sites and is compiled as a
  separate function. Here it has one, so the compiler would inline it. That changes the code generated for the 4150
  preprocessing, and therefore the fuel count: we measured a constant +654,928 fuel per nonce
  without the attribute. With it, the generated code and the fuel count are those of a125 (see
  Local Validation).
* The 4200 kernel of `sat_exact_spfast` (against a123/a127) has two changes, with the same f32
  operations in the same order:
  1. the SP products are indexed directly by the encoded literal (`p[l]`, `p[l ^ 1]`);
  2. the three-literal specialisation of pass 2 is reused from `c001_a128` sat_tailwalk_v7 (zhenglcc).
* `sp128_engine_b.rs` keeps the two diagnostic prints inherited from a123 (unchanged file).

## Hyperparameters

Leave hyperparameters null (or `{}`) on all five tracks. Every route uses its own baked defaults.

The budget keys of the vendored routes keep their meaning:
* `t1_fuel_a`, `t1_fuel_b` (10000), `t4_fuel_b` (5000), `t5_fuel_a`, `t5_fuel_c` (7500), as in a123;
* `max_fuel_low`, `max_fuel_high` (4150), as in a125.

## Local Validation

Binary built from these files in an isolated workspace: registry containing this algorithm
only, dev image 0.0.7, whose toolchain is the one of the 0.0.8 CI image. `cargo check` for
aarch64 passes; exported symbols are the same as `sat_exact_spfast`.

Runs:
* official runtime and verifier 0.0.8 (amd64), hyperparameters null;
* paired with the **official** binaries of a123 and a125, and with the `sat_exact_spfast`
  binary for ratio=4200 (not yet public);
* same instances and same fuel budget for both codes of a pair.

Each run is compared on the runtime exit code, the sha256 of the saved solution, the reported
`fuel_consumed` and the verifier quality.

| Track | Compared with | Fuel budget | Pairs | Identical (exit code, solution, fuel, quality) |
|---|---|---|---:|---:|
| ratio=4200 | sat_exact_spfast | 2.5e12 / 1e9 | 32 / 2 | 32/32, 2/2 |
| ratio=4150 | a125 | 2.5e12 / 3e9 | 32 / 2 | 32/32, 2/2 |
| n_vars=5000 | a123 | 2.5e12 / 5e12 / 2e10 | 8 / 8 / 8 | 8/8, 8/8, 8/8 |
| n_vars=7500 | a123 | 2.5e12 / 5e12 / 2e10 | 8 / 8 / 8 | 8/8, 8/8, 8/8 |
| n_vars=10000 | a123 | 2.5e12 / 5e12 / 2e10 | 8 / 8 / 8 | 8/8, 8/8, 8/8 |

* The reported fuel is identical to the unit on all 140 pairs, including under a binding
  budget, where the out-of-fuel exit happens at the same point.
* Repeated runs reproduce the solution, the fuel and the runtime signature (8/8).
* CPU time is within 1% of the source codes on every track.
* These are finite samples on one host.

## References and Acknowledgments

### 1. Academic Papers

* J. Chavas, C. Furtlehner, M. Mezard, R. Zecchina, "Survey-propagation decimation through
  distributed local computations", J. Stat. Mech. (2005) P11016: reinforced SP (4200 route).
* A. Braunstein, M. Mezard, R. Zecchina, "Survey propagation: an algorithm for satisfiability",
  Random Structures & Algorithms 27 (2005): SP followed by local search.
* N. Een, N. Sorensson, "An Extensible SAT-solver", SAT 2003: `2v + sign` literal encoding.

### 2. Code References

| Code | Author | Use here |
|---|---|---|
| `sat_exact_spfast` | Fred | 4200 route (faster SP kernel) |
| `c001_a127` sat_exact_sp128 | Fred | Base of sat_exact_spfast |
| `c001_a125` sat_exact | ChervovNikita (`0x8e671c8f04fdcd2a519c984784a136594d5e8cd2`) | 4150 route and helper modules; itself vendors sat_imp_giveup2 (a112) |
| `c001_a123` sat_hybrid_sp128 | Fred; donor copyright NVX | 5000, 7500 and 10000 routes and dispatch; 4200 engine_b |
| `c001_a118` sat_hybrid_v4 | NVX | Donor of the a123 engines (engine_iv ports sat_imp_v4, c001_a098) |
| `c001_a128` sat_tailwalk_v7 | zhenglcc (`0xb9a66a0f6c1aedadfd7d03e9d9b2388da67300f5`) | Three-literal specialisation of the SP pass 2 loop |

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
