# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_tailwalk_v5
* **Copyright:** 2026 zhenglcc
* **Identity of Submitter:** zhenglcc
* **Identity of Creator of Algorithmic Method:** zhenglcc
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### Academic Background

- Selman, Kautz, and Cohen, "Noise Strategies for Improving Local Search",
  AAAI 1994.
- Selman and Kautz, "Domain-Independent Extensions to GSAT: Solving Large
  Structured Satisfiability Problems", IJCAI 1993.

These papers are background for randomized local-search SAT, noisy variable
flips, and restart-based escape from stagnation. No paper source code was
copied.

### TIG Public Source References

- TIG Foundation public monorepo:
  https://github.com/tig-foundation/tig-monorepo
- Public `satisfiability/sat_imp_v4` branch inspected at commit
  `58c2ea49cb059442512f737033d0c44518ec3310`.
- Public online binaries and results for `sat_imp_v4` and `sat_vanguard_v3`
  were used as comparison baselines.

This is a derivative/composite submission, not a pure-original implementation.
It adapts the public `sat_imp_v4` implementation for the `5000/4267` and
`10000/4267` routes in `imp_v4_track1.rs` and `imp_v4_track3.rs`. The public
implementation was modified and integrated rather than copied as an unchanged
module.

Development assistance was provided by Codex using GPT-5.

## Algorithm Summary

`sat_tailwalk_v5` is a local-search SAT solver for TIG C001.

Recommended runtime HP by track:

```json
{
  "n_vars=5000,ratio=4267": {
    "target_tail_extend_fuel": 50000000000,
    "target_tail_extend_max_unsat": 1
  },
  "n_vars=7500,ratio=4267": {
    "target_n7500_best_tail_fuel": 75000000000,
    "target_n7500_best_tail_max_unsat": 1
  },
  "n_vars=10000,ratio=4267": {
    "target_n10000_best_tail_fuel": 40000000000,
    "target_n10000_best_tail_max_unsat": 8
  },
  "n_vars=100000,ratio=4150": {},
  "n_vars=100000,ratio=4200": {}
}
```

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
