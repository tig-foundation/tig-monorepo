# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** axiom_quasar
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

`axiom_quasar` runs E1 mode 0 defaults (`lookahead_relink=1`, `ils_rounds_t42=0`,
`t42_elites=4`, `t43_elites=3`) followed by monotonic donor block search with
width 24 and five candidates. T42 uses the donor track4 refiner with up to
twelve rounds; T43 uses axiom Stage-B2 polishing (`48, 38, 18, 30`).
Other tracks use the base E1 path. Each graft is accepted only if feasible
and strictly better. `solve_challenge` and `help` are both exported.

### Code References
- TIG `c003` public submissions (`axiom_pulsar` / `c003_a160`, `vemiral` / `c003_a159`, `packrun` / `c003_a156`, `satchel` / `c003_a154`) under the TIG Open Data License and TIG Innovator Outbound Game License.

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
