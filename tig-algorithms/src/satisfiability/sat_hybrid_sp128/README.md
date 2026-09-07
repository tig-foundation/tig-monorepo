# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** sat_hybrid_sp128
* **Copyright:** 2026 Fred; donor copyright remains with NVX
* **Identity of Submitter:** Fred
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method

This Code is the public `c001_a118 / sat_hybrid_v4` portfolio with one
integration change on a single route. On the `n_vars=100000,ratio=4200`
route (raw dimensions 100000 variables / 420000 clauses), the reinforced
survey-propagation seeding phase (`engine_spr.rs`) hands its current best
assignment to the local-search phase as soon as at least 100 SP iterations
have run and the best assignment violates at most 128 clauses, instead of
continuing SP until convergence or the 150-iteration patience stop. The
hand-off keeps the best prefix assignment; the SP constants, the local
search, the verified-failure fallback, the RNG initialisation and every other
route are unchanged. Every other track (5000, 7500, 10000, 4150) executes the
public a118 code path exactly.

On the tested 100000/4200 route, the candidate preserved 400/400 observed
verified solves while reducing measured runtime CPU by ~38.8% relative to
public a118 (paired, same fixtures, official runtime and verifier).

Leave hyperparameters null. All track routing and settings are internal and
fixed; no hyperparameter JSON is required.

## Public-Source Attribution

* All fourteen files derive from public `c001_a118 / sat_hybrid_v4` by NVX
  (player `0x8bc5ee…`), source commit
  `d2f974477490a028c5e411ffc9c303315dbc16d2`. Twelve files are byte-identical
  to that public source; `engine_b.rs` and `engine_spr.rs` carry the
  route-scoped early hand-off described above and nothing else.
* Stopping message passing early and finishing with local search is a
  classical survey-propagation practice (Braunstein, Mézard, Zecchina 2005;
  Chavas, Furtlehner, Mézard, Zecchina 2005). No new algorithmic method is
  claimed.

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
