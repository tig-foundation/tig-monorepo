# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** titan_v7_ch3b
* **Copyright:** 2026 Fred
* **Identity of Submitter:** 0x003551a316597a1eA553F3498D246499EF76905c
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Fourer, R., *"A simplex algorithm for piecewise-linear programming"*, Mathematical Programming (1985-1992 series)
- Harris, P. M. J., *"Pivot selection methods of the Devex LP code"*, Mathematical Programming 5 (1973)
- Maros, I., *"Computational Techniques of the Simplex Method"*, Kluwer (2003)
- Koberstein, A., *"Progress in the dual simplex algorithm for solving large scale LP problems"*, Computational Optimization and Applications (2008)

### 2. Code References
- titan_v7 (TIG, energy_arbitrage/c008_a039, contributor 0x3ce59606c2c3929d3f3b0f147b900942ae21fffe) —
  https://github.com/tig-foundation/tig-monorepo — this submission is a derivative work of titan_v7 under the TIG
  Innovator Outbound Game License; engines t49/t50/t51/t52 and the surrounding t53 pipeline are unchanged from titan_v7.

## Additional Notes

Derivative of titan_v7. The ONLY algorithmic change is in the capstone engine (`t53_engine.rs`, selected for
num_batteries in (80,150]): the per-step joint dispatch decision, previously a projected-gradient ascent with POCS
projection, is replaced by the solution of a concave-hull piecewise-linear approximation of the same one-step dispatch
objective (sum over batteries of price term plus value-to-go on titan's own value tables), solved by constraint
generation over the DC line constraints and a bounded-variable dual simplex with Harris and bound-flipping ratio tests
(`lpdisp.rs`: pure Rust, deterministic, no RNG, no unsafe, no I/O, no new dependency). titan_v7's unchanged polish
chain (coordinate polish, interval pair polish, basin hop) and final feasibility guard follow. Engines for the other
tracks (t49, t50, t51, t52) are byte-identical to titan_v7.

Method class: known techniques (per-stage LP dispatch with PWL value-to-go under network constraints; standard dual
simplex machinery) adapted to titan_v7's value-function dispatch. It is NOT claimed to be novel, NOT an Advance
submission, NOT a new simplex method, and NOT an exact optimizer of the original non-concave one-step objective
(the concave hull is an upper approximation; the downstream polish chain operates on the exact objective).

Inputs used at step t: current SOCs, RT prices of step t, current exogenous injections (base flows), action bounds,
network PTDF/limits, titan's value tables for t+1, hyperparameters. No future prices, no hidden seed, no simulator
look-ahead, no filesystem/network/process access.

Hyperparameters: identical to titan_v7 (works with null hyperparameters and with existing titan_v7 recipes;
PGA-related hyperparameters are accepted but no longer used on capstone). Measured on the production runtime
(cap 2,500 G): fuel ~163 G/nonce vs ~667 G for titan_v7; wall ~0.36x.

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
