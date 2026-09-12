# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** walkfront
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

Boolean satisfiability solver for random 3-SAT instances near the phase
transition. The entry point dispatches each instance by size to a dedicated
per-track engine; a shared preprocessing stage (`common`) builds a compressed
sparse-row (CSR) representation of the clause database once. Each engine is a
fuel-bounded stochastic local search: it maintains a full assignment and,
guided by clause-satisfaction make/break counts, repeatedly flips variables
drawn from unsatisfied clauses under a probabilistic acceptance rule, with a
double-scan clause pass fused into a single traversal to bound memory traffic.
The effort budget is tuned per size so the search terminates as soon as the
solved fraction stops improving. All tuning is exposed as hyperparameters read
in `from_map`; the defaults are the shipped operating point.

## References and Acknowledgments

### Academic Papers
- B. Selman, H. Kautz, B. Cohen, *"Local Search Strategies for Satisfiability Testing"*, DIMACS Series in Discrete Mathematics and Theoretical Computer Science, 1996 (WalkSAT).
- A. Balint, U. Schöning, *"Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break"*, SAT 2012 (probSAT).
- A. Braunstein, M. Mézard, R. Zecchina, *"Survey propagation: An algorithm for satisfiability"*, Random Structures & Algorithms, 2005.
- U. Schöning, *"A probabilistic algorithm for k-SAT and constraint satisfaction problems"*, FOCS 1999.

### Code References
- TIG baseline (satisfiability).

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
