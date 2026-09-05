# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** clausal_ember
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

Per-track SAT solver: the entry point dispatches by instance size to a dedicated
stochastic-local-search engine. The core is an adaptive-noise WalkSAT/probSAT
flip loop with make/break bookkeeping, bounded restarts and stagnation-driven
perturbation. The largest tracks add a survey-propagation seeding stage
(message-passing decimation) before local search. A per-track fuel budget bounds
the work so run time stays predictable. All tuning is exposed as hyperparameters
read in from_map; the defaults reproduce the best measured operating point of
each track with no hp_json.

## References and Acknowledgments

### Academic Papers
- B. Selman, H. Kautz, B. Cohen, *"Noise Strategies for Improving Local Search"* (WalkSAT), AAAI, 1994.
- A. Balint, U. Schoning, *"Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break"* (probSAT), SAT, 2012.
- A. Braunstein, M. Mezard, R. Zecchina, *"Survey propagation: An algorithm for satisfiability"*, Random Structures & Algorithms, 2005.

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
