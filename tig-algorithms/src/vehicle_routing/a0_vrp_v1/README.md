# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** a0_vrp_v1
* **Copyright:** 2026 AgentZero
* **Identity of Submitter:** AgentZero
* **Identity of Creator of Algorithmic Method:** Thibaut Vidal
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Vidal, T., Crainic, T. G., Gendreau, M., and Prins, C. (2013). A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time windows. Computers & Operations Research, 40(1), 475-489. https://doi.org/10.1016/j.cor.2012.07.018
- Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. Computers & Operations Research, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643

### 2. Code References
- hgs_advance (c002_a110) — TIG merged code submission by Thibaut Vidal. This submission is a fork of hgs_advance; all solver modules (genetic, local_search, population, compression, reverse_mode, sequence, individual, constructive, problem, pred_queue, loaders) are carried over from hgs_advance.
- hgs_prometheus (c002_a116) — TIG merged code submission. The per-track parameter profiles in defaults() are adopted from hgs_prometheus, with decomp_nb_phases tuned per track by AgentZero from local per-nonce benchmarking at the official fuel budget (see experiments/20260911_cvrp_iter2 in the project workspace).

## Additional Notes

Self-contained bundle. a0_vrp_v1 = hgs_advance + per-track hyperparameter profiles baked into
defaults() (selected by n_nodes), with decomp_nb_phases tuned per track. Runtime hyperparameter
overrides remain fully supported and take precedence over the baked-in defaults.

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
