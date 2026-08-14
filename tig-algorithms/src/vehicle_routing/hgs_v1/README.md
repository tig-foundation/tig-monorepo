# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** hgs_v1
* **Copyright:** 2025 Thibaut Vidal
* **Identity of Submitter:** Thibaut Vidal
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments  

This implementation is based on existing work.
This is a Rust re-implementation of the state-of-the-art Hybrid Genetic Search (HGS) algorithm initially introduced for the Capacitated Vehicle Routing Problem (CVRP) by [1], 
and extended to the Vehicle Routing Problem with Time Windows (VRPTW) by [2]. The method also benefits from various speedups as well as an implementation of the SWAP* neighborhood described in [3].

The framework is designed to be easily parameterizable, offering preset configurations that balance runtime and solution quality, from a fast single local-search exploration (exploration_level = 0) to a full-scale HGS exploration consistent with academic benchmarking standards (exploration_level = 5). Besides the exploration_level parameter, all other parameters documented in params.rs can be individually defined.

### 1. Academic Papers

[1] Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., & Rei, W. (2012). A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. Operations Research, 60(3), 611–624. https://doi.org/10.1287/opre.1120.1048

[2] Vidal, T., Crainic, T. G., Gendreau, M., & Prins, C. (2013). A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows. Computers & Operations Research, 40(1), 475–489. https://doi.org/10.1016/j.cor.2012.07.018

[3] Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. Computers and Operations Research, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643

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