# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** sigma_freud_v6
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

sigma_freud_v6 is an even further highly optimised and tuned hypergraph solver inspired by previous iterations.

- The `{"effort":2}` hyperparameter ranges from 0-5 
- Default hyperparameter settings (`effort=2`) 
- For better quality but higher runtime: `{"effort": 4}`
- For faster runtime but lower quality: `{"effort": 1}`

Individual parameters can be overridden: `{"effort": 3, "tabu_tenure": 14, "refinement": 600}`

See ```help_algorithm sigma_freud_v6``` for more details of parameter tuning.

## Related Work

The following prior work informed the design of this algorithm:

- **Fiduccia & Mattheyses (1982)** — "A Linear-Time Heuristic for Improving Network Partitions." *19th ACM/IEEE Design Automation Conference.* Foundational move-based refinement framework.

- **Glover, F. (1989/1990)** — "Tabu Search — Part I & II." *ORSA Journal on Computing.* Tabu search mechanism with aspiration criterion.

- **Lourenço, H.R., Martin, O.C. & Stützle, T. (2003)** — "Iterated Local Search." In *Handbook of Metaheuristics*, Springer. ILS perturbation and re-optimisation framework.

- **Schlag, S. et al. (2023)** — "High-Quality Hypergraph Partitioning." *ACM Journal of Experimental Algorithmics.* Mt-KaHyPar state-of-the-art baseline.

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