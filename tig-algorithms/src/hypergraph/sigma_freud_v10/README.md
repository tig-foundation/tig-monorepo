# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** sigma_freud_v10
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

sigma_freud_v10 is an even further highly optimised and tuned hypergraph solver inspired by previous iterations. It achieves higher qualities than any previous version

- `{"effort":5}` produces the highest quality but also highest runtime
- Previous larger tracks would cause issues with benchmarkers as jobs could not complete and prove in good time
- Average runtime on tracks is ~800 seconds at `{"effort":5}`
- Other hyperparameters are available, however the current defaults have been tuned specifically for the best performance. Refinement can be overidden with `{"refinement:<n>}` however most increases see diminishing returns.

Recommended fuel setting is 5T

The search is direct k-way on the connectivity / (λ−1) metric. GPU refinement is Fiduccia–Mattheyses local search with tabu and iterated local search. Host stages add multilevel V-cycles, Jet refinement, and pairwise max-flow improvement, in the same line as KaHyPar (Schlag et al., High-Quality Hypergraph Partitioning, ACM 2023), flow-based uncoarsening (Heuer, Sanders and Schlag, SEA 2018 / JEA 2019), and Jet (Gilbert et al., SIAM 2023). Coarsening is first-choice heavy-edge matching. 

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
