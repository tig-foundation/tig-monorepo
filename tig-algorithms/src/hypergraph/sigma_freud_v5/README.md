# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** sigma_freud_v5
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

sigma_freud_v5 is an even further highly optimised and tuned hypergraph solver inspired by previous iterations.

- Default hyperparameter settings (`effort=2`) are recommended for most cases
- For better quality but higher runtime: `{"effort": 4}`
- For faster runtime but lower quality: `{"effort": 1}`

Individual parameters can be overridden: `{"effort": 3, "tabu_tenure": 14, "refinement": 600}`

See help_algorithm for more details of parameter tuning.

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