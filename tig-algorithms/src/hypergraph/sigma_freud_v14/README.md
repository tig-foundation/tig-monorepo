# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** sigma_freud_v14
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Further refined version of sigma_freud_high. In testing on actual live benchmarks it achieved higher qualities and faster runtimes than any other solver.

Use `null` for all tracks as hyperparameters, however for higher qualities, the `refinement` hyperparameter can be used at the expense of longer runtime, however, runtimes were still lower than the current top contender.

Fixed track 200k `#define CM_GROUPS_MAX 128` in the 10k kernel which incorrectly compiled in the TIG GitHub CI and originally was set to 16 causing the 200k track to crash.

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
