# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** gordian_v1
* **Copyright:** 2026 Danske
* **Identity of Submitter:** Danske
* **Identity of Creator of Algorithmic Method:** Discovered by private Prometheus swarm (agent DanskVS-1, model Qwen Q8) — a sibling refinement of mainnet's sigma_freud_v8, alongside sigma_freud_v9
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Sibling of `sigma_freud_v9` — both descend from mainnet's `sigma_freud_v8`,
independently tuned. Dispatches on `challenge.num_hyperedges` to a
size-tuned local-search/refinement solver, each paired with its own CUDA
kernel (`track_10k`/`track_20k`/`track_50k`/`track_100k`, with matching
`kernels_*k.cu`); the largest instance size (200,000 hyperedges) shares
the 100k-tier solver rather than using a separate path. Relative to
`sigma_freud_v9`, this variant removes an early-exit condition from the
50k refinement kernel and widens the 100k solver's search window with a
round-dependent aspiration threshold (stricter in early rounds), trading a
small loss on the 100k track for a larger gain on the 200k track that
shares its solver.

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
