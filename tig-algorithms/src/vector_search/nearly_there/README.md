# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** nearly_there
* **Copyright:** 2025 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

GPU-accelerated exhaustive brute-force vector search achieving perfect quality (argmin ceiling) across all tracks.

Key improvements: 32-way loop unrolling with early-exit pruning, adaptive batch sizing based on vector dimensions, and optimized initial bounds. 2.6x faster than previous algorithms with significantly lower fuel requirements.

Requires 100b fuel for tracks 1-3, 200b fuel for tracks 4-5.

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