# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** there_v10
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Main points:
- Exact argmin workflow with per-query best-distance reduction
- Padded FP16 vector representation for the standard tracks
- Track-specialized tensor-core WMMA search kernels for n_queries=7000, 9000, 11000, 13000, and 15000
- Fused or regular preprocessing paths selected per track
- Cached CUDA function handles and reusable execution context to reduce setup overhead
- cuBLAS FP32 GEMM fallback for non-standard query counts

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
