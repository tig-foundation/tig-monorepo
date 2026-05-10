# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** there_v8
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

Main points:
- Exact argmin workflow with per-query best-distance reduction
- cuBLAS dot-product tiles for higher throughput
- FP16 GEMM on the 7000/9000 query tracks
- FP32 SGEMM on larger query tracks
- Dedicated CUDA kernels for norms, FP16 packing, and tile argmin reduction

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
