# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** there_v7
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

`there_v7` is an optimised version of `there_v6` that evaluates all database candidates using tiled GPU matrix multiplication. Inspiration taken from `autovector_final3`

Main points:
- Exhaustive nearest-neighbour search over all database vectors
- Uses cuBLAS GEMM to compute tiled query/database dot products
- Uses precomputed vector norms and a custom CUDA reduction kernel to recover the argmin
- Uses FP16 padded GEMM on the standard tracks for improved tensor-core throughput
- Built to improve runtime and fuel efficiency compared with earlier versions

Fuel note:
- Tested on all 20 benchmark nonces per track with default 100b fuel on an RTX 5070Ti.
- Benchmarker fuel up to 500b should provide additional headroom on supported GPUs.

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