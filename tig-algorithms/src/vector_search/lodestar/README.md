# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** lodestar
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

GPU exact nearest-neighbour search. For each query the solver finds the closest
database vector under Euclidean distance by a chunked distance scan that runs on
the tensor cores: the query and database tiles are packed in fp16 and the inner
products are accumulated with the WMMA API, with the `-1/2*||d||^2` term folded
into padding lanes so a single matrix-multiply yields the distance ranking
directly. Candidates from the fast fp16 pass are re-ranked exactly in fp32, so the
returned neighbour is exact despite the fp16 scan. The PTX module carries only the
kernels actually on the shipped path (it is reloaded per invocation, so unused
kernels cost time). All tuning — base chunk size, warps per block, WMMA fragment
geometry, and the re-rank width — is exposed as hyperparameters read in
`from_map`; the defaults are the shipped operating point.

## References and Acknowledgments

### Academic Papers
- J. Johnson, M. Douze, H. Jégou, *"Billion-scale similarity search with GPUs"*, IEEE Transactions on Big Data, 2019 (GPU brute-force k-selection).
- H. Jégou, M. Douze, C. Schmid, *"Product Quantization for Nearest Neighbor Search"*, IEEE TPAMI, 2011.
- NVIDIA, *"CUDA C++ Programming Guide — Warp Matrix Functions (WMMA)"* (tensor-core matrix multiply-accumulate).

### Code References
- TIG baseline (vector_search).

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
