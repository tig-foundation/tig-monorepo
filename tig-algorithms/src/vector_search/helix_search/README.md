# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** helix_search
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

Exhaustive nearest-neighbour search executed on the GPU through Tensor-Core
(WMMA) fp16 matrix multiplication. The database is streamed in cache-resident
chunks; a fast fp16 accumulation produces a shortlist of candidates per query,
which are then re-ranked exactly in fp32 within a bounded distance filter
(coarse-to-fine retrieval). Vectors are packed as `half2` and padded to a
256-lane tile to feed the Tensor-Core pipeline. All tuning is exposed as
hyperparameters read in `from_map`; the defaults are the shipped operating point.

## Reference Comparison

`helix_search` against two established vector_search algorithms, `there_v10`
and `autovector_i`, measured at **identical recall** (same bundle quality per
track) on a single NVIDIA GeForce RTX 3090 (one worker), 10-nonce timing
batch, default hyperparameters. Wall-clock seconds per batch:

| n_queries | recall (bundle) | helix_search | autovector_i | there_v10 |
|-----------|-----------------|--------------|--------------|-----------|
| 7,000     | 71,608          | 4.24 s       | 4.77 s       | 4.50 s    |
| 9,000     | 73,605          | 4.48 s       | 4.61 s       | 4.87 s    |
| 11,000    | 75,120          | 5.36 s       | 5.22 s       | 5.96 s    |
| 13,000    | 76,312          | 5.40 s       | 5.84 s       | 6.38 s    |
| 15,000    | 77,406          | 6.11 s       | 6.57 s       | 7.01 s    |

Recall is identical across all three; on a saturated challenge, where quality is
capped by the network cutoff, wall-clock throughput is the deciding factor.
`helix_search` is fastest on four of the five sizes.

## References and Acknowledgments

### Academic Papers
- J. Johnson, M. Douze, H. Jégou, *"Billion-scale similarity search with GPUs"*, IEEE Transactions on Big Data, 2019.
- S. Markidis, S. W. D. Chien, E. Laure, I. B. Peng, J. S. Vetter, *"NVIDIA Tensor Core Programmability, Performance and Precision"*, IEEE IPDPSW, 2018.
- H. Jégou, M. Douze, C. Schmid, *"Product Quantization for Nearest Neighbor Search"*, IEEE TPAMI, 2011 — coarse-to-fine reranking principle.

### Code References
- NVIDIA CUDA C++ Programming Guide — Warp-level Matrix Multiply-Accumulate (WMMA) API.
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
