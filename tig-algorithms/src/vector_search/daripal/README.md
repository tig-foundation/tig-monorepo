# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** daripal
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|---|---|---|
| n_queries=7000 | 5T | `null` |
| n_queries=9000 | 5T | `null` |
| n_queries=11000 | 5T | `null` |
| n_queries=13000 | 5T | `null` |
| n_queries=15000 | 5T | `null` |

No hyperparameters; any map is ignored.

## Method

Exact 1-NN. Same pipeline as there_v11: fp16 packing with folded half-norms, an fp16 tensor-core
screen (WMMA m32n8k16, fp16 accumulate) that keeps the best two scores per query in each
database chunk, then an exact fp32 rerank of every candidate within 0.5 of the best score.

Changes from there_v11: a faster screen kernel and a faster exact rescan of close chunks.

Answers are identical to there_v11.

## Results

Official runtime and verifier, 48 nonces per run, 4 rounds alternating the codes. Time per nonce (runtime
plus verifier), ratio daripal / there_v11:

| n_queries | 7000 | 9000 | 11000 | 13000 | 15000 |
|---|---|---|---|---|---|
| daripal / there_v11 | 0.989 | 0.982 | 0.979 | 0.970 | 0.964 |

Same qualities and solution hashes as there_v11 on all 100 nonces compared (20 per track). helix_search
was 1-4% slower than there_v11.

GPU memory per process: 6 MB (7000) to 28 MB (15000) more than there_v11, at most about 1.2%.

## References and Acknowledgments

### 1. Academic Papers
- J. Johnson, M. Douze, H. Jegou, "Billion-scale similarity search with GPUs", IEEE Transactions on Big Data, 2019.

### 2. Code References
- there_v11 (c004_a107), Rootz: packing kernel, exact distance and rerank code, and the screening
  method. Copyright remains with Rootz.
- NVIDIA CUDA C++ Programming Guide, WMMA API.

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
