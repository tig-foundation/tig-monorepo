## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** autovector_v9
* **Copyright:** 2026 NVX
* **Identity of Submitter:** NVX
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

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

## Overview

AutoVector v9 — Warp-per-query nearest neighbor search with frequent inter-batch bound sharing.

32 threads (1 warp) per query with strided database scan. The database is processed in small
batches (default 1024 vectors). After each batch, the warp-reduced global best distance is
persisted to GPU memory and loaded by all 32 threads at the start of the next batch. This
shares the tightest bound across all threads between batches, enabling dramatically more
early exits in the bounded L2 distance computation.
