# AutoVector Native v2.0-beta  
**High-Performance Vector Range Search for the TIG Challenge**  
---
## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** autovector_native_v2
* **Copyright:** 2025 Brent Beane
* **Identity of Submitter:** Brent Beane
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

AutoVector Native is a GPU-accelerated vector range search algorithm engineered for **maximum throughput**, **deterministic execution**, and **resilience under fuel-limited environments**. Designed specifically for the TIG Vector Search Challenge, it combines **adaptive clustering**, **coalesced memory access**, and **batched solution persistence** to deliver state-of-the-art performance without sacrificing compliance.

Unlike conventional approaches that rely on complex synchronization or approximate filtering, AutoVector Native uses a **radical simplification strategy**: fixed small cluster counts, exhaustive intra-cluster search, and minimal kernel overhead — enabling predictable, scalable performance across difficulty levels.

This version (`v2.0-beta`) introduces Sigma I & II compliance, dynamic probe scheduling, and hybrid CPU-GPU coordination for robust operation under real contest constraints.

---

## Key Features

| Feature | Description |
|-------|-------------|
| ✅ **Sigma I Compliant** | No oracle access; only uses permitted challenge fields |
| ✅ **Sigma II Compliant** | Saves partial solutions after every query batch (256 queries) |
| 🔁 **Adaptive Clustering** | Dynamically computes optimal centroid count based on database size, dimensionality, and query load |
| 🧠 **Multi-Probe Search** | Supports both exhaustive and adaptive probe strategies via hyperparameters |
| ⚙️ **GPU-Optimized Kernels** | All core operations (centroid selection, assignment, indexing, search) executed on GPU |
| 💾 **Progress Preservation** | Batched `save_solution()` ensures partial results survive fuel exhaustion |
| 🚀 **High Occupancy** | Uses 256-thread blocks, coalesced memory access, and shared memory where safe |

---

## Algorithm Design

### 1. **Centroid Initialization**
- Centroids selected via strided sampling from database vectors using `select_centroids_strided`.
- Number of centroids adaptively scaled:
  
```rust
  base = √N
  dim_factor = 1.0 + ln(dim / 100)
  query_factor = √(num_queries / 10)
  adaptive_count = (base × dim_factor / query_factor).clamp(8, 64)
  
