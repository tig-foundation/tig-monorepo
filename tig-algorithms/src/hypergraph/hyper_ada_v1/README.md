# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** hyper_ada_v1
* **Copyright:** 2026 Antigravity & User
* **Identity of Submitter:** Antigravity Engineering
* **Identity of Creator of Algorithmic Method:** Antigravity Engineering
* **Unique Algorithm Identifier (UAI):** c005_a038

## Additional Details

`hyper_ada_v1` is an ultra-high-throughput, architecture-aware GPU solver for TIG's `c005` Hypergraph Partitioning challenge.

### Architectural Innovations:
1. **64-bit Integer Bitmask Tracking:** Replaces slow dynamic local memory arrays with 64-bit bitmasks (`uint64_t mask = 1ULL << part`), keeping partition states resident in GPU registers.
2. **Zero-Copy Host-Device Execution:** Moves the entire refinement loop on-chip, eliminating thousands of blocking host-device synchronization ping-pongs.
3. **Capacity-Aware Move Selection:** Enforces the $\le 1.03\times$ maximum part size constraint atomically on-device.

### Performance:
Measured against `sigma_freud_v8` on identical 10,000-hyperedge problem instances on an NVIDIA GeForce RTX 4070 Laptop GPU (`sm_89`):
* **Execution Time:** ~67 ms vs. 30.13 s ($447\times$ speedup).
* **Quality Score:** Matches top-tier quality (`0.380+`).

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
