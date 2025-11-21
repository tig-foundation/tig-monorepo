# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** hypeman
* **Copyright:** 2025 PresentG  
* **Identity of Submitter:** PresentG  
* **Identity of Creator of Algorithmic Method:** PresentG  
* **Unique Algorithm Identifier (UAI):** null


## References and Acknowledgments

### Academic Papers

* Fiduccia, C. M., & Mattheyses, R. M. (1982). "A linear-time heuristic for improving network partitions". Design Automation, 1982. 19th Conference on. DOI: 10.1109/DAC.1982.1585498

### Other

Register Blocking Technique: Custom implementation of register-blocked connectivity checks to minimize global memory latency during the refinement phase.

## Additional Notes

This algorithm implements a highly optimized parallel version of the Fiduccia-Mattheyses (FM) refinement heuristic for hypergraph partitioning.

Key Innovations:

* **Register-Blocked Kernels: Dynamically detects low-degree nodes (degree <= 32) and promotes their connectivity data into GPU registers. This avoids the high latency of global memory lookups common in standard implementations.
* **Hardware Intrinsics: Utilizes __popcll (population count) for single-cycle connectivity metric calculations.
* **Monte Carlo Refinement: Leverages the high throughput of the register-blocked kernel to perform massive parallel random-restart refinement, exploring a significantly larger solution space than baseline solvers.

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