# tig-protocol

A Rust crate that contains the implementation of TIG's core protocol logic.

1. `context.rs` defines the Context trait which the TIG protocol interacts with to query & update data
2. `add_block.rs` implements the state transitions that occur when a new block is created. These transitions include:
    * confirm mempool items (benchmarks, algorithms, and more)
    * update player cutoffs
    * for each challenge, update solution signature threshold, qualifiers, and frontiers
    * using optimisable proof-of-work to calculate influence and adoption
    * distribution block rewards amongst benchmarkers and innovators
    * updating algorithm merge points
3. `submit_algorithm.rs`/`submit_benchmark.rs`/`submit_proof.rs` implements the logic for validating an algorithm/benchmark/proof submission before adding it to the mempool
4. `verify_proof.rs` implements the logic for validating the runtime signature of a solution

# License

[Download Agreement](../docs/agreements/download_agreement.pdf)