# The Innovation Game

This repository contains the implementation of The Innovation Game (TIG).

## Important Links

* [TIG Whitepaper](docs/whitepaper.pdf)
* [TIG Tech Explainer](docs/tech/1_basics.md)
* [TIG Licensing Explainer](docs/guides/anatomy.md)
* [Getting Started with Innovating](docs/guides/innovating.md)
* [Challenge Descriptions](docs/challenges/satisfiability.md)

## Repo Contents

* [tig-algorithms](./tig-algorithms/README.md) - A Rust crate that hosts algorithm submissions made by Innovators in TIG
* [tig-api](./tig-api/README.md) - A Rust crate for making requests to TIG's API
* [tig-benchmarker](./tig-benchmarker/README.md) - A Rust crate that implements a Benchmarker for TIG that can run in the browser
* [tig-challenges](./tig-challenges/README.md) - A Rust crate that contains the implementation of TIG's challenges (computational problems adapted for proof-of-work)
* [tig-protocol](./tig-protocol/README.md) - A Rust crate that contains the implementation of TIG's core protocol logic.
* [tig-structs](./tig-structs/README.md) - A Rust crate that contains the definitions of structs used throughout TIG
* [tig-token](./tig-token/README.md) - Solidity contract for TIG's ERC20 token that is deployed on Ethereum L2 Base chain
* [tig-utils](./tig-utils/README.md) - A Rust crate that contains utility functions used throughout TIG
* [tig-wasm](./tig-wasm/README.md) - A Rust crate for wrapping algorithm submissions for compilation into WASM with an exported `entry_point`
* [tig-worker](./tig-worker/README.md) - A Rust crate for verifying and computing solutions

## Useful Scripts

Under `scripts/` folder is a bunch of useful bash scripts:

* `list_algorithms.sh`
* `list_benchmark_ids.sh`
* `list_challenges.sh`
* `test_performance.sh`
* `verify_benchmark_solutions.sh`

## License

See README for individual folders