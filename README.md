# The Innovation Game

This repository contains the implementation of The Innovation Game (TIG).

## Important Links

* [TIG Documentation](https://docs.tig.foundation/)
* [TIG Whitepaper](docs/whitepaper.pdf)
* [TIG Licensing Explainer](docs/guides/anatomy.md)
* [Getting Started with Innovating](docs/guides/innovating.md)
* [Implementations vs Breakthroughs](docs/guides/breakthroughs.md)
* [Voting Guidelines for Token Holders](docs/guides/voting.md)

## Repo Contents

* [tig-algorithms](./tig-algorithms/README.md) - A Rust crate that hosts algorithm submissions made by Innovators in TIG
* [tig-benchmarker](./tig-benchmarker/README.md) - Python scripts for running TIG's benchmarker in master/slave configuration
* [tig-binary](./tig-binary/README.md) - A Rust crate that wraps an algorithm from [`tig-algorithm`](./tig-algorithms/README.md) for compilation into a shared object.
* [tig-breakthroughs](./tig-breakthroughs/README.md) - A folder that hosts submissions of algorithmic methods made by Innovators in TIG.
* [tig-challenges](./tig-challenges/README.md) - A Rust crate that contains the implementation of TIG's challenges (computational problems adapted for proof-of-work)
* [tig-protocol](./tig-protocol/README.md) - A Rust crate that contains the implementation of TIG's core protocol logic.
* [tig-runtime](./tig-runtime/README.md) - A Rust crate that execute an algorithm (compiled from [`tig-binary`](./tig-binary/README.md)) for a single nonce, generating runtime signature and fuel consumed for verification.
* [tig-structs](./tig-structs/README.md) - A Rust crate that contains the definitions of structs used throughout TIG
* [tig-token](./tig-token/README.md) - Solidity contract for TIG's ERC20 token that is deployed on Ethereum L2 Base chain
* [tig-utils](./tig-utils/README.md) - A Rust crate that contains utility functions used throughout TIG
* [tig-verifier](./tig-verifier/README.md) - A Rust crate that verifies a single solution or Merkle proof.

## Docker Images

TIG docker images are hosted on [Github Packages](https://github.com/orgs/tig-foundation/packages):

* [dev](https://github.com/orgs/tig-foundation/packages/container/package/tig-monorepo%2Fdev) - environment for Innovators who are developing algorithms
* [runtime](https://github.com/orgs/tig-foundation/packages/container/package/tig-monorepo%2Fruntime) - environment for Benchmarkers who are running slaves

## Useful Scripts

Under `scripts/` folder is a bunch of useful scripts:

* `download_algorithm`
* `list_algorithms`
* `list_challenges`
* `test_algorithms`

These are available on `PATH` in the `dev` and `runtime` docker images

## License

See README for individual folders