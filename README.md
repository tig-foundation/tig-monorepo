# The Innovation Game

This repository contains the implementation of The Innovation Game (TIG).

## Important Links

* [Quick Start for Benchmarkers](./tig-benchmarker/README.md#quick-start)
* [Quick Start for Innovators](./tig-algorithms/README.md#quick-start)
* [TIG Documentation](https://docs.tig.foundation/)
* [TIG Whitepaper](docs/whitepaper.pdf)
* [TIG Licensing Explainer](docs/guides/anatomy.md)
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

TIG docker images are hosted on [Github Packages](https://github.com/orgs/tig-foundation/packages), supporting `linux/arm64` and `linux/amd64` platforms.

For Innovators who are developing & compiling algorithms, there is a `dev` image for each challenge, containing the development environment:

* [satisfiability/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fsatisfiability%2Fdev)
* [vehicle_routing/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvehicle_routing%2Fdev)
* [knapsack/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fknapsack%2Fdev)
* [vector_search/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvector_search%2Fdev)
* [hypergraph/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%hypergraph%2Fdev)

For Benchmarkers to spin up slaves, there is a `benchmarker\slave` image along with a `runtime` image for each challenge. It is intended that these images are spun up as part of [`docker-compose-slave.yml`](tig-benchmarker/docker-compose-slave.yml) (see [this README](tig-benchmarker/README.md) for more details):

* [benchmarker/slave](https://github.com/orgs/tig-foundation/packages/container/package/tig-monorepo%2Fbenchmarker%2Fslave)
* [satisfiability/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fsatisfiability%2Fruntime)
* [vehicle_routing/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvehicle_routing%2Fruntime)
* [knapsack/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fknapsack%2Fruntime)
* [vector_search/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvector_search%2Fruntime)
* [hypergraph/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%hypergraph%2Fruntime)

For Benchmarkers to spin up a master, there is a series of images that are spun up as part of [`docker-compose-master.yml`](tig-benchmarker/docker-compose-master.yml) (see [this README](tig-benchmarker/README.md) for more details):

* [benchmarker/master](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fmaster)
* [benchmarker/ui](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fui)
* [benchmarker/postgres](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fpostgres)
* [benchmarker/nginx](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fnginx)

### Useful Scripts

As part of the `runtime` and `dev` images, there are a bunch of useful scripts available on `PATH`:

* `list_algorithms`
* `download_algorithm <algorithm_name_or_id>`
* `test_algorithm <algorithm_name> <difficulty>`

Notes:
* The docker will automatically set a CHALLENGE environment variable used by these scripts
  * e.g. `knapsack/runtime` docker will set `CHALLENGE=knapsack`
* Use `--testnet` option to target testnet

## License

See README for individual folders