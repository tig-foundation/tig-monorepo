<h1 align="center">
  <a href="https://tig.foundation/"><img src="docs/images/logo_black.png" width="75" alt="TIG logo" /></a><br>
  <b>The Innovation Game</b><br>
  <sub>The Network for Algorithmic Breakthroughs </sub>
</h1>
<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-%E2%89%A51.70-orange?logo=rust&logoColor=white" alt="Rust" /></a>
  <a href="https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses"><img src="https://img.shields.io/badge/license-TIG-0ea5e9.svg" alt="License" /></a>
  <a href="https://docs.tig.foundation/"><img src="https://img.shields.io/badge/docs-tig.foundation-6366f1.svg" alt="Docs" /></a>
  <a href="docs/whitepaper.pdf"><img src="https://img.shields.io/badge/whitepaper-PDF-B31B1B.svg" alt="Whitepaper" /></a>
  <a href="https://play.tig.foundation/dashboard"><img src="https://img.shields.io/badge/play-dashboard-10b981.svg" alt="Play" /></a>
  <a href="https://discord.gg/tigfoundation"><img src="https://img.shields.io/badge/Discord-join-5865F2.svg?logo=discord&logoColor=white" alt="Discord" /></a>
  <a href="https://x.com/tigfoundation"><img src="https://img.shields.io/badge/follow-%40tigfoundation-000000.svg?logo=x&logoColor=white" alt="X" /></a>
</p>

The Innovation Game (TIG) creates a new economic framework for algorithmic development - one that aligns incentives, rewards contribution, and keeps innovation open.

At its core, TIG uses a novel proof-of-work variant built around computational challenges grounded in scientifically important problems. Innovators submit algorithms that solve these challenges, and benchmarkers are incentivized to adopt the most efficient ones for proof-of-work, creating a manipulation-resistant signal for rewarding the top-performing algorithms.

In this way, TIG democratizes algorithmic innovation, turning contribution into a sustainable economic opportunity, coordinating global intelligence to compete with centralized incumbents.



## Challenges

TIG currently has 8 active computational challenges:

| ID | Challenge | Description | CPU/GPU |
|----|-----------|-------------|------|
| c001 | [satisfiability](tig-challenges/src/satisfiability/README.md) | Boolean Satisfiability (SAT) | CPU |
| c002 | [vehicle_routing](tig-challenges/src/vehicle_routing/README.md) | Capacitated Vehicle Routing with Time Windows | CPU |
| c003 | [knapsack](tig-challenges/src/knapsack/README.md) | Quadratic Knapsack Problem | CPU |
| c004 | [vector_search](tig-challenges/src/vector_search/README.md) | Vector Range Search | GPU |
| c005 | [hypergraph](tig-challenges/src/hypergraph/README.md) | Hypergraph Partitioning | GPU |
| c006 | [neuralnet_optimizer](tig-challenges/src/neuralnet_optimizer/README.md) | Neural Network Optimizer | GPU |
| c007 | [job_scheduling](tig-challenges/src/job_scheduling/README.md) | Flexible Job Shop Scheduling | CPU |
| c008 | [energy_arbitrage](tig-challenges/src/energy_arbitrage/README.md) | Energy Market Arbitrage | CPU |

## Glossary

| Term | Definition |
|------|-----------|
| **Innovator** | Participant who submits algorithms (code or advances) to solve challenges |
| **Benchmarker** | Participant who runs algorithm benchmarks and submits proofs |
| **Challenge** | A computational problem adapted for optimisable proof-of-work |
| **Code** | An algorithm source code submission by an Innovator |
| **Advance** | An algorithm improvement submission (documentation/paper) by an Innovator |
| **Fuel** | Computational cost metric — algorithms must solve within a fuel budget |
| **OPoW** | Optimisable Proof of Work — TIG's core consensus mechanism |
| **Nonce** | Input seed for a single benchmark run |
| **Runtime Signature** | Hash produced during algorithm execution, used for verification |
| **Binary** | A compiled shared object (`.so`) built from an algorithm submission |

## Important Links

* [Getting Started for Innovators](https://docs.tig.foundation/innovating)
* [Getting Started for Benchmarkers](https://docs.tig.foundation/benchmarking)
* [TIG Documentation](https://docs.tig.foundation/)
* [TIG Whitepaper](docs/whitepaper.pdf)
* [TIG Licensing Explainer](docs/guides/anatomy.md)
* [Code vs Advances](docs/guides/advances.md)
* [Voting Guidelines for Token Holders](docs/guides/voting.md)

## Repo Contents

| Crate | Description |
|-------|-------------|
| [tig-algorithms](./tig-algorithms/README.md) | Hosts algorithm submissions (code and advances) made by Innovators |
| [tig-benchmarker](./tig-benchmarker/README.md) | Python scripts for running TIG's benchmarker in master/slave configuration |
| [tig-binary](./tig-binary/README.md) | Wraps an algorithm submission for compilation into a shared object |
| [tig-challenges](./tig-challenges/README.md) | Implementations of TIG's 8 computational challenges |
| [tig-protocol](./tig-protocol/README.md) | Core protocol logic (block processing, submissions, verification) |
| [tig-runtime](./tig-runtime/README.md) | Executes a compiled algorithm for a single nonce, generating runtime signature and fuel consumed |
| [tig-structs](./tig-structs/README.md) | Shared struct definitions used throughout TIG |
| [tig-token](./tig-token/README.md) | Solidity ERC20 token contract deployed on Ethereum L2 Base chain |
| [tig-utils](./tig-utils/README.md) | Utility functions (hashing, Merkle trees, serialization, etc.) |
| [tig-verifier](./tig-verifier/README.md) | Verifies a single solution or Merkle proof |

## Docker Images

TIG Docker images are hosted on [GitHub Packages](https://github.com/orgs/tig-foundation/packages), supporting `linux/arm64` and `linux/amd64` platforms.

> **Note:** Check `tig-benchmarker/.env` for the current `VERSION` (currently `0.0.5`).

### Dev Images (for Innovators)

Development environment for writing and compiling algorithms:

| Challenge | Image |
|-----------|-------|
| satisfiability | [satisfiability/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fsatisfiability%2Fdev) |
| vehicle_routing | [vehicle_routing/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvehicle_routing%2Fdev) |
| knapsack | [knapsack/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fknapsack%2Fdev) |
| vector_search | [vector_search/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvector_search%2Fdev) |
| hypergraph | [hypergraph/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fhypergraph%2Fdev) |
| neuralnet_optimizer | [neuralnet_optimizer/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fneuralnet_optimizer%2Fdev) |
| job_scheduling | [job_scheduling/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fjob_scheduling%2Fdev) |
| energy_arbitrage | [energy_arbitrage/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fenergy_arbitrage%2Fdev) |

### Slave Images (for Benchmarkers)

Runtime images spun up as part of [`slave.yml`](tig-benchmarker/slave.yml) (see [benchmarker README](tig-benchmarker/README.md)):

| Component | Image |
|-----------|-------|
| Slave orchestrator | [benchmarker/slave](https://github.com/orgs/tig-foundation/packages/container/package/tig-monorepo%2Fbenchmarker%2Fslave) |
| satisfiability | [satisfiability/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fsatisfiability%2Fruntime) |
| vehicle_routing | [vehicle_routing/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvehicle_routing%2Fruntime) |
| knapsack | [knapsack/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fknapsack%2Fruntime) |
| vector_search | [vector_search/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvector_search%2Fruntime) |
| hypergraph | [hypergraph/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fhypergraph%2Fruntime) |
| neuralnet_optimizer | [neuralnet_optimizer/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fneuralnet_optimizer%2Fruntime) |
| job_scheduling | [job_scheduling/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fjob_scheduling%2Fruntime) |
| energy_arbitrage | [energy_arbitrage/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fenergy_arbitrage%2Fruntime) |

### Master Images (for Benchmarkers)

Spun up as part of [`master.yml`](tig-benchmarker/master.yml) (see [benchmarker README](tig-benchmarker/README.md)):

| Component | Image |
|-----------|-------|
| Master orchestrator | [benchmarker/master](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fmaster) |
| Dashboard UI | [benchmarker/ui](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fui) |
| PostgreSQL | [benchmarker/postgres](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fpostgres) |
| Nginx reverse proxy | [benchmarker/nginx](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fnginx) |

### Useful Scripts

The `runtime` and `dev` images include these scripts on `PATH`:

```bash
list_algorithms                              # List available algorithms for the challenge
download_algorithm <algorithm_name_or_id>    # Download an algorithm's source
test_algorithm <algorithm_name> <difficulty>  # Test an algorithm locally
```

> The container automatically sets the `CHALLENGE` environment variable (e.g. `knapsack/runtime` sets `CHALLENGE=knapsack`). Use `--testnet` to target testnet.

## License

See README for individual folders.
