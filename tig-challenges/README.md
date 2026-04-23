# tig-challenges

Implementations of TIG's challenges (computational problems adapted for optimisable proof-of-work).

Each challenge defines a problem instance generator, solution format, and verification logic. Innovators write algorithms that solve these challenges within a fuel budget.

## Challenges

| ID | Challenge | Description | CUDA | Documentation |
|----|-----------|-------------|------|---------------|
| c001 | satisfiability | Boolean Satisfiability (SAT) | No | [README](./src/satisfiability/README.md) |
| c002 | vehicle_routing | Capacitated Vehicle Routing with Time Windows | No | [README](./src/vehicle_routing/README.md) |
| c003 | knapsack | Quadratic Knapsack Problem | No | [README](./src/knapsack/README.md) |
| c004 | vector_search | Vector Range Search | Yes | [README](./src/vector_search/README.md) |
| c005 | hypergraph | Hypergraph Partitioning | Yes | [README](./src/hypergraph/README.md) |
| c006 | neuralnet_optimizer | Neural Network Optimizer | Yes (cublas/cudnn) | [README](./src/neuralnet_optimizer/README.md) |
| c007 | job_scheduling | Flexible Job Shop Scheduling | No | [README](./src/job_scheduling/README.md) |
| c008 | energy_arbitrage | Energy Market Arbitrage | No | [README](./src/energy_arbitrage/README.md) |
| c009 | zk_optimization | ZK Circuit Optimization | No | [README](./src/zk_optimization/README.md) |

## Cargo Features

Each challenge is a Cargo feature (e.g. `--features c003` for knapsack). CUDA-enabled challenges (`c004`, `c005`, `c006`) pull in the `cudarc` dependency.

## License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)
