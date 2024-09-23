# tig-worker

A Rust crate for verifying and computing solutions.

Solutions are computed by executing an algorithm in a WASM virtual machine ([TIG's fork of wasmi](https://github.com/tig-foundation/wasmi)).

# Compiling

```
cargo build -p tig-worker --release
./target/release/tig-worker --help
```

# Usage

`tig-worker` has sub-commands `verify_solution`, `compute_solution` and `compute_batch`. These are used in 2 scripts:

* [Test algorithm performance](../scripts/test_algorithm_performance.sh)
* [Verify benchmark solutions](../scripts/verify_benchmark_solutions.sh)

## Verify Solution

Given settings, nonce and a solution, `tig-worker` verifies the solution is a valid solution for the challenge instance.

* If the solution is valid, `tig-worker` will terminate with exit code 0

* If the solution is invalid, `tig-worker` will terminate with exit code 1

```
Usage: tig-worker verify_solution <SETTINGS> <RAND_HASH> <NONCE> <SOLUTION>

Arguments:
  <SETTINGS>   Settings json string or path to json file
  <RAND_HASH>  A string used in seed generation
  <NONCE>      Nonce value
  <SOLUTION>   Solution json string or path to json file

Options:
  -h, --help  Print help
```

**Example:**
```
SETTINGS='{"challenge_id":"c001","difficulty":[50,300],"algorithm_id":"","player_id":"","block_id":""}'
NONCE=7
SOLUTION='{"variables":[1,0,0,0,0,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0]}'
RAND_HASH=random_string
./target/release/tig-worker verify_solution $SETTINGS $RAND_HASH $NONCE $SOLUTION
```

## Compute Solution

Given settings, nonce and the WASM for an algorithm, `tig-worker` computes the solution data (runtime_signature, fuel_consumed, solution). This sub-command does not verify whether the solution is valid or not.

* If the algorithm results in an error, `tig-worker` will terminate with exit code 1 and print error to stderr.

* If the algorithm returns a solution, `tig-worker` will terminate with exit code 0 and print the solution data to stdout.

```
Usage: tig-worker compute_solution [OPTIONS] <SETTINGS> <RAND_HASH> <NONCE> <WASM>

Arguments:
  <SETTINGS>   Settings json string or path to json file
  <RAND_HASH>  A string used in seed generation
  <NONCE>      Nonce value
  <WASM>       Path to a wasm file

Options:
      --fuel [<FUEL>]  Optional maximum fuel parameter for WASM VM [default: 1000000000]
      --mem [<MEM>]    Optional maximum memory parameter for WASM VM [default: 1000000000]
  -h, --help           Print help
```

**Example:**
```
SETTINGS='{"challenge_id":"c001","difficulty":[50,300],"algorithm_id":"","player_id":"","block_id":""}'
NONCE=7
WASM=./tig-algorithms/wasm/satisfiability/schnoing.wasm
RAND_HASH=random_string
./target/release/tig-worker compute_solution $SETTINGS $RAND_HASH $NONCE $WASM 
```

# Compute Batch
Computes a batch of nonces and generates Merkle root and proofs.

```
Usage: tig-worker compute_batch [OPTIONS] <SETTINGS> <RAND_HASH> <START_NONCE> <NUM_NONCES> <BATCH_SIZE> <WASM>

Arguments:
  <SETTINGS>     Settings json string or path to json file
  <RAND_HASH>    A string used in seed generation
  <START_NONCE>  Starting nonce
  <NUM_NONCES>   Number of nonces to compute
  <BATCH_SIZE>   Batch size for Merkle tree
  <WASM>         Path to a wasm file

Options:
      --fuel [<FUEL>]                Optional maximum fuel parameter for WASM VM [default: 2000000000]
      --mem [<MEM>]                  Optional maximum memory parameter for WASM VM [default: 1000000000]
      --sampled <SAMPLED_NONCES>...  Sampled nonces for which to generate proofs
      --workers [<WORKERS>]          Number of worker threads [default: 1]
  -h, --help                         Print help
```

**Example:**
```
SETTINGS='{"challenge_id":"c001","difficulty":[50,300],"algorithm_id":"","player_id":"","block_id":""}'
START_NONCE=0
NUM_NONCES=1000
BATCH_SIZE=1024
WASM=./tig-algorithms/wasm/satisfiability/schnoing.wasm
RAND_HASH=random_string
./target/release/tig-worker compute_batch $SETTINGS $RAND_HASH $START_NONCE $NUM_NONCES $BATCH_SIZE $WASM 
```

# Notes

* `challenge_id` must be set:
  * `c001` is satisfiability
  * `c002` is vehicle_routing
  * `c003` is knapsack
  * `c004` is vector_search
* Recommended low difficulties for testing are:
  * satisfiability [50,300]
  * vehicle_routing [40, 250]
  * knapsack [50, 10]
  * vector_search [10, 350]
* You can query the latest difficulties by using `scripts/list_challenges.sh`
* You can test the performance of an algorithm using `scripts/test_algorithm.sh`
* You can list all active benchmark ids with `scripts/list_benchmark_ids.sh`
* You can get benchmark data with `scripts/list_benchmark_ids.sh`
* You can verify a benchmark's solutions, runtime_signature and fuel_consumed with `scripts/verify_benchmark.sh`

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)