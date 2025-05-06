# tig-worker

A Rust crate for executing a batch of instances using [`tig-runtime`](../tig-runtime/README.md), aggregating outputs, and calculating Merkle root.

# Getting Started

`tig-worker` executes a number of `tig-runtime` concurrently. Each `tig-runtime` loads an algorithm shared object (compiled from `tig-binary`), which expects a specific version of rust standard libraries to be available on `LD_LIBRARY_PATH`. 

Users who don't intend to customise `tig-worker` are recommended to download pre-compiled version available in [TIG's runtime docker images](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fruntime).

**Example:**
```
docker run -it ghcr.io/tig-foundation/tig-monorepo/runtime:0.0.1-aarch64
# tig-worker is already on PATH
```

## Compiling (using dev docker image)

The required rust environment for development are available via [TIG's development docker images](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fdev).


**Example:**
```
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/dev:0.0.1-aarch64
# cargo build -p tig-worker --release
```

## Compiling (local setup)

Users who intend to customise `tig-worker` need to install a specific version of rust:

1. Install rust version `nightly-2025-02-10`
```
ARCH=$(uname -m)
RUST_TARGET=$(if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "aarch64-unknown-linux-gnu";
else
    echo "x86_64-unknown-linux-gnu";
fi)
rustup install nightly-2025-02-10
rustup default nightly-2025-02-10
rustup component add rust-src
rustup target add $RUST_TARGET
RUST_LIBDIR=$(rustc --print target-libdir --target=$RUST_TARGET)
ln -s $RUST_LIBDIR /usr/local/lib/rust
echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}:/usr/local/lib/rust\"" >> ~/.bashrc
```

2. Compile `tig-worker`
```
# for cuda version, add --features cuda
cargo build -p tig-worker --release --target $RUST_TARGET
```

# Usage

```
Usage: tig-worker [OPTIONS] <RUNTIME> <SETTINGS> <RAND_HASH> <START_NONCE> <NUM_NONCES> <BATCH_SIZE> <BINARY>

Arguments:
  <RUNTIME>      Path to tig-runtime executable
  <SETTINGS>     Settings json string or path to json file
  <RAND_HASH>    A string used in seed generation
  <START_NONCE>  Starting nonce
  <NUM_NONCES>   Number of nonces to compute
  <BATCH_SIZE>   Batch size for Merkle tree
  <BINARY>       Path to a shared object (*.so) file

Options:
      --ptx [<PTX>]               Path to a CUDA ptx file
      --fuel [<FUEL>]             Optional maximum fuel parameter for runtime [default: 2000000000]
      --workers [<WORKERS>]       Number of worker threads [default: 1]
      --output [<OUTPUT_FOLDER>]  If set, the data for nonce will be saved as '<nonce>.json' in this folder
  -h, --help                      Print help
```

**Example:**
```
SETTINGS='{"challenge_id":"c001","difficulty":[5000,415],"algorithm_id":"","player_id":"","block_id":""}'
RANDHASH='rand_hash'$
NONCE=1337
FUEL=987654321123456789
SO_PATH=./tig-algorithms/lib/satisfiability/aarch64/better_sat.so

tig-worker \tig-runtime $SETTINGS $RANDHASH $NONCE $SO_PATH --fuel $FUEL
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
SETTINGS='{"challenge_id":"c001","difficulty":[5000,415],"algorithm_id":"","player_id":"","block_id":""}'
START=0
NUM_NONCES=8
BATCH_SIZE=8
SO_PATH=./tig-algorithms/lib/satisfiability/aarch64/better_sat.so
RAND_HASH=random_string

tig-worker \tig-runtime $SETTINGS $RAND_HASH $START $NUM_NONCES $BATCH_SIZE $SO_PATH --workers 8
```

**Example Output:**
```
{"merkle_root":"ab3f7ea08a2b991217bd9b08299b063bc77a0239af3a826d3b0ea91ca3384f98","solution_nonces":[6,3,4,5,7,2,1,0]}
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)