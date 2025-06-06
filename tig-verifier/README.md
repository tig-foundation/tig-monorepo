# tig-verifier

A Rust crate that verifies a single solution or Merkle proof.

# Getting Started

Users who don't intend to customise `tig-verifier` are recommended to download pre-compiled version available in [TIG's runtime docker images](../README.md#docker-images).

Note there is a different `tig-verifier` for each challenge.

**Example:**
```
CHALLENGE=knapsack
VERSION=0.0.1
docker run -it ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/runtime:$VERSION

# inside docker
tig-verifier --help
```

## Compiling

The required rust environment for development are available via [TIG's development docker images](../README.md#docker-images).

You will need to add `--features <CHALLENGE>` to compile for a specific challenge.

**Example:**
```
# clone this repo
cd tig-monorepo
CHALLENGE=knapsack
VERSION=0.0.1
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION

# inside docker
cargo build -p tig-verifier --release --features knapsack
```

# Usage

`tig-verifier` will have multiple sub-commands, but currently only `verify_solution` is implemented:

```
Usage: tig-verifier verify_solution [OPTIONS] <SETTINGS> <RAND_HASH> <NONCE> <SOLUTION>

Arguments:
  <SETTINGS>   Settings json string or path to json file
  <RAND_HASH>  A string used in seed generation
  <NONCE>      Nonce value
  <SOLUTION>   Solution json string, path to json file, or '-' for stdin

Options:
      --ptx [<PTX>]  Path to a CUDA ptx file
      --gpu [<GPU>]  Which GPU device to use [default: 0]
  -h, --help         Print help
```

**Example:**
```
CHALLENGE=satisfiability
VERSION=0.0.1
docker run -it ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/runtime:$VERSION

# inside docker
SETTINGS='{"challenge_id":"c001","difficulty":[50,300],"algorithm_id":"","player_id":"","block_id":""}'
RANDHASH='rand_hash'
NONCE=1337
SOLUTION='{"variables":[1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,0,0]}'

tig-verifier verify_solution $SETTINGS $RANDHASH $NONCE $SOLUTION
```

**Example Output:**
```
Solution is valid
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)