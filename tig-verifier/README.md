# tig-verifier

A Rust crate that verifies a single solution or Merkle proof.

# Getting Started

Users who don't intend to customise `tig-verifier` are recommended to download pre-compiled version available in [TIG's runtime docker images](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fruntime).

**Example:**
```
docker run -it ghcr.io/tig-foundation/tig-monorepo/runtime:0.0.1-aarch64
# tig-verifier is already on PATH
```

## Compiling (using dev docker image)

The required rust environment for development are available via [TIG's development docker images](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fdev).


**Example:**
```
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/dev:0.0.1-aarch64
# cargo build -p tig-verifier --release
```

## Compiling (local setup)

Users who intend to customise `tig-verifier` need to install a specific version of rust:

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

2. Compile `tig-verifier`
```
# for cuda version, add --features cuda
cargo build -p tig-verifier --release --target $RUST_TARGET
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