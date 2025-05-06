# tig-runtime

A Rust crate that execute an algorithm (compiled from [`tig-binary`](../tig-binary/README.md)) for a single nonce, generating runtime signature and fuel consumed for verification.

# Getting Started

Users who don't intend to customise `tig-runtime` are recommended to download pre-compiled version available in [TIG's runtime docker images](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fruntime).

**Example:**
```
docker run -it ghcr.io/tig-foundation/tig-monorepo/runtime:0.0.1-aarch64
# tig-runtime is already on PATH
```

## Compiling (using dev docker image)

The required rust environment for development are available via [TIG's development docker images](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fdev).


**Example:**
```
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/dev:0.0.1-aarch64
# cargo build -p tig-runtime --release
```

## Compiling (local setup)

Users who intend to customise `tig-runtime` need to install a specific version of rust:

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

2. Compile `tig-runtime`
```
# for cuda version, add --features cuda
cargo build -p tig-runtime --release --target $RUST_TARGET
```

# Usage

```
Usage: tig-runtime [OPTIONS] <SETTINGS> <RAND_HASH> <NONCE> <BINARY>

Arguments:
  <SETTINGS>   Settings json string or path to json file
  <RAND_HASH>  A string used in seed generation
  <NONCE>      Nonce value
  <BINARY>     Path to a shared object (*.so) file

Options:
      --ptx [<PTX>]             Path to a CUDA ptx file
      --fuel [<FUEL>]           Optional maximum fuel parameter [default: 2000000000]
      --output [<OUTPUT_FILE>]  If set, the output data will be saved to this file path (default json)
      --compress [<COMPRESS>]   If output file is set, the output data will be compressed as zlib [default: false] [possible values: true, false]
      --gpu [<GPU>]             Which GPU device to use [default: 0]
  -h, --help                    Print help
```

The following exit codes indicate specific meanings:
* 0 - solution found
* 85 - no solution found
* 86 - runtime error
* 87 - out of fuel

**Example:**
```
SETTINGS='{"challenge_id":"c001","difficulty":[50,300],"algorithm_id":"","player_id":"","block_id":""}'
RANDHASH='rand_hash'$
NONCE=1337
FUEL=987654321123456789
SO_PATH=./tig-algorithms/lib/satisfiability/aarch64/better_sat.so

tig-runtime $SETTINGS $RANDHASH $NONCE $SO_PATH --fuel $FUEL
```

**Example Output:**
```
{"cpu_arch":"aarch64","fuel_consumed":95496,"nonce":1337,"runtime_signature":4125818588297548058,"solution":{"variables":[0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0]}}
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)