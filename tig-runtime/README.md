# tig-runtime

A Rust crate that execute an algorithm (compiled from [`tig-binary`](../tig-binary/README.md)) for a single nonce, generating runtime signature and fuel consumed for verification.

# Getting Started

Users who don't intend to customise `tig-runtime` are recommended to download pre-compiled version available in [TIG's runtime docker images](../README.md#docker-images).

Note there is a different `tig-runtime` for each challenge.

**Example:**
```
CHALLENGE=knapsack
VERSION=0.0.1
docker run -it ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/runtime:$VERSION

# inside docker
tig-runtime --help
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
cargo build -p tig-runtime --release --features knapsack
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
* 84 - runtime error
* 85 - no solution found
* 86 - invalid solution
* 87 - out of fuel

**Example:**
```
CHALLENGE=satisfiability
VERSION=0.0.1
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/runtime:$VERSION

# inside docker
download_algorithm sat_global_opt --testnet

ARCH=$(if [ "$(uname -i)" = "aarch64" ] || [ "$(uname -i)" = "arm64" ] || [ "$(arch 2>/dev/null || echo "")" = "aarch64" ] || [ "$(arch 2>/dev/null || echo "")" = "arm64" ]; then
    echo "arm64"
else
    echo "amd64"
fi)
SETTINGS='{"challenge_id":"c001","difficulty":[50,300],"algorithm_id":"","player_id":"","block_id":""}'
RANDHASH='rand_hash'$
NONCE=1337
FUEL=987654321123456789
SO_PATH=./tig-algorithms/lib/satisfiability/$ARCH/sat_global_opt.so

tig-runtime $SETTINGS $RANDHASH $NONCE $SO_PATH --fuel $FUEL
```

**Example Output:**
```
{"cpu_arch":"arm64","fuel_consumed":97188,"nonce":1337,"runtime_signature":13607024390209669967,"solution":{"variables":[1,0,0,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]}}
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)