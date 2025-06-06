# tig-binary

A Rust crate that wraps an algorithm from [`tig-algorithm`](../tig-algorithms/README.md) for compilation into a shared object.

TIG uses a forked version of [LLVM compiler](https://github.com/tig-foundation/llvm) for injecting runtime signature and fuel tracking into the shared object.

For CUDA, TIG uses `nvcc` to generate ptx using target version `sm_70/compute_70`, before injecting runtime signature and fuel tracking into the ptx file.

# Getting Started

1. Run the [development docker image](../README.md#docker-images) for your challenge. Example:
```
# clone this repo
cd tig-monorepo
CHALLENGE=knapsack
VERSION=0.0.1
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION
```

2. Build shared object using `build_algorithm` script:
  * Expects `tig_algorithm::<CHALLENGE>::<ALGORITHM>::solve_challenge` to be importable
  * Outputs to `tig-algorithms/lib/<CHALLENGE>/<ARCH>/<ALGORITHM>.so`, where `ARCH` is arm64 or amd64
  * `CHALLENGE` is determined by your docker image
  ```
  build_algorithm $ALGORITHM
  ```

Notes:
* To specifically build the shared object, you can use `build_so` script which is on `PATH`
* To specifically build the ptx, you can use `build_ptx` script which is on `PATH`
  * Expects `tig_algorithm/src/<CHALLENGE>/<ALGORITHM>.cu` or `tig_algorithm/src/<CHALLENGE>/<ALGORITHM>/benchmarker_outbound.cu` file to exist
  * Outputs to `tig-algorithms/lib/<CHALLENGE>/ptx/<ALGORITHM>.ptx`
* To test your binary, you can use `test_algorithm` script which is on `PATH`. Example:
  ```
  build_algorithm sat_global_opt
  test_algorithm sat_global_opt [1000,300]
  ```

## Complete Example

```
git clone https://github.com/tig-foundation/tig-monorepo -b test/satisfiability/sat_global_opt
cd tig-monorepo

CHALLENGE=satisfiability
VERSION=0.0.1
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION

# inside docker
build_algorithm sat_global_opt
test_algorithm sat_global_opt [1000,300]
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)