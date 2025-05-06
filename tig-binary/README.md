# tig-binary

A Rust crate that wraps an algorithm from [`tig-algorithm`](../tig-algorithms/README.md) for compilation into a shared object.

TIG uses a forked version of [LLVM compiler](https://github.com/tig-foundation/llvm) for injecting runtime signature and fuel tracking into the shared object.

For CUDA, TIG uses `nvcc` to generate ptx using target version `sm_70/compute_70`, before injecting runtime signature and fuel tracking into the ptx file.

# Getting Started

1. Run the appropiate [development docker image](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fdev). Available flavours are:
  * amd64 (x86_64 compatible)
  * aarch64
  * amd64-cuda12.6.3 (x86_64 compatible)
  * aarch64-cuda12.6.3
```
# example
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/dev:0.0.1-aarch64
# scripts build_so.sh and build_ptx.py are on PATH
```

2. Build shared object using `build_so.sh` script:
  * Expects `tig_algorithm::<CHALLENGE>::<ALGORITHM>::solve_challenge` to be importable
  * Outputs to `tig-algorithms/lib/<CHALLENGE>/<ARCH>/<ALGORITHM>.so`, where `ARCH` is aarch64 or amd64
  ```
  # add '--cuda' flag if building cuda algorithm
  build_so.sh $CHALLENGE $ALGORITHM
  ```

3. (Optional) Build ptx using `build_ptx.py` script:
  * Expects `tig_algorithm/src/<CHALLENGE>/<ALGORITHM>.cu` or `tig_algorithm/src/<CHALLENGE>/<ALGORITHM>/benchmarker_outbound.cu` file to exist
  * Outputs to `tig-algorithms/lib/<CHALLENGE>/ptx/<ALGORITHM>.ptx`
```
build_ptx.py $CHALLENGE $ALGORITHM
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)