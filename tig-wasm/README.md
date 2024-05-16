# tig-wasm

A Rust crate for wrapping algorithm submissions for compilation into WASM with an exported `entry_point`.

## Compiling Algorithm into WASM with entry-point

These steps replicate what TIG's CI does (`.github/workflows/build_algorithm.yml`):

1. Set environment variables to match the algorithm you are compiling:
    ```
    export CHALLENGE=<challenge_name>
    export ALGORITHM=<algorithm_name>
    ```
2. Compile your algorithm
    ```
    cargo build -p tig-wasm --target wasm32-wasi --release --features entry-point
    ```
3. Optimise the WASM and save it into `tig-algorithms/wasm`:
    ```
    mkdir -p tig-algorithms/wasm/${CHALLENGE}
    wasm-opt target/wasm32-wasi/release/tig_wasm.wasm -o tig-algorithms/wasm/${CHALLENGE}/${ALGORITHM}.wasm -O2 --remove-imports
    ```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)