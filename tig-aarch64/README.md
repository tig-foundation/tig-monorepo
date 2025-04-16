# tig-aarch64

A Rust crate for wrapping algorithm submissions for compilation for aarch64 architecture with an exported `entry_point`.

## Compiling Algorithm into aarch64 shared library with entry-point

1. Build tig-aarch64 docker image
    ```
    cd tig-aarch64
    docker build -t tig-aarch64 .
    ```
2. Set environment variables to match the algorithm you are compiling:
    ```
    export CHALLENGE=<challenge_name>
    export ALGORITHM=<algorithm_name>
    ```
3. Compile your algorithm
    ```
    # navigate to tig-monorepo
    docker run -v $(pwd):/app -w /app tig-aarch64 $CHALLENGE $ALGORITHM
    ```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)