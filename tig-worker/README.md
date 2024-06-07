# tig-worker

A Rust crate for verifying and computing solutions.

Solutions are computed by executing an algorithm in a WASM virtual machine ([TIG's fork of wasmi](https://github.com/tig-foundation/wasmi)).

## Testing Performance of Algorithms

Performance testing is done in a sandboxed WASM Virtual Machine.

**IMPORTANT**: You can compile / test existing algorithms as binary executables, but be sure to throughly vet the code beforehand for malicious routines!

1. Pull an existing algorithm or compile your own algorithm to WASM
2. Set environment variables to match the algorithm you are testing:
    ```
    export CHALLENGE=<challenge_name>
    export ALGORITHM=<algorithm_name>
    ```
3. Pick a difficulty & create `settings.json`:
    * [Difficulty for Satisfiability](../tig-challenges/src/satisfiability.rs#L12). Recommend `[50, 300]` for initial tests
    * [Difficulty for Vehicle Routing](../tig-challenges/src/vehicle_routing.rs#L7). Recommend `[40, 250]` for initial tests
    * [Difficulty for Knapsack](../tig-challenges/src/knapsack.rs#L8). Recommend `[50, 10]` for initial tests

    ```
    {
        "block_id": "",
        "algorithm_id": "",
        "challenge_id": "",
        "player_id": "",
        "difficulty": <your difficulty>
    }
    ```
4. Test the algorithm:
    ```
    cargo run -p tig-worker --release -- settings.json tig-algorithms/wasm/${CHALLENGE}/${ALGORITHM}.wasm
    ```

Notes:
* You can query the latest difficulty ranges via TIG's API:
    ```
    query https://mainnet-api.tig.foundation/get-block for <block_id>
    query https://mainnet-api.tig.foundation/get-challenges?block_id=<block_id> for qualifier_difficulties
    ```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)