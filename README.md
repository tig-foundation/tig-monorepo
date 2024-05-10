# The Innovation Game

This repository contains the implementation of The Innovation Game (TIG).

## Important Links

* [TIG's tech explainer](docs/1_basics.md)
* [Getting Started with Innovating](docs/1_basics.md)
* [Challenge descriptions](tig-challenges/docs/knapsack.md)

## Repo Contents
### tig-algorithms

A Rust crate that hosts algorithm submissions made by Innovators in TIG.

Submissions are committed to their own branch with name:

`<challenge_name>\<algorithm_name>` 

Submissions only get merged to the main branch after earning sufficient merge points.

WASM blobs for an algorithm are stored in the `wasm` subfolder and can be downloaded via:

`https://raw.githubusercontent.com/tig-foundation/tig-monorepo/<branch_name>/tig-algorithms/wasm/<branch_name>.wasm`
    
### tig-api

A Rust crate for making requests to TIG's API.

Developers must either enable feature `request` (uses `reqwest`) or `request-js` (uses `web-sys`) 

### tig-benchmarker

A Rust crate that implements a Benchmarker for TIG that can run in the browser. 

### tig-challenges

A Rust crate that contains the implementation of TIG's challenges (computational problems adapted for proof-of-work).

### tig-protocol

A Rust crate that contains the implementation of TIG's core protocol logic.

### tig-structs

A Rust crate that contains the definitions of structs used throughout TIG.

### tig-token

Solidity contract for TIG's ERC20 token that is deployed on Ethereum L2 Base chain.

### tig-utils

A Rust crate that contains utility functions used throughout TIG.

### tig-wasm

A Rust crate for wrapping algorithm submissions for compilation into WASM with an exported `entry_point`.

### tig-worker

A Rust crate for verifying and computing solutions.

Solutions are computed by executing an algorithm in a WASM virtual machine ([TIG's fork of wasmi](https://github.com/tig-foundation/wasmi)).

## Getting Started with Innovating

### Setting up Private Fork

Innovators will want to create a private fork so that they can test that their algorithm can be successfully compiled into WASM by the CI.

1. Create private repository on GitHub
2. Create empty git repository on your local machine
    ```
    mkdir tig-monorepo
    cd tig-monorepo
    git init
    ```
3. Setup remotes with origin pointed to your private repository
    ```
    git remote add origin <your private repo>
    git remote add public https://github.com/tig-foundation/tig-monorepo.git
    ```
    
4. Pulling `blank_slate` from TIG public repository (branch with no algorithms)
    ```
    git fetch public
    git checkout -b blank_slate
    git pull public blank_slate
    ```
    
5. Push to your private repository
    ```
    git push origin blank_slate
    ```

### Checking out Existing Algorithms

Every algorithm has its own `<branch>` with name `<challenge_name>/<algorithm_name>`.

Only algorithms that are successfully compiled into WASM have their branch pushed to this public repository.

Each algorithm branch will have 2 files:
1. Rust code @ `tig-algorithms/src/<branch>.rs`
2. Wasm blob @ `tig-algorithms/wasm/<branch>.wasm`

To pull an existing algorithm from TIG public repository, run the following command:
```
git fetch public
git pull public <branch>
```

### Developing Your Algorithm

1. Pick a challenge (`<challenge_name>`) to develop an algorithm for
2. Make a copy of an existing algorithm's rust code or `tig-algorithms/<challenge_name>/template.rs`
3. Rename the file with your own `<algorithm_name>`
4. Edit `tig-algorithms/<challenge_name>/mod.rs` to export your algorithm and test it:
    ```
    pub mod <algorithm_name>;

    #[cfg(test)]
    mod tests {
        use super::*;
        use tig_challenges::{<challenge_name>::*, *};

        #[test]
        fn test_<algorithm_name>() {
            let difficulty = Difficulty {
                // Uncomment the relevant fields.

                // -- satisfiability --
                // num_variables: 50,
                // clauses_to_variables_percent: 300,
                
                // -- vehicle_routing --
                // num_nodes: 40,
                // better_than_baseline: 250,
                
                // -- knapsack --
                // num_items: 50,
                // better_than_baseline: 10,
            };
            let seed = 0;
            let challenge = Challenge::generate_instance(seed, &difficulty).unwrap();    
            <algorithm_name>::solve_challenge(&challenge).unwrap();
        }
    }
    ```
5. Check that your algorithm compiles & runs:
    ```
    cargo test -p tig-algorithms
    ```

Notes:
* Do not include tests in your algorithm file. TIG will reject your algorithm submission.
* Only your algorithm's rust code gets submitted. You should not be adding dependencies to `tig-algorithms` as they will not be available when TIG compiles your algorithm

### Locally Compiling Your Algorithm into WASM 

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

### Testing Performance of Algorithms

Performance testing is done by `tig-worker` in a sandboxed WASM Virtual Machine.

**IMPORTANT**: You can compile / test existing algorithms as binary executables, but be sure to throughly vet the code beforehand for malicious routines!

1. Pull an existing algorithm or compile your own algorithm to WASM
2. Set environment variables to match the algorithm you are testing:
    ```
    export CHALLENGE=<challenge_name>
    export ALGORITHM=<algorithm_name>
    ```
3. Pick a difficulty & create `settings.json`:
    ```
    {
        "block_id": "",
        "algorithm_id": "",
        "challenge_id": "",
        "player_id": "",
        "difficulty": [50, 300]
    }
    ```
4. Test the algorithm:
    ```
    cargo run -p tig-worker --release -- settings.json tig-algorithms/wasm/${CHALLENGE}/${ALGORITHM}.wasm
    ```

Notes:
* You can query the latest difficulty ranges via TIG's API:
    ```
    query https://api.tig.foundation/play/get-block for <block_id>
    query https://api.tig.foundation/play/get-challenges?block_id=<block_id> for qualifier_difficulties
    ```

### Checking CI Successfully Compiles Your Algorithm

TIG pushes all algorithms to their own branch which triggers the CI (`.github/workflows/build_algorithm.yml`).

To trigger the CI on your private repo, your branch just needs to have a particular name:
```
git checkout -b <challenge_name>/<algorithm_name>
git push origin <challenge_name>/<algorithm_name>
```

### Making Your Submission

You will need to burn 0.001 ETH to make a submission. Visit https://play.tig.foundation/innovator and follow the instructions.

## License

Placeholder