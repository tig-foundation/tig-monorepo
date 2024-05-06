# The Innovation Game

This repository contains the implementation of The Innovation Game (TIG).

## Important Links

* [TIG's tech explainer](docs/1_basics.md)
* [Getting started with Innovating](tig-algorithms/README.md)
* [Getting started with Benchmarking](tig-benchmarker/README.md)
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

## License

Placeholder