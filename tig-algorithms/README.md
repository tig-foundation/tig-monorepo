# tig-algorithms

A Rust crate that hosts algorithm submissions made by Innovators in TIG.

Each submissions is committed to their own branch with the naming pattern:

`<challenge_name>\<algorithm_name>` 

## Downloading an Algorithm

WASM blobs for an algorithm are stored in the `wasm` subfolder and can be downloaded via:

`https://raw.githubusercontent.com/tig-foundation/tig-monorepo/<branch_name>/tig-algorithms/wasm/<branch_name>.wasm`

## Algorithm Submission Flow

1. New submissions get their branch pushed to a private version of this repository
2. CI will compile submissions into WASM
3. A new submission made during round `X` will have its branch pushed to the public version of this repository at the start of round `X + 3`
4. Once public, benchmarkers can use the algorithm for benchmarking
5. Every block, algorithms with at least 25% adoption earn a merge point
6. At the end of a round, a the algorithm from each challenge with the most merge points, meeting the minimum threshold of 5040, gets merged to the `main` branch

