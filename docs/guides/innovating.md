# Getting Started with Innovating

## Setting up Private Fork

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

## Checking out Existing Algorithms

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

## Developing Your Algorithm

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

## Locally Compiling Your Algorithm into WASM 

See the [README](../../tig-wasm/README.md) for `tig-wasm`

## Testing Performance of Algorithms

See the [README](../../tig-worker/README.md) for `tig-worker`

## Checking CI Successfully Compiles Your Algorithm

TIG pushes all algorithms to their own branch which triggers the CI (`.github/workflows/build_algorithm.yml`).

To trigger the CI on your private repo, your branch just needs to have a particular name:
```
git checkout -b <challenge_name>/<algorithm_name>
git push origin <challenge_name>/<algorithm_name>
```

## Making Your Submission

You will need to burn 0.001 ETH to make a submission. Visit https://play.tig.foundation/innovator and follow the instructions.