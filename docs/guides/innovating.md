# Getting Started with Innovating

## Developer Environment

TIG compiles all algorithms into WASM for sandboxing and for verifiable execution.

TIG currently requires all algorithms to be written in [Rust](https://www.rust-lang.org/tools/install).

We recommend developing using [Visual Studio Code](https://code.visualstudio.com/) with Rust plugins:
* rust-analyzer
* Even Better TOML
* crates
* CodeLLDB

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

Each algorithm branch will have 6 key files:
1. Rust code with TIG commercial license header @ `tig-algorithms/src/<branch>/commercial.rs`
2. Rust code with TIG open data license header @ `tig-algorithms/src/<branch>/open_data.rs`
3. Rust code with TIG benchmarker outbound license header @ `tig-algorithms/src/<branch>/benchmarker_outbound.rs`
4. Rust code with TIG innovator outbound license header @ `tig-algorithms/src/<branch>/innovator_outbound.rs`
5. Rust code with TIG inbound license header @ `tig-algorithms/src/<branch>/inbound.rs`
6. Wasm blob @ `tig-algorithms/wasm/<branch>.wasm`

To pull an existing algorithm from TIG public repository, run the following command:
```
git fetch public
git pull public <branch>
```

## Developing Your Algorithm

**READ THE IMPORTANT NOTES AT THE BOTTOM OF THIS SECTION**

1. Pick a challenge (`<challenge_name>`) to develop an algorithm for
2. Make a copy of `tig-algorithms/<challenge_name>/template.rs` or an existing algorithm (see notes)
3. Make sure your file has the following notice in its header if you intend to submit it to TIG:
```
Copyright [yyyy] [name of copyright owner]

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
```
4. Rename the file with your own `<algorithm_name>`
5. Edit `tig-algorithms/<challenge_name>/mod.rs` to export your algorithm and test it:
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
                // Modify the values for different difficulties

                // -- satisfiability --
                // num_variables: 50,
                // clauses_to_variables_percent: 300,

                // -- vehicle_routing --
                // num_nodes: 40,
                // better_than_baseline: 250,

                // -- knapsack --
                // num_items: 50,
                // better_than_baseline: 10,
                
                // -- vector_search --
                // num_queries: 10,
                // better_than_baseline: 350,
            };
            let seeds = [0; 8]; // change this to generate different instances
            let challenge = Challenge::generate_instance(seeds, &difficulty).unwrap();
            match <algorithm_name>::solve_challenge(&challenge) {
                Ok(Some(solution)) => match challenge.verify_solution(&solution) {
                    Ok(_) => println!("Valid solution"),
                    Err(e) => println!("Invalid solution: {}", e),
                },
                Ok(None) => println!("No solution"),
                Err(e) => println!("Algorithm error: {}", e),
            };
        }
    }
    ```
    * See notes for CUDA tests
6. Use the above test to debug your algorithm:
    ```
    cargo test -p tig-algorithms -- --nocapture
    ```

**IMPORTANT (READ THIS):**
* Not all challenge instances have solutions. Algorithms that can detect such cases and exit early (`return Ok(None)`) will potentially have better performance than algorithms that don't exit early
* You can find the current qualifying difficulties by:
    * Query https://mainnet-api.tig.foundation/get-block for <block_id>
    * Query https://mainnet-api.tig.foundation/get-challenges?block_id=<block_id> for <qualifier_difficulties>
* If you are copying and modifying an algorithm that has been submitted to TIG, make sure to use the `innovator_outbound` version
* Do not include tests in your algorithm file. TIG will reject your algorithm submission.
* Only your algorithm's rust code gets submitted. You should not be modifying `Cargo.toml` in `tig-algorithms`. Any extra dependencies you add will not be available when TIG compiles your algorithm
* If you need to use random number generation be sure to use `let mut rng = StdRng::seed_from_u64(challenge.seed as u64)` to ensure your algorithm is deterministic.
* To test cuda, edit the following test, and use the command `cargo test -p tig-algorithms --features cuda -- --nocapture`:
```
#[cfg(feature = "cuda")]
#[cfg(test)]
mod cuda_tests {
    use std::collections::HashMap;

    use super::*;
    use cudarc::driver::*;
    use cudarc::nvrtc::compile_ptx;
    use std::{sync::Arc, collections::HashMap};
    use tig_challenges::{<challenge_name>::*, *};

    fn load_cuda_functions(
        dev: &Arc<CudaDevice>,
        kernel: &CudaKernel,
        key: &str,
    ) -> HashMap<&'static str, CudaFunction> {
        let start = std::time::Instant::now();
        println!("Compiling CUDA kernels for {}", key);
        let ptx = compile_ptx(kernel.src).expect("Cuda Kernel failed to compile");
        dev.load_ptx(ptx, key, &kernel.funcs)
            .expect("Failed to load CUDA functions");
        let funcs = kernel
            .funcs
            .iter()
            .map(|&name| (name, dev.get_func(key, name).unwrap()))
            .collect();
        println!(
            "CUDA kernels for '{}' compiled in {}ms",
            key,
            start.elapsed().as_millis()
        );
        funcs
    }

    #[test]
    fn test_cuda_<algorithm_name>() {
        let dev = CudaDevice::new(0).expect("Failed to create CudaDevice");
        let challenge_cuda_funcs = match &<challenge_name>::KERNEL {
            Some(kernel) => load_cuda_functions(&dev, &kernel, "challenge"),
            None => {
                println!("No CUDA kernel for challenge");
                HashMap::new()
            }
        };
        let algorithm_cuda_funcs = match &<algorithm_name>::KERNEL {
            Some(kernel) => load_cuda_functions(&dev, &kernel, "algorithm"),
            None => {
                println!("No CUDA kernel for algorithm");
                HashMap::new()
            }
        };

        let difficulty = Difficulty {
            // Uncomment the relevant fields.
            // Modify the values for different difficulties

            // -- satisfiability --
            // num_variables: 50,
            // clauses_to_variables_percent: 300,
            // -- vehicle_routing --
            // num_nodes: 40,
            // better_than_baseline: 250,

            // -- knapsack --
            // num_items: 50,
            // better_than_baseline: 10,

            // -- vector_search --
            // num_queries: 10,
            // better_than_baseline: 350,
        };
        let seeds = [0; 8]; // change this to generate different instances
        let challenge =
            Challenge::cuda_generate_instance(seeds, &difficulty, &dev, challenge_cuda_funcs)
                .unwrap();
        match <algorithm_name>::cuda_solve_challenge(&challenge, &dev, algorithm_cuda_funcs) {
            Ok(Some(solution)) => match challenge.verify_solution(&solution) {
                Ok(_) => println!("Valid solution"),
                Err(e) => println!("Invalid solution: {}", e),
            },
            Ok(None) => println!("No solution"),
            Err(e) => println!("Algorithm error: {}", e),
        };
    }
}
```

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

**IMPORTANT:** 
* Submissions are final and cannot be modified after they are made
* Be sure to adhere to the notes in [developing your algorithm](#developing-your-algorithm)
* We highly recommend [compiling your algorithm into WASM](#locally-compiling-your-algorithm-into-wasm) and [testing its performance](#testing-performance-of-algorithms) with `tig-worker`.
* We highly recommend [checking the CI can compile your algorithm](#checking-ci-successfully-compiles-your-algorithm)
