# Getting Started with Innovating

## Developer Environment

TIG compiles all algorithms into shared objects and ptx for verifiable execution.

TIG currently requires all algorithms to be written in [Rust](https://www.rust-lang.org/tools/install). Algorithms can optionally have a .cu with Cuda kernels, which are launched from the Rust code.

We recommend developing using [Visual Studio Code](https://code.visualstudio.com/) with Rust plugins:
* rust-analyzer
* Even Better TOML
* crates
* CodeLLDB

## Checking out Existing Algorithms

Every algorithm has its own `<branch>` with name `<challenge_name>/<algorithm_name>`.

To pull an existing algorithm from tig-monorepo repository, run the following command:
```
git pull origin <branch>
```

Only algorithms that are successfully compiled have their branch pushed to this public repository.

Each algorithm branch will have 6 key files (11 if there is CUDA code):
1. Rust code with TIG commercial license header @ `tig-algorithms/src/<branch>/commercial.rs`
2. Rust code with TIG open data license header @ `tig-algorithms/src/<branch>/open_data.rs`
3. Rust code with TIG benchmarker outbound license header @ `tig-algorithms/src/<branch>/benchmarker_outbound.rs`
4. Rust code with TIG innovator outbound license header @ `tig-algorithms/src/<branch>/innovator_outbound.rs`
5. Cuda code with TIG inbound license header @ `tig-algorithms/src/<branch>/inbound.cu`
6. Tarball with shared objects and (optionally) ptx @ `tig-algorithms/lib/<branch>.tar.gz`
7. Cuda code with TIG commercial license header @ `tig-algorithms/src/<branch>/commercial.cu`
8. Cuda code with TIG open data license header @ `tig-algorithms/src/<branch>/open_data.cu`
9. Cuda code with TIG benchmarker outbound license header @ `tig-algorithms/src/<branch>/benchmarker_outbound.cu`
10. Cuda code with TIG innovator outbound license header @ `tig-algorithms/src/<branch>/innovator_outbound.cu`
11. Cuda code with TIG inbound license header @ `tig-algorithms/src/<branch>/inbound.cu`

## Developing Your Algorithm

**READ THE IMPORTANT NOTES AT THE BOTTOM OF THIS SECTION**

1. Pick a challenge (`<challenge_name>`) to develop an algorithm for
2. Make a copy of `tig-algorithms/<challenge_name>/template.rs` or an existing algorithm (see notes)
    * (Optional) for Cuda, additionally make a copy of `tig-algorithms/<challenge_name>/template.cu`
3. Make sure your file has the following notice in its header if you intend to submit it to TIG:
```
Copyright [year copyright work created] [name of copyright owner]

Identity of Submitter [name of person or entity that submits the Work to TIG]

UAI [UAI (if applicable)]

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
```
  * If your implementation is based on an algorithmic method submitted to TIG, you must attribute your implementation to it (example UAI: `c001_b001`)
    * UAI of a method is detailed inside `tig-breakthroughs/<challenge_name>/<method_name>.md`
    * Methods have branch name `<challenge_name>/method/<method_name>`
  * If your implementation is based on an algorithmic method outside of TIG, set UAI to `null`
4. Rename the file with your own `<algorithm_name>`
5. Edit `tig-algorithms/<challenge_name>/mod.rs` to export your algorithm and test it:
    ```
    pub mod <algorithm_name>;

    #[cfg(test)]
    mod tests {
        #[cfg(feature = "cuda")]
        use cudarc::{driver::CudaContext, nvrtc::Ptx, runtime::result::device::get_device_prop};
        use super::<algorithm_name>::solve_challenge;
        use tig_challenges::<challenge_name>::*;

        #[cfg(not(feature = "cuda"))]
        #[test]
        fn test_algorithm() {
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
            let seed = [0u8; 32]; // change this to generate different instances
            let challenge = Challenge::generate_instance(seed, &difficulty).unwrap();
            match solve_challenge(&challenge) {
                Ok(Some(solution)) => match challenge.verify_solution(&solution) {
                    Ok(_) => println!("Valid solution"),
                    Err(e) => println!("Invalid solution: {}", e),
                },
                Ok(None) => println!("No solution"),
                Err(e) => println!("Algorithm error: {}", e),
            };
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn test_algorithm() {
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
            let seed = [0u8; 32]; // change this to generate different instances

            let ptx_path = "/app/tig-algorithms/lib/vector_search/ptx/simple_search.ptx"; // change this to the path of your PTX file
            let ptx = Ptx::from_file(ptx_path);
            let gpu_device = 0; // change this to the desired GPU device ID
            let ctx = CudaContext::new(gpu_device).unwrap();
            ctx.set_blocking_synchronize().unwrap();
            let module = ctx.load_module(ptx).unwrap();
            let stream = ctx.default_stream();
            let prop = get_device_prop(gpu_device as i32).unwrap();

            let challenge =
                Challenge::generate_instance(seed, &difficulty, module.clone(), stream.clone(), &prop)
                    .unwrap();

            match solve_challenge(&challenge, module.clone(), stream.clone(), &prop) {
                Ok(Some(solution)) => {
                    match challenge.verify_solution(&solution, module.clone(), stream.clone(), &prop) {
                        Ok(_) => println!("Valid solution"),
                        Err(e) => println!("Invalid solution: {}", e),
                    }
                }
                Ok(None) => println!("No solution"),
                Err(e) => println!("Algorithm error: {}", e),
            };
        }
    }
    ```
6. Run the appropiate [development docker image](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fdev). Available flavours are:
    * amd64 (x86_64 compatible)
    * aarch64
    * amd64-cuda12.6.3 (x86_64 compatible)
    * aarch64-cuda12.6.3
    ```
    # example
    docker run -it -v $(pwd):/app --gpus all ghcr.io/tig-foundation/tig-monorepo/dev:0.0.1-amd64-cuda12.6.3
    ```
6. If you have Cuda code, use `build_ptx.py` to compile it
    ```
    build_ptx.py <CHALLENGE> <ALGORITHM>
    ```
7. Run the test
    * No cuda:
    ```
    cargo test -p tig-algorithms --release -- --nocapture
    ```
    * With cuda:
    ```
    cargo test -p tig-algorithms --release --features cuda -- --nocapture
    ```

**IMPORTANT (READ THIS):**
* Not all challenge instances have solutions.
* You can find the current qualifying difficulties by:
    * Query https://mainnet-api.tig.foundation/get-block for <block_id>
    * Query https://mainnet-api.tig.foundation/get-challenges?block_id=<block_id> for <qualifier_difficulties>
* If you are copying and modifying an algorithm that has been submitted to TIG, make sure to use the `innovator_outbound` version
* Do not include tests in your algorithm file. TIG will reject your algorithm submission.
* Only your algorithm's code gets submitted. You should not be modifying `Cargo.toml` in `tig-algorithms`. Any extra dependencies you add will not be available when TIG compiles your algorithm
* If you need to use random number generation,  ensure that it is seeded so that your algorithm is deterministic.
    * Suggest to use `let mut rng = SmallRng::from_seed(StdRng::from_seed(challenge.seed).gen())`

## Locally Compiling Your Algorithm into Shared Object 

See the [README](../../tig-binary/README.md) for `tig-binary`

## Testing Performance of Algorithms

TODO

## Checking CI Successfully Compiles Your Algorithm

TIG pushes all algorithms to their own branch which triggers the CI (`.github/workflows/build_algorithm.yml`).

To replicate this, Innovators will want to create a private fork:

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

6.  Trigger the CI on your private repo with your algorithm branch:
    ```
    git checkout -b <challenge_name>/<algorithm_name> blank_slate
    git push origin <challenge_name>/<algorithm_name>
    ```

## Making Your Submission

10 TIG will be deducted from your Available Fee Balance to make a submission. You can topup via the [Benchmarker page](https://play.tig.foundation/benchmarker)

**IMPORTANT:** 
* Submissions are final and cannot be modified after they are made
* Be sure to adhere to the notes in [developing your algorithm](#developing-your-algorithm)
* We highly recommend [checking the CI can compile your algorithm](#checking-ci-successfully-compiles-your-algorithm)
