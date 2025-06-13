# tig-algorithms

A Rust crate that hosts algorithm submissions made by Innovators in TIG.

Each submissions is committed to their own branch with the naming pattern:

`<challenge_name>\<algorithm_name>` 

## Table of Contents

1. [Quick Start](#quick-start)
2. [Developer Environment](#developer-environment)
3. [Improving Algorithms](#improving-algorithms)
4. [GPU Algorithms](#gpu-algorithms)
5. [Attributing Breakthroughs](#attributing-breakthroughs)
6. [License](#license)

# Quick Start

1. Clone this repo
    ```
    git clone https://github.com/tig-foundation/tig-monorepo
    ```

2. Create an algorithm
   * Create your algorithm file `tig-algorithms/src/satisfiability/my_algo.rs` and copy/paste [this code](https://github.com/tig-foundation/tig-monorepo/blob/test/satisfiability/schnoing/tig-algorithms/src/satisfiability/schnoing/benchmarker_outbound.rs)
   * Edit `tig-algorithms/src/satisfiability/mod.rs` and add the line:
   ```
   pub mod my_algo;
   ```

3. Compile your algorithm:
    ```
    cd tig-monorepo

    CHALLENGE=satisfiability
    VERSION=0.0.1
    docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION

    # inside docker
    build_algorithm my_algo
    ```

4. Test your algorithm:
    ```
    # inside docker
    test_algorithm my_algo [100,300]
    ```
    * Use `--help` to see more options
    * Use `--verbose` to dump individual instance commands that you can run. Example:
    ```
    /usr/local/bin/tig-runtime '{"algorithm_id":"","challenge_id":"c001","difficulty":[50,300],"block_id":"","player_id":""}' rand_hash 99 ./tig-algorithms/lib/satisfiability/arm64/my_algo.so --fuel 100000000000
    ```

5. Submitting your algorithm to Testnet:
    * Add and edit the following license header to `my_algo.rs`:
    ```
    /*!
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
    */
    ```

    * See [Attributing Breakthroughs](#attributing-breakthroughs) for how to set UAI field
    * Visit https://test.tig.foundation/innovation/submission and follow the instructions
    * Each new address on testnet gets 10 TIG balance
    * Algorithms on testnet can be used straight away after they are successfully compiled

6. Try benchmarking! See [Benchmarking Quick Start](../tig-benchmarker/README.md#quick-start)

# Developer Environment

We recommend developing using [Visual Studio Code](https://code.visualstudio.com/) with Rust plugins:
* rust-analyzer
* Even Better TOML
* crates
* CodeLLDB

# Improving Algorithms

All algorithms in TIG are open! You can develop your own de-novo algorithm or improve upon existing ones. 

**Tip:** there is a `tig-algorithms/src/<challenge>/template.rs` for each challenge. Inside it contains valuable tips on how to make your algorithm deterministic. Determinism is important in TIG because results need to be reproducible by everyone!

## Challenge Descriptions

See the [README in tig-challenges](../tig-challenges/README.md) for links to descriptions for each challenge.

## Obtaining Code

Every algorithm has its own `<branch>` with name `<challenge_name>/<algorithm_name>`. You should visit [the repository](https://github.com/tig-foundation/tig-monorepo) to look at code submitted by other Innovators (look in `tig-algorithm/src/<challenge_name>/<algorithm_name>`).

## Compiling & Testing Your Algorithm

See [Quick Start](README.md#quick-start)

## Testing Existing Algorithms

TIG has a [dev docker image](../README.md#docker-images) for each challenge. To test an existing algorithm, you can edit & follow this example:

```
CHALLENGE=satisfiability
VERSION=0.0.1
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION

# inside docker
list_algorithms --testnet
download_algorithm sat_global_opt --testnet
test_algorithm sat_global_opt [1000,300]
```

# GPU Algorithms

For challenges `vector_search` and `hypergraph`, a GPU with CUDA 12.6.3+ is required.

Additionally, you will need to run your docker with `--gpus all` option. Example:

```
CHALLENGE=vector_search
VERSION=0.0.1
docker run -it --gpus all -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION

# inside docker
nvidia-smi
```

To be competitive in these challenges, you should develop cuda kernels in a `.cu` file which can be launched from your rust `.rs` code.

# Attributing Breakthroughs

Innovators in TIG can also [submit breakthroughs](../tig-breakthroughs/README.md) (algorithmic methods).

If your implementation is based on breakthrough that has been submitted to TIG, you must attribute your implementation to it (example UAI: `c001_b001`)
* UAI of a method is detailed inside `tig-breakthroughs/<challenge_name>/<method_name>.md`
* If your implementation is based on an algorithmic method outside of TIG, set UAI to `null`

# License

Each algorithm submission will have 5 versions, each under a specific license:

* `commercial` will be under TIG commercial license
* `open_data` will be under TIG open data license
* `benchmarker_outbound` will be under TIG benchmarker outbound license
* `innovator_outbound` will be under TIG innovator outbound license
* `inbound` will be under TIG inbound license