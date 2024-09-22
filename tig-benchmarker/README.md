# tig-benchmarker

A Rust crate that implements a Benchmarker for TIG. 

## Compiling Your Benchmarker

`tig-benchmarker` can be compiled into an executable for running standalone, or in slave mode (see notes)

There are two ways to start the master benchmarker:

1. Compile into exectuable and then run the executable:
    ```
    ALGOS_TO_COMPILE="" # See notes
    # USE_CUDA="cuda" # See notes
    cargo build -p tig-benchmarker --release --no-default-features --features "${ALGOS_TO_COMPILE} ${USE_CUDA}"
    # edit below line for your own algorithm selection
    SELECTED_ALGORITHMS='{"satisfiability":"schnoing","vehicle_routing":"clarke_wright","knapsack":"dynamic","vector_search":"basic"}'
    ./target/release/tig-benchmarker <address> <api_key> $SELECTED_ALGORITHMS
    ```

2. Compile executable in a docker, and run the docker:
    ```
    ALGOS_TO_COMPILE="" # See notes
    # USE_CUDA="cuda" # See notes
    docker build -f tig-benchmarker/Dockerfile --build-arg features="${ALGOS_TO_COMPILE} ${USE_CUDA}" -t tig-benchmarker .
    # edit below line for your own algorithm selection
    SELECTED_ALGORITHMS='{"satisfiability":"schnoing","vehicle_routing":"clarke_wright","knapsack":"dynamic","vector_search":"optimal_ann"}'
    docker run -it -v $(pwd):/app tig-benchmarker <address> <api_key> $SELECTED_ALGORITHMS
    ```

**Notes:**

* Setting `ALGOS_TO_COMPILE` will run the selected algorithms directly in your execution environment to filter for nonces that results in solutions. Nonces that are filtered will then be re-run in the WASM virtual machine to compute the necessary solution data for submission.

    * **WARNING** before setting `ALGOS_TO_COMPILE`, be sure to thoroughly review the algorithm code for malicious routines as it will be ran directly in your execution environment (not within a sandboxed WASM virtual machine)!

    * `ALGOS_TO_COMPILE` is a space separated string of algorithms with format `<challenge_name>_<algorithm_name>`. Example: 
    
        ```
        ALGOS_TO_COMPILE="satisfiability_schnoing vehicle_routing_clarke_wright knapsack_dynamic vector_search_optimal_ann"
        ```

* You can see available algorithms in the dropdowns of the [Benchmarker UI](https://play.tig.foundation/benchmarker)
    * Alternatively, you can use [`script\list_algorithms.sh`](../scripts/list_algorithms.sh)
* `tig-benchmarker` starts a master node by default. The port can be set with `--port <port>` (default 5115)
* `tig-benchmarker` that are started with the option `--master <hostname>` are ran as slaves and will poll the master for jobs
    * slaves will ignore any job that doesn't match their algorithm selection
    * one possible setup is to run the master with `--workers 0`, and then run a separate slave for each challenge with different number of workers
* `tig-benchmarker` can be executed with `--help` to see all options including setting the number of workers, and setting the duration of a benchmark
* Uncomment `# USE_CUDA="cuda"` to compile `tig-benchmarker` to use CUDA optimisations where they are available. 
    * You must have a CUDA compatible GPU with CUDA toolkit installed
    * You must have set `ALGOS_TO_COMPILE`
    * Not all algorithms have CUDA optimisations. If they don't, it will default to using the CPU version
    * CUDA optimisations may or may not be more performant

# Python Master Benchmarker

There is the option of running all your standalone benchmarkers in slave mode, and running `main.py` to act as your master. The key benefits of such a setup is:
1. Much easier to change settings for specific challenges/algorithms such as benchmark duration, and duration to wait for slaves to submit solutions
2. Much easier to integrate with dashboards and other tools
3. Much easier to modify to dump logs and other stats for refining your strategy

## Running Python Master

```
cd tig-monorepo/tig-benchmarker
pip3 install -r requirements.txt
python3 main.py
```

## Customising Your Algorithms

Edit [tig-benchmarker/master/config.py](./master/config.py)

Example:
```
PLAYER_ID = "0x1234567890123456789012345678901234567890" # your player_id
API_KEY = "11111111111111111111111111111111" # your api_key
TIG_WORKER_PATH = "/<path to tig-monorepo>/target/release/tig-worker" # path to executable tig-worker
TIG_ALGORITHMS_FOLDER = "/<path to tig-monorepo>/tig-algorithms" # path to tig-algorithms folder
API_URL = "https://mainnet-api.tig.foundation"

if PLAYER_ID is None or API_KEY is None or TIG_WORKER_PATH is None or TIG_ALGORITHMS_FOLDER is None:
    raise Exception("Please set the PLAYER_ID, API_KEY, and TIG_WORKER_PATH, TIG_ALGORITHMS_FOLDER variables in 'tig-benchmarker/master/config.py'")

PORT = 5115
JOBS = dict(
    # add an entry for each challenge you want to benchmark
    satisfiability=dict(
        # add an entry for each algorithm you want to benchmark
        schnoing=dict(
            benchmark_duration=10000, # amount of time to run the benchmark in milliseconds
            wait_slave_duration=5000, # amount of time to wait for slaves to post solutions before submitting
            num_jobs=1, # number of jobs to create. each job will sample its own difficulty
            weight=1.0, # weight of jobs for this algorithm. more weight = more likely to be picked
        )
    ),
    vehicle_routing=dict(
        clarke_wright=dict(
            benchmark_duration=10000,
            wait_slave_duration=5000,
            num_jobs=1,
            weight=1.0,
        )
    ),
    knapsack=dict(
        dynamic=dict(
            benchmark_duration=10000,
            wait_slave_duration=5000,
            num_jobs=1,
            weight=1.0,
        )
    ),
    vector_search=dict(
        optimal_ann=dict(
            benchmark_duration=30000, # recommend a high duration
            wait_slave_duration=30000, # recommend a high duration
            num_jobs=1,
            weight=1.0,
        )
    ),
)
```

Notes:
  * `weight` determine how likely a slave will benchmark that algorithm (if your slave is setup with all algorithms). If algorithm A has weight of 10, and algorithm B has weight of 1, algorithm A is 10x more likely to be picked
  * `num_jobs` can usually be left at `1` unless you are running a lot of slaves and want to spread out your compute across different difficulties
  * `vector_search` challenge may require longer durations due to its resource requirements
  * See [tig-worker/README.md](../tig-worker/README.md) for instructions on compiling `tig-worker`

# Finding your API Key

## Mainnet

1. Navigate to https://play.tig.foundation/
2. Connect your wallet
3. Run the following command in the console: `JSON.parse(Cookies.get("account"))`
    * `address` is your Mainnet `player_id`
    * `api_key` is your Mainnet API key

## Testnet

1. Navigate to https://test.tig.foundation/
2. Connect your wallet
3. Run the following command in the console: `JSON.parse(Cookies.get("account"))`
    * `address` is your Testnet `player_id`
    * `api_key` is your Testnet API key

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)
