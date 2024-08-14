# tig-benchmarker

A Rust crate that implements a Benchmarker for TIG. 

## Browser Benchmarker

`tig-benchmarker` can be compiled to WASM with bindings for browsers. 

The browser version is deployed to https://play.tig.foundation/benchmarker

To build & run it locally, run the following commands before visiting localhost in your browser:

```
# uncomment below to install wasm-pack
# curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
cd tig-benchmarker
wasm-pack build --release --target web
python3 -m http.server 80
```

## Standalone Benchmarker

`tig-benchmarker` can be compiled into an executable for running standalone, or in slave mode (see notes)

There are two ways to start the master benchmarker:

1. Compile into exectuable and then run the executable:
    ```
    ALGOS_TO_COMPILE="" # See notes
    # USE_CUDA="cuda" # See notes
    cargo build -p tig-benchmarker --release --no-default-features --features "standalone ${ALGOS_TO_COMPILE} ${USE_CUDA}"
    # edit below line for your own algorithm selection
    echo '{"satisfiability":"schnoing","vehicle_routing":"clarke_wright","knapsack":"dynamic","vector_search":"optimal_ann"}' > algo_selection.json
    ./target/release/tig-benchmarker <address> <api_key> algo_selection.json
    ```

2. Compile executable in a docker, and run the docker:
    ```
    ALGOS_TO_COMPILE="" # See notes
    # USE_CUDA="cuda" # See notes
    docker build -f tig-benchmarker/Dockerfile --build-arg features="${ALGOS_TO_COMPILE} ${USE_CUDA}" -t tig-benchmarker .
    # edit below line for your own algorithm selection
    echo '{"satisfiability":"schnoing","vehicle_routing":"clarke_wright","knapsack":"dynamic","vector_search":"optimal_ann"}' > algo_selection.json
    docker run -it -v $(pwd):/app tig-benchmarker <address> <api_key> algo_selection.json
    ```

**Notes:**

* Setting `ALGOS_TO_COMPILE` will run the selected algorithms directly in your execution environment to filter for nonces that results in solutions. Nonces that are filtered will then be re-run in the WASM virtual machine to compute the necessary solution data for submission.

    * **WARNING** before setting `ALGOS_TO_COMPILE`, be sure to thoroughly review the algorithm code for malicious routines as it will be ran directly in your execution environment (not within a sandboxed WASM virtual machine)!

    * `ALGOS_TO_COMPILE` is a space separated string of algorithms with format `<challenge_name>_<algorithm_name>`. Example: 
    
        ```
        ALGOS_TO_COMPILE="satisfiability_schnoing vehicle_routing_clarke_wright knapsack_dynamic vector_search_optimal_ann"
        ```

* Every 10 seconds, the benchmarker reads your json file path and uses the contents to update its algorithm selection. 
* You can see available algorithms in the dropdowns of the [Benchmarker UI](https://play.tig.foundation/benchmarker)
    * Alternatively, you can use [`script\list_algorithms.sh`](../scripts/list_algorithms.sh)
* `tig-benchmarker` starts a master node by default. The port can be set with `--port <port>` (default 5115)
* `tig-benchmarker` that are started with the option `--master <hostname>` are ran as slaves and will poll the master for jobs
* `tig-benchmarker` can be executed with `--help` to see all options including setting the number of workers, and setting the duration of a benchmark
* Uncomment `# USE_CUDA="cuda"` to compile `tig-benchmarker` to use CUDA optimisations where they are available. 
    * You must have a CUDA compatible GPU with CUDA toolkit installed
    * You must have set `ALGOS_TO_COMPILE`
    * Not all algorithms have CUDA optimisations. If they don't, it will default to using the CPU version
    * CUDA optimisations may or may not be more performant

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
