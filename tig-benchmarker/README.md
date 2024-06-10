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

`tig-benchmarker` can be compiled into an executable for running locally.

There are two ways to run locally:

1. Compile into exectuable and then run the executable:
    ```
    cargo build -p tig-benchmarker --release --no-default-features --features standalone
    # edit below line for your own algorithm selection
    echo '{"satisfiability":"schnoing","vehicle_routing":"clarke_wright","knapsack":"dynamic"}' >> algo_selection.json
    ./target/release/tig-benchmarker <address> <api_key> algo_selection.json
    ```

2. Compile executable in a docker, and run the docker:
    ```
    docker build -f tig-benchmarker/Dockerfile -t tig-benchmarker .
    # edit below line for your own algorithm selection
    echo '{"satisfiability":"schnoing","vehicle_routing":"clarke_wright","knapsack":"dynamic"}' >> algo_selection.json
    docker run -it -v $(pwd):/app tig-benchmarker <address> <api_key> algo_selection.json
    ```

**Note:**

* Every 10 seconds, the benchmarker reads your json file path and uses the contents to update its algorithm selection. 
* You can see available algorithms in the dropdowns of the [Benchmarker UI](https://play.tig.foundation/benchmarker)
* `tig-benchmarker` can be executed with `--help` to see all options including setting the number of workers, and setting the duration of a benchmark

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
