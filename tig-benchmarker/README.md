# tig-benchmarker

A Rust crate that implements a Benchmarker for TIG that can run in the browser. 

This browser benchmarker is deployed to https://play.tig.foundation/benchmarker

To run it locally, run the following commands before visiting localhost in your browser:

```
# uncomment below to install wasm-pack
# curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
cd tig-benchmarker
wasm-pack build --release --target web
python3 -m http.server 80
```

See `tig-api` [README](../tig-api/README.md) for API Urls.

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