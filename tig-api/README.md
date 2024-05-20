# tig-api

A Rust crate for making requests to TIG's API.

Developers must either enable feature `request` (uses `reqwest`) or `request-js` (uses `web-sys`) 

## API Url
* Mainnet https://api.tig.foundation/play
* Testnet https://api.tig.foundation/test

## GET Endpoints

* `get-algorithms`
* `get-benchmarks`
* `get-benchmark-data`
* `get-block`
* `get-challenges`
* `get-players`

## POST Endpoints

Requires `x-api-key` header to be set

* `submit-algorithm`
* `submit-benchmark`
* `submit-proof`

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)