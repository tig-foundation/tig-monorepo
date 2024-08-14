#!/bin/bash
set -e
curl -s https://mainnet-api.tig.foundation/get-block?include_data | jq -r '.block.data.active_benchmark_ids[]' | nl