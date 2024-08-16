#!/bin/bash
set -e

REPO_DIR=$(dirname $(dirname "$(realpath "$0")"))
TIG_WORKER_PATH="$REPO_DIR/target/release/tig-worker"

if [ ! -f $TIG_WORKER_PATH ]; then
    echo "Error: tig-worker binary not found at ./target/release/tig-worker"
    echo "Run: cd $REPO_DIR && cargo build -p tig-worker --release"
    exit 1
fi

read -p "Enter benchmark_id: " benchmark_id

echo "Fetching benchmark data"
response=$(curl -s "https://mainnet-api.tig.foundation/get-benchmark-data?benchmark_id=$benchmark_id")

# parse data from resp
proof=$(echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for item in data['proof']['solutions_data']:
    item['nonce'] = str(item['nonce'])
print(json.dumps(data))
" | jq '.proof')
if [ "$proof" == "null" ]; then
    echo "No proofs found"
    exit 0
fi
solutions_count=$(echo "$proof" | jq -r '.solutions_data | length')
settings=$(echo "$response" | jq -r '.benchmark.settings')
algorithm_id=$(echo "$settings" | jq -r '.algorithm_id')
echo "Found $solutions_count solutions to verify"

# Fetch block id
echo "Fetching block data"
block_response=$(curl -s "https://mainnet-api.tig.foundation/get-block")
block_id=$(echo "$block_response" | jq -r '.block.id')

# Fetch algorithms for the block
echo "Fetching algorithms data"
algorithms_response=$(curl -s "https://mainnet-api.tig.foundation/get-algorithms?block_id=$block_id")
wasms=$(echo "$algorithms_response" | jq -c '.wasms[]')

wasm=$(echo "$wasms" | jq -c "select(.algorithm_id == \"$algorithm_id\")")

if [ -z "$wasm" ]; then
    echo "No matching WASM found for $algorithm_id"
    exit 0
fi

compile_success=$(echo "$wasm" | jq -r '.details.compile_success')
if [ "$compile_success" == "false" ]; then
    echo "WASM was not successful compiled"
    exit 0
fi

download_url=$(echo "$wasm" | jq -r '.details.download_url')
echo "Downloading WASM from $download_url"
curl -s -o "$algorithm_id.wasm" "$download_url"

# verify solutions
for i in $(seq 0 $(($solutions_count - 1))); do
    solution_data=$(echo "$proof" | jq -c -S ".solutions_data[$i]")
    echo Verifying $solution_data
    nonce=$(echo "$solution_data" | jq -r '.nonce')
    solution=$(echo "$solution_data" | jq -r '.solution')
    echo Verifying solution is valid
    ./target/release/tig-worker verify_solution "$settings" $nonce "$solution"
    echo Verifying runtime_signature and fuel_consumed
    compute_output=$(./target/release/tig-worker compute_solution "$settings" $nonce ./$algorithm_id.wasm | python3 -c "
import sys, json
data = json.load(sys.stdin)
data['nonce'] = str(data['nonce'])
print(json.dumps(data))
    " | jq -c -S)
    if [[ "$compute_output" == "$solution_data" ]]; then
        echo "Ok"
    else
        echo "Mismatch. Actual: $compute_output"
    fi
done
rm $algorithm_id.wasm