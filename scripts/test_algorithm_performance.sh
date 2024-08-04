#!/bin/bash
set -e

echo "Enter settings (JSON string):"
read settings

echo "Enter starting nonce:"
read start_nonce

echo "Enter number of nonces:"
read num_nonces

challenge_id=$(echo "$settings" | jq -r '.challenge_id')
algorithm_id=$(echo "$settings" | jq -r '.algorithm_id')

block_id=$(curl -s https://testnet-api.tig.foundation/get-block | jq -r '.block.id')
algorithm_name=$(curl -s "https://testnet-api.tig.foundation/get-algorithms?block_id=$block_id" | jq -r --arg ID "$algorithm_id" '.algorithms[] | select(.id == $ID) | .details.name')
challenge_name=$(curl -s "https://testnet-api.tig.foundation/get-challenges?block_id=$block_id" | jq -r --arg ID "$challenge_id" '.challenges[] | select(.id == $ID) | .details.name')

branch="test_performance/${challenge_name}/${algorithm_name}"
if git show-ref --quiet $branch; then
    echo "Branch $branch already exists. Switching"
    git checkout $branch
else 
    echo "Create new branch with name $branch? (y/n)"
    read confirm
    if [ "$confirm" != "y" ]; then
        echo "Aborting."
        exit 1
    fi
    git fetch origin
    git checkout -b $branch origin/test/blank_slate
fi
git pull origin test/$challenge_name/$algorithm_name --no-edit

cargo build -p tig-worker --release --features ${challenge_name}_${algorithm_name}

solutions=0
invalid=0
total_ms=0

echo "------------------------------------------------------------"
echo "Testing performance of $challenge_name/$algorithm_name"
echo "Settings: $settings"
echo "Starting nonce: $start_nonce"
echo "Number of nonces: $num_nonces"
echo -ne ""
for ((nonce=start_nonce; nonce<start_nonce+num_nonces; nonce++)); do
    start_time=$(date +%s%3N)
    ./target/release/tig-worker compute_solution "$settings" $nonce ./tig-algorithms/wasm/$challenge_name/$algorithm_name.wasm > /dev/null 2>&1
    exit_code=$?
    end_time=$(date +%s%3N)
    duration=$((end_time - start_time))
    total_ms=$((total_ms + duration))
    if [ $exit_code -eq 0 ]; then
        solutions=$((solutions + 1))
    else
        invalid=$((invalid + 1))
    fi
    if [ $((solutions)) -eq 0 ]; then
        avg_ms_per_solution=0
    else
        avg_ms_per_solution=$((total_ms / solutions))
    fi
    echo -ne "#instances: $((solutions + invalid)), #solutions: $solutions, #invalid: $invalid, average ms/solution: $avg_ms_per_solution, average ms/nonce: $((total_ms / (solutions + invalid)))\r"
done
echo
echo "------------------------------------------------------------"