#!/bin/bash
set -e

# Ask for player_id as input
read -p "Enter player ID: " PLAYER_ID

BLOCK_ID=$(curl -s https://mainnet-api.tig.foundation/get-block | jq -r '.block.id')
RESP=$(curl -s "https://mainnet-api.tig.foundation/get-benchmarks?block_id=$BLOCK_ID&player_id=$PLAYER_ID")

BENCHMARKS=$(echo $RESP | jq -c '[.benchmarks[]] | sort_by(.settings.challenge_id, -.details.num_solutions)')

for BENCHMARK in $(echo $BENCHMARKS | jq -c '.[]'); do
    ID=$(echo $BENCHMARK | jq -r '.id')
    SETTINGS=$(echo $BENCHMARK | jq -c '.settings')
    NUM_SOLUTIONS=$(echo $BENCHMARK | jq -r '.details.num_solutions')
    printf "ID: %-38s #Solutions: %-5s Settings: %-50s \n" "$ID" "$NUM_SOLUTIONS" "$SETTINGS"
done