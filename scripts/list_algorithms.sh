#!/bin/bash
set -e

BLOCK_ID=$(curl -s https://mainnet-api.tig.foundation/get-block | jq -r '.block.id')
RESP=$(curl -s "https://mainnet-api.tig.foundation/get-algorithms?block_id=$BLOCK_ID")

ALGORITHMS=$(echo $RESP | jq -c '.algorithms[]' | jq -s 'sort_by(.id)')
WASMS_DICT=$(echo $RESP | jq -c '[.wasms[] | {key: .algorithm_id, value: .}] | from_entries')

for ALGO in $(echo $ALGORITHMS | jq -c '.[]'); do
    ID=$(echo $ALGO | jq -r '.id')
    A_NAME=$(echo $ALGO | jq -r '.details.name')
    case $(echo $ALGO | jq -r '.details.challenge_id') in
        "c001") C_NAME="satisfiability" ;;
        "c002") C_NAME="vehicle_routing" ;;
        "c003") C_NAME="knapsack" ;;
        "c004") C_NAME="vector_search" ;;
        *) echo "unknown" ;;
    esac
    ROUND_SUBMITTED=$(echo $ALGO | jq -r '.state.round_submitted')
    ROUND_PUSHED=$(echo $ALGO | jq -r '.state.round_pushed')
    COMPILE_SUCCESS=$(echo $WASMS_DICT | jq -c --arg ID "$ID" '.[$ID] | .details.compile_success')
    printf "(%-9s) %-40s %-20s %-20s %-20s\n" "$ID" "$C_NAME/$A_NAME" "round_submitted: $ROUND_SUBMITTED" "round_pushed: $ROUND_PUSHED" "compile_success: $COMPILE_SUCCESS"
done