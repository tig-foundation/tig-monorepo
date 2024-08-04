#!/bin/bash
set -e

BLOCK_ID=$(curl -s https://mainnet-api.tig.foundation/get-block | jq -r '.block.id')
RESP=$(curl -s "https://mainnet-api.tig.foundation/get-algorithms?block_id=$BLOCK_ID")

ALGORITHMS=$(echo $RESP | jq -c '.algorithms[]' | jq -s 'sort_by(.id)')
WASMS_DICT=$(echo $RESP | jq -c '[.wasms[] | {key: .algorithm_id, value: .}] | from_entries')

for ALGO in $(echo $ALGORITHMS | jq -c '.[]'); do
    ID=$(echo $ALGO | jq -r '.id')
    NAME=$(echo $ALGO | jq -r '.details.name')
    ROUND_SUBMITTED=$(echo $ALGO | jq -r '.state.round_submitted')
    ROUND_PUSHED=$(echo $ALGO | jq -r '.state.round_pushed')
    COMPILE_SUCCESS=$(echo $WASMS_DICT | jq -c --arg ID "$ID" '.[$ID] | .details.compile_success')
    printf "(%-9s) %-25s %-20s %-20s %-20s\n" "$ID" "$NAME" "round_submitted: $ROUND_SUBMITTED" "round_pushed: $ROUND_PUSHED" "compile_success: $COMPILE_SUCCESS"
done