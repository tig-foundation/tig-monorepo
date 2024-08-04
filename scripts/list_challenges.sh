#!/bin/bash
set -e

BLOCK_ID=$(curl -s https://mainnet-api.tig.foundation/get-block | jq -r '.block.id')
RESP=$(curl -s "https://mainnet-api.tig.foundation/get-challenges?block_id=$BLOCK_ID")

CHALLENGES=$(echo $RESP | jq -c '.challenges[]' | jq -s 'sort_by(.id)')

for C in $(echo $CHALLENGES | jq -c '.[]'); do
    ID=$(echo $C | jq -r '.id')
    NAME=$(echo $C | jq -r '.details.name')
    ROUND_ACTIVE=$(echo $C | jq -r '.state.round_active')
    NUM_QUALIFIERS=$(echo $C | jq -c '.block_data.num_qualifiers')
    DIFFICULTIES=$(echo $C | jq -c '.block_data.qualifier_difficulties')
    printf "(%-4s) %-20s %-20s %-20s\n%-s\n\n" "$ID" "$NAME" "round_active: $ROUND_ACTIVE" "num_qualifiers: $NUM_QUALIFIERS" "qualifier_difficulties: $DIFFICULTIES"
done