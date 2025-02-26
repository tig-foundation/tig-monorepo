#!/bin/bash
set -e

BLOCK_ID=$(curl -s https://mainnet-api.tig.foundation/get-block | jq -r '.block.id')
RESP=$(curl -s "https://mainnet-api.tig.foundation/get-opow?block_id=$BLOCK_ID")

PLAYERS=$(echo $RESP | jq -c '[.opow[] | .block_data.reward = (if .block_data.reward == null then 0 else (.block_data.reward | tonumber) end)] | sort_by(.block_data.reward) | reverse')

for PLAYER in $(echo $PLAYERS | jq -c '.[]'); do
    ID=$(echo $PLAYER | jq -r '.player_id')
    REWARD=$(echo $PLAYER | jq -r '.block_data.reward | if . == null then "null" else tonumber / 1e18 end')
    printf "Player ID: %-25s Reward: %-20s\n" "$ID" "$REWARD"
done