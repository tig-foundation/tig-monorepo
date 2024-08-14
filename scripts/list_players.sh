#!/bin/bash
set -e

BLOCK_ID=$(curl -s https://mainnet-api.tig.foundation/get-block | jq -r '.block.id')
RESP=$(curl -s "https://mainnet-api.tig.foundation/get-players?player_type=benchmarker&block_id=$BLOCK_ID")

PLAYERS=$(echo $RESP | jq -c '[.players[] | .block_data.reward = (if .block_data.reward == null then 0 else (.block_data.reward | tonumber) end)] | sort_by(.block_data.reward) | reverse')

for PLAYER in $(echo $PLAYERS | jq -c '.[]'); do
    ID=$(echo $PLAYER | jq -r '.id')
    ROUND_EARNINGS=$(echo $PLAYER | jq -r '.block_data.round_earnings | tonumber / 1e18')
    REWARD=$(echo $PLAYER | jq -r '.block_data.reward | if . == null then "null" else tonumber / 1e18 end')
    printf "Player ID: %-25s Round Earnings: %-20s Reward: %-20s\n" "$ID" "$ROUND_EARNINGS" "$REWARD"
done