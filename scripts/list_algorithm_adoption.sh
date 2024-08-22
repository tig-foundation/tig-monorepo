#!/bin/bash
set -e

echo "Fetching latest block..."
BLOCK_ID=$(curl -s https://mainnet-api.tig.foundation/get-block | jq -r '.block.id')
echo "Fetching algorithms..."
RESP=$(curl -s "https://mainnet-api.tig.foundation/get-algorithms?block_id=$BLOCK_ID")
echo "Filtering algorithms..."
ALGORITHMS=$(echo $RESP | jq -c '.algorithms[]' | jq 'select(.block_data.adoption | length > 15)' | jq -s 'sort_by(.block_data.adoption|tonumber)' | jq 'reverse')

declare -A CHALLENGES

echo "Calculating adoption percentage..."
for ALGO in $(echo $ALGORITHMS | jq -c '.[]'); do
    A_NAME=$(echo $ALGO | jq -r '.details.name')
    ADOPTION=$(echo $ALGO | jq -r '.block_data.adoption')
    
    ADOPTION_PERCENT=$(awk "BEGIN {printf \"%.2f\", $ADOPTION / 10000000000000000}")

    case $(echo $ALGO | jq -r '.details.challenge_id') in
        "c001") C_NAME="satisfiability" ;;
        "c002") C_NAME="vehicle_routing" ;;
        "c003") C_NAME="knapsack" ;;
        "c004") C_NAME="vector_search" ;;
        *) C_NAME="unknown" ;;
    esac

    CHALLENGES[$C_NAME]+="$(printf "   %-30s %-20s\n" "$A_NAME" "adoption: $ADOPTION_PERCENT%")\n"
done
echo "Printing..."
echo ""
for CHALLENGE in "${!CHALLENGES[@]}"; do
    echo "   Challenge: $CHALLENGE"
    echo -e "${CHALLENGES[$CHALLENGE]}"
done

