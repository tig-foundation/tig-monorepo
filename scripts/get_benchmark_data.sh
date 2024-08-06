#!/bin/bash
set -e
read -p "Enter benchmark_id: " benchmark_id
curl -s https://mainnet-api.tig.foundation/get-benchmark-data?benchmark_id=$benchmark_id