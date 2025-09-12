import numpy as np
import json
import requests
from common.structs import *
from common.calcs import *

API_URL = "https://mainnet-api.tig.foundation"

deposit = input("Enter deposit in TIG (leave blank to fetch deposit from your player_id): ")
if deposit != "":
    lock_period = input("Enter number of weeks to lock (longer lock will have higher APY): ")
    deposit = float(deposit)
    lock_period = float(lock_period)
    player_id = None
else:
    player_id = input("Enter player_id: ").lower()

print("Fetching data...")
block = Block.from_dict(requests.get(f"{API_URL}/get-block?include_data").json()["block"])
opow_data = {
    x["player_id"]: OPoW.from_dict(x)
    for x in requests.get(f"{API_URL}/get-opow?block_id={block.id}").json()["opow"]
}

factors = {
    benchmarker: {
        **{
            f: (
                opow_data[benchmarker].block_data.num_qualifiers_by_challenge.get(f, 0)
            )
            for f in block.data.active_ids["challenge"]
        },
        "weighted_deposit": opow_data[benchmarker].block_data.delegated_weighted_deposit.to_float()
    }
    for benchmarker in opow_data
}

if player_id is None:
    blocks_till_round_ends = block.config["rounds"]["blocks_per_round"] - (block.details.height % block.config["rounds"]["blocks_per_round"])
    seconds_till_round_ends = blocks_till_round_ends * block.config["rounds"]["seconds_between_blocks"]
    weighted_deposit = calc_weighted_deposit(deposit, seconds_till_round_ends, lock_period * 604800)
else:
    player_data = Player.from_dict(requests.get(f"{API_URL}/get-player-data?player_id={player_id}&block_id={block.id}").json()["player"])
    deposit = sum(x.to_float() for x in player_data.block_data.deposit_by_locked_period)
    weighted_deposit = player_data.block_data.weighted_deposit.to_float()
    for delegatee, fraction in player_data.block_data.delegatees.items():
        factors[delegatee]["weighted_deposit"] -= fraction * weighted_deposit

total_factors = {
    f: sum(factors[benchmarker][f] for benchmarker in opow_data)
    for f in list(block.data.active_ids["challenge"]) + ["weighted_deposit"]
}
reward_shares = {
    benchmarker: opow_data[benchmarker].block_data.reward_share.to_float() / (opow_data[benchmarker].block_data.reward.to_float() + 1e-12)
    for benchmarker in opow_data
}

print("Optimising delegation by splitting into 100 chunks...")
chunk = weighted_deposit / 100
delegate = {}
for i in range(100):
    print(f"Chunk {i + 1}: simulating delegation...")
    total_factors["weighted_deposit"] += chunk
    if len(delegate) == 10:
        potential_delegatees = list(delegate)
    else:
        potential_delegatees = [benchmarker for benchmarker in opow_data if opow_data[benchmarker].block_data.self_deposit.to_float() >= 10000]
    highest_apy_benchmarker = max(
        potential_delegatees,
        key=lambda delegatee: (
            calc_influence({
                benchmarker: {
                    f: (factors[benchmarker][f] + chunk * (benchmarker == delegatee and f == "weighted_deposit")) / total_factors[f]
                    for f in total_factors
                }
                for benchmarker in opow_data
            }, block.config["opow"])[delegatee] * 
            reward_shares[delegatee] * chunk / (factors[delegatee]["weighted_deposit"] + chunk)
        )
    )
    print(f"Chunk {i + 1}: best delegatee is {highest_apy_benchmarker}")
    if highest_apy_benchmarker not in delegate:
        delegate[highest_apy_benchmarker] = 0
    delegate[highest_apy_benchmarker] += 1
    factors[highest_apy_benchmarker]["weighted_deposit"] += chunk

influences = calc_influence({
    benchmarker: {
        f: factors[benchmarker][f] / total_factors[f]
        for f in total_factors
    }
    for benchmarker in opow_data
}, block.config["opow"])
print("")


print("Optimised delegation split:")
reward_pool = block.config["rewards"]["distribution"]["opow"] * next(x["block_reward"] for x in reversed(block.config["rewards"]["schedule"]) if x["round_start"] <= block.details.round)
deposit_chunk = deposit / 100
total_reward = 0
for delegatee, num_chunks in delegate.items():
    share_fraction = influences[delegatee] * reward_shares[delegatee] * (num_chunks * chunk) / factors[delegatee]["weighted_deposit"]
    reward = share_fraction * reward_pool
    total_reward += reward
    apy = reward * block.config["rounds"]["blocks_per_round"] * 52 / (num_chunks * deposit_chunk)
    print(f"{delegatee}: %delegated = {num_chunks}%, apy = {apy * 100:.2f}%")

print(f"average_apy = {total_reward * 10080 * 52 / deposit * 100:.2f}% on your deposit of {deposit} TIG")

print("")
print("To set this delegation split, run the following command:")
req = {"delegatees": {k: v / 100 for k, v in delegate.items()}}
print("API_KEY=<YOUR API KEY HERE>")
print(f"curl -H \"X-Api-Key: $API_KEY\" -X POST -d '{json.dumps(req)}' {API_URL}/set-delegatees")