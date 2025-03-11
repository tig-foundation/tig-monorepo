import numpy as np
import requests
from common.structs import *
from common.calcs import *
from copy import deepcopy


API_URL = "https://mainnet-api.tig.foundation"

player_id = input("Enter player_id: ").lower()
print("Fetching data...")
block = Block.from_dict(requests.get(f"{API_URL}/get-block?include_data").json()["block"])
opow_data = {
    x["player_id"]: OPoW.from_dict(x)
    for x in requests.get(f"{API_URL}/get-opow?block_id={block.id}").json()["opow"]
}
player_data = Player.from_dict(requests.get(f"{API_URL}/get-player-data?player_id={player_id}&block_id={block.id}").json()["player"])

factors = {
    benchmarker: {
        **{
            f: (
                opow_data[benchmarker].block_data.num_qualifiers_by_challenge.get(f, 0) * 
                opow_data[benchmarker].block_data.solution_ratio_by_challenge.get(f, 0)
            )
            for f in block.data.active_ids["challenge"]
        },
        "weighted_deposit": opow_data[benchmarker].block_data.delegated_weighted_deposit.to_float()
    }
    for benchmarker in opow_data
}
total_factors = {
    f: sum(factors[benchmarker][f] for benchmarker in opow_data)
    for f in list(block.data.active_ids["challenge"]) + ["weighted_deposit"]
}
fractions = {
    benchmarker: {
        f: factors[benchmarker][f] / total_factors[f]
        for f in total_factors
    }
    for benchmarker in opow_data
}
sum_other_deposit_fractions = sum(
    fractions[benchmarker]["weighted_deposit"] for benchmarker in opow_data if benchmarker != player_id
)

def simulate_deposit_fraction(deposit_fraction):
    delta = fractions[player_id]["weighted_deposit"] - deposit_fraction
    fractions2 = deepcopy(fractions)
    for benchmarker in opow_data:
        if benchmarker != player_id:
            fractions2[benchmarker]["weighted_deposit"] += delta * fractions[benchmarker]["weighted_deposit"] / sum_other_deposit_fractions
        else:
            fractions2[benchmarker]["weighted_deposit"] = deposit_fraction
    influence = calc_influence(fractions2, block.config["opow"])
    return influence[player_id]


print("Calculating rewards...")
average_fraction_of_qualifiers = sum(fractions[player_id][f] for f in block.data.active_ids["challenge"]) / len(block.data.active_ids["challenge"])
self_only_deposit_fraction = player_data.block_data.weighted_deposit.to_float() / total_factors["weighted_deposit"]
influence = {
    "current": opow_data[player_id].block_data.influence.to_float(),
    "only_self_deposit": simulate_deposit_fraction(self_only_deposit_fraction),
    "at_parity": simulate_deposit_fraction(average_fraction_of_qualifiers)
}

current_reward = opow_data[player_id].block_data.reward.to_float()
reward = {
    "current": current_reward,
    "only_self_deposit": influence["only_self_deposit"] / influence["current"] * current_reward,
    "at_parity": influence["at_parity"] / influence["current"] * current_reward
}

print("-----Current State-----")
for f in sorted(block.data.active_ids["challenge"]):
    print(f"%qualifiers for {f}: {fractions[player_id][f] * 100:.2f}%")
print(f"average %qualifiers: {average_fraction_of_qualifiers * 100:.2f}%")
print(f"%weighted_deposit (self only): {self_only_deposit_fraction * 100:.2f}%")
print(f"%weighted_deposit (current self + delegated): {fractions[player_id]['weighted_deposit'] * 100:.2f}%")
print("----------------------")
print(f"Scenario 1 (only self-deposit)")
print(f"%weighted_deposit = {self_only_deposit_fraction * 100:.2f}%")
print(f"reward = {reward['only_self_deposit']:.4f} TIG per block")
print("")
print("Scenario 2 (current self + delegated deposit)")
print(f"%weighted_deposit = {fractions[player_id]['weighted_deposit'] * 100:.2f}%")
print(f"reward = {reward['current']:.4f} TIG per block (recommended max reward_share = {100 - reward['only_self_deposit'] / reward['current'] * 100:.2f}%*)")
print("")
print(f"Scenario 3 (self + delegated deposit at parity)")
print(f"%weighted_deposit = average %qualifiers = {average_fraction_of_qualifiers * 100:.2f}%")
print(f"reward = {reward['at_parity']:.4f} TIG per block (recommended max reward_share = {100 - reward['only_self_deposit'] / reward['at_parity'] * 100:.2f}%*)")
print("")
print("*Recommend not setting reward_share above the max. You will not benefit from delegation (earn the same as Scenario 1 with zero delegation).")
print("")
print("Note: the imbalance penalty is such that your reward increases at a high rate when moving up to parity")