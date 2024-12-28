import numpy as np
import json
import requests
from typing import List, Dict
from common.structs import *
from copy import deepcopy


def calc_influence(fractions, opow_config) -> Dict[str, float]:
    benchmarkers = list(fractions)
    factors = list(next(iter(fractions.values())))
    num_challenges = len(factors) - 1
    avg_qualifier_fractions = {
        benchmarker: sum(
            fractions[benchmarker][f]
            for f in factors
            if f != "weighted_deposit"
        ) / num_challenges
        for benchmarker in benchmarkers
    }
    deposit_fraction_cap = {
        benchmarker: avg_qualifier_fractions[benchmarker] * opow_config["max_deposit_to_qualifier_ratio"]
        for benchmarker in benchmarkers
    }
    capped_fractions = {
        benchmarker: {
            **fractions[benchmarker], 
            "weighted_deposit": min(
                fractions[benchmarker]["weighted_deposit"], 
                deposit_fraction_cap[benchmarker]
            )
        }
        for benchmarker in benchmarkers
    }
    avg_fraction = {
        benchmarker: np.mean(list(capped_fractions[benchmarker].values()))
        for benchmarker in benchmarkers
    }
    var_fraction = {
        benchmarker: np.var(list(capped_fractions[benchmarker].values()))
        for benchmarker in benchmarkers
    }
    imbalance = {
        benchmarker: (var_fraction[benchmarker] / np.square(avg_fraction[benchmarker]) / num_challenges) if avg_fraction[benchmarker] > 0 else 0
        for benchmarker in benchmarkers
    }
    imbalance_penalty = {
        benchmarker: 1.0 - np.exp(-opow_config["imbalance_multiplier"] * imbalance[benchmarker])
        for benchmarker in benchmarkers
    }
    weighted_avg_fraction = {
        benchmarker: ((avg_qualifier_fractions[benchmarker] * num_challenges) + capped_fractions[benchmarker]["weighted_deposit"] * opow_config["deposit_multiplier"]) / (num_challenges + opow_config["deposit_multiplier"])
        for benchmarker in benchmarkers
    }
    unormalised_influence = {
        benchmarker: weighted_avg_fraction[benchmarker] * (1.0 - imbalance_penalty[benchmarker])
        for benchmarker in benchmarkers
    }
    total = sum(unormalised_influence.values())
    influence = {
        benchmarker: unormalised_influence[benchmarker] / total
        for benchmarker in benchmarkers
    }
    return influence

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
            f: opow_data[benchmarker].block_data.num_qualifiers_by_challenge.get(f, 0)
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
print(f"reward = {reward['current']:.4f} TIG per block ({reward['current'] / reward['only_self_deposit'] * 100 - 100:.2f}% difference*)")
print("")
print(f"Scenario 3 (self + delegated deposit at parity)")
print(f"%weighted_deposit = average %qualifiers = {average_fraction_of_qualifiers * 100:.2f}%")
print(f"reward = {reward['at_parity']:.4f} TIG per block ({reward['at_parity'] / reward['only_self_deposit'] * 100 - 100:.2f}% difference*)")
print("")
print("*These are percentage differences in reward compared with relying only on self-deposit (Scenario 1).")
print("")
print("Note: the imbalance penalty is such that your reward increases at a high rate when moving up to parity")