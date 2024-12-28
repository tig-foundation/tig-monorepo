import numpy as np
from typing import List, Dict


def calc_influence(fractions: Dict[str, Dict[str, float]], opow_config: dict) -> Dict[str, float]:
    """
    Calculate the influence of each benchmarker based on their fractions and the OPoW configuration.

    Args:
        fractions: A dictionary of dictionaries, mapping benchmarker_ids to their fraction of each factor (challenges & weighted_deposit).
        opow_config: A dictionary containing configuration parameters for the calculation.

    Returns:
        Dict[str, float]: A dictionary mapping each benchmarker_id to their calculated influence.
    """
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


def calc_weighted_deposit(deposit: float, seconds_till_round_end: int, lock_seconds: int) -> float:
    """
    Calculate weighted deposit
    
    Args:
        deposit: Amount to deposit
        seconds_till_round_end: Seconds remaining in current round
        lock_seconds: Total lock duration in seconds
    
    Returns:
        Weighted deposit
    """
    weighted_deposit = 0
    
    if lock_seconds <= 0:
        return weighted_deposit
        
    # Calculate first chunk (partial week)
    weighted_deposit += deposit * min(seconds_till_round_end, lock_seconds) // lock_seconds

    remaining_seconds = lock_seconds - min(seconds_till_round_end, lock_seconds)
    weight = 2
    while remaining_seconds > 0:
        chunk_seconds = min(remaining_seconds, 604800)
        chunk = deposit * chunk_seconds // lock_seconds
        weighted_deposit += chunk * weight
        remaining_seconds -= chunk_seconds
        weight = min(weight + 1, 26)
    
    return weighted_deposit