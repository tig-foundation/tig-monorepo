import logging
import math
import numpy as np
import os
import random
from common.structs import *
from typing import List, Tuple, Dict
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

Point = List[int]
Frontier = List[Point]

def calc_valid_difficulties(upper_frontier: List[Point], lower_frontier: List[Point]) -> List[Point]:
    """
    Calculates a list of all difficulty combinations within the base and scaled frontiers
    """
    hardest_difficulty = np.max(upper_frontier, axis=0)
    min_difficulty = np.min(lower_frontier, axis=0)

    weights = np.zeros(hardest_difficulty - min_difficulty + 1, dtype=float)
    lower_cutoff_points = np.array(lower_frontier) - min_difficulty
    upper_cutoff_points = np.array(upper_frontier) - min_difficulty

    lower_cutoff_points = lower_cutoff_points[np.argsort(lower_cutoff_points[:, 0]), :]
    upper_cutoff_points = upper_cutoff_points[np.argsort(upper_cutoff_points[:, 0]), :]
    lower_cutoff_idx = 0
    lower_cutoff1 = lower_cutoff_points[lower_cutoff_idx]
    if len(lower_cutoff_points) > 1:
        lower_cutoff2 = lower_cutoff_points[lower_cutoff_idx + 1]
    else:
        lower_cutoff2 = lower_cutoff1
    upper_cutoff_idx = 0
    upper_cutoff1 = upper_cutoff_points[upper_cutoff_idx]
    if len(upper_cutoff_points) > 1:
        upper_cutoff2 = upper_cutoff_points[upper_cutoff_idx + 1]
    else:
        upper_cutoff2 = upper_cutoff1

    for i in range(weights.shape[0]):
        if lower_cutoff_idx + 1 < len(lower_cutoff_points) and i == lower_cutoff_points[lower_cutoff_idx + 1, 0]:
            lower_cutoff_idx += 1
            lower_cutoff1 = lower_cutoff_points[lower_cutoff_idx]
            if lower_cutoff_idx + 1 < len(lower_cutoff_points):
                lower_cutoff2 = lower_cutoff_points[lower_cutoff_idx + 1]
            else:
                lower_cutoff2 = lower_cutoff1
        if upper_cutoff_idx + 1 < len(upper_cutoff_points) and i == upper_cutoff_points[upper_cutoff_idx + 1, 0]:
            upper_cutoff_idx += 1
            upper_cutoff1 = upper_cutoff_points[upper_cutoff_idx]
            if upper_cutoff_idx + 1 < len(upper_cutoff_points):
                upper_cutoff2 = upper_cutoff_points[upper_cutoff_idx + 1]
            else:
                upper_cutoff2 = upper_cutoff1
        if i > lower_cutoff1[0] and lower_cutoff1[0] != lower_cutoff2[0]:
            start = lower_cutoff2[1] + 1
        else:
            start = lower_cutoff1[1]
        if i <= upper_cutoff2[0]:
            weights[i, start:upper_cutoff2[1] + 1] = 1.0
        if i < upper_cutoff2[0]:
            weights[i, start:upper_cutoff1[1]] = 1.0
        if i == upper_cutoff1[0]:
            weights[i, upper_cutoff1[1]] = 1.0

    valid_difficulties = np.stack(np.where(weights), axis=1) + min_difficulty
    return valid_difficulties.tolist()

def calc_pareto_frontier(points: List[Point]) -> Frontier:
    """
    Calculates a single Pareto frontier from a list of points
    Adapted from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    points_ = points
    points = np.array(points)
    frontier_idxs = np.arange(points.shape[0])
    n_points = points.shape[0]
    next_point_index = 0  # Next index in the frontier_idxs array to search for
    while next_point_index < len(points):
        nondominated_point_mask = np.any(points < points[next_point_index], axis=1)
        nondominated_point_mask[np.all(points == points[next_point_index], axis=1)] = True
        frontier_idxs = frontier_idxs[nondominated_point_mask]  # Remove dominated points
        points = points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    return [points_[idx] for idx in frontier_idxs]

def calc_all_frontiers(points: List[Point]) -> List[Frontier]:
    """
    Calculates a list of Pareto frontiers from a list of points
    """
    buckets = {}
    r = np.max(points, axis=0) - np.min(points, axis=0) 
    dim1, dim2 = (1, 0) if r[0] > r[1] else (0, 1)
    for p in points:
        if p[dim1] not in buckets:
            buckets[p[dim1]] = []
        buckets[p[dim1]].append(p)
    for bucket in buckets.values():
        bucket.sort(reverse=True, key=lambda x: x[dim2])
    frontiers = []
    while len(buckets) > 0:
        points = [bucket[-1] for bucket in buckets.values()]
        frontier = calc_pareto_frontier(points)
        for p in frontier:
            x = p[dim1]
            buckets[x].pop()
            if len(buckets[x]) == 0:
                buckets.pop(x)
        frontiers.append(frontier)
    return frontiers

class DifficultySampler:
    def __init__(self):
        self.valid_difficulties = {}
        self.frontiers = {}
        self.challenges = {}

    def on_new_block(self, challenges: Dict[str, Challenge], **kwargs):
        for c in challenges.values():
            if c.block_data is None:
                continue
            logger.debug(f"Calculating valid difficulties and frontiers for challenge {c.details.name}")
            if c.block_data.scaling_factor >= 1:
                upper_frontier, lower_frontier = c.block_data.scaled_frontier, c.block_data.base_frontier
            else:
                upper_frontier, lower_frontier = c.block_data.base_frontier, c.block_data.scaled_frontier
            self.valid_difficulties[c.details.name] = calc_valid_difficulties(list(upper_frontier), list(lower_frontier))
            self.frontiers[c.details.name] = calc_all_frontiers(self.valid_difficulties[c.details.name])

        self.challenges = [c.details.name for c in challenges.values()]

    def run(self) -> Dict[str, Point]:
        samples = {}
        config = CONFIG["difficulty_sampler_config"]

        for c_name in self.challenges:
            found_valid = False

            if len(selected_difficulties := config["selected_difficulties"].get(c_name, [])) > 0:
                valid_difficulties = set(tuple(d) for d in self.valid_difficulties[c_name])
                selected_difficulties = [tuple(d) for d in selected_difficulties]
                selected_difficulties = [
                    d for d in selected_difficulties if d in valid_difficulties
                ]
            
            if len(selected_difficulties) > 0:
                random.shuffle(selected_difficulties)
                samples[c_name] = selected_difficulties[0]
                logger.debug(f"Selected difficulty {samples[c_name]} for challenge {c_name}")
                found_valid = True
            else:
                logger.debug(f"No valid difficulties found for {c_name} - skipping selected difficulties")

            if not found_valid:
                frontiers = self.frontiers[c_name]
                difficulty_range = config["difficulty_ranges"][c_name]
                idx1 = math.floor(difficulty_range[0] * (len(frontiers) - 1))
                idx2 = math.ceil(difficulty_range[1] * (len(frontiers) - 1))
                difficulties = [p for frontier in frontiers[idx1:idx2 + 1] for p in frontier]
                difficulty = random.choice(difficulties)
                samples[c_name] = difficulty
                logger.debug(f"Sampled difficulty {difficulty} for challenge {c_name}")
                
        return samples