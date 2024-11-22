import asyncio
import logging
import math
import numpy as np
import os
import random
from tig_benchmarker.structs import *
from typing import List, Tuple, Dict

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

def calc_pareto_frontier(points: List[Point]) -> Tuple[Frontier, List[bool]]:
    if not points:
        return [], []

    indices                     = list(range(len(points)))
    indices.sort(key=lambda i: (points[i][1], points[i][0]))
    
    on_front                    = [True] * len(points)
    stack                       = []
    
    for curr_idx in indices:
        while stack and points[stack[-1]][0] > points[curr_idx][0]:
            stack.pop()

        if stack and points[stack[-1]][0] <= points[curr_idx][0]:
            on_front[curr_idx]  = False
            
        stack.append(curr_idx)

    i                           = 0
    while i < len(indices):
        j                       = i + 1
        while j < len(indices) and points[indices[j]][1] == points[indices[i]][1]:
            j                   += 1

        if j - i > 1:
            min_x_idx           = min(indices[i:j], key=lambda k: points[k][0])
            for k in indices[i:j]:
                if k != min_x_idx:
                    on_front[k] = False
                    
        i                       = j
    
    frontier                    = [points[i] for i in range(len(points)) if on_front[i]]
    
    return frontier, on_front


def calc_all_frontiers(points: List[Point]) -> List[Frontier]:
    """
    Calculates a list of Pareto frontiers from a list of points
    """
    if not points:
        return []
    
    frontiers                   = []
    remaining_points            = None
    
    while True:
        points_                 = remaining_points if remaining_points is not None else points
        frontier, on_front      = calc_pareto_frontier(points_)

        frontiers.append(frontier)
        
        # Get remaining points not on frontier
        remaining_points        = [points_[i] for i in range(len(points_)) if not on_front[i]]
        
        # Break if no more points to process
        if not remaining_points:
            break
            
    return frontiers

@dataclass
class DifficultySamplerConfig(FromDict):
    difficulty_ranges: Dict[str, Tuple[float, float]]

    def __post_init__(self):
        for c_name, (start, end) in self.difficulty_ranges.items():
            if start < 0 or start > 1 or end < 0 or end > 1 or start > end:
                raise ValueError(f"Invalid difficulty range for challenge {c_name}. Must be (start, end) where '0 <= start <= end <= 1'")

class DifficultySampler:
    def __init__(self, config: DifficultySamplerConfig):
        self.config = config
        self.valid_difficulties = {}
        self.frontiers = {}

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

    def run(self) -> Dict[str, Point]:
        samples = {}
        for c_name, frontiers in self.frontiers.items():
            difficulty_range = self.config.difficulty_ranges[c_name] # FIXME
            idx1 = math.floor(difficulty_range[0] * (len(frontiers) - 1))
            idx2 = math.ceil(difficulty_range[1] * (len(frontiers) - 1))
            difficulties = [p for frontier in frontiers[idx1:idx2 + 1] for p in frontier]
            difficulty = random.choice(difficulties)
            samples[c_name] = difficulty
            logger.debug(f"Sampled difficulty {difficulty} for challenge {c_name}")
        return samples