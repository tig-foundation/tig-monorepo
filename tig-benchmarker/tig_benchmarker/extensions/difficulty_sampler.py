import asyncio
import logging
import numpy as np
import os
from tig_benchmarker.event_bus import *
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from typing import List, Tuple, Dict

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class DifficultySamplerConfig(FromDict):
    num_samples: int = 100
    padding_factor: float = 0.2
    decay: float = 0.7
    initial_solutions_weight: float = 500.0
    solutions_multiplier: float = 10.0

class DifficultySampler:
    def __init__(self, config: DifficultySamplerConfig):
        self.config = config
        self.min_difficulty = None
        self.padding = None
        self.dimensions = None
        self.weights = np.empty((0,0,2), dtype=float)
        self.distribution = None

    def sample(self):
        if self.distribution is None:
            raise ValueError("You must update sampler first")
        p = self.distribution.flatten()
        idx = np.random.choice(len(p), p=p)
        num_cols = self.dimensions[1] + self.padding[1]
        x = idx // num_cols
        y = idx % num_cols
        return [int(x + self.min_difficulty[0]), int(y + self.min_difficulty[1])]

    def update_with_block_data(self, min_difficulty: List[int], block_data):
        assert len(min_difficulty) == 2, "Only difficulty with 2 parameters are supported"
        min_difficulty = np.array(min_difficulty)

        if self.min_difficulty is None:
            left_pad = np.zeros(2, dtype=int)
        else:
            left_pad = min_difficulty - self.min_difficulty
        self.min_difficulty = min_difficulty
        self.update_dimensions_and_padding(block_data)
        size = self.dimensions + self.padding
        self.resize_weights(left_pad, size)

        self.update_valid_range(block_data)
        self.update_distributions()

    def update_with_solutions(self, difficulty: List[int], num_solutions: int):
        center = np.array(difficulty) - self.min_difficulty

        x_min = max(0, center[0] - self.padding[0])
        x_max = min(self.weights.shape[0] - 1, center[0] + self.padding[0])
        y_min = max(0, center[1] - self.padding[1])
        y_max = min(self.weights.shape[1] - 1, center[1] + self.padding[1])
        if x_min > x_max or y_min > y_max:
            return
        y, x = np.meshgrid(
            np.arange(y_min, y_max + 1, dtype=float),
            np.arange(x_min, x_max + 1, dtype=float)
        )
        position = np.stack((x, y), axis=-1)
        dist = np.linalg.norm((position - center) / self.padding, axis=-1)
        decay = dist * (1.0 - self.config.decay) + self.config.decay
        delta = (1.0 - decay) * num_solutions * self.config.solutions_multiplier
        decay[np.where(dist > 1.0)] = 1.0
        delta[np.where(dist > 1.0)] = 0.0

        self.weights[x_min:x_max + 1, y_min:y_max + 1, 1] *= decay
        self.weights[x_min:x_max + 1, y_min:y_max + 1, 1] += delta

    def update_valid_range(self, block_data):
        lower_cutoff_points = np.array(list(block_data.base_frontier)) - self.min_difficulty
        upper_cutoff_points = np.array(list(block_data.scaled_frontier)) - self.min_difficulty
        
        if block_data.scaling_factor < 1.0:
            lower_cutoff_points, upper_cutoff_points = upper_cutoff_points, lower_cutoff_points

        lower_cutoff_points = lower_cutoff_points[np.argsort(lower_cutoff_points[:, 0]), :]
        upper_cutoff_points = upper_cutoff_points[np.argsort(upper_cutoff_points[:, 0]), :]
        lower_cutoff_idx = 0
        lower_cutoff = lower_cutoff_points[lower_cutoff_idx]
        upper_cutoff_idx = 0
        upper_cutoff1 = upper_cutoff_points[upper_cutoff_idx]
        if len(upper_cutoff_points) > 1:
            upper_cutoff2 = upper_cutoff_points[upper_cutoff_idx + 1]
        else:
            upper_cutoff2 = upper_cutoff1
        self.weights[:, :, 0] = 0.0
        for i in range(self.weights.shape[0]):
            if lower_cutoff_idx + 1 < len(lower_cutoff_points) and i == lower_cutoff_points[lower_cutoff_idx + 1, 0]:
                lower_cutoff_idx += 1
                lower_cutoff = lower_cutoff_points[lower_cutoff_idx]
            if upper_cutoff_idx + 1 < len(upper_cutoff_points) and i == upper_cutoff_points[upper_cutoff_idx + 1, 0]:
                upper_cutoff_idx += 1
                upper_cutoff1 = upper_cutoff_points[upper_cutoff_idx]
                if upper_cutoff_idx + 1 < len(upper_cutoff_points):
                    upper_cutoff2 = upper_cutoff_points[upper_cutoff_idx + 1]
                else:
                    upper_cutoff2 = upper_cutoff1

            if i >= lower_cutoff[0]:
                start = lower_cutoff[1]
            else:
                start = lower_cutoff[1] + 1
            if i <= upper_cutoff2[0]:
                self.weights[i, start:upper_cutoff2[1] + 1, 0] = 1.0
            if i < upper_cutoff2[0]:
                self.weights[i, start:upper_cutoff1[1], 0] = 1.0
            if i == upper_cutoff1[0]:
                self.weights[i, upper_cutoff1[1], 0] = 1.0

    def update_distributions(self):
        distribution = np.prod(self.weights, axis=2)
        distribution /= np.sum(distribution)
        self.distribution = distribution

    def resize_weights(self, left_pad: np.ndarray, size: np.ndarray):
        default_values = [0.0, self.config.initial_solutions_weight]
        if left_pad[0] > 0:
            pad = np.full((left_pad[0], self.weights.shape[1], 2), default_values)
            self.weights = np.vstack((pad, self.weights))
        elif left_pad[0] < 0:
            self.weights = self.weights[-left_pad[0]:, :, :]
        if left_pad[1] > 0:
            pad = np.full((self.weights.shape[0], left_pad[1], 2), default_values)
            self.weights = np.hstack((pad, self.weights))
        elif left_pad[1] < 0:
            self.weights = self.weights[:, -left_pad[1]:, :]
        right_pad = size - self.weights.shape[:2]
        if right_pad[0] > 0:
            pad = np.full((right_pad[0], self.weights.shape[1], 2), default_values)
            self.weights = np.vstack((self.weights, pad))
        elif right_pad[0] < 0:
            self.weights = self.weights[:size[0], :, :]
        if right_pad[1] > 0:
            pad = np.full((self.weights.shape[0], right_pad[1], 2), default_values)
            self.weights = np.hstack((self.weights, pad))
        elif right_pad[1] < 0:
            self.weights = self.weights[:, :size[1], :]

    def update_dimensions_and_padding(self, block_data):
        hardest_difficulty = np.max([
            np.max(list(block_data.scaled_frontier), axis=0),
            np.max(list(block_data.base_frontier), axis=0),
        ], axis=0)
        if block_data.qualifier_difficulties is not None and len(block_data.qualifier_difficulties) > 0:
            hardest_difficulty = np.max([
                hardest_difficulty, 
                np.max(list(block_data.qualifier_difficulties), axis=0)
            ], axis=0)
        self.dimensions = hardest_difficulty - self.min_difficulty + 1
        self.padding = np.ceil(self.dimensions * self.config.padding_factor).astype(int)

class Extension:
    def __init__(self, difficulty_sampler: dict, **kwargs):
        self.config = DifficultySamplerConfig.from_dict(difficulty_sampler)
        self.samplers = {}
        self.lock = True

    async def on_new_block(
        self, 
        block: Block, 
        challenges: Dict[str, Challenge],
        **kwargs # ignore other data
    ):
        logger.debug("new block, updating difficulty samplers")
        for challenge in challenges.values():
            if challenge.block_data is None:
                continue
            if challenge.id not in self.samplers:
                self.samplers[challenge.id] = (challenge.details.name, DifficultySampler(self.config))
            logger.debug(f"updating sampler for challenge {challenge.details.name}")
            self.samplers[challenge.id][1].update_with_block_data(
                min_difficulty=[
                    param["min_value"]
                    for param in block.config["difficulty"]["parameters"][challenge.id]
                ],
                block_data=challenge.block_data
            )
            logger.info(f"emitting {self.config.num_samples} difficulty samples for challenge {challenge.details.name}")
            await emit(
                "difficulty_samples", 
                challenge_id=challenge.id,
                block_id=block.id, 
                samples=[self.samplers[challenge.id][1].sample() for _ in range(self.config.num_samples)]
            )
        self.lock = False

    async def on_benchmark_confirmed(self, precommit: Precommit, benchmark: Benchmark):
        while self.lock:
            await asyncio.sleep(0.5)
        challenge_id = precommit.settings.challenge_id
        num_solutions = benchmark.details.num_solutions
        if challenge_id not in self.samplers:
            logger.warning(f"no sampler for challenge {challenge_id}")
        else:
            challenge_name, sampler = self.samplers[challenge_id]
            logger.info(f"updating sampler for challenge {challenge_name} with {num_solutions} solutions @ difficulty {precommit.settings.difficulty}")
            sampler.update_with_solutions(
                difficulty=precommit.settings.difficulty,
                num_solutions=num_solutions
            )