import numpy as np
from typing import List, Tuple

PADDING_FACTOR = 0.2
DECAY = 0.7
INITIAL_SOLUTIONS_WEIGHT = 500.0
SOLUTIONS_MULTIPLIER = 10.0

class DifficultySampler:
    def __init__(self):
        self.min_difficulty = np.array([], dtype=int)
        self.padding = np.array([], dtype=int)
        self.dimensions = np.array([], dtype=int)
        self.weights = np.array([])
        self.distribution = None

    def sample(self, rng):
        if self.distribution is None:
            raise ValueError("You must update sampler first")
        idx = rng.choice(len(self.distribution), p=self.distribution)
        num_cols = self.dimensions[1] + self.padding[1]
        x = idx // num_cols
        y = idx % num_cols
        return [x + self.min_difficulty[0], y + self.min_difficulty[1]]

    def update_with_block_data(self, min_difficulty: List[int], block_data):
        assert len(min_difficulty) == 2, "Only difficulty with 2 parameters are supported"

        left_pad = np.subtract(self.min_difficulty, min_difficulty, where=self.min_difficulty.size > 0)
        self.min_difficulty = np.array(min_difficulty)
        self.update_dimensions_and_padding(block_data)
        size = self.dimensions + self.padding
        self.resize_weights(left_pad, size)

        self.update_qualifier_weights(block_data)
        self.update_valid_range(block_data)
        self.update_distributions()

    def update_with_solutions(self, difficulty: List[int], num_solutions: int):
        x, y = np.subtract(difficulty, self.min_difficulty)
        if x < 0 or y < 0 or x >= self.dimensions[0] + self.padding[0] or y >= self.dimensions[1] + self.padding[1]:
            return

        for x_offset in range(self.padding[0]):
            for y_offset in range(self.padding[1]):
                dist = np.sqrt((x_offset / self.padding[0])**2 + (y_offset / self.padding[1])**2)
                if dist > 1.0:
                    break
                decay = dist * (1.0 - DECAY) + DECAY
                delta = (1.0 - decay) * num_solutions * SOLUTIONS_MULTIPLIER
                coords = [
                    (x + x_offset, y + y_offset),
                    (x - x_offset, y + y_offset),
                    (x + x_offset, y - y_offset),
                    (x - x_offset, y - y_offset)
                ]
                for cx, cy in coords:
                    if 0 <= cx < self.weights.shape[0] and 0 <= cy < self.weights.shape[1]:
                        self.weights[cx, cy, 1] *= decay
                        self.weights[cx, cy, 1] += delta

    def update_valid_range(self, block_data):
        lower_cutoff_points = np.subtract(block_data.base_frontier(), self.min_difficulty)
        upper_cutoff_points = np.subtract(block_data.scaled_frontier(), self.min_difficulty)
        
        if block_data.scaling_factor() < 1.0:
            lower_cutoff_points, upper_cutoff_points = upper_cutoff_points, lower_cutoff_points

        lower_cutoff_points = lower_cutoff_points[lower_cutoff_points[:, 0].argsort()]
        upper_cutoff_points = upper_cutoff_points[upper_cutoff_points[:, 0].argsort()]

        for i in range(self.weights.shape[0]):
            lower_cutoff = lower_cutoff_points[lower_cutoff_points[:, 0] <= i][-1]
            upper_cutoff1 = upper_cutoff_points[upper_cutoff_points[:, 0] <= i][-1]
            upper_cutoff2 = upper_cutoff_points[upper_cutoff_points[:, 0] > i][0] if i < upper_cutoff_points[-1, 0] else upper_cutoff1

            for j in range(self.weights.shape[1]):
                within_lower = j > lower_cutoff[1] or (j == lower_cutoff[1] and i >= lower_cutoff[0])
                within_upper = (j <= upper_cutoff2[1] and i <= upper_cutoff2[0]) or \
                               (j < upper_cutoff1[1] and i < upper_cutoff2[0]) or \
                               (j == upper_cutoff1[1] and i == upper_cutoff1[0])
                self.weights[i, j, 2] = within_lower and within_upper

    def update_distributions(self):
        self.distribution = np.where(self.weights[:, :, 2], 
                                     self.weights[:, :, 0] * self.weights[:, :, 1], 
                                     0).flatten()
        self.distribution /= self.distribution.sum()

    def update_qualifier_weights(self, block_data):
        cutoff_points = np.subtract(block_data.cutoff_frontier(), self.min_difficulty)
        cutoff_points = cutoff_points[cutoff_points[:, 0].argsort()]

        for i in range(self.weights.shape[0]):
            cutoff = cutoff_points[cutoff_points[:, 0] <= i][-1] if len(cutoff_points) > 0 else [0, 0]
            self.weights[i, :, 0] *= 0.9
            self.weights[i, cutoff[1]:, 0] += 0.1
            if i >= cutoff[0]:
                self.weights[i, cutoff[1], 0] += 0.1

    def resize_weights(self, left_pad: np.ndarray, size: np.ndarray):
        new_weights = np.full((*size, 3), fill_value=1.0)
        new_weights[:, :, 1] = INITIAL_SOLUTIONS_WEIGHT

        x_start = max(0, left_pad[0])
        y_start = max(0, left_pad[1])
        x_end = min(size[0], size[0] + left_pad[0])
        y_end = min(size[1], size[1] + left_pad[1])

        old_x_start = max(0, -left_pad[0])
        old_y_start = max(0, -left_pad[1])

        new_weights[x_start:x_end, y_start:y_end] = self.weights[
            old_x_start:old_x_start + x_end - x_start,
            old_y_start:old_y_start + y_end - y_start
        ]

        self.weights = new_weights

    def update_dimensions_and_padding(self, block_data):
        hardest_difficulty = np.maximum(
            np.maximum(
                block_data.scaled_frontier().max(axis=0),
                block_data.base_frontier().max(axis=0)
            ),
            block_data.qualifier_difficulties().max(axis=0) if len(block_data.qualifier_difficulties()) > 0 else -np.inf
        )
        self.dimensions = hardest_difficulty - self.min_difficulty + 1
        self.padding = np.ceil(self.dimensions * PADDING_FACTOR).astype(int)