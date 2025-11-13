import logging
import os
import random
from common.structs import *
from typing import List, Dict
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

Point = List[int]

class SizeSampler:
    def __init__(self):
        self.valid_size_ranges = {}

    def on_new_block(self, block: Block, **kwargs):
        self.allowed_sizes = {
            c_id: d["difficulty"]["allowed_sizes"]
            for c_id, d in block.config["challenges"].items()
        }

    def run(self) -> Dict[str, int]:
        samples = {}

        for config in CONFIG["algo_selection"]:
            a_id = config["algorithm_id"]
            c_id = a_id[:4]
            allowed_sizes = self.allowed_sizes[c_id]

            selected_sizes = list(set(config["selected_sizes"]) & set(allowed_sizes))
            if len(selected_sizes) == 0:
                selected_sizes = list(allowed_sizes)
            config["selected_sizes"] = selected_sizes

            samples[a_id] = random.choice(selected_sizes)
            logger.debug(f"Selected size {samples[a_id]} for algorithm {a_id} in challenge {c_id}")
                
        return samples