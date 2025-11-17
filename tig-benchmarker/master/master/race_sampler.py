import logging
import os
import random
from common.structs import *
from typing import List, Dict
from master.client_manager import CONFIG

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

Point = List[int]

class RaceSampler:
    def __init__(self):
        self.valid_size_ranges = {}

    def on_new_block(self, block: Block, **kwargs):
        self.active_race_ids = {
            c_id: d["active_race_ids"]
            for c_id, d in block.config["challenges"].items()
        }

    def run(self) -> Dict[str, int]:
        samples = {}

        for config in CONFIG["algo_selection"]:
            a_id = config["algorithm_id"]
            c_id = a_id[:4]
            active_races = self.active_race_ids[c_id]

            selected_races = sorted(set(config["selected_races"]) & set(active_races))
            if len(selected_races) == 0:
                selected_races = list(active_races)
            config["selected_races"] = selected_races

            samples[a_id] = random.choice(selected_races)
            logger.debug(f"Selected race '{samples[a_id]}' for algorithm {a_id} in challenge {c_id}")
                
        return samples