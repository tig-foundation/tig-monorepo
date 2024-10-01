import asyncio
import os
import logging
import random
import time
from dataclasses import dataclass
from tig_benchmarker.event_bus import *
from tig_benchmarker.structs import *
from tig_benchmarker.utils import FromDict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class AlgorithmSelectionConfig(FromDict):
    algorithm: str
    num_nonces: int
    base_fee_limit: PreciseNumber
    weight: float

@dataclass
class PrecommitManagerConfig(FromDict):
    max_precommits_per_block: int
    algo_selection: Dict[str, AlgorithmSelectionConfig]

class Extension:
    def __init__(self, player_id: str, backup_folder: str, precommit_manager: dict, **kwargs):
        self.player_id = player_id
        self.config = PrecommitManagerConfig.from_dict(precommit_manager)
        self.num_precommits = 0
        self.last_block_id = None
        self.curr_base_fees = {}
        self.difficulty_samples = {}
        self.algorithm_name_2_id = {}
        self.challenge_name_2_id = {}
        self.lock = True

    async def on_new_block(self, block: Block, challenges: Dict[str, Challenge], algorithms: Dict[str, Algorithm], **kwargs):
        if self.last_block_id != block.id:
            self.num_precommits_submitted = 0
            self.last_block_id = block.id
        for c in challenges.values():
            c_name = c.details.name
            assert c_name in self.config.algo_selection, f"missing algorithm selection for challenge {c_name}"
            self.challenge_name_2_id[c_name] = c.id
            a_name = self.config.algo_selection[c_name].algorithm
            a = next(
                (
                    a for a in algorithms.values()
                    if a.details.challenge_id == c.id and a.details.name == a_name
                ),
                None
            )
            assert a is not None, f"selected non-existent algorithm {a_name} for challenge {c_name}"
            self.algorithm_name_2_id[f"{c_name}_{a_name}"] = a.id
            if c.block_data is not None:
                self.curr_base_fees[c_name] = c.block_data.base_fee
        logger.info(f"current base fees: {self.curr_base_fees}")
        self.lock = False

    async def on_difficulty_samples(self, challenge_id: str, samples: list, **kwargs):
        self.difficulty_samples[challenge_id] = samples

    async def on_update(self):
        if self.lock or self.num_precommits_submitted >= self.config.max_precommits_per_block:
            return
        algo_selection = [
            (c_name, selection)
            for c_name, selection in self.config.algo_selection.items()
            if self.curr_base_fees[c_name] <= selection.base_fee_limit
        ]
        if len(algo_selection) == 0:
            logger.info("no challenges within base fee limit")
            return
        logger.debug(f"randomly selecting a challenge + algorithm from {self.config.algo_selection}")
        selection = random.choices(algo_selection, weights=[x[1].weight for x in algo_selection])[0]
        c_name = selection[0]
        c_id = self.challenge_name_2_id[c_name]
        a_name = selection[1].algorithm
        a_id = self.algorithm_name_2_id[f"{c_name}_{a_name}"]
        num_nonces = selection[1].num_nonces
        difficulty_samples = self.difficulty_samples.get(c_id, [])
        if len(difficulty_samples) == 0:
            return
        difficulty = difficulty_samples.pop()
        self.num_precommits_submitted += 1
        logger.info(f"creating precommit {self.num_precommits_submitted} of {self.config.max_precommits_per_block} for block {self.last_block_id}: (challenge: {c_name}, algorithm: {a_name}, difficulty: {difficulty}, num_nonces: {num_nonces})")
        await emit(
            "precommit_ready",
            settings=BenchmarkSettings(
                player_id=self.player_id,
                block_id=self.last_block_id,
                algorithm_id=a_id,
                challenge_id=c_id,
                difficulty=difficulty
            ),
            num_nonces=num_nonces
        )
        