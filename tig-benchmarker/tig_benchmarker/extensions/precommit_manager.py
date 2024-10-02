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
    base_fee_limit: PreciseNumber
    weight: float
    initial_num_nonces: int
    max_num_nonces: int
    target_duration: int

@dataclass
class PrecommitManagerConfig(FromDict):
    max_precommits_per_block: int
    max_unresolved_precommits: int
    algo_selection: Dict[str, AlgorithmSelectionConfig]

class Extension:
    def __init__(self, player_id: str, backup_folder: str, precommit_manager: dict, **kwargs):
        self.player_id = player_id
        self.config = PrecommitManagerConfig.from_dict(precommit_manager)
        self.num_precommits = 0
        self.last_block_id = None
        self.num_unresolved_precommits = None
        self.curr_base_fees = {}
        self.difficulty_samples = {}
        self.algorithm_name_2_id = {}
        self.challenge_name_2_id = {}
        self.performance_data = {}
        self.lock = True

    async def on_new_block(
        self, 
        block: Block, 
        challenges: Dict[str, Challenge], 
        algorithms: Dict[str, Algorithm], 
        precommits: Dict[str, Precommit],
        proofs: Dict[str, Proof],
        **kwargs
    ):
        if self.last_block_id == block.id:
            return
        self.num_precommits_submitted = 0
        self.num_unresolved_precommits = sum(1 for benchmark_id in precommits if benchmark_id not in proofs)
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

    async def on_precommit_confirmed(self, precommit: Precommit, **kwargs):
        self.performance_data[precommit.benchmark_id] = [
            precommit.settings.challenge_id,
            precommit.settings.difficulty,
            precommit.details.num_nonces,
            int(time.time() * 1000),
            None
        ]

    async def on_benchmark_ready(self, benchmark_id: str, **kwargs):
        if benchmark_id in self.performance_data:
            self.performance_data[benchmark_id][-1] = int(time.time() * 1000)

    async def on_difficulty_samples(self, challenge_id: str, samples: list, **kwargs):
        self.difficulty_samples[challenge_id] = samples

    async def on_update(self):
        if self.lock:
            return
        if self.num_precommits_submitted >= self.config.max_precommits_per_block:
            logger.info(f"reached max precommits per block: {self.config.max_precommits_per_block}")
            return
        if self.num_unresolved_precommits >= self.config.max_unresolved_precommits:
            logger.info(f"reached max unresolved precommits: {self.config.max_precommits_per_block}")
            return
        now = int(time.time() * 1000)
        prune_benchmark_ids = [
            benchmark_id
            for benchmark_id, data in self.performance_data.items()
            if now - data[3] >= 3600000 # 1 hr
        ]
        for benchmark_id in prune_benchmark_ids:
            self.performance_data.pop(benchmark_id, None)
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

        historic_data = [
            data
            for data in self.performance_data.values()
            if data[0] == c_id and data[4] is not None
        ]
        if len(historic_data) == 0:
            logger.info(f"insufficient historic data for challenge {c_name}. defaulting to initial num nonces {selection[1].initial_num_nonces}")
            num_nonces = selection[1].initial_num_nonces
        else:
            average_duration_per_nonce = sum([
                (data[4] - data[3]) / data[2]
                for data in historic_data
            ]) / len(historic_data)
            target_num_nonces = selection[1].target_duration / average_duration_per_nonce
            logger.info(f"average duration per nonce for challenge {c_name}: {average_duration_per_nonce}ms, target num nonces: {target_num_nonces}")
            num_nonces = min(int(target_num_nonces), selection[1].max_num_nonces)

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
        